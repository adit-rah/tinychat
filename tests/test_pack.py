"""Pack format v1 tests — roundtrip, BitLinear fidelity, forward parity, and the
reconciliation of shipped file size against count_bytes (spec §6)."""

import json
import os
import struct

import numpy as np
import pytest
import torch

from nanofable.bitlinear import BitLinear
from nanofable.bytes import count_bytes
from nanofable.config import TIERS, ModelConfig
from nanofable.model import build_model
from nanofable.pack import (
    ALIGN,
    MAGIC,
    _align,
    expected_count_bytes,
    pack_model,
    pack_trits,
    read_pack,
    recover_ternary,
    state_dict_from_pack,
    tensor_names,
    unpack_trits,
)

_REPO = os.path.join(os.path.dirname(__file__), "..")


def _seeded_model(precision, cfg=TIERS["tiny"], seed=0):
    torch.manual_seed(seed)
    return build_model(cfg, precision)


# --- 1. trit packing roundtrip ---------------------------------------------------

@pytest.mark.parametrize("numel", [5, 100, 7, 1, 851968 % 1000 + 3])
def test_trit_roundtrip(numel):
    rng = np.random.default_rng(numel)
    trits = rng.integers(-1, 2, size=numel).astype(np.int8)
    packed = pack_trits(trits)
    assert len(packed) == -(-numel // 5)  # ceil(n/5)
    assert max(packed) <= 242
    assert np.array_equal(unpack_trits(packed, numel), trits)


# --- 2. header integrity ----------------------------------------------------------

def test_header_integrity():
    m = _seeded_model("ternary")
    data = pack_model(m.state_dict(), m.cfg, "ternary")

    assert data[:4] == MAGIC
    (header_len,) = struct.unpack_from("<I", data, 4)
    header = json.loads(data[8:8 + header_len])
    assert header["format"] == "nanofable-pack" and header["version"] == 1
    assert header["precision"] == "ternary" and header["tied_head"] is True
    assert header["config"] == {"n_layer": 4, "n_embd": 128, "n_head": 4,
                                "ctx": 512, "vocab": 4096}

    tensors = header["tensors"]
    assert [t["name"] for t in tensors] == tensor_names(m.cfg)
    assert all("lm_head" not in t["name"] for t in tensors)
    end = 0
    for t in tensors:
        assert t["offset"] % ALIGN == 0
        assert t["offset"] >= end  # in order, non-overlapping
        end = t["offset"] + t["bytes"]
        sd_shape = list(m.state_dict()[t["name"]].shape)
        assert t["shape"] == sd_shape
        assert ("scale" in t) == (t["dtype"] == "trit5")
    assert _align(8 + header_len) + end == len(data)


def test_fp16_arm_has_no_trits_or_scales():
    m = _seeded_model("fp16")
    header, _ = read_pack(pack_model(m.state_dict(), m.cfg, "fp16"))
    assert all(t["dtype"] == "f16" and "scale" not in t for t in header["tensors"])


# --- 3. quantization fidelity vs BitLinear ---------------------------------------

def test_quantization_matches_bitlinear():
    m = _seeded_model("ternary")
    header, _ = read_pack(pack_model(m.state_dict(), m.cfg, "ternary"))
    scales = {t["name"]: t["scale"] for t in header["tensors"] if t["dtype"] == "trit5"}
    bitlinears = {n + ".weight": mod for n, mod in m.named_modules()
                  if isinstance(mod, BitLinear)}
    assert set(scales) == set(bitlinears)
    for name, mod in bitlinears.items():
        assert scales[name] == float(np.float16(mod.scale().item()))


def test_unpacked_trits_match_bitlinear_rule():
    m = _seeded_model("ternary")
    data = pack_model(m.state_dict(), m.cfg, "ternary")
    header, tensors = read_pack(data)
    for t in header["tensors"]:
        if t["dtype"] != "trit5":
            continue
        mod_name = t["name"].rsplit(".weight", 1)[0]
        mod = dict(m.named_modules())[mod_name]
        expected = mod.quantized_weight().detach().numpy() / mod.scale().item()
        got = tensors[t["name"]] / t["scale"]
        assert np.array_equal(np.round(got), np.round(expected))


# --- 3b. packing the release form (already-quantized) ------------------------------

def test_prequantized_pack_is_byte_identical():
    # checkpoint -> pack must equal (checkpoint -> dequantized release form -> pack):
    # the safetensors route may not drift from the .pt route by a single byte.
    m = _seeded_model("ternary")
    from_latents = pack_model(m.state_dict(), m.cfg, "ternary")
    header, tensors = read_pack(from_latents)  # dequantized fp32, fp16-exact values
    release_form = {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}
    from_release = pack_model(release_form, m.cfg, "ternary", prequantized=True)
    assert from_latents == from_release


def test_recover_ternary_rejects_non_ternary():
    with pytest.raises(ValueError):
        recover_ternary(np.array([0.5, -0.25, 0.0], dtype=np.float32))


# --- 4. forward parity -------------------------------------------------------------

def _parity_logits(precision, atol):
    cfg = ModelConfig(n_layer=2, n_embd=64, n_head=4, ctx=64, vocab=512)
    m = _seeded_model(precision, cfg)
    header, tensors = read_pack(pack_model(m.state_dict(), cfg, precision))
    m2 = build_model(cfg, "fp16")  # dequantized weights load into plain linears
    m2.load_state_dict(state_dict_from_pack(header, tensors))
    ids = torch.arange(32).unsqueeze(0) % cfg.vocab
    with torch.no_grad():
        ref, _ = m(ids)
        got, _ = m2(ids)
    assert torch.allclose(ref, got, atol=atol), (ref - got).abs().max()


def test_forward_parity_ternary():
    # fp16 scale rounding + fp16 casts of embeddings/norms bound the error.
    _parity_logits("ternary", atol=0.05)


def test_forward_parity_fp16():
    _parity_logits("fp16", atol=0.02)


def test_fp16_weight_roundtrip_exact():
    m = _seeded_model("fp16")
    _, tensors = read_pack(pack_model(m.state_dict(), m.cfg, "fp16"))
    for name, arr in tensors.items():
        w16 = m.state_dict()[name].numpy().astype(np.float16).astype(np.float32)
        assert np.array_equal(arr, w16)


# --- 5. size reconciliation vs count_bytes ----------------------------------------

@pytest.mark.parametrize("tier", ["tiny", "large"])
@pytest.mark.parametrize("precision", ["fp16", "ternary"])
def test_size_reconciles_with_count_bytes(tier, precision):
    m = _seeded_model(precision, TIERS[tier])
    data = pack_model(m.state_dict(), m.cfg, precision)
    accounted = count_bytes(m, precision)["total"]
    assert expected_count_bytes(m.cfg, precision) == accounted

    # Exact expected delta: trit ceil padding + norm gains + header/alignment.
    (header_len,) = struct.unpack_from("<I", data, 4)
    header = json.loads(data[8:8 + header_len])
    trit_pad = sum(
        t["bytes"] - int(np.prod(t["shape"])) * 1.58 / 8
        for t in header["tensors"] if t["dtype"] == "trit5"
    )
    norm_gains = sum(t["bytes"] for t in header["tensors"]
                     if t["name"].endswith("norm.weight"))
    payload = sum(t["bytes"] for t in header["tensors"])
    container = len(data) - payload  # magic + header (incl. scales) + alignment
    scale_bytes = count_bytes(m, precision)["scale_bytes"]

    delta = len(data) - accounted
    assert delta == pytest.approx(trit_pad + norm_gains + container - scale_bytes)
    assert delta / accounted < 0.02


# --- 6. frozen tokenizer copy ------------------------------------------------------

def test_web_tokenizer_is_frozen_copy():
    src = os.path.join(_REPO, "artifacts", "tokenizer", "tokenizer.json")
    web = os.path.join(_REPO, "web", "tokenizer.json")
    with open(src, "rb") as f_src, open(web, "rb") as f_web:
        assert f_src.read() == f_web.read(), \
            "web/tokenizer.json must be byte-identical to the frozen artifact"
