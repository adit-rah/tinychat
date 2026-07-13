from torch import nn

from nanofable.bitlinear import BitLinear
from nanofable.bytes import count_bytes
from nanofable.config import TIERS
from nanofable.model import build_model


def _block_linear_weight_count(model):
    return sum(
        m.weight.numel()
        for m in model.blocks.modules()
        if isinstance(m, (nn.Linear, BitLinear))
    )


def test_fp16_total_is_blocks_plus_embed_head():
    m = build_model(TIERS["tiny"], "fp16")
    n_block = _block_linear_weight_count(m)
    cfg = m.cfg
    expected = 2 * n_block + 2 * cfg.vocab * cfg.n_embd
    assert count_bytes(m, "fp16")["total"] == expected


def test_fp16_has_no_scale_bytes():
    m = build_model(TIERS["tiny"], "fp16")
    assert count_bytes(m, "fp16")["scale_bytes"] == 0


def test_ternary_block_uses_1p58():
    mt = build_model(TIERS["small"], "ternary")
    nq = sum(
        l.weight.numel() for l in mt.modules() if isinstance(l, BitLinear)
    )
    assert count_bytes(mt, "ternary")["block_bytes"] == nq * 1.58 / 8


def test_scale_bytes_counts_every_quantized_layer():
    mt = build_model(TIERS["tiny"], "ternary")
    nlayers = sum(isinstance(l, BitLinear) for l in mt.modules())
    assert count_bytes(mt, "ternary")["scale_bytes"] == nlayers * 2


def test_embed_head_counted_once():
    m = build_model(TIERS["tiny"], "fp16")
    cfg = m.cfg
    assert count_bytes(m, "fp16")["embed_head_bytes"] == 2 * cfg.vocab * cfg.n_embd


def test_no_norm_key_in_accounting():
    # §6 literal: three terms only, norms not counted.
    m = build_model(TIERS["tiny"], "fp16")
    keys = set(count_bytes(m, "fp16"))
    assert keys == {"block_bytes", "embed_head_bytes", "scale_bytes", "total"}


def test_ternary_smaller_than_fp16_with_4k_vocab():
    # The whole experiment depends on this holding at the 4k vocab, at every tier.
    for tier in TIERS:
        cfg = TIERS[tier]
        bt = count_bytes(build_model(cfg, "ternary"), "ternary")["total"]
        bf = count_bytes(build_model(cfg, "fp16"), "fp16")["total"]
        assert bt < bf, f"ternary not smaller at tier {tier}"
