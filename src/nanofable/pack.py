"""Pack format v1 ("TCP1", .tpack) — the phase-2 shippable model file.

Self-describing container the static site's JS loader reads (docs/demo_site.md;
docs/phase2_static_site_target.md deliverable #1). Layout, little-endian:

    magic b"TCP1" | uint32 header_len | JSON header | zero-pad to 8 | tensor payload

The JSON header carries the ModelConfig, the precision arm, and a tensor manifest
(name/dtype/shape/offset/bytes, offsets 8-aligned relative to the payload start).
`lm_head.weight` is never stored — the head is tied, so loaders reuse `tok_emb.weight`.

Ternary arm: in-block linears are quantized at pack time with BitLinear's rule
(absmean scale, round(clamp(w/scale, -1, 1))) and stored as `trit5` — 5 trits/byte,
byte = sum(t_i * 3^i) with t = w+1, last group padded with t=1 (zero weight). The
per-tensor scale is rounded to fp16 *before* being written to the header, so the
Python unpacker and the JS loader dequantize with bit-identical values. Everything
else (embeddings, norm gains, fp16-arm linears) is stored as raw IEEE fp16.

The shipped file must reconcile with count_bytes (spec §6): the only deltas are the
1.6-vs-1.58-bit ceil padding, the norm gains (deliberately uncounted there), and the
header/alignment overhead. tests/test_pack.py pins this decomposition exactly.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

from .bytes import FP16_BYTES, TERNARY_BITS
from .config import ModelConfig

MAGIC = b"TCP1"
VERSION = 1
ALIGN = 8

_POWERS = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def tensor_names(cfg: ModelConfig) -> list[str]:
    """Canonical tensor order in the pack (lm_head excluded — tied)."""
    names = ["tok_emb.weight"]
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        names += [
            p + "attn_norm.weight",
            p + "attn.q.weight", p + "attn.k.weight",
            p + "attn.v.weight", p + "attn.o.weight",
            p + "mlp_norm.weight",
            p + "mlp.gate.weight", p + "mlp.up.weight", p + "mlp.down.weight",
        ]
    names.append("final_norm.weight")
    return names


def _is_block_linear(name: str) -> bool:
    """The seven in-block linears (q,k,v,o,gate,up,down) — the ternarized set."""
    return ".attn." in name or ".mlp." in name


def pack_trits(trits: np.ndarray) -> bytes:
    """Pack a flat array of {-1, 0, +1} into 5 trits/byte (ceil(n/5) bytes)."""
    t = (trits.astype(np.int16).ravel() + 1).astype(np.uint8)
    pad = -len(t) % 5
    if pad:
        t = np.concatenate([t, np.ones(pad, dtype=np.uint8)])  # t=1 == zero weight
    return (t.reshape(-1, 5) @ _POWERS).astype(np.uint8).tobytes()


def unpack_trits(data: bytes, numel: int) -> np.ndarray:
    """Inverse of pack_trits: bytes -> int8 array of {-1, 0, +1}, length numel."""
    b = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    out = np.empty((len(b), 5), dtype=np.int8)
    for j in range(5):
        out[:, j] = (b % 3).astype(np.int8) - 1
        b //= 3
    return out.ravel()[:numel]


def quantize_ternary(w: np.ndarray) -> tuple[np.ndarray, float]:
    """BitLinear's pack-time rule: (trits, fp16-rounded absmean scale)."""
    scale = max(float(np.abs(w).mean()), 1e-8)
    trits = np.round(np.clip(w / scale, -1.0, 1.0)).astype(np.int8)
    return trits, float(np.float16(scale))


def recover_ternary(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact inverse for an already-quantized tensor (values scale * {-1,0,+1}, i.e.
    the release/safetensors form). Recovers scale as max|w| — re-running the absmean
    rule here would compute a smaller, wrong scale (mean over trits < 1). Raises if
    the tensor isn't genuinely ternary."""
    scale = float(np.abs(w).max())
    if scale == 0.0:
        return np.zeros(w.shape, dtype=np.int8), float(np.float16(1e-8))
    trits = np.round(w / scale).astype(np.int8)
    if not np.array_equal(trits.astype(np.float32) * np.float32(scale),
                          w.astype(np.float32)):
        raise ValueError("tensor is not ternary-quantized (scale * {-1,0,+1})")
    return trits, float(np.float16(scale))


def expected_count_bytes(cfg: ModelConfig, precision: str) -> float:
    """count_bytes (spec §6) computed from the config alone — no model needed."""
    n_lin = cfg.n_layer * (4 * cfg.n_embd * cfg.n_embd + 3 * cfg.n_embd * cfg.mlp_hidden)
    embed_head = FP16_BYTES * cfg.vocab * cfg.n_embd
    if precision == "ternary":
        return n_lin * TERNARY_BITS / 8 + embed_head + 7 * cfg.n_layer * FP16_BYTES
    return n_lin * FP16_BYTES + embed_head


def pack_model(state_dict: dict, cfg: ModelConfig, precision: str,
               prequantized: bool = False) -> bytes:
    """prequantized=True: ternary weights already hold scale * {-1,0,+1} (the
    release/safetensors form) — recover trits+scale exactly instead of quantizing."""
    if precision not in ("fp16", "ternary"):
        raise ValueError(f"unknown precision: {precision}")

    tensors = []
    chunks: list[bytes] = []
    offset = 0
    for name in tensor_names(cfg):
        w = state_dict[name].detach().cpu().float().numpy()
        entry = {"name": name, "shape": list(w.shape)}
        if precision == "ternary" and _is_block_linear(name):
            trits, scale = (recover_ternary if prequantized else quantize_ternary)(w)
            data = pack_trits(trits)
            entry.update(dtype="trit5", scale=scale)
        else:
            data = w.astype(np.float16).tobytes()
            entry["dtype"] = "f16"
        entry.update(offset=offset, bytes=len(data))
        tensors.append(entry)
        chunks.append(data)
        offset = _align(offset + len(data))

    header = {
        "format": "nanofable-pack",
        "version": VERSION,
        "precision": precision,
        "config": {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
                   "ctx": cfg.ctx, "vocab": cfg.vocab},
        "tied_head": True,
        "tensors": tensors,
        "accounting": {"count_bytes_total": expected_count_bytes(cfg, precision)},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(header_bytes))
    out += header_bytes
    out += b"\0" * (_align(len(out)) - len(out))
    payload_start = len(out)
    for entry, data in zip(tensors, chunks):
        out += b"\0" * (payload_start + entry["offset"] - len(out))
        out += data
    return bytes(out)


def read_pack(data: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse a pack into (header, {name: float32 array}) — mirrors the JS loader."""
    if data[:4] != MAGIC:
        raise ValueError("not a TCP1 pack")
    (header_len,) = struct.unpack_from("<I", data, 4)
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    payload = _align(8 + header_len)

    tensors: dict[str, np.ndarray] = {}
    for t in header["tensors"]:
        raw = data[payload + t["offset"]: payload + t["offset"] + t["bytes"]]
        numel = int(np.prod(t["shape"]))
        if t["dtype"] == "trit5":
            arr = unpack_trits(raw, numel).astype(np.float32) * t["scale"]
        else:
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        tensors[t["name"]] = arr.reshape(t["shape"])
    return header, tensors


def state_dict_from_pack(header: dict, tensors: dict[str, np.ndarray]) -> dict:
    """Torch state dict loadable into a precision="fp16" Transformer (weights are
    already dequantized floats), with the tied lm_head re-materialized."""
    sd = {name: torch.from_numpy(arr.copy()) for name, arr in tensors.items()}
    sd["lm_head.weight"] = sd["tok_emb.weight"].clone()
    return sd


def write_pack(path: str, state_dict: dict, cfg: ModelConfig, precision: str,
               prequantized: bool = False) -> bytes:
    data = pack_model(state_dict, cfg, precision, prequantized=prequantized)
    with open(path, "wb") as f:
        f.write(data)
    return data
