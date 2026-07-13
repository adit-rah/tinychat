"""Honest byte accounting (spec §6) — the headline result depends on this being correct.

Three terms, exactly as §6 states (norm gains are intentionally NOT counted — see
docs/byte_accounting.md and docs/design_notes.md item 2):

    total_bytes = block_bytes + embed_head_bytes + scale_bytes

- in-block linear weights: ternary -> n_w * 1.58/8 (packed); fp16 -> n_w * 2.
- embeddings + tied head: 2 * vocab * n_embd (counted once; head is tied).
- per-layer ternary scales: num_quantized_layers * 2 (fp16 scalar each); fp16 arm -> 0.
"""

from __future__ import annotations

from torch import nn

from .bitlinear import BitLinear

TERNARY_BITS = 1.58
FP16_BYTES = 2


def _block_linears(model):
    """All linear layers living inside the transformer blocks (q,k,v,o,gate,up,down)."""
    return [m for m in model.blocks.modules() if isinstance(m, (nn.Linear, BitLinear))]


def count_bytes(model, precision: str) -> dict:
    linears = _block_linears(model)
    n_block_weights = sum(l.weight.numel() for l in linears)

    if precision == "ternary":
        n_quantized = sum(isinstance(l, BitLinear) for l in linears)
        block_bytes = n_block_weights * TERNARY_BITS / 8
        scale_bytes = n_quantized * FP16_BYTES
    elif precision == "fp16":
        block_bytes = n_block_weights * FP16_BYTES
        scale_bytes = 0
    else:
        raise ValueError(f"unknown precision: {precision}")

    cfg = model.cfg
    embed_head_bytes = FP16_BYTES * cfg.vocab * cfg.n_embd  # embeddings + tied head, once

    total = block_bytes + embed_head_bytes + scale_bytes
    return {
        "block_bytes": block_bytes,
        "embed_head_bytes": embed_head_bytes,
        "scale_bytes": scale_bytes,
        "total": total,
    }
