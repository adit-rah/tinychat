"""Rotary position embeddings (RoPE).

GPT-NeoX style: the head dimension is split in half and rotated. Held fixed across the
sweep (a controlled variable).
"""

from __future__ import annotations

import torch


def build_rope_cache(head_dim: int, ctx: int, base: float = 10000.0):
    """Return (cos, sin), each of shape [ctx, head_dim]."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(ctx).float()
    freqs = torch.outer(pos, inv_freq)  # [ctx, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [ctx, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE to q, k of shape [B, H, T, head_dim].

    `cos`/`sin` are [T_total, head_dim]; only the first T rows are used.
    """
    t = q.shape[-2]
    cos = cos[:t].to(q.dtype)[None, None, :, :]
    sin = sin[:t].to(q.dtype)[None, None, :, :]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out
