"""Training compute estimate (spec §3/§4): FLOPs ≈ 6 · N_params · N_tokens."""

from __future__ import annotations


def flops(n_params: int, n_tokens: int) -> float:
    return 6.0 * n_params * n_tokens
