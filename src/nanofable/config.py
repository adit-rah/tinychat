"""Model + tier configuration.

The sweep's controlled variables live here. Only `n_layer`/`n_embd`/`n_head` vary
across tiers; `ctx` and `vocab` are held fixed for the whole sweep (spec §4, and the
ctx-conflict resolution recorded in docs/design_notes.md).
"""

from __future__ import annotations

from dataclasses import dataclass


def _round_up_to_multiple(x: float, m: int) -> int:
    """Smallest multiple of `m` that is >= x."""
    return ((int(x) + m - 1) // m) * m


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    n_embd: int
    n_head: int
    ctx: int = 512
    vocab: int = 4096

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        """SwiGLU hidden width: 8/3 * n_embd rounded up to a multiple of 64."""
        return _round_up_to_multiple(8 / 3 * self.n_embd, 64)


# Tiers — only size varies. ctx=512 and vocab=4096 are frozen across the sweep.
TIERS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(n_layer=4, n_embd=128, n_head=4),
    "small": ModelConfig(n_layer=6, n_embd=256, n_head=8),
    "medium": ModelConfig(n_layer=8, n_embd=384, n_head=8),
    "large": ModelConfig(n_layer=8, n_embd=512, n_head=8),
}
