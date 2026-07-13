"""Ternary BitLinear layer (spec §7) — a project non-negotiable.

Full-precision latent weight that the optimizer updates; forward quantizes to ternary
{-1, 0, +1} scaled by a per-tensor absmean scale; backward is a straight-through
estimator (identity through round/clip, masked to the active region |w| <= scale).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        ctx.save_for_backward(w, scale)
        return scale * torch.round(torch.clamp(w / scale, -1.0, 1.0))

    @staticmethod
    def backward(ctx, g):
        w, scale = ctx.saved_tensors
        # STE: pass gradient through, but only where the latent weight is in the
        # clip-active region (|w| <= scale); zero it where round/clip saturates.
        mask = (w.abs() <= scale).to(g.dtype)
        return g * mask, None


class BitLinear(nn.Module):
    """Linear layer with ternary quantized weights. bias is not supported (v1)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert not bias, "BitLinear is bias-free (v1)"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def scale(self) -> torch.Tensor:
        """Per-tensor absmean scale (fp16-representable scalar)."""
        return self.weight.detach().abs().mean().clamp_min(1e-8)

    def quantized_weight(self) -> torch.Tensor:
        return _TernarySTE.apply(self.weight, self.scale())

    def forward(self, x):
        return F.linear(x, self.quantized_weight())
