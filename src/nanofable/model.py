"""Decoder-only transformer with a global precision switch.

Pre-norm RMSNorm + RoPE + SwiGLU, tied embedding/LM head, all linears bias-free. The
`precision` flag swaps every *in-block* linear (q, k, v, o, gate, up, down) between
`nn.Linear` (fp16 arm) and `BitLinear` (ternary arm). Embeddings, the final RMSNorm, the
LM head, and every RMSNorm gain stay fp16 in both arms (spec §7).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .bitlinear import BitLinear
from .config import ModelConfig
from .rope import apply_rope, build_rope_cache

PAD_ID = 2
Precision = Literal["fp16", "ternary"]


def _linear_factory(precision: Precision):
    if precision == "ternary":
        return lambda i, o: BitLinear(i, o, bias=False)
    return lambda i, o: nn.Linear(i, o, bias=False)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, linear):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.q = linear(cfg.n_embd, cfg.n_embd)
        self.k = linear(cfg.n_embd, cfg.n_embd)
        self.v = linear(cfg.n_embd, cfg.n_embd)
        self.o = linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig, linear):
        super().__init__()
        hidden = cfg.mlp_hidden
        self.gate = linear(cfg.n_embd, hidden)
        self.up = linear(cfg.n_embd, hidden)
        self.down = linear(hidden, cfg.n_embd)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, linear):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg, linear)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg, linear)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, precision: Precision):
        super().__init__()
        self.cfg = cfg
        self.precision = precision
        linear = _linear_factory(precision)

        self.tok_emb = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, linear) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        cos, sin = build_rope_cache(cfg.head_dim, cfg.ctx)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss


def build_model(cfg: ModelConfig, precision: Precision) -> Transformer:
    return Transformer(cfg, precision)
