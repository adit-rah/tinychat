"""Autoregressive completion sampler for eval.

Deterministic given `seed` (uses a dedicated torch.Generator). Generates up to
`max_new_tokens`, stopping early at the EOS token, and returns the decoded continuation
(the new tokens only).
"""

from __future__ import annotations

import torch

from .data import EOS_ID


@torch.no_grad()
def next_token(model, ids, temperature: float = 1.0, top_k: int = 40,
               generator: torch.Generator | None = None) -> int:
    """Sample the id of the single next token given context ids (last ctx are used).

    The step-wise primitive: `generate` is a loop over this, so both paths share one
    sampling implementation. Pass a seeded CPU `generator` for determinism.
    """
    device = next(model.parameters()).device
    ctx = model.cfg.ctx
    cur = torch.tensor([list(ids)], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    logits, _ = model(cur[:, -ctx:])
    if was_training:
        model.train()
    logits = logits[0, -1] / max(temperature, 1e-6)
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float("inf")
    probs = torch.softmax(logits, dim=-1).cpu()
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate(model, tokenizer, prefix: str, max_new_tokens: int = 200, seed: int = 0,
             temperature: float = 1.0, top_k: int = 40) -> str:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(tokenizer.encode(prefix).ids)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        nxt = next_token(model, ids + out_ids, temperature=temperature, top_k=top_k,
                         generator=gen)
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
    return tokenizer.decode(out_ids)
