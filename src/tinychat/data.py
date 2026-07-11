"""Data pipeline: tokenize TinyStories to a flat uint16 memmap, then sample batches.

Documents are joined with `<|eos|>` and packed contiguously. The batch iterator is fully
deterministic given its seed — apples-to-apples across the sweep depends on it.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

import numpy as np
import torch

EOS_ID = 1


def build_token_memmap(texts: Iterable[str], tokenizer, out_path: str) -> int:
    """Tokenize `texts`, append `<|eos|>` after each, write uint16 to `out_path`.

    Returns the total number of tokens written. Writes incrementally so the full
    corpus need not fit in memory. Writes to a temp file and renames on completion, so
    an interrupted build never leaves a truncated file that callers (which use the
    file's existence as the "already built" check) would silently accept.
    """
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        eos = EOS_ID
    n = 0
    docs = 0
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            for text in texts:
                ids = tokenizer.encode(text).ids
                ids.append(eos)
                arr = np.asarray(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                n += arr.size
                docs += 1
                if docs % 200_000 == 0:
                    print(f"tokenized {docs:,} docs, {n:,} tokens", flush=True)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    os.replace(tmp_path, out_path)
    return n


def batch_iterator(
    memmap_path: str, ctx: int, tokens_per_step: int, seed: int, start_step: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (x, y) batches of shape [tokens_per_step // ctx, ctx], y = x shifted by 1.

    Each step draws from an independent RNG substream `default_rng([seed, step])`, so the
    stream is both deterministic *and* indexable by step: resuming at `start_step` produces
    exactly the batches that a fresh run would have at that step (O(1) resume, no skipping).
    """
    data = np.memmap(memmap_path, dtype=np.uint16, mode="r")
    rows = tokens_per_step // ctx
    max_start = data.shape[0] - ctx - 1
    if max_start <= 0:
        raise ValueError("memmap too small for the requested ctx")
    step = start_step
    while True:
        rng = np.random.default_rng([seed, step])
        starts = rng.integers(0, max_start, size=rows)
        x = np.stack([data[s : s + ctx] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1 : s + ctx + 1] for s in starts]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)
        step += 1
