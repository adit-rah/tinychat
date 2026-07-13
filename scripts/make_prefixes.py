"""Generate the FROZEN set of exactly 200 held-out prefixes (spec §8).

Samples 200 TinyStories validation stories with a fixed seed, truncates each to its first
~40 tokens (the prefix), and records the gold continuation. Run ONCE; commit
eval/prefixes.jsonl, then never edit it.

    python scripts/make_prefixes.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanofable.tokenizer import load_tokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TOK_PATH = os.path.join(ROOT, "artifacts", "tokenizer", "tokenizer.json")
OUT_PATH = os.path.join(ROOT, "eval", "prefixes.jsonl")
N_PREFIXES = 200
PREFIX_TOKENS = 40
SEED = 12345


def main():
    import numpy as np
    from datasets import load_dataset

    tok = load_tokenizer(TOK_PATH)
    ds = load_dataset("roneneldan/TinyStories", split="validation")

    rng = np.random.default_rng(SEED)
    # sample candidate indices, keep stories long enough to have a real continuation
    idxs = rng.permutation(len(ds))
    chosen = []
    for i in idxs:
        text = ds[int(i)]["text"]
        ids = tok.encode(text).ids
        if len(ids) <= PREFIX_TOKENS + 10:
            continue
        prefix = tok.decode(ids[:PREFIX_TOKENS])
        gold = tok.decode(ids[PREFIX_TOKENS:])
        chosen.append({"id": int(i), "prefix": prefix, "gold_continuation": gold})
        if len(chosen) == N_PREFIXES:
            break

    assert len(chosen) == N_PREFIXES, f"only found {len(chosen)} usable stories"
    with open(OUT_PATH, "w") as f:
        for row in chosen:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(chosen)} prefixes -> {OUT_PATH}")


if __name__ == "__main__":
    main()
