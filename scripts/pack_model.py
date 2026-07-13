"""Pack model weights into a shippable .tpack (pack format v1).

Input is either a training checkpoint (.pt, latent weights — quantized here) or a
release export (model.safetensors from scripts/export_model.py, already-quantized
weights — trits and scale recovered exactly; both routes produce identical bytes).
With a .safetensors input, --tier/--precision default from the sibling config.json.

    python scripts/pack_model.py --ckpt local/tiny_ternary_0.pt --tier tiny \
        --precision ternary --out web/models/tiny_ternary.tpack
    python scripts/pack_model.py --ckpt local/hf/tiny_ternary/model.safetensors \
        --out local/hf/tiny_ternary/model.tpack

Prints the reconciliation table: shipped file size vs count_bytes (spec §6), with the
delta decomposed (trit ceil padding, norm gains, header/alignment). CPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from nanofable.config import TIERS  # noqa: E402
from nanofable.pack import expected_count_bytes, read_pack, write_pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="training checkpoint (.pt) or release export (.safetensors)")
    ap.add_argument("--tier", choices=sorted(TIERS))
    ap.add_argument("--precision", choices=["fp16", "ternary"])
    ap.add_argument("--out", required=True, help="output .tpack path")
    args = ap.parse_args()

    prequantized = args.ckpt.endswith(".safetensors")
    if prequantized:
        from safetensors.torch import load_file

        state_dict = load_file(args.ckpt)
        cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
        if (args.tier is None or args.precision is None) and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                meta = json.load(f)
            args.tier = args.tier or meta["tier"]
            args.precision = args.precision or meta["precision"]
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
    if args.tier is None or args.precision is None:
        ap.error("--tier and --precision are required (no config.json to infer from)")

    cfg = TIERS[args.tier]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data = write_pack(args.out, state_dict, cfg, args.precision,
                      prequantized=prequantized)
    header, _ = read_pack(data)

    tensors = header["tensors"]
    block = sum(t["bytes"] for t in tensors if t["dtype"] == "trit5"
                or (".attn." in t["name"] or ".mlp." in t["name"]))
    embed = sum(t["bytes"] for t in tensors if t["name"] == "tok_emb.weight")
    norms = sum(t["bytes"] for t in tensors if t["name"].endswith("norm.weight"))
    overhead = len(data) - sum(t["bytes"] for t in tensors)
    accounted = expected_count_bytes(cfg, args.precision)

    print(f"packed {args.tier}/{args.precision} -> {args.out}")
    print(f"  block linears   {block:>10,} B")
    print(f"  embed+tied head {embed:>10,} B")
    print(f"  norm gains      {norms:>10,} B  (uncounted by count_bytes)")
    print(f"  header+align    {overhead:>10,} B  (incl. per-tensor scales)")
    print(f"  file total      {len(data):>10,} B")
    print(f"  count_bytes     {accounted:>12,.1f} B")
    print(f"  delta           {len(data) - accounted:>12,.1f} B "
          f"({(len(data) - accounted) / accounted * 100:.2f}%)")


if __name__ == "__main__":
    main()
