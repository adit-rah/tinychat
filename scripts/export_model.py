"""Export a training checkpoint's model weights in their shipped form, for release
(e.g. HuggingFace): quantization applied, training state stripped.

    python scripts/export_model.py --ckpt local/tiny_ternary_0.pt --tier tiny \
        --precision ternary --out-dir local/hf/tiny_ternary

Writes <out-dir>/model.safetensors + <out-dir>/config.json.

- ternary arm: in-block linears become scale * {-1,0,+1} in fp16, using the same
  fp16-rounded absmean scale as the .tpack packer, so released weights are bit-identical
  to what the demo site computes. This is one-way — latents are gone; keep the original
  checkpoint as the master copy.
- everything else (and the whole fp16 arm): cast to fp16.
- lm_head.weight is not stored (tied to tok_emb.weight; safetensors also forbids shared
  tensors). Load with:

    model = build_model(TIERS[tier], "fp16")   # dequantized ternary loads into fp16 arm
    model.load_state_dict(load_file("model.safetensors"), strict=False)  # head re-ties
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from nanofable.config import TIERS  # noqa: E402
from nanofable.pack import _is_block_linear, quantize_ternary, tensor_names  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--precision", required=True, choices=["fp16", "ternary"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = TIERS[args.tier]
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    out = {}
    for name in tensor_names(cfg):
        w = state_dict[name].detach().cpu().float()
        if args.precision == "ternary" and _is_block_linear(name):
            trits, scale = quantize_ternary(w.numpy())
            w = torch.from_numpy(trits.astype("float32")) * scale
        out[name] = w.to(torch.float16).contiguous()

    os.makedirs(args.out_dir, exist_ok=True)
    save_file(out, os.path.join(args.out_dir, "model.safetensors"))
    config = {
        "model_type": "nanofable",
        "tier": args.tier,
        "precision": args.precision,
        "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
        "ctx": cfg.ctx, "vocab": cfg.vocab,
        "tied_head": True,
        "torch_dtype": "float16",
        "quantization": (
            "ternary: in-block linears are absmean-quantized to scale * {-1,0,+1} "
            "(per-tensor fp16 scale), stored dequantized in fp16"
            if args.precision == "ternary" else "none (fp16 weights)"
        ),
        "step": ckpt.get("step"),
        "tokens_seen": ckpt.get("tokens_seen"),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    size = os.path.getsize(os.path.join(args.out_dir, "model.safetensors"))
    print(f"{args.out_dir}: model.safetensors {size:,} B "
          f"({len(out)} tensors, fp16) + config.json")


if __name__ == "__main__":
    main()
