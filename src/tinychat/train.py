"""Training loop: seedable, hard-kill resumable, CSV-logged, periodic val perplexity.

A run is identified by its `run_dir`. Checkpoints are written atomically (tmp + os.replace)
so a `kill -9` mid-write never corrupts the resumable state. Re-invoking `train_run` with the
same `run_dir` resumes from the last checkpoint; reaching the token budget writes a `DONE`
marker. The data order, init, and dropout are all seeded so two runs of the same config are
identical (apples-to-apples, spec §4).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time

import numpy as np
import torch

from .bytes import count_bytes
from .config import ModelConfig
from .data import batch_iterator
from .flops import flops
from .model import build_model

CSV_COLUMNS = [
    "step",
    "tokens_seen",
    "train_loss",
    "val_loss",
    "val_ppl",
    "flops",
    "wall_clock_s",
    "lr",
    "timestamp",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- checkpoints


def save_checkpoint(run_dir, step, tokens_seen, model, opt, sched, periodic=False) -> None:
    """Atomically write ckpt_latest.pt (+ optional periodic ckpt_step{N}.pt)."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    latest = os.path.join(run_dir, "ckpt_latest.pt")
    tmp = latest + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, latest)  # atomic on POSIX
    if periodic:
        torch.save(payload, os.path.join(run_dir, f"ckpt_step{step}.pt"))


def load_latest(run_dir) -> dict | None:
    """Load ckpt_latest.pt, or None if absent. A leftover .tmp is ignored."""
    latest = os.path.join(run_dir, "ckpt_latest.pt")
    if not os.path.exists(latest):
        return None
    return torch.load(latest, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- evaluation


def evaluate_ppl(model, val_path, ctx, n_batches, eval_rows=16, seed=1234) -> float:
    """Mean held-out perplexity = exp(mean per-batch cross-entropy)."""
    device = next(model.parameters()).device
    it = batch_iterator(val_path, ctx, eval_rows * ctx, seed)
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = next(it)
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
    if was_training:
        model.train()
    return math.exp(sum(losses) / len(losses))


# --------------------------------------------------------------------------- schedule


def _make_scheduler(opt, total_steps, warmup_frac=0.03, final_frac=0.1):
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return final_frac + (1 - final_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# --------------------------------------------------------------------------- training


def _truncate_csv(csv_path, max_step):
    """Keep only CSV rows with step <= max_step (used on resume)."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        kept = [r for r in reader if int(r["step"]) <= max_step]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(kept)


def _append_csv(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def train_run(
    cfg: ModelConfig,
    precision: str,
    seed: int,
    run_dir: str,
    train_path: str,
    val_path: str,
    total_tokens: int = 500_000_000,
    tokens_per_step: int = 65536,
    peak_lr: float = 3e-4,
    micro_rows: int = 16,
    eval_every: int = 250,
    eval_batches: int = 50,
    ckpt_every: int = 500,
) -> None:
    """Train one (cfg, precision, seed) run to `total_tokens`, resuming if interrupted."""
    os.makedirs(run_dir, exist_ok=True)
    device = _device()
    ctx = cfg.ctx
    total_steps = total_tokens // tokens_per_step
    rows = tokens_per_step // ctx

    set_seed(seed)
    model = build_model(cfg, precision).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8
    )
    sched = _make_scheduler(opt, total_steps)

    # meta.json — config + frozen byte accounting for this run.
    meta = {
        "tier": {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
                  "ctx": cfg.ctx, "vocab": cfg.vocab},
        "precision": precision,
        "seed": seed,
        "n_params": n_params,
        "total_tokens": total_tokens,
        "tokens_per_step": tokens_per_step,
        "peak_lr": peak_lr,
        "total_bytes": count_bytes(model, precision)["total"],
        "bytes_breakdown": count_bytes(model, precision),
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Resume if a checkpoint exists.
    ckpt = load_latest(run_dir)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
    else:
        step = 0
        tokens_seen = 0

    if step >= total_steps:
        open(os.path.join(run_dir, "DONE"), "w").close()
        return

    csv_path = os.path.join(run_dir, "metrics.csv")
    # Logging can run ahead of checkpointing (eval_every < ckpt_every). On resume, drop any
    # CSV rows logged after the checkpoint we restored, so the step column stays strictly
    # monotonic (never re-logs or restarts a step).
    if ckpt is not None:
        _truncate_csv(csv_path, step)
    data = batch_iterator(train_path, ctx, tokens_per_step, seed, start_step=step)
    micro = max(1, micro_rows)
    n_micro = max(1, math.ceil(rows / micro))
    # fp16 autocast on GPU (frozen_config): ~2-3x faster, fits the large tier in 16GB.
    # No-op on CPU so the test suite is unaffected. Latent weights stay fp32; only the
    # in-block matmuls run in fp16 — the weight-precision variable is unchanged.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0 = time.time()
    model.train()

    while step < total_steps:
        x, y = next(data)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for i in range(n_micro):
            xb = x[i * micro : (i + 1) * micro].to(device)
            yb = y[i * micro : (i + 1) * micro].to(device)
            if xb.numel() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(xb, yb)
            scaler.scale(loss / n_micro).backward()
            step_loss += loss.item() / n_micro
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        tokens_seen += tokens_per_step

        is_last = step >= total_steps
        if step % eval_every == 0 or is_last:
            val_ppl = evaluate_ppl(model, val_path, ctx, eval_batches)
            _append_csv(csv_path, {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": round(step_loss, 6),
                "val_loss": round(math.log(val_ppl), 6),
                "val_ppl": round(val_ppl, 6),
                "flops": flops(n_params, tokens_seen),
                "wall_clock_s": round(time.time() - t0, 3),
                "lr": sched.get_last_lr()[0],
                "timestamp": time.time(),
            })
        if step % ckpt_every == 0 or is_last:
            save_checkpoint(run_dir, step, tokens_seen, model, opt, sched,
                            periodic=(step % ckpt_every == 0))

    save_checkpoint(run_dir, step, tokens_seen, model, opt, sched)
    open(os.path.join(run_dir, "DONE"), "w").close()
