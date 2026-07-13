"""Idempotent sweep orchestration (spec §9): the 4×2×2 matrix.

Each run gets a deterministic directory `runs/<tier>_<precision>_<seed>`. A run whose dir
already contains a `DONE` marker is skipped, so re-running `run_sweep` after a session
timeout is safe and resumes only the unfinished work.

`run_sweep_parallel` trains the same matrix with one worker subprocess per GPU
(CUDA_VISIBLE_DEVICES pins each worker). Workers pull from a shared queue — each atomically
claims the next pending run via a CLAIM file — so uneven run times (ternary is slower than
fp16) self-balance. Kaggle bills session wall-clock regardless of GPU count, so pairing runs
on a T4×2 halves quota cost per run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .config import TIERS
from .train import train_run

PRECISIONS = ("fp16", "ternary")
SEEDS = (0, 1)


def sweep_matrix() -> list[tuple[str, str, int]]:
    """The full (tier, precision, seed) matrix — fp16 before ternary so the fp16 baseline
    (needed for the PPL gate) lands first."""
    return [
        (tier, precision, seed)
        for tier in TIERS
        for precision in PRECISIONS
        for seed in SEEDS
    ]


def run_dir_for(runs_root: str, tier: str, precision: str, seed: int) -> str:
    return os.path.join(runs_root, f"{tier}_{precision}_{seed}")


def claim_run(run_dir: str, name: str = "CLAIM") -> bool:
    """Atomically claim a run for this process (O_EXCL create of the `name` marker).
    Training and eval use distinct marker names so leftover training claims never block
    eval workers. Claims only coordinate workers within one session; the parent clears
    stale ones at startup."""
    os.makedirs(run_dir, exist_ok=True)
    try:
        fd = os.open(os.path.join(run_dir, name),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def run_sweep(
    runs_root: str,
    train_path: str,
    val_path: str,
    only: list[tuple[str, str, int]] | None = None,
    eval_fn=None,
    claim: bool = False,
    **train_kwargs,
) -> list[str]:
    """Train (and optionally eval) every matrix entry, skipping finished runs.

    `only` restricts to a subset of the matrix. `eval_fn(run_dir)` runs after training when
    provided (the CLI passes the judge-backed evaluator). `claim=True` makes this loop safe
    to run in multiple worker processes at once (each run trains in exactly one worker).
    Returns the list of run dirs that were trained this call.
    """
    trained = []
    for tier, precision, seed in sweep_matrix():
        if only is not None and (tier, precision, seed) not in only:
            continue
        run_dir = run_dir_for(runs_root, tier, precision, seed)
        if os.path.exists(os.path.join(run_dir, "DONE")):
            continue
        if claim and not claim_run(run_dir):
            continue
        train_run(TIERS[tier], precision, seed, run_dir, train_path, val_path,
                  **train_kwargs)
        if eval_fn is not None:
            eval_fn(run_dir)
        trained.append(run_dir)
    return trained


def _tail_line(path: str) -> str:
    """Last non-blank line of a log file ('' if unreadable/empty)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [ln for ln in f.read().decode(errors="replace").splitlines()
                     if ln.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def run_sweep_parallel(
    runs_root: str,
    train_path: str,
    val_path: str,
    only: list[tuple[str, str, int]] | None = None,
    workers: int | None = None,
    poll_s: int = 60,
    **train_kwargs,
) -> None:
    """Train the matrix with one claiming worker subprocess per GPU.

    Worker i sees only GPU i (CUDA_VISIBLE_DEVICES) and logs to `runs_root/worker{i}.log`;
    the parent echoes each worker's latest log line every `poll_s` seconds. Falls back to
    the in-process `run_sweep` when only one GPU is visible. Raises if any worker exits
    nonzero (the other worker's finished runs keep their DONE markers — re-run to resume).
    """
    if workers is None:
        import torch

        workers = max(1, torch.cuda.device_count())
    if workers == 1:
        run_sweep(runs_root, train_path, val_path, only=only, **train_kwargs)
        return

    # Claims only mean "a live worker owns this" — clear leftovers from killed sessions.
    for tier, precision, seed in sweep_matrix():
        stale = os.path.join(run_dir_for(runs_root, tier, precision, seed), "CLAIM")
        if os.path.exists(stale):
            os.remove(stale)

    os.makedirs(runs_root, exist_ok=True)
    cfg = json.dumps({"runs_root": runs_root, "train_path": train_path,
                      "val_path": val_path, "only": only, "train_kwargs": train_kwargs})
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    procs, log_paths, log_files = [], [], []
    for i in range(workers):
        log_path = os.path.join(runs_root, f"worker{i}.log")
        logf = open(log_path, "a")
        env = {**os.environ,
               "CUDA_VISIBLE_DEVICES": str(i),
               "PYTHONPATH": src_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        procs.append(subprocess.Popen([sys.executable, "-m", "nanofable.sweep", cfg],
                                      env=env, stdout=logf, stderr=subprocess.STDOUT))
        log_paths.append(log_path)
        log_files.append(logf)

    last = [""] * workers
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(poll_s)
            for i, log_path in enumerate(log_paths):
                line = _tail_line(log_path)
                if line and line != last[i]:
                    print(f"[gpu{i}] {line}", flush=True)
                    last[i] = line
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for f in log_files:
            f.close()

    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        raise RuntimeError(
            f"worker(s) {failed} failed — see "
            + ", ".join(log_paths[i] for i in failed)
            + " (finished runs are DONE-marked; re-run to resume the rest)")


if __name__ == "__main__":
    _cfg = json.loads(sys.argv[1])
    _only = [tuple(o) for o in _cfg["only"]] if _cfg["only"] is not None else None
    run_sweep(_cfg["runs_root"], _cfg["train_path"], _cfg["val_path"], only=_only,
              claim=True, **_cfg["train_kwargs"])
