"""Kaggle setup helpers — keep the boilerplate out of the notebook.

The notebook only syncs the repo and puts it on ``sys.path``; everything else (deps,
frozen-artifact checks, data build, judge passes, deliverables) lives here so each notebook
cell stays a one-liner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

FROZEN_ARTIFACTS = [
    "artifacts/tokenizer/tokenizer.json",
    "eval/prefixes.jsonl",
    "eval/rubric.md",
    "eval/judge_prompt.md",
]


@dataclass
class Ctx:
    """Paths the notebook cells need, returned by :func:`bootstrap`."""

    repo_dir: str
    data_dir: str
    runs_dir: str
    train_path: str
    val_path: str
    tokenizer_path: str


def _pip(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def install_deps() -> None:
    """Install only what Kaggle lacks. Never torch/transformers — upgrading Kaggle's torch
    breaks its preinstalled transformers (ImportError: _maybe_view_chunk_cat)."""
    # Some Kaggle images pair torch with a triton whose layout torch doesn't expect, so any
    # lazy `import torch._dynamo` (e.g. optimizer construction) dies with
    # AttributeError: module 'triton.backends' has no attribute 'compiler'. We never use
    # triton (eager only), and torch skips its triton block when the package is absent —
    # probe in a subprocess (before torch is imported here) and drop triton if broken.
    probe = subprocess.run([sys.executable, "-c", "import torch._dynamo"],
                           capture_output=True)
    if probe.returncode != 0:
        print("torch._dynamo import broken — uninstalling triton (unused; eager-only)")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "triton"],
                       check=True)
        subprocess.run([sys.executable, "-c", "import torch._dynamo"], check=True)
    _pip(["bitsandbytes"])  # 4-bit judge
    for mod in ("tokenizers", "datasets", "transformers", "accelerate"):
        try:
            __import__(mod)
        except ImportError:
            _pip(["--no-deps", mod])


def verify_frozen_artifacts(repo_dir: str = ".") -> int:
    """Assert the frozen eval set is present and the prefix count is exactly 200."""
    for rel in FROZEN_ARTIFACTS:
        assert os.path.exists(os.path.join(repo_dir, rel)), f"MISSING frozen artifact: {rel}"
    path = os.path.join(repo_dir, "eval/prefixes.jsonl")
    n = sum(1 for line in open(path) if line.strip())
    assert n == 200, f"expected 200 prefixes, found {n}"
    return n


def build_data(ctx: Ctx) -> None:
    """Tokenize TinyStories train/val into memmaps if not already built (idempotent)."""
    from datasets import load_dataset

    from nanofable.data import build_token_memmap
    from nanofable.tokenizer import load_tokenizer

    tok = load_tokenizer(ctx.tokenizer_path)
    if not os.path.exists(ctx.train_path):
        n = build_token_memmap(
            (r["text"] for r in load_dataset("roneneldan/TinyStories", split="train")),
            tok, ctx.train_path)
        print(f"train tokens: {n:,}")
    if not os.path.exists(ctx.val_path):
        n = build_token_memmap(
            (r["text"] for r in load_dataset("roneneldan/TinyStories", split="validation")),
            tok, ctx.val_path)
        print(f"val tokens: {n:,}")


def bootstrap(repo_dir: str | None = None, work: str = "/kaggle/working",
              build: bool = True) -> Ctx:
    """Install deps, check a GPU is visible, verify frozen artifacts, build data.

    Returns a :class:`Ctx` with the paths the rest of the notebook uses.
    """
    repo_dir = repo_dir or os.getcwd()
    # HF downloads (Qwen judge ~15GB) stay in the default ephemeral cache: /kaggle/working
    # has a hard ~19.5GiB quota and must be reserved for data + run checkpoints. The judge
    # re-downloads in sessions that need it (~minutes on Kaggle's pipe).
    install_deps()

    import torch

    assert torch.cuda.is_available(), "GPU not visible — set the accelerator in the panel"
    n = verify_frozen_artifacts(repo_dir)

    data_dir = os.path.join(work, "data")
    runs_dir = os.path.join(work, "runs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    ctx = Ctx(
        repo_dir=repo_dir,
        data_dir=data_dir,
        runs_dir=runs_dir,
        train_path=os.path.join(data_dir, "train.bin"),
        val_path=os.path.join(data_dir, "val.bin"),
        tokenizer_path=os.path.join(repo_dir, "artifacts/tokenizer/tokenizer.json"),
    )
    if build:
        build_data(ctx)

    import transformers

    print(f"ready | torch {torch.__version__} | transformers {transformers.__version__} "
          f"| prefixes {n} | runs -> {runs_dir}")
    return ctx


def _tinystories_ref_fn(model_id: str = "roneneldan/TinyStories-33M",
                        sampled: bool = False):
    """Completion fn for a published TinyStories reference model (33M, 1M, ...).

    `sampled=True` mirrors the sweep models' original eval decoding (temperature 1.0,
    top_k 40) instead of greedy, so the reference locates the gate under the same
    decoding policy the models are judged with.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m_id = model_id
    m_tok = AutoTokenizer.from_pretrained(m_id)
    m = AutoModelForCausalLM.from_pretrained(m_id).to("cuda").eval()

    def fn(item):
        enc = m_tok(item["prefix"], return_tensors="pt").to("cuda")
        gen_kwargs = ({"do_sample": True, "temperature": 1.0, "top_k": 40}
                      if sampled else {"do_sample": False})
        if sampled:
            torch.manual_seed(0)
        # explicit attention_mask + pad_token_id: silences the per-call transformers
        # warning spam (GPT-Neo defines no pad token); output is identical for batch-of-1.
        out = m.generate(**enc, max_new_tokens=200, pad_token_id=m_tok.eos_token_id,
                         **gen_kwargs)
        n_in = enc["input_ids"].shape[1]
        return m_tok.decode(out[0, n_in:], skip_special_tokens=True)

    return fn


def calibrate(ctx: Ctx, use_33m: bool = False, repeats: int = 3) -> None:
    """Run the one-time sweep-blind judge calibration and print eval/calibration.md."""
    import importlib.util

    from eval.judge import LocalQwenJudge

    spec = importlib.util.spec_from_file_location(
        "run_calibration", os.path.join(ctx.repo_dir, "scripts/run_calibration.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)

    judge = LocalQwenJudge()
    model33m_fn = _tinystories_ref_fn() if use_33m else None
    cal.main(judge=judge, model33m_fn=model33m_fn, repeats=repeats)
    print(open(os.path.join(ctx.repo_dir, "eval/calibration.md")).read())


def calibrate_reference(ctx: Ctx, model_id: str = "roneneldan/TinyStories-33M",
                        sampled: bool = False, n_prefixes: int | None = None,
                        judge=None, detail_path: str | None = None) -> None:
    """Score a published TinyStories reference model through the frozen judge pipeline
    and append the result to calibration.md.

    Used for spec §8 reference (b) (33M) and for the reference-anchored secondary
    capability line (TinyStories-1M, greedy — see the frozen block in calibration.md).
    `sampled=True` decodes with the original sweep eval policy (temp 1.0 / top_k 40)
    instead of greedy — it measures how much of the gap between a reference model and the
    gate is the decoding policy, not the model. `n_prefixes` caps the set for a quick
    diagnostic pass; the anchor itself must use the full 200. Pass `judge` to reuse one
    loaded judge across a ladder of reference models (it loads ~5GB per call otherwise).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_calibration", os.path.join(ctx.repo_dir, "scripts/run_calibration.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)

    if judge is None:
        from eval.judge import LocalQwenJudge

        judge = LocalQwenJudge()
    mean, half, n = cal.score_reference(judge, model_id, sampled=sampled,
                                        n_prefixes=n_prefixes, detail_path=detail_path)
    line = cal.reference_line(model_id, sampled, n, mean, half)
    with open(os.path.join(ctx.repo_dir, "eval/calibration.md"), "a") as f:
        f.write(line + "\n")
    print(line)


# Back-compat name for earlier notebook cells.
calibrate_33m = calibrate_reference

LADDER = tuple(f"roneneldan/TinyStories-{m}" for m in ("1M", "3M", "8M", "28M", "33M"))


def calibrate_ladder(ctx: Ctx, model_ids: tuple[str, ...] = LADDER,
                     sampled: bool = False, n_prefixes: int | None = None,
                     workers: int | None = None, poll_s: int = 60) -> None:
    """Score the published reference ladder, one judge per GPU (anchor rule,
    eval/calibration.md).

    Models are split round-robin across workers; each worker writes JSON results to its
    own file (`runs/ladder_worker{i}.jsonl`, logs alongside) and the parent appends the
    ledger lines to calibration.md in ladder order — concurrent workers never touch the
    ledger directly. Single GPU falls back to one in-process judge.
    """
    import json as _json
    import time

    if workers is None:
        import torch

        workers = max(1, torch.cuda.device_count())
    workers = min(workers, len(model_ids))
    detail_dir = os.path.join(ctx.runs_dir, "ladder_axes")
    os.makedirs(detail_dir, exist_ok=True)
    if workers == 1:
        import importlib.util

        from eval.judge import LocalQwenJudge

        judge = LocalQwenJudge()
        for mid in model_ids:
            short = mid.split("/")[-1]
            calibrate_reference(ctx, model_id=mid, sampled=sampled,
                                n_prefixes=n_prefixes, judge=judge,
                                detail_path=os.path.join(detail_dir,
                                                         f"{short}.axes.jsonl"))
        spec = importlib.util.spec_from_file_location(
            "run_calibration", os.path.join(ctx.repo_dir, "scripts/run_calibration.py"))
        cal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cal)
        cal.score_gold_detailed(judge, os.path.join(detail_dir, "gold.axes.jsonl"),
                                n_prefixes=n_prefixes)
        return

    from huggingface_hub import snapshot_download

    from nanofable.sweep import _tail_line

    snapshot_download("Qwen/Qwen2.5-7B-Instruct")  # avoid a two-worker fetch race
    runs_dir = ctx.runs_dir
    os.makedirs(runs_dir, exist_ok=True)
    procs, out_paths, log_paths, log_files = [], [], [], []
    for i in range(workers):
        out_path = os.path.join(runs_dir, f"ladder_worker{i}.jsonl")
        log_path = os.path.join(runs_dir, f"ladder_worker{i}.log")
        if os.path.exists(out_path):
            os.remove(out_path)
        cfg = _json.dumps({"model_ids": list(model_ids[i::workers]), "sampled": sampled,
                           "n_prefixes": n_prefixes, "out": out_path,
                           "detail_dir": detail_dir, "gold": i == 0})
        logf = open(log_path, "a")
        env = {**os.environ,
               "CUDA_VISIBLE_DEVICES": str(i),
               "PYTHONPATH": os.pathsep.join(
                   [os.path.join(ctx.repo_dir, "src"), ctx.repo_dir,
                    os.environ.get("PYTHONPATH", "")])}
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(ctx.repo_dir, "scripts/run_calibration.py"),
             cfg], env=env, cwd=ctx.repo_dir, stdout=logf, stderr=subprocess.STDOUT))
        out_paths.append(out_path)
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
            f"ladder worker(s) {failed} failed — see "
            + ", ".join(log_paths[i] for i in failed)
            + " (completed model results are in the .jsonl files)")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_calibration", os.path.join(ctx.repo_dir, "scripts/run_calibration.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    results = {}
    for out_path in out_paths:
        with open(out_path) as f:
            for row in map(_json.loads, f):
                results[row["model_id"]] = row
    with open(os.path.join(ctx.repo_dir, "eval/calibration.md"), "a") as f:
        for mid in model_ids:  # ledger lines land in ladder order, not finish order
            r = results[mid]
            line = cal.reference_line(mid, sampled, r["n"], r["mean"], r["ci_half"])
            f.write(line + "\n")
            print(line)


def evaluate_all(ctx: Ctx, workers: int | None = None, poll_s: int = 60,
                 **eval_kwargs) -> None:
    """Score every finished run that lacks an eval.json, one judge per GPU.

    With multiple GPUs, spawns one eval worker subprocess per GPU (CUDA_VISIBLE_DEVICES
    pins each; EVAL_CLAIM markers keep runs exclusive; logs in runs/eval_worker{i}.log).
    Single GPU falls back to one in-process pass. `eval_kwargs` (e.g. temperature/top_k)
    pass through to `evaluate_run` → `generate`, so the eval decoding policy is settable
    from the notebook without a code push.
    """
    import json as _json
    import subprocess
    import sys as _sys
    import time

    from eval.run_eval import evaluate_pending

    from nanofable.sweep import _tail_line, run_dir_for, sweep_matrix

    if workers is None:
        import torch

        workers = max(1, torch.cuda.device_count())
    if workers == 1:
        evaluate_pending(ctx.runs_dir, **eval_kwargs)
        return

    # Pre-download the judge once so parallel workers don't race the 15GB fetch.
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen2.5-7B-Instruct")
    for tier, prec, seed in sweep_matrix():
        stale = os.path.join(run_dir_for(ctx.runs_dir, tier, prec, seed), "EVAL_CLAIM")
        if os.path.exists(stale):
            os.remove(stale)

    cfg = _json.dumps({"runs_root": ctx.runs_dir, "eval_kwargs": eval_kwargs})
    src_root = os.path.join(ctx.repo_dir, "src")
    procs, log_paths, log_files = [], [], []
    for i in range(workers):
        log_path = os.path.join(ctx.runs_dir, f"eval_worker{i}.log")
        logf = open(log_path, "a")
        env = {**os.environ,
               "CUDA_VISIBLE_DEVICES": str(i),
               "PYTHONPATH": os.pathsep.join(
                   [src_root, ctx.repo_dir, os.environ.get("PYTHONPATH", "")])}
        procs.append(subprocess.Popen(
            [_sys.executable, "-m", "eval.run_eval", cfg], env=env, cwd=ctx.repo_dir,
            stdout=logf, stderr=subprocess.STDOUT))
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
            f"eval worker(s) {failed} failed — see "
            + ", ".join(log_paths[i] for i in failed)
            + " (scored runs keep their eval.json; re-run to resume)")


def save_deliverables(ctx: Ctx, out: str = "/kaggle/working/deliverables") -> str:
    """Copy the plot, calibration, and per-run metrics/eval into one folder for download."""
    import glob
    import shutil

    os.makedirs(out, exist_ok=True)
    for f in ["/kaggle/working/frontier.png",
              os.path.join(ctx.repo_dir, "eval/calibration.md")]:
        if os.path.exists(f):
            shutil.copy(f, out)
    for f in (glob.glob(os.path.join(ctx.runs_dir, "*", "metrics.csv"))
              + glob.glob(os.path.join(ctx.runs_dir, "*", "eval.json"))):
        run = os.path.basename(os.path.dirname(f))
        shutil.copy(f, os.path.join(out, f"{run}_{os.path.basename(f)}"))
    print("deliverables ->", out)
    return out
