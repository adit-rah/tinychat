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

    from tinychat.data import build_token_memmap
    from tinychat.tokenizer import load_tokenizer

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


def _tinystories_33m_fn():
    """Completion fn for the optional TinyStories-33M mediocre reference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m_id = "roneneldan/TinyStories-33M"
    m_tok = AutoTokenizer.from_pretrained(m_id)
    m = AutoModelForCausalLM.from_pretrained(m_id).to("cuda").eval()

    def fn(item):
        enc = m_tok(item["prefix"], return_tensors="pt").to("cuda")
        # explicit attention_mask + pad_token_id: silences the per-call transformers
        # warning spam (GPT-Neo defines no pad token); output is identical for batch-of-1.
        out = m.generate(**enc, max_new_tokens=200, do_sample=False,
                         pad_token_id=m_tok.eos_token_id)
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
    model33m_fn = _tinystories_33m_fn() if use_33m else None
    cal.main(judge=judge, model33m_fn=model33m_fn, repeats=repeats)
    print(open(os.path.join(ctx.repo_dir, "eval/calibration.md")).read())


def calibrate_33m(ctx: Ctx) -> None:
    """Append the spec §8 reference (b) — TinyStories-33M completions — to calibration.md.

    For when calibrate() ran with use_33m=False: scores the 200 frozen prefixes completed
    by the published 33M model and appends the mediocre line. Pre-sweep only — it locates
    where achievable small-model quality sits relative to the 4.0 gate.
    """
    import importlib.util
    import statistics

    from eval.judge import LocalQwenJudge

    spec = importlib.util.spec_from_file_location(
        "run_calibration", os.path.join(ctx.repo_dir, "scripts/run_calibration.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)

    judge = LocalQwenJudge()
    med = cal.score_set(judge, cal._load_prefixes(), _tinystories_33m_fn(), label="33M")
    line = f"- mediocre (TinyStories-33M) mean: {statistics.fmean(med):.3f}  (addendum)"
    with open(os.path.join(ctx.repo_dir, "eval/calibration.md"), "a") as f:
        f.write(line + "\n")
    print(line)


def evaluate_all(ctx: Ctx) -> None:
    """Load the judge once and score every finished run that lacks an eval.json."""
    from eval.judge import LocalQwenJudge
    from eval.run_eval import evaluate_run

    from tinychat.sweep import sweep_matrix

    judge = LocalQwenJudge()
    for tier, prec, seed in sweep_matrix():
        rd = os.path.join(ctx.runs_dir, f"{tier}_{prec}_{seed}")
        if not os.path.exists(os.path.join(rd, "DONE")):
            print("skip (not trained):", os.path.basename(rd))
            continue
        if os.path.exists(os.path.join(rd, "eval.json")):
            print("already evaluated:", os.path.basename(rd))
            continue
        r = evaluate_run(rd, judge=judge)
        print(os.path.basename(rd), "mean", round(r["mean"], 3),
              f"CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


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
