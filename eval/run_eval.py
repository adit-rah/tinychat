"""Evaluate a trained run: complete the 200 frozen prefixes, judge them, write eval.json.

    python eval/run_eval.py runs/tiny_fp16_0

Writes runs/<...>/eval.json with per-prefix axis scores, the mean per-completion score, and
its 95% CI. The judge defaults to the frozen LocalQwenJudge (GPU). Requires the committed
tokenizer and eval/prefixes.jsonl.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.gate import mean_and_ci  # noqa: E402
from eval.judge import per_completion_score  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PREFIXES = os.path.join(ROOT, "eval", "prefixes.jsonl")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizer", "tokenizer.json")


def load_prefixes(path: str | None = None) -> list[dict]:
    with open(path or PREFIXES) as f:  # resolved at call time, not def time
        return [json.loads(line) for line in f if line.strip()]


def load_model_from_run(run_dir: str):
    import torch

    from nanofable.config import ModelConfig
    from nanofable.model import build_model
    from nanofable.train import load_latest

    meta = json.loads(open(os.path.join(run_dir, "meta.json")).read())
    cfg = ModelConfig(**meta["tier"])
    model = build_model(cfg, meta["precision"])
    ckpt = load_latest(run_dir)
    model.load_state_dict(ckpt["model"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def _generate(model, tok, prefix, **kwargs):
    """Indirection over nanofable.generate.generate (patchable in tests)."""
    from nanofable.generate import generate

    return generate(model, tok, prefix, **kwargs)


def evaluate_run(run_dir: str, judge=None, max_new_tokens: int = 200,
                 **decode_kwargs) -> dict:
    """Judge one run. `decode_kwargs` (temperature/top_k) pass through to `generate`.

    eval.json stores, per prefix, the completion text and the judge's raw output —
    zeros must be auditable (real verdict vs parse failure), so the evidence is kept.
    """
    from nanofable.tokenizer import load_tokenizer

    if judge is None:
        from eval.judge import LocalQwenJudge

        judge = LocalQwenJudge()

    tok = load_tokenizer(TOKENIZER)
    model = load_model_from_run(run_dir)
    prefixes = load_prefixes()

    per_prefix = []
    completion_scores = []
    for item in prefixes:
        completion = _generate(model, tok, item["prefix"],
                               max_new_tokens=max_new_tokens, seed=0, **decode_kwargs)
        scored = dict(judge.score(item["prefix"], completion))
        raw = scored.pop("raw", "")
        parsed = scored.pop("parsed", True)
        score = per_completion_score(scored)
        per_prefix.append({"id": item["id"], "completion": completion, "axes": scored,
                           "parsed": parsed, "raw": raw, "score": score})
        completion_scores.append(score)

    mean, half = mean_and_ci(completion_scores)
    result = {
        "run_dir": os.path.basename(run_dir.rstrip("/")),
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n": len(completion_scores),
        "n_parse_failures": sum(1 for r in per_prefix if not r["parsed"]),
        "decode_kwargs": {"max_new_tokens": max_new_tokens, **decode_kwargs},
        "per_prefix": per_prefix,
    }
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def evaluate_pending(runs_root: str, judge=None, claim: bool = False,
                     **eval_kwargs) -> list[str]:
    """Score every DONE run under `runs_root` that lacks an eval.json.

    `claim=True` makes the loop safe to run in multiple worker processes at once (each
    run is scored by exactly one worker, via an EVAL_CLAIM marker — distinct from the
    training CLAIM). The judge is created once if not supplied. Returns the run dirs
    scored by this call.
    """
    from nanofable.sweep import claim_run, run_dir_for, sweep_matrix

    if judge is None:
        from eval.judge import LocalQwenJudge

        judge = LocalQwenJudge()
    done = []
    for tier, prec, seed in sweep_matrix():
        rd = run_dir_for(runs_root, tier, prec, seed)
        if not os.path.exists(os.path.join(rd, "DONE")):
            print("skip (not trained):", os.path.basename(rd), flush=True)
            continue
        if os.path.exists(os.path.join(rd, "eval.json")):
            print("already evaluated:", os.path.basename(rd), flush=True)
            continue
        if claim and not claim_run(rd, name="EVAL_CLAIM"):
            continue
        r = evaluate_run(rd, judge=judge, **eval_kwargs)
        print(os.path.basename(rd), "mean", round(r["mean"], 3),
              f"CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
              f"parse_failures {r['n_parse_failures']}", flush=True)
        done.append(rd)
    return done


if __name__ == "__main__":
    if sys.argv[1].lstrip().startswith("{"):  # eval-worker mode (see kaggle.evaluate_all)
        _cfg = json.loads(sys.argv[1])
        evaluate_pending(_cfg["runs_root"], claim=True, **_cfg["eval_kwargs"])
    else:
        out = evaluate_run(sys.argv[1])
        print(f"{out['run_dir']}: mean={out['mean']:.3f} "
              f"CI=[{out['ci_low']:.3f}, {out['ci_high']:.3f}]")
