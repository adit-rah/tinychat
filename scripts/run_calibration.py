"""Sweep-blind calibration + judge-reliability pass (spec §8). ONE-TIME, before the sweep.

Scores three reference sets through the EXACT frozen rubric + judge, over the 200 frozen
prefixes, and reports the numbers needed to confirm where 4.0 sits and whether the judge is
reliable enough to trust the gate:

  (a) real TinyStories gold continuations  (good)
  (b) a published small model's completions (e.g. TinyStories-33M)  (mediocre/good)
  (c) deliberately degenerate text (shuffled / truncated / repetitive)  (bad)

Outputs the means per set, the rank-ordering check (a > b > c), the intra-judge std
(re-scoring the same set), and the 95% CI width on N=200. Writes a summary to
eval/calibration.md (commit it; it freezes the gate's validity, not just its number).

Requires GPU (judge) + the committed tokenizer + eval/prefixes.jsonl. Deferred to the
Kaggle session; this file is the runnable spec of that pass.

    python scripts/run_calibration.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.gate import mean_and_ci  # noqa: E402
from eval.judge import per_completion_score  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PREFIXES = os.path.join(ROOT, "eval", "prefixes.jsonl")
OUT = os.path.join(ROOT, "eval", "calibration.md")


def _load_prefixes():
    with open(PREFIXES) as f:
        return [json.loads(line) for line in f if line.strip()]


def _degenerate(text: str) -> str:
    """A deliberately bad continuation: repeat the first word many times."""
    first = (text.split() or ["the"])[0]
    return (first + " ") * 30


def score_set(judge, prefixes, completion_fn, label: str = "") -> list[float]:
    scores = []
    for i, item in enumerate(prefixes, 1):
        comp = completion_fn(item)
        scores.append(per_completion_score(judge.score(item["prefix"], comp)))
        if i % 50 == 0 or i == len(prefixes):
            print(f"[{label}] scored {i}/{len(prefixes)} | running mean "
                  f"{statistics.fmean(scores):.3f}", flush=True)
    return scores


def reference_line(model_id: str, sampled: bool, n: int, mean: float, half: float) -> str:
    """The calibration.md ledger line for one published reference model."""
    short = model_id.split("/")[-1]
    decode = "sampled temp1.0/topk40" if sampled else "greedy"
    return (f"- reference ({short}, {decode}, n={n}) mean: "
            f"{mean:.3f}  (95% CI ±{half:.3f})  (addendum)")


def score_set_detailed(judge, prefixes, completion_fn, label: str = "",
                       detail_path: str | None = None) -> list[dict]:
    """Like score_set, but keeps per-completion axis scores (for axis-level anchors,
    e.g. the grammar/T1 analysis). Appends one JSON row per completion to
    `detail_path` when given."""
    from eval.judge import AXES

    rows = []
    f = open(detail_path, "a") if detail_path else None
    for i, item in enumerate(prefixes, 1):
        axes = judge.score(item["prefix"], completion_fn(item))
        row = {"id": item.get("id", i), **{a: axes[a] for a in AXES},
               "score": per_completion_score(axes)}
        rows.append(row)
        if f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        if i % 50 == 0 or i == len(prefixes):
            mean = statistics.fmean(r["score"] for r in rows)
            print(f"[{label}] scored {i}/{len(prefixes)} | running mean {mean:.3f}",
                  flush=True)
    if f:
        f.close()
    return rows


def score_reference(judge, model_id: str, sampled: bool = False,
                    n_prefixes: int | None = None,
                    detail_path: str | None = None) -> tuple[float, float, int]:
    """Score one published TinyStories checkpoint; returns (mean, ci_half, n)."""
    from nanofable.kaggle import _tinystories_ref_fn

    prefixes = _load_prefixes()[:n_prefixes]
    short = model_id.split("/")[-1]
    rows = score_set_detailed(judge, prefixes,
                              _tinystories_ref_fn(model_id, sampled=sampled),
                              label=f"{short}-{'sampled' if sampled else 'greedy'}",
                              detail_path=detail_path)
    mean, half = mean_and_ci([r["score"] for r in rows])
    return mean, half, len(prefixes)


def score_gold_detailed(judge, detail_path: str, n_prefixes: int | None = None) -> None:
    """Per-axis scores for the gold continuations (T1 anchor needs gold's grammar)."""
    prefixes = _load_prefixes()[:n_prefixes]
    score_set_detailed(judge, prefixes, lambda it: it["gold_continuation"],
                       label="gold-axes", detail_path=detail_path)


def main(judge=None, model33m_fn=None, repeats: int = 3):
    if judge is None:
        from eval.judge import LocalQwenJudge

        judge = LocalQwenJudge()
    prefixes = _load_prefixes()

    good = score_set(judge, prefixes, lambda it: it["gold_continuation"], label="good")
    bad = score_set(judge, prefixes, lambda it: _degenerate(it["gold_continuation"]),
                    label="bad")
    mediocre = (
        score_set(judge, prefixes, model33m_fn, label="33M") if model33m_fn else None
    )

    # intra-judge stability: re-score the good set `repeats` times, report std of the mean.
    repeat_means = [
        statistics.fmean(score_set(judge, prefixes,
                                   lambda it: it["gold_continuation"],
                                   label=f"stability {r + 1}/{repeats}"))
        for r in range(repeats)
    ]
    intra_std = statistics.pstdev(repeat_means) if len(repeat_means) > 1 else 0.0

    gmean, ghalf = mean_and_ci(good)
    bmean, _ = mean_and_ci(bad)
    good_bad_gap = gmean - bmean

    lines = [
        "# Calibration & Judge Reliability (FROZEN)\n",
        f"- good (gold) mean: {gmean:.3f}  (95% CI ±{ghalf:.3f})",
        f"- bad (degenerate) mean: {bmean:.3f}",
    ]
    if mediocre is not None:
        lines.append(f"- mediocre (TinyStories-33M) mean: {statistics.fmean(mediocre):.3f}")
    lines += [
        f"- good−bad gap: {good_bad_gap:.3f}",
        f"- intra-judge std (mean over {repeats} re-scores): {intra_std:.4f}",
        "",
        f"- rank-ordering good > bad: {gmean > bmean}",
        f"- judge reliable (intra_std < good−bad gap): {intra_std < good_bad_gap}",
        "",
        "If rank-ordering fails OR intra_std >= good−bad gap, STOP and upgrade the judge "
        "before freezing the gate (spec §8).",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lstrip().startswith("{"):
        # Ladder-worker mode (see nanofable.kaggle.calibrate_ladder): score the given
        # reference models with one judge and write JSON lines to cfg["out"].
        from eval.judge import LocalQwenJudge

        _cfg = json.loads(sys.argv[1])
        _judge = LocalQwenJudge()
        _detail_dir = _cfg.get("detail_dir")
        if _detail_dir:
            os.makedirs(_detail_dir, exist_ok=True)
        with open(_cfg["out"], "a") as _f:
            for _mid in _cfg["model_ids"]:
                _short = _mid.split("/")[-1]
                _mean, _half, _n = score_reference(
                    _judge, _mid, sampled=_cfg["sampled"],
                    n_prefixes=_cfg["n_prefixes"],
                    detail_path=(os.path.join(_detail_dir, f"{_short}.axes.jsonl")
                                 if _detail_dir else None))
                _f.write(json.dumps({"model_id": _mid, "mean": _mean,
                                     "ci_half": _half, "n": _n}) + "\n")
                _f.flush()
            if _cfg.get("gold") and _detail_dir:
                score_gold_detailed(_judge,
                                    os.path.join(_detail_dir, "gold.axes.jsonl"),
                                    n_prefixes=_cfg["n_prefixes"])
    else:
        main()
