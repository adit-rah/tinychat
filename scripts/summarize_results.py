"""Distill the sweep into one committed CSV: results/summary.csv.

Reads every run dir (eval.json + metrics.csv), pools per-prefix judge scores across
seeds, recomputes params/bytes from the frozen configs via count_bytes(), and writes
one row per (tier, precision). This is the committed, inspectable form of the numbers
the README reports; raw per-run outputs stay local.

    python scripts/summarize_results.py                 # reads local/hf_runs, writes results/summary.csv
    python scripts/summarize_results.py <runs_dir> <out_csv>
"""

from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from eval.gate import mean_and_ci  # noqa: E402
from nanofable.bytes import count_bytes  # noqa: E402
from nanofable.config import TIERS  # noqa: E402
from nanofable.model import Transformer  # noqa: E402

AXES = ("grammar", "consistency", "completes")


def _final_val_ppl(run_dir: str) -> float | None:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["val_ppl"]) if rows else None


def _run_dirs(runs_dir: str, tier: str, precision: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(runs_dir)):
        parts = name.rsplit("_", 2)
        if len(parts) == 3 and parts[0] == tier and parts[1] == precision:
            out.append(os.path.join(runs_dir, name))
    return out


def summarize(runs_dir: str) -> list[dict]:
    rows = []
    for tier, cfg in TIERS.items():
        for precision in ("fp16", "ternary"):
            dirs = _run_dirs(runs_dir, tier, precision)
            scores, axes = [], {a: [] for a in AXES}
            ppls, seeds = [], 0
            for d in dirs:
                ev_path = os.path.join(d, "eval.json")
                if not os.path.isfile(ev_path):
                    continue
                seeds += 1
                ev = json.load(open(ev_path))
                for p in ev["per_prefix"]:
                    scores.append(p["score"])
                    for a in AXES:
                        axes[a].append(p["axes"][a])
                ppl = _final_val_ppl(d)
                if ppl is not None:
                    ppls.append(ppl)
            if not scores:
                continue
            mean, half = mean_and_ci(scores)
            model = Transformer(cfg, precision)
            rows.append({
                "config": f"{tier}_{precision}",
                "tier": tier,
                "precision": precision,
                "params": sum(p.numel() for p in model.parameters()),
                "total_bytes": round(count_bytes(model, precision)["total"]),
                "coherence": round(mean, 4),
                "ci_low": round(mean - half, 4),
                "ci_high": round(mean + half, 4),
                **{a: round(sum(v) / len(v), 4) for a, v in axes.items()},
                "val_ppl": round(sum(ppls) / len(ppls), 4) if ppls else None,
                "n_completions": len(scores),
                "n_seeds": seeds,
            })
    return rows


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "local", "hf_runs")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "summary.csv")
    rows = summarize(runs_dir)
    if not rows:
        sys.exit(f"no runs with eval.json found under {runs_dir}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
