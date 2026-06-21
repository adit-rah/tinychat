"""Headline deliverable (spec §4/§10): coherence vs total-bytes frontier.

`collect_results` reads every run dir (meta.json + metrics.csv + optional eval.json) into
flat rows. `plot_frontier` draws coherence (y) against total bytes (log x) as two curves
(fp16, ternary), one marker per tier, applies the capability gate, annotates the
smallest-bytes capable point on each curve, and titles the figure with the global-minimum
headline.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

from .config import TIERS

COHERENCE_THRESHOLD = 4.0
PPL_MULTIPLIER = 1.5
_TIER_ORDER = list(TIERS)


def _last_val_ppl(run_dir: str):
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["val_ppl"]) if rows else None


def collect_results(runs_dir: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        meta_p = os.path.join(d, "meta.json")
        if not os.path.isfile(meta_p):
            continue
        meta = json.load(open(meta_p))
        tier, precision, seed = name.rsplit("_", 2)
        row = {
            "tier": tier,
            "precision": precision,
            "seed": int(seed),
            "total_bytes": meta["total_bytes"],
            "val_ppl": _last_val_ppl(d),
            "judge_mean": None,
            "ci_low": None,
            "ci_high": None,
        }
        ev = os.path.join(d, "eval.json")
        if os.path.isfile(ev):
            e = json.load(open(ev))
            row.update(judge_mean=e["mean"], ci_low=e["ci_low"], ci_high=e["ci_high"])
        rows.append(row)
    return rows


def _aggregate(rows):
    """Mean across seeds -> one point per (precision, tier)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["precision"], r["tier"])].append(r)
    points = {}
    for (precision, tier), rs in groups.items():
        bytes_ = sum(r["total_bytes"] for r in rs) / len(rs)
        ppls = [r["val_ppl"] for r in rs if r["val_ppl"] is not None]
        judges = [r["judge_mean"] for r in rs if r["judge_mean"] is not None]
        points[(precision, tier)] = {
            "total_bytes": bytes_,
            "val_ppl": (sum(ppls) / len(ppls)) if ppls else None,
            "judge_mean": (sum(judges) / len(judges)) if judges else None,
        }
    return points


def plot_frontier(results: list[dict], out_png: str) -> dict:
    """Draw the frontier, annotate the smallest capable point per curve, return the headline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _aggregate(results)
    fp16_ppls = [
        r["val_ppl"] for r in results
        if r["precision"] == "fp16" and r["val_ppl"] is not None
    ]
    best_fp16 = min(fp16_ppls) if fp16_ppls else None
    ppl_gate = PPL_MULTIPLIER * best_fp16 if best_fp16 is not None else None

    def capable(p):
        if p["judge_mean"] is None or p["judge_mean"] < COHERENCE_THRESHOLD:
            return False
        if ppl_gate is None or p["val_ppl"] is None:
            return False
        return p["val_ppl"] <= ppl_gate

    fig, ax = plt.subplots(figsize=(7, 5))
    headline = None
    colors = {"fp16": "tab:blue", "ternary": "tab:red"}
    for precision in ("fp16", "ternary"):
        pts = [
            (points[(precision, t)]["total_bytes"], points[(precision, t)]["judge_mean"],
             t, points[(precision, t)])
            for t in _TIER_ORDER
            if (precision, t) in points and points[(precision, t)]["judge_mean"] is not None
        ]
        pts.sort(key=lambda z: z[0])
        if pts:
            xs, ys = [z[0] for z in pts], [z[1] for z in pts]
            ax.plot(xs, ys, "-o", color=colors[precision], label=precision)
        capables = [z for z in pts if capable(z[3])]
        if capables:
            bx, by, btier, _ = min(capables, key=lambda z: z[0])
            ax.annotate(f"{precision} capable\n{btier} ({bx/1e6:.1f} MB)",
                        (bx, by), textcoords="offset points", xytext=(6, -18),
                        fontsize=8, color=colors[precision])
            if headline is None or bx < headline["total_bytes"]:
                headline = {"precision": precision, "tier": btier, "total_bytes": bx}

    ax.axhline(COHERENCE_THRESHOLD, ls="--", color="gray", lw=1, label="gate (4.0)")
    ax.set_xscale("log")
    ax.set_xlabel("total model size (bytes, log scale)")
    ax.set_ylabel("coherence (mean judge score, 0–5)")
    if headline:
        title = (f"Smallest capable English LM: {headline['total_bytes']/1e6:.2f} MB "
                 f"({headline['precision']}, {headline['tier']})")
    else:
        title = "Coherence vs bytes frontier (no config cleared the capability gate)"
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return headline or {}
