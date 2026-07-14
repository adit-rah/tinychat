"""Render the README frontier figure from the committed results distillate.

Unlike scripts/plot_frontier.py (the spec deliverable, which reads raw run dirs),
this reads results/summary.csv so the committed figure is reproducible from
committed data alone. Emits a light and a dark variant for GitHub's <picture>.

    python scripts/plot_frontier_showcase.py    # reads results/summary.csv,
                                                # writes docs/frontier.png + docs/frontier-dark.png
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SUMMARY = os.path.join(ROOT, "results", "summary.csv")

GATE = 4.0
PARAM_LABEL = {"tiny": "1M", "small": "6M", "medium": "16M", "large": "28M"}

THEMES = {
    "light": {
        "out": os.path.join(ROOT, "docs", "frontier.png"),
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e8e7e2",
        "fp16": "#2a78d6",
        "ternary": "#1baf7a",
    },
    "dark": {
        "out": os.path.join(ROOT, "docs", "frontier-dark.png"),
        "surface": "#0d1117",
        "ink": "#f0f0ee",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#24292f",
        "fp16": "#3987e5",
        "ternary": "#199e70",
    },
}


def load_summary():
    with open(SUMMARY) as f:
        rows = list(csv.DictReader(f))
    series = {"fp16": [], "ternary": []}
    for r in rows:
        series[r["precision"]].append({
            "tier": r["tier"],
            "mb": float(r["total_bytes"]) / 1e6,
            "y": float(r["coherence"]),
            "lo": float(r["ci_low"]),
            "hi": float(r["ci_high"]),
        })
    for pts in series.values():
        pts.sort(key=lambda p: p["mb"])
    return series


def render(series, t):
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    ax.set_xscale("log")
    ax.set_ylim(0, 5)
    ax.set_xlim(0.95, 90)

    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(which="both", colors=t["muted"], labelsize=10, length=0, pad=6)

    ticks = [1, 2, 5, 10, 20, 50]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([f"{v} MB" for v in ticks])
    ax.set_yticks(range(6))

    # capability gate
    ax.axhline(GATE, color=t["muted"], linewidth=1.2, linestyle=(0, (5, 4)))
    ax.text(0.985, GATE + 0.09, "capability gate (4.0) · smallest published model to clear it: TinyStories-8M",
            transform=ax.get_yaxis_transform(), ha="right",
            fontsize=9.5, color=t["ink2"])

    for name in ("fp16", "ternary"):
        pts = series[name]
        xs = [p["mb"] for p in pts]
        ys = [p["y"] for p in pts]
        color = t[name]
        # 95% CI whiskers (pooled seeds, n=400)
        for p in pts:
            ax.plot([p["mb"], p["mb"]], [p["lo"], p["hi"]],
                    color=color, linewidth=1.4, alpha=0.55,
                    solid_capstyle="butt", zorder=2)
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=1.6,
                solid_joinstyle="round", solid_capstyle="round",
                label=name, zorder=3)
        # per-marker parameter counts, muted (same params, very different bytes)
        dy = 0.30 if name == "fp16" else -0.34
        va = "bottom" if name == "fp16" else "top"
        for p in pts:
            ax.annotate(PARAM_LABEL[p["tier"]], (p["mb"], p["y"]),
                        textcoords="offset points", xytext=(0, dy * 36),
                        ha="center", va=va, fontsize=8.5, color=t["muted"])
        # direct label at the line end
        end = pts[-1]
        ax.annotate(name, (end["mb"], end["y"]), textcoords="offset points",
                    xytext=(10, -3), fontsize=11, fontweight="bold",
                    color=t["ink"], va="center")

    ax.set_ylabel("coherence (frozen judge, 0–5)", fontsize=10.5, color=t["ink2"])

    leg = ax.legend(loc="upper left", frameon=False, fontsize=10,
                    handlelength=1.6, borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color(t["ink2"])

    fig.text(0.065, 0.955, "How small can English get?",
             fontsize=16, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.065, 0.905,
             "Story coherence vs honest packed bytes · 4 sizes × 2 precisions × 2 seeds on TinyStories · "
             "whiskers: 95% CI, n=400",
             fontsize=10, color=t["ink2"], ha="left")

    fig.subplots_adjust(left=0.065, right=0.93, top=0.84, bottom=0.09)
    fig.savefig(t["out"], facecolor=t["surface"])
    plt.close(fig)
    print(f"Wrote {t['out']}")


def main():
    series = load_summary()
    for theme in THEMES.values():
        render(series, theme)


if __name__ == "__main__":
    main()
