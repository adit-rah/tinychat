"""Produce the headline coherence-vs-bytes frontier plot from all run CSVs.

    python scripts/plot_frontier.py            # reads runs/, writes docs/frontier.png
    python scripts/plot_frontier.py runs out.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinychat.plotting import collect_results, plot_frontier  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs")
    out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "frontier.png")
    results = collect_results(runs_dir)
    headline = plot_frontier(results, out_png)
    print(f"Wrote {out_png}")
    print(f"Headline: {headline}")


if __name__ == "__main__":
    main()
