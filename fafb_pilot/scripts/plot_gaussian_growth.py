#!/usr/bin/env python3
"""Plot Gaussian population (n_gauss) vs. epoch from one or more blocks'
log.json files, to visualise whether growth (clone/split) and pruning
reach a self-limiting equilibrium or simply saturate at max_gaussians.

Works on partially-complete training runs (plots whatever epochs have been
logged so far), so it can be checked while a run is still in progress.

Usage:
    python plot_gaussian_growth.py \
        --log fafb_pilot/models/blocks_v4_test/b_000/log.json:b_000 \
        --log fafb_pilot/models/blocks_v4_test/b_133/log.json:b_133 \
        --out fafb_pilot/results/figures/gaussian_growth_v4_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order (validated palette, light mode) -- assign in this
# order, never cycled/re-picked per run.
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"


def load_trace(path: Path) -> tuple[list[int], list[int]]:
    with path.open() as f:
        records = json.load(f)
    epochs = [r["epoch"] for r in records]
    n_gauss = [r["n_gauss"] for r in records]
    return epochs, n_gauss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", action="append", required=True,
        help="path/to/log.json:label, repeatable",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-gaussians", type=float, default=None,
                         help="draw a reference line at the configured ceiling")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, spec in enumerate(args.log):
        path_str, _, label = spec.partition(":")
        epochs, n_gauss = load_trace(Path(path_str))
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(epochs, n_gauss, linewidth=2, color=color, label=label or path_str)

    if args.max_gaussians:
        ax.axhline(args.max_gaussians, linestyle="--", linewidth=1,
                   color=MUTED, zorder=0)
        ax.annotate("max_gaussians ceiling", xy=(0, args.max_gaussians),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=MUTED)

    ax.set_xlabel("Epoch", fontsize=11, color=INK)
    ax.set_ylabel("Active Gaussian count", fontsize=11, color=INK)
    ax.set_title("Gaussian population vs. training epoch", fontsize=12, color=INK, pad=14)

    ax.grid(True, linewidth=0.6, color=GRID, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)

    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.tick_params(colors=MUTED, labelsize=9.5)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor="white")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
