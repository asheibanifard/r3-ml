"""
python plot_rq1_multiblock.py \
  --input results/rq1_multiblock/rq1_multiblock_raw.csv \
  --output-dir results/rq1_multiblock/figures

"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_RATIOS = np.array([0.10,0.20,0.30,0.40,0.50,0.60,0.75,1.00])


def validate_input(df: pd.DataFrame, minimum_blocks: int = 5) -> None:
    required = {
        "block_id", "repeat_id", "retention_ratio", "active_gaussians",
        "active_payload_mib", "gaussian_tile_pairs", "median_render_ms",
        "median_fps", "p5_fps",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input CSV contains no rows.")
    if df["block_id"].nunique() < minimum_blocks:
        raise ValueError(
            f"Expected at least {minimum_blocks} blocks, "
            f"found {df['block_id'].nunique()}."
        )
    observed = np.sort(df["retention_ratio"].unique())
    if len(observed) != len(EXPECTED_RATIOS) or not np.allclose(observed, EXPECTED_RATIOS, atol=1e-9):
        raise ValueError(f"Unexpected retention ratios: {observed.tolist()}")
    if df.duplicated(["block_id", "retention_ratio", "repeat_id"]).any():
        raise ValueError("Duplicate block/ratio/repeat rows detected.")
    for (block_id, ratio), group in df.groupby(["block_id", "retention_ratio"]):
        repeats = set(group["repeat_id"].astype(int))
        if repeats != {0,1,2,3,4}:
            raise ValueError(
                f"{block_id}, ratio {ratio}: expected repeats 0--4, "
                f"found {sorted(repeats)}"
            )


def summarise(df: pd.DataFrame):
    block_ratio = (
        df.groupby(["block_id", "retention_ratio"], as_index=False)
        .agg(
            active_gaussians=("active_gaussians", "mean"),
            active_payload_mib=("active_payload_mib", "mean"),
            gaussian_tile_pairs=("gaussian_tile_pairs", "mean"),
            median_render_ms_mean=("median_render_ms", "mean"),
            median_render_ms_std=("median_render_ms", "std"),
            median_fps_mean=("median_fps", "mean"),
            median_fps_std=("median_fps", "std"),
            p5_fps_mean=("p5_fps", "mean"),
        )
        .sort_values(["block_id", "retention_ratio"])
        .reset_index(drop=True)
    )
    block_ratio["active_gaussians"] = block_ratio["active_gaussians"].round().astype(int)
    block_ratio["gaussian_tile_pairs"] = block_ratio["gaussian_tile_pairs"].round().astype(int)

    aggregate = (
        block_ratio.groupby("retention_ratio", as_index=False)
        .agg(
            active_gaussians=("active_gaussians", "mean"),
            active_payload_mib=("active_payload_mib", "mean"),
            gaussian_tile_pairs_mean=("gaussian_tile_pairs", "mean"),
            gaussian_tile_pairs_std=("gaussian_tile_pairs", "std"),
            median_render_ms_mean=("median_render_ms_mean", "mean"),
            median_render_ms_between_block_std=("median_render_ms_mean", "std"),
            median_fps_mean=("median_fps_mean", "mean"),
            median_fps_between_block_std=("median_fps_mean", "std"),
            p5_fps_mean=("p5_fps_mean", "mean"),
        )
        .sort_values("retention_ratio")
        .reset_index(drop=True)
    )
    aggregate["active_gaussians"] = aggregate["active_gaussians"].round().astype(int)
    return block_ratio, aggregate


def save_figure(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fps(block_ratio, aggregate, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for block_id, group in block_ratio.groupby("block_id"):
        group = group.sort_values("active_gaussians")
        ax.errorbar(
            group["active_gaussians"], group["median_fps_mean"],
            yerr=group["median_fps_std"].fillna(0), marker="o",
            linewidth=1.2, capsize=2, alpha=0.8, label=str(block_id)
        )
    ax.errorbar(
        aggregate["active_gaussians"], aggregate["median_fps_mean"],
        yerr=aggregate["median_fps_between_block_std"].fillna(0),
        marker="s", linewidth=2.8, capsize=4, label="Cross-block mean"
    )
    ax.set_xlabel("Active Gaussian count")
    ax.set_ylabel("Median FPS")
    ax.set_title("RQ1: Rendering speed versus active Gaussian count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, outdir / "rq1_fps_vs_active_gaussians")


def plot_time_vs_count(block_ratio, aggregate, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    pooled_x, pooled_y = [], []
    for block_id, group in block_ratio.groupby("block_id"):
        group = group.sort_values("active_gaussians")
        x = group["active_gaussians"].to_numpy(float)
        y = group["median_render_ms_mean"].to_numpy(float)
        ax.errorbar(
            x, y, yerr=group["median_render_ms_std"].fillna(0),
            marker="o", linewidth=1.2, capsize=2, alpha=0.8,
            label=str(block_id)
        )
        pooled_x.extend(x); pooled_y.extend(y)
    pooled_x = np.asarray(pooled_x); pooled_y = np.asarray(pooled_y)
    slope, intercept = np.polyfit(pooled_x, pooled_y, 1)
    fit_x = np.linspace(pooled_x.min(), pooled_x.max(), 200)
    ax.plot(
        fit_x, intercept + slope * fit_x, linestyle="--", linewidth=2.5,
        label=f"Pooled linear fit: {intercept:.3f} + {slope*1000:.3f} ms/1k"
    )
    ax.errorbar(
        aggregate["active_gaussians"], aggregate["median_render_ms_mean"],
        yerr=aggregate["median_render_ms_between_block_std"].fillna(0),
        marker="s", linewidth=2.8, capsize=4, label="Cross-block mean"
    )
    ax.set_xlabel("Active Gaussian count")
    ax.set_ylabel("Median render time (ms)")
    ax.set_title("RQ1: Render time versus active Gaussian count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, outdir / "rq1_render_time_vs_active_gaussians")


def plot_time_vs_pairs(block_ratio, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    all_x, all_y = [], []
    for block_id, group in block_ratio.groupby("block_id"):
        x = group["gaussian_tile_pairs"].to_numpy(float)
        y = group["median_render_ms_mean"].to_numpy(float)
        ax.scatter(x, y, s=40, alpha=0.85, label=str(block_id))
        all_x.extend(x); all_y.extend(y)
    all_x = np.asarray(all_x); all_y = np.asarray(all_y)
    slope, intercept = np.polyfit(all_x, all_y, 1)
    pred = intercept + slope * all_x
    r2 = 1 - np.sum((all_y - pred)**2) / np.sum((all_y - all_y.mean())**2)
    fit_x = np.linspace(all_x.min(), all_x.max(), 200)
    ax.plot(fit_x, intercept + slope*fit_x, linestyle="--", linewidth=2.5,
            label=f"Linear fit, $R^2={r2:.5f}$")
    ax.set_xlabel("Gaussian--tile pair count")
    ax.set_ylabel("Median render time (ms)")
    ax.set_title("RQ1 diagnostic: Render time versus Gaussian--tile pairs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, outdir / "rq1_render_time_vs_tile_pairs")


def plot_normalised(block_ratio, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for block_id, group in block_ratio.groupby("block_id"):
        group = group.sort_values("retention_ratio").copy()
        full_time = float(group.loc[np.isclose(group["retention_ratio"], 1.0), "median_render_ms_mean"].iloc[0])
        ax.plot(group["retention_ratio"]*100, group["median_render_ms_mean"]/full_time,
                marker="o", linewidth=1.4, label=str(block_id))
    ax.plot([10,100],[0.10,1.00], linestyle="--", linewidth=2.0,
            label="Ideal linear scaling")
    ax.set_xlabel("Retained Gaussian payload (%)")
    ax.set_ylabel("Render time / full-payload render time")
    ax.set_title("RQ1: Normalised scalability across blocks")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, outdir / "rq1_normalised_time_vs_retention")


def plot_full_payload(block_ratio, outdir):
    full = block_ratio[np.isclose(block_ratio["retention_ratio"], 1.0)].copy()
    full = full.sort_values("median_render_ms_mean")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(full))
    ax.bar(x, full["median_render_ms_mean"],
           yerr=full["median_render_ms_std"].fillna(0), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(full["block_id"], rotation=30, ha="right")
    ax.set_ylabel("Median render time (ms)")
    ax.set_title("RQ1: Full-payload rendering cost by block")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (_, row) in enumerate(full.iterrows()):
        ax.annotate(f"{int(row['gaussian_tile_pairs']):,} pairs",
                    (i, row["median_render_ms_mean"]), xytext=(0,6),
                    textcoords="offset points", ha="center", fontsize=8)
    save_figure(fig, outdir / "rq1_full_payload_block_comparison")


def block_regressions(block_ratio):
    rows = []
    for block_id, group in block_ratio.groupby("block_id"):
        x = group["active_gaussians"].to_numpy(float)
        y = group["median_render_ms_mean"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope*x
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        rows.append({
            "block_id": block_id,
            "intercept_ms": intercept,
            "slope_ms_per_gaussian": slope,
            "slope_ms_per_1000_gaussians": slope*1000,
            "r_squared": r2,
        })
    return pd.DataFrame(rows).sort_values("block_id")


def main():
    parser = argparse.ArgumentParser(description="Plot multi-block RQ1 results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/rq1_multiblock/figures")
    parser.add_argument("--minimum-blocks", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    validate_input(df, args.minimum_blocks)
    block_ratio, aggregate = summarise(df)
    regressions = block_regressions(block_ratio)

    block_ratio.to_csv(outdir / "rq1_block_ratio_summary.csv", index=False)
    aggregate.to_csv(outdir / "rq1_aggregate_summary.csv", index=False)
    regressions.to_csv(outdir / "rq1_block_regression.csv", index=False)

    plot_fps(block_ratio, aggregate, outdir)
    plot_time_vs_count(block_ratio, aggregate, outdir)
    plot_time_vs_pairs(block_ratio, outdir)
    plot_normalised(block_ratio, outdir)
    plot_full_payload(block_ratio, outdir)

    print(f"Input rows: {len(df)}")
    print(f"Blocks: {df['block_id'].nunique()}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
