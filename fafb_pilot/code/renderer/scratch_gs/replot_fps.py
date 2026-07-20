"""Paper-ready figures from the repeated-benchmark data already saved by
benchmark_repeated.py: FPS vs. resolution, and a companion latency (ms/frame)
figure. Regenerates from the saved CSVs -- does not re-run the (expensive)
20-repeat sweep.

Latency is computed per-repeat (latency_ms = 1000/fps for each of the 20 raw
samples, THEN mean/std/CI taken over those) rather than by transforming the
already-aggregated FPS mean/CI -- avoids bias from applying a nonlinear
transform (1/x) to a point estimate instead of to the underlying samples.

USAGE
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/scratch_gs/replot_fps.py \\
        --raw_csv results_bench_stats/fps_raw_repeats.csv \\
        --out_dir results_bench_stats
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

# Fixed series identity: color + linestyle + marker together, so overlapping
# curves (GT DVR and Baked+DVR use the same underlying DVR kernel and read
# almost identically) stay visually distinguishable rather than one hiding
# the other. Draw order matters here: Baked+DVR (filled square) is drawn
# FIRST so it sits underneath; GT DVR (dashed, HOLLOW circle) is drawn on
# TOP so the filled square is still visible inside/around the open circle
# even where the two series' values are numerically indistinguishable.
PLOT_SERIES = [
    ("Baked + DVR", "baked_gpu_fps", "#008300", "-", "s", False),
    ("GT DVR", "dvr_gpu_fps", "#2a78d6", "--", "o", True),
    ("Live Gaussian rasterizer", "rasterizer_gpu_fps", "#e87ba4", "-", "^", False),
]

FPS_CAPTION = (
    "Figure: GPU-only rendering throughput vs. output resolution. Throughput measured "
    "using CUDA events; the Gaussian rasterizer timing includes tile-count "
    "initialization, Gaussian-to-tile list construction, and tile-based rendering. "
    "Device-to-host transfers, frame export, video generation, and disk I/O are "
    "excluded. Each point reports the mean and 95% confidence interval (Student's "
    "t-distribution) over 20 repeated camera sweeps (fresh process per repeat, 60 "
    "camera angles per sweep) on a synthetic, cache-resident 64³ voxel volume. "
    "Results characterize performance on this specific synthetic benchmark, not a "
    "general claim about large-scale volume rendering."
)
LATENCY_CAPTION = (
    "Figure: Per-frame latency (ms) vs. output resolution, computed as 1000/FPS from "
    "the same measurements as the throughput figure (latency taken per repeat, then "
    "aggregated, not derived by inverting the aggregated FPS). Same timing scope and "
    "caveats as above: GPU-only via CUDA events, synthetic cache-resident 64³ volume, "
    "mean and 95% CI (Student's t) over 20 repeats."
)


def read_raw(path):
    # samples[metric][size] = list of raw fps values across repeats
    samples = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            samples[row["metric"]][int(row["screen_size"])].append(float(row["fps"]))
    return samples


def mean_std_ci95(values):
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half_width = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
    else:
        half_width = float("nan")
    return mean, mean - half_width, mean + half_width


def style_axes(ax, all_sizes, xlabel, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([str(s) for s in all_sizes], color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK)
    ax.set_xlabel(xlabel, color=PRIMARY_INK)
    ax.set_ylabel(ylabel, color=PRIMARY_INK)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)


def make_figure(per_size_stats, all_sizes, ylabel, title, subtitle, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    handles_by_label = {}
    for label, key, color, linestyle, marker, hollow in PLOT_SERIES:
        if key not in per_size_stats:
            continue
        sizes_k = sorted(per_size_stats[key])
        means_k = np.array([per_size_stats[key][s][0] for s in sizes_k])
        los_k = np.array([per_size_stats[key][s][1] for s in sizes_k])
        his_k = np.array([per_size_stats[key][s][2] for s in sizes_k])
        yerr = [means_k - los_k, his_k - means_k]
        mfc = "none" if hollow else color
        mew = 2.2 if hollow else 1.0
        container = ax.errorbar(sizes_k, means_k, yerr=yerr, label=label, color=color,
                     linestyle=linestyle, marker=marker, markersize=9,
                     markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=mew,
                     linewidth=2.2, capsize=7, capthick=2, elinewidth=2)
        handles_by_label[label] = container

    style_axes(ax, all_sizes, "Output resolution (pixels per dimension)", ylabel)
    fig.suptitle(title, fontsize=14, fontweight="bold", color=PRIMARY_INK, y=0.98)
    ax.set_title(subtitle, fontsize=10, color=MUTED_INK, pad=10)
    # Legend in the requested reading order (GT DVR, Baked+DVR, rasterizer)
    # regardless of draw order (which controls z-stacking, not legend order).
    legend_order = ["GT DVR", "Baked + DVR", "Live Gaussian rasterizer"]
    ordered_labels = [lbl for lbl in legend_order if lbl in handles_by_label]
    ax.legend([handles_by_label[l] for l in ordered_labels], ordered_labels,
              frameon=False, labelcolor=PRIMARY_INK)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    ap.add_argument("--raw_csv", type=Path, default=script_dir / "results_bench_stats" / "fps_raw_repeats.csv")
    ap.add_argument("--out_dir", type=Path, default=script_dir / "results_bench_stats")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_raw(args.raw_csv)
    all_sizes = sorted({s for sizes in raw.values() for s in sizes})
    n_repeats = len(next(iter(next(iter(raw.values())).values())))

    fps_stats = defaultdict(dict)
    latency_stats = defaultdict(dict)
    for metric, by_size in raw.items():
        for size, fps_values in by_size.items():
            fps_stats[metric][size] = mean_std_ci95(fps_values)
            latency_values = [1000.0 / v for v in fps_values]
            latency_stats[metric][size] = mean_std_ci95(latency_values)

    make_figure(
        fps_stats, all_sizes, "Throughput (FPS)",
        "GPU rendering throughput versus output resolution",
        f"Mean and 95% confidence interval over {n_repeats} repeated benchmarks",
        args.out_dir / "fps_vs_resolution.png",
    )
    make_figure(
        latency_stats, all_sizes, "Latency (ms / frame)",
        "GPU rendering latency versus output resolution",
        f"Mean and 95% confidence interval over {n_repeats} repeated benchmarks",
        args.out_dir / "latency_vs_resolution.png",
    )

    captions_path = args.out_dir / "figure_captions.txt"
    with open(captions_path, "w") as f:
        f.write(FPS_CAPTION + "\n\n" + LATENCY_CAPTION + "\n")
    print(f"Saved {captions_path}")

    # Latency summary table (handy for sanity-checking numbers quoted in text)
    print("\nLatency (ms/frame), mean [95% CI]:")
    for size in all_sizes:
        parts = []
        for label, key, *_ in PLOT_SERIES:
            if size in latency_stats.get(key, {}):
                mean, lo, hi = latency_stats[key][size]
                parts.append(f"{label}={mean:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"  {size:5d}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
