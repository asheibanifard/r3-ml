"""
Statistically rigorous FPS benchmarking for gaussian_splat_scratch_corrected.

Repeats the FULL benchmark program (fresh process each time, reusing the same
trained checkpoint) N times per screen size -- the process boundary matters:
each invocation experiences its own scheduling, thermal state, cache-cold
effects, and background-process interference, which is exactly the "system
noise" a single process's internal-only repetition (this program's own
20-round GPU-event average) cannot capture on its own. Internal averaging
reduces GPU-launch jitter; external repetition is what mitigates the rest.

For each screen size, collects one FPS sample per repeat for every method
reported in fps_summary.txt, then reports:
  - mean
  - sample standard deviation (ddof=1)
  - 95% confidence interval via the t-distribution (not a fixed z=1.96 --
    correct for the 10-30 sample range this is designed for; converges to the
    z-based interval as n grows, but is meaningfully wider at small n)

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/scratch_gs/benchmark_repeated.py \\
        --sizes 64 128 256 512 1024 2048 \\
        --n_repeats 20
"""
import argparse
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def read_fps_summary(path):
    values = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            key, val = parts
            try:
                values[key] = float(val)
            except ValueError:
                pass
    return values


def run_one_trial(binary, frames_dir, size, checkpoint):
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    subprocess.run(
        [str(binary), str(frames_dir), str(size), str(size), str(checkpoint)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    return read_fps_summary(frames_dir / "fps_summary.txt")


def mean_std_ci95(samples):
    n = len(samples)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        half_width = t_crit * std / np.sqrt(n)
    else:
        half_width = float("nan")
    return mean, std, mean - half_width, mean + half_width


def main():
    ap = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    ap.add_argument("--binary", type=Path, default=script_dir / "gaussian_splat_scratch_corrected")
    ap.add_argument("--checkpoint", type=Path, default=script_dir / "checkpoint.bin")
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n_repeats", type=int, default=20, help="10-30 recommended for statistical rigor")
    ap.add_argument("--scratch_dir", type=Path, default=script_dir / "frames_bench_scratch")
    ap.add_argument("--out_dir", type=Path, default=script_dir / "results_bench_stats")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_repeats < 10:
        print(f"WARNING: --n_repeats={args.n_repeats} is below the recommended 10-30 range "
              f"for stable mean/std/CI estimates.")

    method_keys = None  # discovered from the first trial's fps_summary.txt
    raw_rows = []       # (size, repeat, key, value)
    summary_rows = []   # (size, key, n, mean, std, ci_lo, ci_hi)

    for size in args.sizes:
        print(f"=== size {size}: {args.n_repeats} repeats ===")
        per_key_samples = {}
        for rep in range(args.n_repeats):
            result = run_one_trial(args.binary, args.scratch_dir, size, args.checkpoint)
            if method_keys is None:
                method_keys = [k for k in result if k.endswith("_fps")]
                print(f"  discovered FPS metrics: {method_keys}")
            for k in method_keys:
                if k in result:
                    per_key_samples.setdefault(k, []).append(result[k])
                    raw_rows.append((size, rep, k, result[k]))
            print(f"  repeat {rep + 1}/{args.n_repeats}: "
                  + ", ".join(f"{k}={result.get(k, float('nan')):.2f}" for k in method_keys))

        for k, samples in per_key_samples.items():
            mean, std, lo, hi = mean_std_ci95(samples)
            summary_rows.append((size, k, len(samples), mean, std, lo, hi))
            print(f"  {k}: n={len(samples)} mean={mean:.2f} std={std:.2f} 95% CI=[{lo:.2f}, {hi:.2f}]")

    # ---- raw per-repeat CSV ----
    raw_path = args.out_dir / "fps_raw_repeats.csv"
    with open(raw_path, "w") as f:
        f.write("screen_size,repeat,metric,fps\n")
        for size, rep, k, v in raw_rows:
            f.write(f"{size},{rep},{k},{v:.4f}\n")
    print(f"Saved {raw_path}")

    # ---- summary CSV ----
    summary_path = args.out_dir / "fps_summary_stats.csv"
    with open(summary_path, "w") as f:
        f.write("screen_size,metric,n,mean_fps,std_fps,ci95_lo,ci95_hi\n")
        for size, k, n, mean, std, lo, hi in summary_rows:
            f.write(f"{size},{k},{n},{mean:.4f},{std:.4f},{lo:.4f},{hi:.4f}\n")
    print(f"Saved {summary_path}")

    # ---- plot: mean +/- 95% CI vs screen size ----
    # Only the 3 semantically distinct series (all *_fps duplicate their
    # *_gpu_fps counterpart exactly in this build's benchmark design, so
    # plotting all 6 metric keys would just overdraw 3 pairs of identical
    # lines). Fixed categorical order/colors (never reassigned by rank),
    # validated for CVD-safety: blue / green / magenta.
    def series_for(key):
        by_size = {s: (m, lo, hi) for s, k, n, m, std, lo, hi in summary_rows if k == key}
        sizes_k = sorted(by_size)
        means_k = np.array([by_size[s][0] for s in sizes_k])
        los_k = np.array([by_size[s][1] for s in sizes_k])
        his_k = np.array([by_size[s][2] for s in sizes_k])
        return sizes_k, means_k, [means_k - los_k, his_k - means_k]

    plot_series = [
        ("GT DVR", "dvr_gpu_fps", "#2a78d6"),
        ("Baked + DVR", "baked_gpu_fps", "#008300"),
        ("Live Gaussian rasterizer", "rasterizer_gpu_fps", "#e87ba4"),
    ]

    GRID_COLOR = "#e1e0d9"
    AXIS_COLOR = "#c3c2b7"
    MUTED_INK = "#898781"
    PRIMARY_INK = "#0b0b0b"

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for label, key, color in plot_series:
        if key not in method_keys:
            continue
        sizes_k, means_k, yerr = series_for(key)
        ax.errorbar(sizes_k, means_k, yerr=yerr, marker="o", markersize=8,
                     linewidth=2, capsize=4, color=color, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(args.sizes)
    ax.set_xticklabels([str(s) for s in args.sizes], color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK)
    ax.set_xlabel("Screen size (pixels, square)", color=PRIMARY_INK)
    ax.set_ylabel("FPS", color=PRIMARY_INK)
    ax.set_title(f"FPS vs screen size (mean ± 95% CI, n={args.n_repeats} repeats)", color=PRIMARY_INK)
    legend = ax.legend(frameon=False, labelcolor=PRIMARY_INK)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    plot_path = args.out_dir / "fps_mean_ci95.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")

    print(f"\nAll outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
