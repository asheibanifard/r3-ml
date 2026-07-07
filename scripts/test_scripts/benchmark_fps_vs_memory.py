#!/usr/bin/env python3
"""Benchmark splat-MIP rendering FPS vs. block size (N Gaussians) under
emulated GPU-memory budgets, to derive a hardware-transferable metric:

    Gaussians per GB of GPU memory  (= N / memory_budget_GB)

"FPS" = 1 / (time for one render_splatted_mips() call), which renders all
three canonical orthographic MIP views (xy, xz, yz) via the fused CUDA
splat_mip kernel — the actual production splatting path (scripts/render_scripts/render_mip.py).

"Block size" = N, the number of Gaussians — what actually drives the splat
kernel's per-frame cost (this process has one physical GPU, so instead of
testing different hardware we cap this GPU's visible memory at several
budgets via torch.cuda.set_per_process_memory_fraction, and sweep N at each
budget until it OOMs).

Usage
-----
    /venv/r3-ml/bin/python3 scripts/test_scripts/benchmark_fps_vs_memory.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()
from _3dgs._3dgs import AABB, GaussianCloud, VolumeDataset, render_splatted_mips


def make_cfg():
    return argparse.Namespace(
        scale_min_clamp=1e-5,
        mahal_max_clamp=20.0,
        init_scale=0.05,
        init_inten=0.1,
        init_scale_z_factor=1.0,
        swc_path=None,
    )


def time_one_config(n: int, dataset, aabb, cfg, device, reps: int, warmup: int):
    """Build an N-Gaussian cloud and time render_splatted_mips(). Returns
    (elapsed_ms_per_call, peak_mem_gb) or raises torch.cuda.OutOfMemoryError."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    gc = GaussianCloud(n, aabb, device, cfg)  # random init (no SWC)

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(warmup):
            render_splatted_mips(gc, dataset, cfg)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            render_splatted_mips(gc, dataset, cfg)
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / reps * 1000
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    del gc
    torch.cuda.empty_cache()
    return elapsed_ms, peak_mem_gb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=128, help="H=W=D of the dummy volume (view resolution)")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--memory_caps_gb", type=float, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--n_values", type=int, nargs="+",
                   default=[500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
                            200000, 500000, 1000000, 2000000, 5000000, 10000000])
    p.add_argument("--out", default="results/fps_vs_memory_benchmark.json")
    cfg_args = p.parse_args()

    device = torch.device("cuda")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_properties(0).name}  total_mem={total_mem_gb:.1f}GB")

    aabb = AABB.unit()
    cfg = make_cfg()
    res = cfg_args.resolution
    dataset = VolumeDataset(torch.zeros(res, res, res), aabb, cfg)

    results = []
    for cap_gb in cfg_args.memory_caps_gb:
        fraction = min(cap_gb / total_mem_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"\n=== memory cap: {cap_gb:g} GB (fraction={fraction:.4f}) ===")

        for n in cfg_args.n_values:
            try:
                ms, peak_gb = time_one_config(n, dataset, aabb, cfg, device, cfg_args.reps, cfg_args.warmup)
                fps = 1000.0 / ms
                results.append({
                    "memory_cap_gb": cap_gb, "n": n, "ms_per_frame": round(ms, 3),
                    "fps": round(fps, 2), "peak_mem_gb": round(peak_gb, 4), "oom": False,
                })
                print(f"  N={n:>9,d}  {ms:8.3f} ms/frame  {fps:8.2f} FPS  peak_mem={peak_gb:.3f} GB")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results.append({"memory_cap_gb": cap_gb, "n": n, "ms_per_frame": None,
                                 "fps": None, "peak_mem_gb": None, "oom": True})
                print(f"  N={n:>9,d}  OOM at {cap_gb:g} GB cap — stopping this series")
                break

    # reset to unrestricted before exiting
    torch.cuda.set_per_process_memory_fraction(1.0, device=0)

    out_path = Path(cfg_args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "gpu": torch.cuda.get_device_properties(0).name,
        "total_mem_gb": total_mem_gb,
        "resolution": res,
        "results": results,
    }, indent=2))
    print(f"\nSaved {len(results)} data points -> {out_path}")


if __name__ == "__main__":
    main()
