#!/usr/bin/env python3
"""
Stitch an NxNxN neighborhood of smoke_data blocks into one volume and train
a single GaussianCloud on it (as opposed to training each block separately).

Steps:
  1. Stitch the `raw` datasets of the N^3 neighboring blocks at
     (base_z..base_z+N-1, base_y..base_y+N-1, base_x..base_x+N-1) into one
     (N*50)^3 volume, written to data/smoke_data/stitched_blocks/.
  2. Train via scripts/_3dgs/_3dgs.py (subprocess) using configs/smoke_config.yml,
     saving a checkpoint every --ckpt_epoch_interval epochs.
  3. Post-process: compute {psnr, ssim, lpips} at init, every saved epoch
     checkpoint, and best; save init.pdf / best.pdf (mid-slice GT/pred/diff)
     and best_splat_render_mip.pdf (true CUDA splat-MIP render, xy/xz/yz).
  4. Delete everything else (last.pth, epoch_*.pth, train.log, log.json,
     config.json) once metrics/visuals have been extracted from it.

Usage
-----
    /venv/r3-ml/bin/python3 scripts/train_scripts/train_stitched_neighborhood.py \\
        --base_z 0 --base_y 1 --base_x 6 --n 3
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks_dir", default="data/smoke_data/blocks")
    p.add_argument("--stitched_dir", default="data/smoke_data/stitched_blocks")
    p.add_argument("--models_dir", default="models_smoke")
    p.add_argument("--config", default="configs/smoke_config.yml")
    p.add_argument("--base_z", type=int, required=True)
    p.add_argument("--base_y", type=int, required=True)
    p.add_argument("--base_x", type=int, required=True)
    p.add_argument("--n", type=int, default=3, help="neighborhood is n x n x n blocks")
    p.add_argument("--ckpt_epoch_interval", type=int, default=50)
    return p.parse_args()


def stitch_raw_blocks(blocks_dir: Path, base_z: int, base_y: int, base_x: int, n: int) -> np.ndarray:
    """Stitch the `raw` datasets of the n^3 neighboring blocks into one volume.

    Blocks at the far edge of the original volume can be smaller than
    block_size (e.g. smoke_data's z axis is 125 = 2x50 + 25, so the last
    z-block is only 25 deep) — sizes are read per-block rather than assumed
    uniform, and offsets accumulate accordingly along each axis.
    """
    raws = {}
    for dz in range(n):
        for dy in range(n):
            for dx in range(n):
                path = blocks_dir / f"block_z{base_z+dz}_y{base_y+dy}_x{base_x+dx}.h5"
                if not path.exists():
                    raise FileNotFoundError(f"Missing neighbor block: {path}")
                with h5py.File(path, "r") as fh:
                    raws[(dz, dy, dx)] = fh["raw"][:]

    z_sizes = [raws[(dz, 0, 0)].shape[0] for dz in range(n)]
    y_sizes = [raws[(0, dy, 0)].shape[1] for dy in range(n)]
    x_sizes = [raws[(0, 0, dx)].shape[2] for dx in range(n)]
    z_off = np.concatenate([[0], np.cumsum(z_sizes)])
    y_off = np.concatenate([[0], np.cumsum(y_sizes)])
    x_off = np.concatenate([[0], np.cumsum(x_sizes)])

    combined = np.zeros((z_off[-1], y_off[-1], x_off[-1]), dtype=next(iter(raws.values())).dtype)
    for (dz, dy, dx), raw in raws.items():
        combined[z_off[dz]:z_off[dz+1], y_off[dy]:y_off[dy+1], x_off[dx]:x_off[dx+1]] = raw
    return combined


def train_stitched(volume_path: Path, model_dir: Path, log_file: Path,
                    python: str, script: str, config_path: Path, ckpt_epoch_interval: int) -> tuple[bool, float]:
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, script,
        "--config", str(config_path),
        "--volume", str(volume_path),
        "--out",    str(model_dir),
        "--device", "cuda",
        "--use_kernel",
        "--flat_out",
        "--no_swc_init",
        "--no_wandb",
        "--ckpt_epoch_interval", str(ckpt_epoch_interval),
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def _load_3dgs_modules(project_root: Path):
    sys.path.insert(0, str(project_root / "scripts"))
    import _3dgs._3dgs as _mod
    _mod.USE_CUDA_KERNEL = True
    _mod._load_3dgs_kernel()
    from _3dgs._3dgs import AABB, GaussianCloud, VolumeDataset, vol_psnr, _ssim_2d, render_splatted_mips
    from _3dgs._3dgs_training import _load_volume, _visualize_middle_slices
    return {
        "AABB": AABB, "GaussianCloud": GaussianCloud, "VolumeDataset": VolumeDataset,
        "vol_psnr": vol_psnr, "_ssim_2d": _ssim_2d, "render_splatted_mips": render_splatted_mips,
        "_load_volume": _load_volume, "_visualize_middle_slices": _visualize_middle_slices,
    }


def _mid_xy_slice(gc, dataset, device):
    D, H, W = dataset.D, dataset.H, dataset.W
    mid_d = D // 2
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    zz = torch.full_like(xx, (2 * mid_d / (D - 1) - 1))
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    with torch.no_grad():
        pred = gc.forward(pts, chunk_n=1024).reshape(H, W).clamp(0.0, 1.0)
    gt = dataset.vol[mid_d].to(device)
    return pred, gt


def compute_metrics(gc, dataset, cfg, device, lpips_model, mods) -> dict:
    psnr = mods["vol_psnr"](gc, dataset, cfg)
    pred, gt = _mid_xy_slice(gc, dataset, device)
    pred_b, gt_b = pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0)
    ssim = 1.0 - mods["_ssim_2d"](pred_b, gt_b).item()
    pred_rgb = pred_b.repeat(1, 3, 1, 1) * 2 - 1
    gt_rgb = gt_b.repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        lp = lpips_model(pred_rgb, gt_rgb).item()
    return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def save_splat_mip_pdf(gc, dataset, cfg, out_path: Path, mods):
    mips = mods["render_splatted_mips"](gc, dataset, cfg, depth_samples=32)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, img) in zip(axs, mips.items()):
        ax.imshow(img.cpu().numpy(), cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(f'splat MIP: {name}')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def postprocess(model_dir: Path, volume_path: Path, device, lpips_model, mods: dict) -> dict:
    cfg_dict = json.loads((model_dir / "config.json").read_text())
    cfg = argparse.Namespace(**cfg_dict)

    volume, _, _ = mods["_load_volume"](str(volume_path))
    aabb = mods["AABB"].unit()
    dataset = mods["VolumeDataset"](volume, aabb, cfg)

    # every checkpoint we want metrics for: init, periodic epoch_NNNN, best
    ckpts = {"init": model_dir / "init.pth"}
    for f in sorted(model_dir.glob("epoch_*.pth")):
        ckpts[f.stem] = f
    ckpts["best"] = model_dir / "best.pth"

    metrics = {}
    gc_init = gc_best = None
    for tag, ckpt_path in ckpts.items():
        gc = mods["GaussianCloud"].load(ckpt_path, aabb, device, cfg)
        metrics[tag] = compute_metrics(gc, dataset, cfg, device, lpips_model, mods)
        if tag == "init":
            gc_init = gc
        if tag == "best":
            gc_best = gc

    for tag, gc in [("init", gc_init), ("best", gc_best)]:
        vis_path = mods["_visualize_middle_slices"](dataset.vol, gc, model_dir, 0, device, ext="pdf")
        if vis_path:
            Path(vis_path).replace(model_dir / f"{tag}.pdf")

    save_splat_mip_pdf(gc_best, dataset, cfg, model_dir / "best_splat_render_mip.pdf", mods)

    keep = {"init.pth", "best.pth", "init.pdf", "best.pdf", "best_splat_render_mip.pdf", "metrics.json"}
    for f in model_dir.iterdir():
        if f.name not in keep:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    cfg = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    blocks_dir = project_root / cfg.blocks_dir
    stitched_dir = project_root / cfg.stitched_dir
    models_root = project_root / cfg.models_dir
    config_path = project_root / cfg.config
    script_path = project_root / "scripts" / "_3dgs" / "_3dgs.py"
    python_exe = sys.executable

    tag = f"z{cfg.base_z}_y{cfg.base_y}_x{cfg.base_x}"
    model_dir = models_root / f"stitched_{tag}"
    log_file = model_dir / "train_subprocess.log"
    stitched_dir.mkdir(parents=True, exist_ok=True)
    stitched_path = stitched_dir / f"stitched_{tag}.h5"

    print(f"Stitching {cfg.n}x{cfg.n}x{cfg.n} = {cfg.n**3} blocks at base ({cfg.base_z},{cfg.base_y},{cfg.base_x}) ...")
    combined = stitch_raw_blocks(blocks_dir, cfg.base_z, cfg.base_y, cfg.base_x, cfg.n)
    print(f"Stitched volume shape: {combined.shape}")
    with h5py.File(stitched_path, "w") as fh:
        fh.create_dataset("raw", data=combined)
    print(f"Saved -> {stitched_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on stitched volume (this may take a while) ...")
    success, elapsed = train_stitched(
        stitched_path, model_dir, log_file, python_exe, str(script_path), config_path, cfg.ckpt_epoch_interval
    )
    if not success:
        sys.exit(f"Training FAILED after {elapsed:.0f}s — see {log_file}")
    print(f"Training done in {elapsed:.0f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    mods = _load_3dgs_modules(project_root)

    print("Post-processing (metrics + visualizations) ...")
    metrics = postprocess(model_dir, stitched_path, device, lpips_model, mods)
    print(json.dumps(metrics, indent=2))
    print(f"Done. Outputs in: {model_dir}")


if __name__ == "__main__":
    main()
