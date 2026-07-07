#!/usr/bin/env python3
"""
Batch 3DGS training over all smoke blocks in data/smoke_data/blocks/.

Each block is trained via scripts/_3dgs/_3dgs.py using configs/smoke_config.yml
(subprocess, so JIT-compiled CUDA state and any crash is isolated per block).
Afterward, only these are kept — everything else the training script writes
(last.pth, epoch_*.pth, train.log, log.json, config.json, slices_ep*.png) is
deleted once metrics/visuals have been extracted from it:

models_smoke/
  block_z000_y000_x000/
    init.pth       <- Gaussian cloud at step 0
    best.pth       <- best vol_PSNR checkpoint
    init.png       <- GT/pred/|diff| middle-slice visualization at init
    best.png       <- GT/pred/|diff| middle-slice visualization at best
    metrics.json   <- {"init": {psnr, ssim, lpips}, "best": {psnr, ssim, lpips}}

logs/smoke/blocks/
  z000_y000_x000.log  <- full stdout/stderr of each training run
  training_log.jsonl  <- master log: one JSON line per block

Resumable: blocks with an existing metrics.json are skipped.

Usage
-----
    /venv/r3-ml/bin/python3 scripts/train_scripts/train_all_smoke_blocks.py [OPTIONS]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch 3DGS smoke-block training")
    p.add_argument("--blocks_dir", default="data/smoke_data/blocks",
                   help="Directory containing block_z*_y*_x*.h5 files")
    p.add_argument("--models_dir", default="models_smoke")
    p.add_argument("--logs_dir",   default="logs/smoke/blocks")
    p.add_argument("--config",     default="configs/smoke_config.yml")
    p.add_argument("--start", type=int, default=0,
                   help="Start at this 0-based block index (for resuming a range)")
    p.add_argument("--end",   type=int, default=None,
                   help="Stop before this block index (exclusive)")
    p.add_argument("--z_min", type=int, default=None, help="Inclusive z filter")
    p.add_argument("--z_max", type=int, default=None, help="Inclusive z filter")
    p.add_argument("--y_min", type=int, default=None, help="Inclusive y filter")
    p.add_argument("--y_max", type=int, default=None, help="Inclusive y filter")
    p.add_argument("--x_min", type=int, default=None, help="Inclusive x filter")
    p.add_argument("--x_max", type=int, default=None, help="Inclusive x filter")
    p.add_argument("--dry_run", action="store_true",
                   help="Print planned actions without training")
    return p.parse_args()


def filter_blocks(blocks: list[tuple[int, int, int, Path]], cfg: argparse.Namespace) -> list[tuple[int, int, int, Path]]:
    """Restrict to an inclusive (z_min..z_max, y_min..y_max, x_min..x_max) box, if given."""
    def in_range(v, lo, hi):
        return (lo is None or v >= lo) and (hi is None or v <= hi)

    return [
        (z, y, x, p) for (z, y, x, p) in blocks
        if in_range(z, cfg.z_min, cfg.z_max)
        and in_range(y, cfg.y_min, cfg.y_max)
        and in_range(x, cfg.x_min, cfg.x_max)
    ]


def discover_blocks(blocks_dir: Path) -> list[tuple[int, int, int, Path]]:
    """Return sorted (z, y, x, path) for all block_z*_y*_x*.h5 files."""
    pat = re.compile(r"block_z(\d+)_y(\d+)_x(\d+)\.h5$")
    blocks = []
    for f in blocks_dir.iterdir():
        m = pat.match(f.name)
        if m:
            blocks.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), f))
    blocks.sort()
    return blocks


def is_done(model_dir: Path) -> bool:
    """True if this block already has extracted metrics (fully post-processed)."""
    return (model_dir / "metrics.json").exists()


def train_block(volume: Path, model_dir: Path, log_file: Path,
                 python: str, script: str, config_path: Path) -> tuple[bool, float]:
    """Run one block's training via subprocess. Returns (success, elapsed_seconds)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, script,
        "--config", str(config_path),
        "--volume", str(volume),
        "--out",    str(model_dir),
        "--device", "cuda",
        "--use_kernel",
        "--flat_out",
        "--no_swc_init",
        "--no_wandb",
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def _load_3dgs_modules(project_root: Path):
    """Import the 3DGS model module once, matching render_camera.py/render_mip.py's pattern."""
    sys.path.insert(0, str(project_root / "scripts"))
    import _3dgs._3dgs as _mod
    _mod.USE_CUDA_KERNEL = True
    _mod._load_3dgs_kernel()
    from _3dgs._3dgs import AABB, GaussianCloud, VolumeDataset, vol_psnr, _ssim_2d
    from _3dgs._3dgs_training import _load_volume, _visualize_middle_slices
    return {
        "AABB": AABB, "GaussianCloud": GaussianCloud, "VolumeDataset": VolumeDataset,
        "vol_psnr": vol_psnr, "_ssim_2d": _ssim_2d,
        "_load_volume": _load_volume, "_visualize_middle_slices": _visualize_middle_slices,
    }


def _mid_xy_slice(gc, dataset, device):
    """Predicted + GT middle-Z xy slice — same view as _visualize_middle_slices' xy row."""
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


def postprocess_block(model_dir: Path, volume_path: Path, device, lpips_model, mods: dict) -> dict:
    """Extract {psnr, ssim, lpips} + a visualization for init and best checkpoints,
    then delete every other artifact the training run left behind."""
    cfg_dict = json.loads((model_dir / "config.json").read_text())
    cfg = argparse.Namespace(**cfg_dict)

    volume, _, _ = mods["_load_volume"](str(volume_path))
    aabb = mods["AABB"].unit()
    dataset = mods["VolumeDataset"](volume, aabb, cfg)

    metrics = {}
    for tag, ckpt_name, epoch_tag in [("init", "init.pth", 0), ("best", "best.pth", 1)]:
        gc = mods["GaussianCloud"].load(model_dir / ckpt_name, aabb, device, cfg)

        psnr = mods["vol_psnr"](gc, dataset, cfg)

        pred, gt = _mid_xy_slice(gc, dataset, device)
        pred_b = pred.unsqueeze(0).unsqueeze(0)
        gt_b = gt.unsqueeze(0).unsqueeze(0)
        ssim = 1.0 - mods["_ssim_2d"](pred_b, gt_b).item()

        pred_rgb = pred_b.repeat(1, 3, 1, 1) * 2 - 1
        gt_rgb = gt_b.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lp = lpips_model(pred_rgb, gt_rgb).item()

        metrics[tag] = {"psnr": psnr, "ssim": ssim, "lpips": lp}

        vis_path = mods["_visualize_middle_slices"](dataset.vol, gc, model_dir, epoch_tag, device)
        if vis_path:
            Path(vis_path).replace(model_dir / f"{tag}.png")

    keep = {"init.pth", "best.pth", "init.png", "best.png", "metrics.json"}
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
    blocks_dir  = project_root / cfg.blocks_dir
    models_root = project_root / cfg.models_dir
    logs_root   = project_root / cfg.logs_dir
    config_path = project_root / cfg.config
    script_path = project_root / "scripts" / "_3dgs" / "_3dgs.py"
    python_exe  = sys.executable

    if not blocks_dir.exists():
        sys.exit(f"Blocks directory not found: {blocks_dir}")
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    blocks = discover_blocks(blocks_dir)
    total = len(blocks)
    blocks = filter_blocks(blocks, cfg)
    matching = len(blocks)
    end_idx = cfg.end if cfg.end is not None else matching
    blocks = blocks[cfg.start:end_idx]

    print(f"Found {total} smoke blocks total")
    print(f"Selected range: z[{cfg.z_min},{cfg.z_max}] y[{cfg.y_min},{cfg.y_max}] x[{cfg.x_min},{cfg.x_max}] "
          f"-> {matching} matching, training [{cfg.start}, {end_idx}) -> {len(blocks)} blocks")
    print(f"Config : {config_path}")
    print(f"Models -> {models_root}")
    print(f"Logs   -> {logs_root}")
    print()

    if cfg.dry_run:
        for i, (z, y, x, vol) in enumerate(blocks[:10]):
            tag = f"z{z:03d}_y{y:03d}_x{x:03d}"
            mdir = models_root / f"block_{tag}"
            print(f"  [{cfg.start + i:6d}] block_{tag}  {'SKIP (done)' if is_done(mdir) else 'TRAIN'}")
        if len(blocks) > 10:
            print(f"  ... ({len(blocks) - 10} more)")
        return

    models_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    master_log = logs_root / "training_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    mods = _load_3dgs_modules(project_root)

    n_done = n_skipped = n_failed = 0
    t_start = time.time()

    with open(master_log, "a") as mlog:
        for i, (z, y, x, vol) in enumerate(blocks):
            global_idx = cfg.start + i
            tag = f"z{z:03d}_y{y:03d}_x{x:03d}"
            model_dir = models_root / f"block_{tag}"
            block_log = logs_root / f"{tag}.log"

            if is_done(model_dir):
                n_skipped += 1
                if n_skipped <= 5 or n_skipped % 200 == 0:
                    print(f"[{global_idx:6d}/{total}] block_{tag} — skipped (already done)")
                continue

            print(f"[{global_idx:6d}/{total}] block_{tag} — training ...", flush=True)
            success, elapsed = train_block(vol, model_dir, block_log, python_exe, str(script_path), config_path)

            metrics, post_err = None, None
            if success:
                try:
                    metrics = postprocess_block(model_dir, vol, device, lpips_model, mods)
                except Exception as e:
                    post_err = str(e)

            entry = {
                "block": tag, "z": z, "y": y, "x": x, "idx": global_idx,
                "success": success, "elapsed": round(elapsed, 1),
                "metrics": metrics, "post_error": post_err,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            mlog.write(json.dumps(entry) + "\n")
            mlog.flush()

            if success and metrics is not None:
                n_done += 1
                done_so_far = n_done + n_failed
                total_elapsed = time.time() - t_start
                remaining = len(blocks) - i - 1 - n_skipped
                eta_h = (total_elapsed / done_so_far * remaining) / 3600 if done_so_far else 0
                print(
                    f"  ✓  {elapsed:.0f}s  best_PSNR={metrics['best']['psnr']:.2f}dB  |  "
                    f"done={n_done}  failed={n_failed}  skipped={n_skipped}  |  ETA ≈ {eta_h:.1f}h",
                    flush=True,
                )
            else:
                n_failed += 1
                reason = post_err if post_err else f"training failed, see {block_log}"
                print(f"  ✗  FAILED after {elapsed:.0f}s → {reason}", flush=True)

    total_min = (time.time() - t_start) / 60
    print(f"\nFinished: trained={n_done}  failed={n_failed}  skipped={n_skipped}  total={total_min:.1f}min")
    print(f"Master log: {master_log}")


if __name__ == "__main__":
    main()
