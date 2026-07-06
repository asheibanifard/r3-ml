#!/usr/bin/env python3
"""
Batch 3DGS training over all image blocks in data/fafb/blocks/.

Output layout
-------------
models/
  z000_y000_x000/
    init.pth          ← Gaussian cloud at step 0
    best.pth          ← best PSNR checkpoint
    last.pth          ← final checkpoint
    epoch_0100.pth    ← periodic snapshots (every --ckpt_epoch_interval epochs)
    epoch_0200.pth
    ...
    train.log         ← per-epoch training log (text)
    log.json          ← per-epoch JSON log
    config.json       ← hyperparameters

logs/3dgs/blocks/
  z000_y000_x000.log  ← full stdout/stderr of each training run
  training_log.jsonl  ← master log: one JSON line per block (status, elapsed, PSNR)

Usage
-----
    /venv/r3-ml/bin/python3 scripts/train_all_blocks.py [OPTIONS]

Resume support: blocks with an existing last.pth are skipped automatically.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch 3DGS block training")
    p.add_argument("--blocks_dir",          default="data/fafb/blocks",
                   help="Directory containing image_z*_y*_x*.tif files")
    p.add_argument("--models_dir",          default="models",
                   help="Root directory for per-block checkpoint folders")
    p.add_argument("--logs_dir",            default="logs/3dgs/blocks",
                   help="Directory for per-block stdout/stderr logs and master log")
    p.add_argument("--epochs",              type=int,   default=200)
    p.add_argument("--steps",               type=int,   default=50,
                   help="Steps per epoch")
    p.add_argument("--n_init",              type=int,   default=1000,
                   help="Initial number of Gaussians per block")
    p.add_argument("--max_gauss",           type=int,   default=5000,
                   help="Maximum Gaussians per block")
    p.add_argument("--batch",               type=int,   default=2048)
    p.add_argument("--chunk_n",             type=int,   default=5000)
    p.add_argument("--ckpt_epoch_interval", type=int,   default=100,
                   help="Save epoch_NNNN.pth every N epochs")
    p.add_argument("--start",               type=int,   default=0,
                   help="Start at this 0-based block index (for resuming a range)")
    p.add_argument("--end",                 type=int,   default=None,
                   help="Stop before this block index (exclusive)")
    p.add_argument("--dry_run",             action="store_true",
                   help="Print planned actions without training")
    return p.parse_args()


def discover_blocks(blocks_dir: Path) -> list[tuple[int, int, int, Path]]:
    """Return sorted (z, y, x, path) for all image_z*_y*_x*.tif files."""
    pat = re.compile(r"image_z(\d+)_y(\d+)_x(\d+)\.tif$")
    blocks = []
    for f in blocks_dir.iterdir():
        m = pat.match(f.name)
        if m:
            blocks.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), f))
    blocks.sort()
    return blocks


def is_done(model_dir: Path) -> bool:
    """True if the block already has a finished training run (last.pth exists)."""
    return (model_dir / "last.pth").exists()


def train_block(
    volume: Path,
    model_dir: Path,
    log_file: Path,
    python: str,
    script: str,
    cfg: argparse.Namespace,
) -> tuple[bool, float]:
    """Run one block's training. Returns (success, elapsed_seconds)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, script,
        "--volume",              str(volume),
        "--out",                 str(model_dir),
        "--flat_out",
        "--device",              "cuda",
        "--use_kernel",
        "--no_swc_init",
        "--no_wandb",
        "--epochs",              str(cfg.epochs),
        "--steps_per_epoch",     str(cfg.steps),
        "--n_init",              str(cfg.n_init),
        "--max_gaussians",       str(cfg.max_gauss),
        "--batch",               str(cfg.batch),
        "--chunk_n",             str(cfg.chunk_n),
        "--ckpt_epoch_interval", str(cfg.ckpt_epoch_interval),
        "--ckpt_interval",       "0",   # disable per-step checkpoints
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def best_psnr_from_log(model_dir: Path) -> float | None:
    """Parse the best PSNR from train.log in the model directory."""
    log_path = model_dir / "train.log"
    if not log_path.exists():
        return None
    best = None
    for line in log_path.read_text().splitlines():
        if line.startswith("# best PSNR"):
            try:
                best = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return best


def main():
    cfg = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    blocks_dir   = project_root / cfg.blocks_dir
    models_root  = project_root / cfg.models_dir
    logs_root    = project_root / cfg.logs_dir
    script_path  = project_root / "scripts" / "_3dgs" / "_3dgs.py"
    python_exe   = sys.executable

    if not blocks_dir.exists():
        sys.exit(f"Blocks directory not found: {blocks_dir}")
    if not script_path.exists():
        sys.exit(f"Training script not found: {script_path}")

    blocks = discover_blocks(blocks_dir)
    total  = len(blocks)
    end_idx = cfg.end if cfg.end is not None else total
    blocks  = blocks[cfg.start:end_idx]

    print(f"Found {total} image blocks total")
    print(f"Training blocks [{cfg.start}, {end_idx}) → {len(blocks)} blocks")
    print(f"Config : epochs={cfg.epochs}  steps/ep={cfg.steps}  "
          f"n_init={cfg.n_init}  max_gauss={cfg.max_gauss}")
    print(f"Models → {models_root}")
    print(f"Logs   → {logs_root}")
    print()

    if cfg.dry_run:
        for i, (z, y, x, vol) in enumerate(blocks[:10]):
            tag   = f"z{z:03d}_y{y:03d}_x{x:03d}"
            mdir  = models_root / tag
            done  = is_done(mdir)
            print(f"  [{cfg.start+i:6d}] {tag}  {'SKIP (done)' if done else 'TRAIN'}")
        if len(blocks) > 10:
            print(f"  ... ({len(blocks) - 10} more)")
        return

    models_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    master_log = logs_root / "training_log.jsonl"

    n_done    = 0
    n_skipped = 0
    n_failed  = 0
    t_start   = time.time()

    with open(master_log, "a") as mlog:
        for i, (z, y, x, vol) in enumerate(blocks):
            global_idx = cfg.start + i
            tag        = f"z{z:03d}_y{y:03d}_x{x:03d}"
            model_dir  = models_root / tag
            block_log  = logs_root   / f"{tag}.log"

            if is_done(model_dir):
                n_skipped += 1
                if n_skipped <= 5 or n_skipped % 5000 == 0:
                    print(f"[{global_idx:6d}/{total}] {tag} — skipped (already done)")
                continue

            print(f"[{global_idx:6d}/{total}] {tag} — training ...", flush=True)
            success, elapsed = train_block(
                vol, model_dir, block_log, python_exe, str(script_path), cfg
            )

            psnr = best_psnr_from_log(model_dir) if success else None
            entry = {
                "block":   tag, "z": z, "y": y, "x": x,
                "idx":     global_idx,
                "success": success,
                "elapsed": round(elapsed, 1),
                "best_psnr": psnr,
                "ts":      datetime.now().isoformat(timespec="seconds"),
            }
            mlog.write(json.dumps(entry) + "\n")
            mlog.flush()

            if success:
                n_done += 1
                done_so_far   = n_done + n_failed
                total_elapsed = time.time() - t_start
                remaining     = len(blocks) - i - 1 - n_skipped
                eta_h = (total_elapsed / done_so_far * remaining) / 3600 if done_so_far else 0
                psnr_str = f"  PSNR={psnr:.2f}dB" if psnr is not None else ""
                print(
                    f"  ✓  {elapsed:.0f}s{psnr_str}  |  "
                    f"done={n_done}  failed={n_failed}  skipped={n_skipped}  |  "
                    f"ETA ≈ {eta_h:.1f}h",
                    flush=True,
                )
            else:
                n_failed += 1
                print(
                    f"  ✗  FAILED after {elapsed:.0f}s → {block_log}",
                    flush=True,
                )

    total_min = (time.time() - t_start) / 60
    print(f"\nFinished: trained={n_done}  failed={n_failed}  skipped={n_skipped}  "
          f"total={total_min:.1f}min")
    print(f"Master log: {master_log}")


if __name__ == "__main__":
    main()
