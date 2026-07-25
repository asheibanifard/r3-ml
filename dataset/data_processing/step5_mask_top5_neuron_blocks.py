#!/usr/bin/env python3
"""Mask image blocks by the top-5 neuron ids' binary segmentation.

For each block, the binary mask is isin(segment_volume, target_ids) — the
same mask recorded (as sparse voxel coords) in results/top5_binary_voxels.json.
Rather than reconstructing that 24GB coordinate list, this recomputes the
mask directly from segment_*.tif (cheap: one array compare per block) and
multiplies it elementwise against the matching image_*.tif: voxels where the
mask is 1 keep their original image intensity, voxels where it's 0 become 0.

Blocks with no target-id voxels are skipped entirely (masked result would be
all zero).
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import tifffile


def discover_blocks(blocks_dir: Path, prefix: str = "image"):
    """Return blocks sorted by (x, y, z), each as (x, y, z, path)."""
    pat = re.compile(rf"{prefix}_z(\d+)_y(\d+)_x(\d+)\.tif$")
    blocks = []
    for f in blocks_dir.iterdir():
        m = pat.match(f.name)
        if m:
            z, y, x = int(m.group(1)), int(m.group(2)), int(m.group(3))
            blocks.append((x, y, z, f))
    blocks.sort()
    return blocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks_dir", default="data/fafb/blocks")
    p.add_argument("--target_ids_file", required=True,
                   help="JSON file of records with an 'id' field, e.g. top5_neuron_ids.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--progress_every", type=int, default=5000)
    cfg = p.parse_args()

    with open(cfg.target_ids_file) as fh:
        target_ids = sorted({int(r["id"]) for r in json.load(fh)})
    print(f"target ids: {target_ids}", flush=True)

    blocks_dir = Path(cfg.blocks_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = discover_blocks(blocks_dir, prefix="image")
    total = len(blocks)
    print(f"Found {total} image blocks, masking by top-5 neuron ids", flush=True)

    blocks_written = 0
    t0 = time.time()
    for block_id, (x, y, z, image_path) in enumerate(blocks):
        seg_path = blocks_dir / f"segment_z{z}_y{y}_x{x}.tif"
        seg = tifffile.imread(seg_path)
        mask = np.isin(seg, target_ids)
        if mask.any():
            image = tifffile.imread(image_path)
            masked = np.where(mask, image, 0).astype(image.dtype)
            tifffile.imwrite(out_dir / f"image_z{z}_y{y}_x{x}.tif", masked)
            blocks_written += 1

        if (block_id + 1) % cfg.progress_every == 0 or (block_id + 1) == total:
            elapsed = time.time() - t0
            rate = (block_id + 1) / elapsed
            eta = (total - block_id - 1) / rate if rate > 0 else float("nan")
            print(f"[{block_id + 1}/{total}] blocks_written={blocks_written} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"Done. {blocks_written} masked blocks written -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
