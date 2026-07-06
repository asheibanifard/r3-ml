#!/usr/bin/env python3
"""Crawl segment_*.tif blocks and record local voxel coords for one or more neuron/segment ids.

Block ids are assigned by sorting all discovered blocks by (x, y, z) — x varies
fastest in the sort, then y, then z — and taking the index in that order. Blocks
can be discovered via either the image_*.tif or segment_*.tif filenames (they
share the same (z, y, x) coordinates); segmentation values are always read from
the matching segment_*.tif regardless of which prefix is used for discovery.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import tifffile


def discover_blocks(blocks_dir: Path, prefix: str = "segment"):
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


def load_target_ids(cfg):
    if cfg.target_ids_file:
        with open(cfg.target_ids_file) as fh:
            records = json.load(fh)
        return sorted({int(r["id"]) for r in records})
    return [cfg.target_id]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks_dir", default="data/fafb/blocks")
    p.add_argument("--target_id", type=int, default=None,
                   help="Single neuron/segment id to search for")
    p.add_argument("--target_ids_file", default=None,
                   help="JSON file of records with an 'id' field (e.g. top5_neuron_ids.json); "
                        "voxels matching ANY of these ids are recorded")
    p.add_argument("--block_name_prefix", default="segment", choices=["segment", "image"],
                   help="Filename prefix used to discover/order blocks and assign block_id; "
                        "voxel values are always read from the segment_*.tif regardless")
    p.add_argument("--out", required=True)
    p.add_argument("--progress_every", type=int, default=5000)
    cfg = p.parse_args()

    if cfg.target_id is None and cfg.target_ids_file is None:
        p.error("must pass --target_id or --target_ids_file")

    target_ids = load_target_ids(cfg)

    blocks_dir = Path(cfg.blocks_dir)
    blocks = discover_blocks(blocks_dir, prefix=cfg.block_name_prefix)
    total = len(blocks)
    print(f"Found {total} blocks (ordered via {cfg.block_name_prefix}_*.tif), "
          f"scanning for ids={target_ids}", flush=True)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    blocks_with_hits = 0
    t0 = time.time()
    for block_id, (x, y, z, f) in enumerate(blocks):
        seg_path = blocks_dir / f"segment_z{z}_y{y}_x{x}.tif"
        vol = tifffile.imread(seg_path)
        mask = np.isin(vol, target_ids)  # 1 where id is one of target_ids, 0 elsewhere
        coords = np.argwhere(mask)
        if coords.size:
            blocks_with_hits += 1
            for zz, yy, xx in coords.tolist():
                results.append([block_id, zz, yy, xx])

        if (block_id + 1) % cfg.progress_every == 0 or (block_id + 1) == total:
            elapsed = time.time() - t0
            rate = (block_id + 1) / elapsed
            eta = (total - block_id - 1) / rate if rate > 0 else float("nan")
            print(f"[{block_id + 1}/{total}] blocks_with_hits={blocks_with_hits} "
                  f"voxels={len(results)} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    with open(out_path, "w") as fh:
        json.dump(results, fh)

    print(f"Done. {len(results)} voxels across {blocks_with_hits} blocks -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
