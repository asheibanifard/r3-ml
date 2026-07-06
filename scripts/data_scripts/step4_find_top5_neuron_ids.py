#!/usr/bin/env python3
"""Step 4 of ROADMAP.md: find the ids with the highest total voxel count
across the whole dataset.

Crawls every segment_*.tif block and tallies a running voxel-count total
per neuron/segment id (background id 0 excluded), then writes the top-N
ids by voxel count to a JSON file consumed by scripts/data_scripts/step5_mask_top5_neuron_blocks.py.
"""
import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import tifffile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks_dir", default="data/fafb/blocks")
    p.add_argument("--top_n", type=int, default=5)
    p.add_argument("--out", default="results/top5_neuron_ids.json")
    p.add_argument("--progress_every", type=int, default=5000)
    cfg = p.parse_args()

    blocks_dir = Path(cfg.blocks_dir)
    pat = re.compile(r"segment_z(\d+)_y(\d+)_x(\d+)\.tif$")
    seg_paths = [f for f in blocks_dir.iterdir() if pat.match(f.name)]
    print(f"scanning {len(seg_paths)} segment blocks for a global voxel-count histogram...", flush=True)

    id_counts = Counter()
    t0 = time.time()
    for i, f in enumerate(seg_paths):
        vol = tifffile.imread(f)
        ids, counts = np.unique(vol, return_counts=True)
        for uid, c in zip(ids.tolist(), counts.tolist()):
            if uid != 0:
                id_counts[uid] += c
        if (i + 1) % cfg.progress_every == 0 or (i + 1) == len(seg_paths):
            elapsed = time.time() - t0
            print(f"[{i + 1}/{len(seg_paths)}] unique ids so far: {len(id_counts)} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    top_records = [{"id": uid, "voxel_count": c} for uid, c in id_counts.most_common(cfg.top_n)]
    print(top_records, flush=True)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(top_records, fh, indent=2)

    print(f"Done. Top {cfg.top_n} ids -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
