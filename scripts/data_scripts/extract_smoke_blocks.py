#!/usr/bin/env python3
"""Smoke test: break the smoke dataset into small blocks for a quick
end-to-end sanity check of the 3DGS CUDA training pipeline.

Splits data/smoke_data/sample_A_20160501.hdf's `volumes/raw` (uint8) and
`volumes/labels/neuron_ids` (uint64) into block_size^3 (z,y,x) blocks, each
written as its own HDF5 file with two datasets: `raw` and `neuron_ids`.
_load_volume() in scripts/_3dgs/_3dgs_training.py reads the `raw` dataset
directly, so these blocks can be passed straight to --volume without any
extra conversion step.

Resumable: skips any (Z, Y, X) whose block file already exists.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def save_smoke_blocks(raw, neuron_ids, block_size=50, output_dir="data/smoke_data/blocks"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_dim, y_dim, x_dim = raw.shape

    n_z = (z_dim + block_size - 1) // block_size
    n_y = (y_dim + block_size - 1) // block_size
    n_x = (x_dim + block_size - 1) // block_size
    total = n_z * n_y * n_x

    with tqdm(total=total, unit="block", desc="Saving smoke blocks") as pbar:
        for zi in range(n_z):
            for yi in range(n_y):
                for xi in range(n_x):
                    block_path = out_dir / f"block_z{zi}_y{yi}_x{xi}.h5"
                    if block_path.exists():
                        pbar.update(1)
                        continue

                    z, y, x = zi * block_size, yi * block_size, xi * block_size
                    raw_block = np.asarray(raw[z:z + block_size, y:y + block_size, x:x + block_size])
                    id_block = np.asarray(neuron_ids[z:z + block_size, y:y + block_size, x:x + block_size])

                    with h5py.File(block_path, "w") as fh:
                        fh.create_dataset("raw", data=raw_block)
                        fh.create_dataset("neuron_ids", data=id_block)

                    pbar.update(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5_path", default="data/smoke_data/sample_A_20160501.hdf")
    p.add_argument("--output_dir", default="data/smoke_data/blocks")
    p.add_argument("--block_size", type=int, default=50)
    cfg = p.parse_args()

    with h5py.File(cfg.h5_path, "r") as f:
        raw = f["volumes"]["raw"]
        neuron_ids = f["volumes"]["labels"]["neuron_ids"]
        print(f"raw shape: {raw.shape} {raw.dtype}, neuron_ids shape: {neuron_ids.shape} {neuron_ids.dtype}")
        save_smoke_blocks(raw, neuron_ids, block_size=cfg.block_size, output_dir=cfg.output_dir)

    print(f"Done. Blocks written to {Path(cfg.output_dir)}")


if __name__ == "__main__":
    main()
