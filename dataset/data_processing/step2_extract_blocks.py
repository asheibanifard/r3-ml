#!/usr/bin/env python3
"""Step 2 of ROADMAP.md: chop the raw HDF5 volume into a grid of small blocks.

Splits the `image` and `annotations` datasets into a grid of block_size^3
blocks and writes each as a separate TIF pair, named by block index (not
voxel offset):

  {output_dir}/image_z{Z}_y{Y}_x{X}.tif      # raw intensity, uint8
  {output_dir}/segment_z{Z}_y{Y}_x{X}.tif    # segmentation ids, uint64

Resumable: skips any (Z, Y, X) whose image+segment files already exist, so
re-running after an interruption picks up where it left off.
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tifffile import imwrite
from tqdm import tqdm


def save_blocks_to_tif(image, segment, block_size=64, output_dir="data/fafb/blocks"):
    os.makedirs(output_dir, exist_ok=True)
    z_dim, y_dim, x_dim = image.shape

    n_z = (z_dim + block_size - 1) // block_size
    n_y = (y_dim + block_size - 1) // block_size
    n_x = (x_dim + block_size - 1) // block_size
    total = n_z * n_y * n_x

    with tqdm(total=total, unit="block", desc="Saving blocks") as pbar:
        for zi in range(n_z):
            for yi in range(n_y):
                for xi in range(n_x):
                    img_path = os.path.join(output_dir, f"image_z{zi}_y{yi}_x{xi}.tif")
                    seg_path = os.path.join(output_dir, f"segment_z{zi}_y{yi}_x{xi}.tif")

                    if os.path.exists(img_path) and os.path.exists(seg_path):
                        pbar.update(1)
                        continue

                    z, y, x = zi * block_size, yi * block_size, xi * block_size
                    img_block = np.array(image[z:z + block_size, y:y + block_size, x:x + block_size])
                    seg_block = np.array(segment[z:z + block_size, y:y + block_size, x:x + block_size])

                    if not os.path.exists(img_path):
                        imwrite(img_path, img_block)
                    if not os.path.exists(seg_path):
                        imwrite(seg_path, seg_block)

                    pbar.update(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5_path", default="data/fafb/fafb_v14_ffn1_z2000-6096.h5")
    p.add_argument("--output_dir", default="data/fafb/blocks")
    p.add_argument("--block_size", type=int, default=64)
    cfg = p.parse_args()

    with h5py.File(cfg.h5_path, "r") as h5_file:
        image = h5_file["image"]
        segment = h5_file["annotations"]
        save_blocks_to_tif(image, segment, block_size=cfg.block_size, output_dir=cfg.output_dir)

    print(f"Done. Blocks written to {Path(cfg.output_dir)}")


if __name__ == "__main__":
    main()
