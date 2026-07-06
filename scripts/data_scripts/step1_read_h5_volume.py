#!/usr/bin/env python3
"""Step 1 of ROADMAP.md: open the raw FAFB HDF5 volume and sanity-check it.

Reads a single z-slice from both the `image` (raw EM intensity) and
`annotations` (FFN1 instance-segmentation ids) datasets, and saves an
overlay so image and segmentation can be visually confirmed to line up
before any further processing.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5_path", default="data/fafb/fafb_v14_ffn1_z2000-6096.h5")
    p.add_argument("--slice_index", type=int, default=15)
    p.add_argument("--out", default="results/h5_overlay_slice.png")
    cfg = p.parse_args()

    with h5py.File(cfg.h5_path, "r") as h5_file:
        image = h5_file["image"]
        segment = h5_file["annotations"]
        print(f"Image shape: {image.shape}")
        print(f"Segment shape: {segment.shape}")

        i = cfg.slice_index
        img_slice = np.array(image[i, :4096, :4096])
        seg_slice = np.array(segment[i, :4096, :4096])

    # Segment IDs are large uint64 values; remap to a small palette so each
    # neuron gets a visually distinct colour instead of a near-identical gray level.
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 256, size=(256, 3), dtype=np.uint8)
    palette[0] = 0  # background stays black
    seg_rgb = palette[seg_slice % 256]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_slice, cmap="gray")
    ax.imshow(seg_rgb, alpha=0.5)
    ax.set_title(f"Image + segment overlay (z={i})")
    plt.tight_layout()

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
