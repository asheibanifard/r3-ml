#!/usr/bin/env python3
"""Stitch per-block Gaussian checkpoints (each fit independently in its own
local [-1,1]^3 frame) into one combined checkpoint in a shared global frame.

Because the GaussianCloud density field is additive
(f(x) = sum_k v_k * exp(-1/2 Mahalanobis)), concatenating remapped Gaussians
from every block reproduces the correct combined field exactly -- no
block-aware max-combination renderer is needed. The existing single flat
Gaussian list renderer (Mip_Render_Inside_Volume.cu, pretrained_gaussian
mode) can render the result directly.

Axis convention: means are stored (x, y, z), matching VolumeDataset's
_indices_to_pts, where x <-> block W/X-index, y <-> block H/Y-index,
z <-> block D/Z-index.

Remap (uniform cubic grid of N blocks per axis, each block spanning
1/N of the 2-unit-wide global [-1,1]^3 domain):
    global = -1 + (block_index + 0.5) * (2/N) + local * (1/N)
    log_scale_global = log_scale_local + log(1/N)
Quaternions and intensities are unaffected by a uniform (isotropic) rescale.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def load_block(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def remap_block(ckpt: dict, iz: int, iy: int, ix: int, n_per_axis: int) -> dict:
    extent = 1.0 / n_per_axis
    log_extent = math.log(extent)

    means = ckpt["means"].clone()
    log_scales = ckpt["log_scales"].clone()

    centers = torch.tensor(
        [
            -1.0 + (ix + 0.5) * (2.0 / n_per_axis),
            -1.0 + (iy + 0.5) * (2.0 / n_per_axis),
            -1.0 + (iz + 0.5) * (2.0 / n_per_axis),
        ],
        dtype=means.dtype,
    )

    means_global = centers.unsqueeze(0) + means * extent
    log_scales_global = log_scales + log_extent

    return {
        "means": means_global,
        "log_scales": log_scales_global,
        "quats": ckpt["quats"].clone(),
        "intensities": ckpt["intensities"].clone(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-dir", required=True,
                         help="Directory containing b_<iz><iy><ix>/ subdirs")
    parser.add_argument("--n-per-axis", type=int, required=True)
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    blocks_dir = Path(args.blocks_dir)
    n = args.n_per_axis

    all_means, all_scales, all_quats, all_inten = [], [], [], []
    loaded, skipped = 0, 0

    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                bname = f"b_{iz}{iy}{ix}"
                ckpt_path = blocks_dir / bname / args.checkpoint_name
                if not ckpt_path.exists():
                    print(f"[skip] {bname}: no {args.checkpoint_name}")
                    skipped += 1
                    continue

                ckpt = load_block(ckpt_path)
                remapped = remap_block(ckpt, iz, iy, ix, n)

                all_means.append(remapped["means"])
                all_scales.append(remapped["log_scales"])
                all_quats.append(remapped["quats"])
                all_inten.append(remapped["intensities"])
                loaded += 1

    if loaded == 0:
        raise RuntimeError("No block checkpoints found -- nothing to stitch.")

    combined = {
        "means": torch.cat(all_means, dim=0),
        "log_scales": torch.cat(all_scales, dim=0),
        "quats": torch.cat(all_quats, dim=0),
        "intensities": torch.cat(all_inten, dim=0),
        "inten_param": "softplus",
    }

    print(f"Stitched {loaded} blocks ({skipped} skipped), "
          f"total Gaussians: {combined['means'].shape[0]}")
    print(f"means range: {combined['means'].min(dim=0).values.tolist()} "
          f"to {combined['means'].max(dim=0).values.tolist()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, out_path)
    print(f"Saved combined checkpoint: {out_path}")


if __name__ == "__main__":
    main()
