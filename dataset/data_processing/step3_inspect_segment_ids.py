#!/usr/bin/env python3
"""Step 3 of ROADMAP.md: stitch a small neighbourhood of blocks and inspect
what segment ids actually look like before trusting any of them.

Stitches the 2x2 xy tile at z=0 and the 2x2 xy tile at z=1 (around a given
base block coordinate) into one 128^3 volume, reports the unique segment
ids and their voxel counts, and saves depth-coded MIPs — coloring a MIP
projection by the depth of the first non-background voxel, so flattening
a 3D segmentation crop to 2D doesn't erase all sense of neurite depth
(a flat `np.max(mask, axis=...)` MIP would collapse that entirely).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.colors import hsv_to_rgb

# tuples are (z, y, x) block-index offsets from the base coordinate, tiled 2x2
# in y/x; the "xy" plane stitch is done in two z-slabs and joined along z.
XY_PLANE_OFFSETS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]


def stitch_blocks(blocks: list) -> np.ndarray:
    """Tile 4 same-shape blocks into a 2x2 grid along their last two axes."""
    assert len(blocks) == 4, f"Expected 4 blocks for stitching, got {len(blocks)}"
    shape = blocks[0].shape
    stitched = np.zeros((shape[0], 2 * shape[1], 2 * shape[2]), dtype=blocks[0].dtype)
    for idx, (_, y, x) in enumerate(XY_PLANE_OFFSETS):
        stitched[:, y * shape[1]:(y + 1) * shape[1], x * shape[2]:(x + 1) * shape[2]] = blocks[idx]
    return stitched


def load_segment_tiles(blocks_dir: Path, base_z: int, base_y: int, base_x: int):
    def load(z_offset):
        paths = [
            blocks_dir / f"segment_z{base_z + z_offset}_y{base_y + dy}_x{base_x + dx}.tif"
            for (_, dy, dx) in XY_PLANE_OFFSETS
        ]
        return [tiff.imread(p) for p in paths]

    z0_tile = stitch_blocks(load(0))
    z1_tile = stitch_blocks(load(1))
    return np.concatenate([z0_tile, z1_tile], axis=0)


def depth_coded_mip(mask_volume, axis, cmap_name='gray'):
    """MIP along `axis`, coloured by the depth (index along `axis`) of the first
    non-background voxel hit — shallow surfaces map to one end of the colormap,
    deep ones to the other, instead of collapsing to flat white/black."""
    depth = mask_volume.shape[axis]
    vol = np.moveaxis(mask_volume, axis, 0)
    present = vol > 0
    has_any = present.any(axis=0)
    first_hit = np.argmax(present, axis=0)
    depth_norm = np.where(has_any, first_hit / max(depth - 1, 1), 0.0)

    cmap = plt.get_cmap(cmap_name)
    rgb = cmap(depth_norm)[..., :3]
    rgb[~has_any] = 0.0
    return rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks_dir", default="data/fafb/blocks")
    p.add_argument("--base_z", type=int, default=0)
    p.add_argument("--base_y", type=int, default=0)
    p.add_argument("--base_x", type=int, default=0)
    p.add_argument("--top_n", type=int, default=5)
    p.add_argument("--out_dir", default="results/segment_id_inspection")
    cfg = p.parse_args()

    blocks_dir = Path(cfg.blocks_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xy_full = load_segment_tiles(blocks_dir, cfg.base_z, cfg.base_y, cfg.base_x)
    print(f"Stitched volume shape: {xy_full.shape}")

    unique_ids, counts = np.unique(xy_full, return_counts=True)
    voxel_counts = dict(sorted(zip(unique_ids.tolist(), counts.tolist()),
                                key=lambda kv: kv[1], reverse=True))
    print(f"Unique ids in this crop (incl. background): {len(voxel_counts)}")

    top_ids = [i for i in voxel_counts if i != 0][:cfg.top_n]
    print(f"Top {cfg.top_n} local ids by voxel count: "
          f"{[(i, voxel_counts[i]) for i in top_ids]}")

    top_id = top_ids[0]
    top_id_mask = (xy_full == top_id).astype(int)

    top_id_rgb = depth_coded_mip(top_id_mask, axis=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(top_id_rgb)
    plt.title(f"Depth-coded MIP, id={top_id} (color = depth along x)")
    plt.savefig(out_dir / f"depth_mip_id_{top_id}.png", dpi=150)
    plt.close()

    axis = 2
    vol = np.moveaxis(xy_full, axis, 0)
    present = np.isin(vol, top_ids)
    has_any = present.any(axis=0)
    first_hit = np.argmax(present, axis=0)
    yx = np.indices(has_any.shape)
    label_at_depth = vol[first_hit, yx[0], yx[1]]

    id_rank = {rid: i for i, rid in enumerate(top_ids)}
    hue = np.where(has_any,
                   np.vectorize(lambda v: id_rank.get(v, 0))(label_at_depth) / max(len(top_ids) - 1, 1),
                   0.0)
    value = np.where(has_any, 1.0 - first_hit / max(vol.shape[0] - 1, 1), 0.0)
    saturation = has_any.astype(float)
    rgb_multi = hsv_to_rgb(np.stack([hue, saturation, value], axis=-1))

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_multi)
    plt.title("Depth-coded MIP, top ids (hue = id, brightness = depth)")
    plt.savefig(out_dir / "depth_mip_top_ids.png", dpi=150)
    plt.close()

    print(f"Saved depth-coded MIPs -> {out_dir}")


if __name__ == "__main__":
    main()
