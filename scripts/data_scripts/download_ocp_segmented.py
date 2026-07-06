"""Download segmented z-slice volumes from the Open Connectome Project.

Supports kasthuri14s1colEM (S3) and FAFB v14 FFN1 (GCS).

Writes each chunk directly into a pre-allocated HDF5 file so the process
is crash-safe and resumable.  Re-running with the same config skips any
z-slices that were already written.

Usage:
    python scripts/data_scripts/download_ocp_segmented.py --config configs/ocp_kasthuri.yml
    python scripts/data_scripts/download_ocp_segmented.py --config configs/ocp_fafb.yml
    python scripts/data_scripts/download_ocp_segmented.py --config configs/ocp_fafb.yml --z-start 4082 --z-stop 4200
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from cloudvolume import CloudVolume
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def open_volume(url: str, mip: int, use_https: bool) -> CloudVolume:
    return CloudVolume(url, mip=mip, use_https=use_https, parallel=True, progress=False)


def _chunk_z_ranges(z_start: int, z_stop: int, chunk_z: int):
    for z0 in range(z_start, z_stop, chunk_z):
        yield z0, min(z0 + chunk_z, z_stop)


def _chunk_index(z0: int, z_start: int, chunk_z: int) -> int:
    return (z0 - z_start) // chunk_z


def init_hdf5(path: Path, n_z: int, n_y: int, n_x: int,
              img_dtype, seg_dtype, cfg: dict, chunk_z: int) -> h5py.File:
    """Create (or open existing) HDF5 file with pre-allocated datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    b = cfg["bounds"]
    ds = cfg.get("dataset", {})

    if path.exists():
        f = h5py.File(path, "a")
        print(f"Resuming into existing file: {path}")
    else:
        f = h5py.File(path, "w")
        hdf5_chunk = (min(chunk_z, n_z), min(256, n_y), min(256, n_x))
        f.create_dataset("image",       shape=(n_z, n_y, n_x), dtype=img_dtype,
                         compression="gzip", chunks=hdf5_chunk)
        f.create_dataset("annotations", shape=(n_z, n_y, n_x), dtype=seg_dtype,
                         compression="gzip", chunks=hdf5_chunk)
        # progress sentinels: 0 = not written, 1 = written
        n_chunks = len(list(_chunk_z_ranges(b["z"][0], b["z"][1], chunk_z)))
        f.create_dataset("_img_done",  data=np.zeros(n_chunks, dtype=np.uint8))
        f.create_dataset("_ann_done",  data=np.zeros(n_chunks, dtype=np.uint8))
        f.attrs["dataset"]       = ds.get("name", "")
        f.attrs["voxel_size_nm"] = str(ds.get("voxel_size_nm", []))
        f.attrs["bounds_x"]      = str(b["x"])
        f.attrs["bounds_y"]      = str(b["y"])
        f.attrs["bounds_z"]      = str(b["z"])
        print(f"Created: {path}  shape=({n_z},{n_y},{n_x})")
    return f


def download_channel_incremental(
    vol: CloudVolume,
    f: h5py.File,
    dataset_name: str,
    done_key: str,
    xs: slice,
    ys: slice,
    z_start: int,
    z_stop: int,
    chunk_z: int,
    n_y: int,
    n_x: int,
    desc: str,
):
    """Download chunk-by-chunk; write each slab directly to HDF5; skip done chunks."""
    done = f[done_key][:]
    chunks = list(_chunk_z_ranges(z_start, z_stop, chunk_z))
    pending = [(i, z0, z1) for i, (z0, z1) in enumerate(chunks) if not done[i]]

    if not pending:
        print(f"  {desc}: all {len(chunks)} chunks already written, skipping.")
        return

    skipped = len(chunks) - len(pending)
    if skipped:
        print(f"  {desc}: resuming — {skipped} chunks already done, {len(pending)} remaining.")

    for i, z0, z1 in tqdm(pending, desc=desc, unit="chunk"):
        cutout = vol[xs, ys, z0:z1]           # → (X, Y, Z, C)
        slab = np.squeeze(cutout, axis=-1)     # → (X, Y, Z)
        slab = np.moveaxis(slab, -1, 0)        # → (Z, Y, X)

        # slab may be smaller in Y/X at volume edges — pad if needed
        actual_z, actual_y, actual_x = slab.shape
        if actual_y != n_y or actual_x != n_x:
            padded = np.zeros((actual_z, n_y, n_x), dtype=slab.dtype)
            padded[:, :actual_y, :actual_x] = slab
            slab = padded

        rel_z0 = z0 - z_start
        rel_z1 = rel_z0 + actual_z
        f[dataset_name][rel_z0:rel_z1, :n_y, :n_x] = slab
        f[done_key][i] = 1
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/ocp_kasthuri.yml")
    parser.add_argument("--z-start", type=int, default=None)
    parser.add_argument("--z-stop",  type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dl  = cfg["download"]
    b   = cfg["bounds"]

    if args.z_start is not None:
        b["z"][0] = args.z_start
    if args.z_stop is not None:
        b["z"][1] = args.z_stop

    xs = slice(*b["x"])
    ys = slice(*b["y"])
    z_start, z_stop = b["z"]
    chunk_z   = dl["chunk_z"]
    use_https = dl["use_https"]

    img_mip = dl.get("image_mip", dl.get("mip", 0))
    seg_mip = dl.get("seg_mip",   dl.get("mip", 0))

    n_z = z_stop - z_start
    n_y = b["y"][1] - b["y"][0]
    n_x = b["x"][1] - b["x"][0]

    print(f"Dataset : {cfg['dataset']['name']}")
    print(f"Z range : [{z_start}, {z_stop})  ({n_z} slices)")
    print(f"MIP     : image={img_mip}  seg={seg_mip}")

    img_vol = open_volume(cfg["channels"]["image"],       img_mip, use_https)
    seg_vol = open_volume(cfg["channels"]["annotations"], seg_mip, use_https)

    print(f"Image vol : {img_vol.volume_size}  {img_vol.dtype}")
    print(f"Seg   vol : {seg_vol.volume_size}  {seg_vol.dtype}")

    fmt     = dl["save_format"]
    ds_name = cfg["dataset"]["name"]
    z_tag   = f"z{z_start}-{z_stop}"
    out_dir = Path(dl["output_dir"])

    if fmt == "hdf5":
        out_path = out_dir / f"{ds_name}_{z_tag}.h5"
        f = init_hdf5(out_path, n_z, n_y, n_x,
                      img_vol.dtype, seg_vol.dtype, cfg, chunk_z)
        try:
            download_channel_incremental(
                img_vol, f, "image", "_img_done",
                xs, ys, z_start, z_stop, chunk_z, n_y, n_x, "image")
            download_channel_incremental(
                seg_vol, f, "annotations", "_ann_done",
                xs, ys, z_start, z_stop, chunk_z, n_y, n_x, "annotations")
        finally:
            f.close()
        print(f"Done: {out_path}")
    else:
        # npy path: keep original in-memory approach (small datasets only)
        from scripts.download_ocp_segmented import download_channel  # noqa: F401
        slabs_img, slabs_ann = [], []
        for z0, z1 in tqdm(list(_chunk_z_ranges(z_start, z_stop, chunk_z)),
                           desc="image", unit="chunk"):
            c = img_vol[xs, ys, z0:z1]
            slabs_img.append(np.moveaxis(np.squeeze(c, -1), -1, 0))
        for z0, z1 in tqdm(list(_chunk_z_ranges(z_start, z_stop, chunk_z)),
                           desc="annotations", unit="chunk"):
            c = seg_vol[xs, ys, z0:z1]
            slabs_ann.append(np.moveaxis(np.squeeze(c, -1), -1, 0))
        image = np.concatenate(slabs_img, axis=0)
        seg   = np.concatenate(slabs_ann, axis=0)
        npy_dir = out_dir / z_tag
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / "image.npy",       image)
        np.save(npy_dir / "annotations.npy", seg)
        print(f"Saved npy in {npy_dir}  image={image.shape}  seg={seg.shape}")


if __name__ == "__main__":
    main()
