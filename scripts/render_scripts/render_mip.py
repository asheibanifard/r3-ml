#!/usr/bin/env python3
"""Render orthographic MIP views from a 3D Gaussian checkpoint via splatting.

This script keeps the camera path separate from the MIP rasterisation path.
It loads a checkpoint, evaluates the fused splat-MIP kernel, and writes the
three canonical orthographic projections (xy, xz, yz) to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import _3dgs._3dgs as _mod

from _3dgs._3dgs import AABB, GaussianCloud, VolumeDataset, render_splatted_mips
from _3dgs._3dgs_training import _load_volume


def _make_cfg() -> argparse.Namespace:
    return argparse.Namespace(
        scale_min_clamp=1e-5,
        mahal_max_clamp=20.0,
        init_scale=0.05,
        init_inten=0.1,
        init_scale_z_factor=1.0,
        n_init=5000,
        swc_path=None,
        chunk_n=5000,
        eval_samples=200_000,
        ssim_crop=64,
        batch=2048,
        grad_sample_weight=0.0,
        lambda_ssim=0.2,
    )


def _save_image(path: Path, image: torch.Tensor, cmap: str = 'gray') -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(image.cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')
    fig.savefig(str(path), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default='models/z000_y000_x000/best.pth')
    parser.add_argument('--volume', default='data/fafb/blocks/image_z0_y0_x0.tif')
    parser.add_argument('--out_dir', default='models/z000_y000_x000/mip_render')
    parser.add_argument('--cmap', default='gray')
    parser.add_argument('--depth_samples', type=int, default=32,
                        help='CPU-fallback ray samples; CUDA ignores this value')
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else root / args.ckpt
    volume_path = Path(args.volume) if Path(args.volume).is_absolute() else root / args.volume
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('render_mip.py requires CUDA for the splat-MIP kernel')

    device = torch.device('cuda')
    cfg = _make_cfg()

    volume, _, _ = _load_volume(str(volume_path))
    aabb = AABB.unit()
    dataset = VolumeDataset(volume, aabb, cfg)

    _mod.USE_CUDA_KERNEL = True
    _mod._load_3dgs_kernel()

    gc = GaussianCloud.load(str(ckpt_path), aabb, device, cfg)

    print(f'Volume   : {dataset.D}x{dataset.H}x{dataset.W}  ({volume_path.name})')
    print(f'Model    : {gc.N} Gaussians  ({ckpt_path.name})')
    print('Rendering splatted MIP views ...')

    mips = render_splatted_mips(gc, dataset, cfg, depth_samples=args.depth_samples)

    for name, image in mips.items():
        png_path = out_dir / f'{name}.png'
        npy_path = out_dir / f'{name}.npy'
        np.save(str(npy_path), image.cpu().numpy())
        _save_image(png_path, image, cmap=args.cmap)
        print(f'  {name}: saved {png_path.name} and {npy_path.name}')

    print(f'Done. Outputs in: {out_dir}')


if __name__ == '__main__':
    main()