#!/usr/bin/env python3
"""
render_camera.py — First-person camera renderer for a 3DGS Gaussian cloud.

The camera sits at the block centre (0,0,0) in normalised [-1,1]^3 space and
rotates around its own vertical (Y) axis.  For each frame a perspective image
is formed by ray-marching outward from the centre through the reconstructed
voxel volume and accumulating via MIP along each ray.

Outputs:
  <out_dir>/frame_0000.png  ...  frame_NNNN.png
  <out_dir>/orbit.gif       (requires Pillow)

Usage:
  python scripts/render_camera.py \
      --ckpt    models/z000_y000_x000/best.pth \
      --volume  data/fafb/blocks/image_z0_y0_x0.tif \
      --out_dir models/z000_y000_x000/renders \
      --n_frames 36 --fov 90 --img_size 256
"""

import sys, argparse
from pathlib import Path

import numpy as np
import torch
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()
from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_cfg():
    return argparse.Namespace(
        scale_min_clamp=1e-5, mahal_max_clamp=20.0, init_scale=0.05,
        init_inten=0.1, init_scale_z_factor=1.0, n_init=5000,
        swc_path=None, chunk_n=5000, eval_samples=200_000, ssim_crop=64,
        batch=2048, grad_sample_weight=0.0, lambda_ssim=0.2,
    )


def reconstruct_volume(gc, dataset, cfg, device):
    """Query every voxel centre slice-by-slice; returns float32 (D, H, W)."""
    D, H, W = dataset.D, dataset.H, dataset.W
    vol = np.empty((D, H, W), dtype=np.float32)
    with torch.no_grad():
        for z in range(D):
            pts = dataset._indices_to_pts(
                torch.full((H * W,), z, dtype=torch.long),
                torch.arange(H, dtype=torch.long).repeat_interleave(W),
                torch.arange(W, dtype=torch.long).tile(H),
                device,
            )
            pred = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
            vol[z] = pred.cpu().numpy().reshape(H, W)
    return vol


def render_frame(vol, azimuth_deg, elev_deg=0.0,
                 fov_deg=90.0, img_h=256, img_w=256, n_steps=128):
    """
    Camera at (0,0,0) inside the [-1,1]^3 block, looking outward.

    azimuth_deg : yaw rotation around the Y axis (0 = +Z direction)
    elev_deg    : pitch up/down from the horizontal plane
    fov_deg     : horizontal field of view
    n_steps     : ray-march samples along each ray
    returns     : (img_h, img_w) float32 MIP image
    """
    D, H, W = vol.shape

    az = np.radians(azimuth_deg)
    el = np.radians(elev_deg)

    # Camera basis vectors
    fwd = np.array([np.cos(el) * np.sin(az),
                    np.sin(el),
                    np.cos(el) * np.cos(az)], dtype=np.float64)

    world_up = np.array([0.0, 1.0, 0.0])
    # handle degenerate case when fwd is parallel to world_up
    if abs(np.dot(fwd, world_up)) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])

    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)

    # Pixel ray directions (img_h, img_w, 3)
    half_w = np.tan(np.radians(fov_deg / 2.0))
    half_h = half_w * img_h / img_w
    px = np.linspace(-half_w,  half_w, img_w)
    py = np.linspace( half_h, -half_h, img_h)
    px_g, py_g = np.meshgrid(px, py)   # (img_h, img_w)

    dirs = (fwd[None, None, :]
            + px_g[:, :, None] * right[None, None, :]
            + py_g[:, :, None] * up[None, None, :])   # (img_h, img_w, 3)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs /= norms                                       # normalise

    # t_max: furthest distance from origin before leaving [-1,1]^3
    # p = origin + t*d = t*d  (origin is 0,0,0)
    # solve |t*d_i| = 1  => t = 1/|d_i|; take minimum over axes
    t_max = np.min(1.0 / np.maximum(np.abs(dirs), 1e-8), axis=-1)  # (img_h, img_w)

    # Sample along every ray at once: shape (n_steps, img_h, img_w, 3)
    ts = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    # pts[s, h, w] = ts[s] * t_max[h,w] * dirs[h,w]
    pts = ts[:, None, None, None] * t_max[None, :, :, None] * dirs[None, :, :, :]

    x_all = pts[:, :, :, 0].ravel()   # world x -> iw
    y_all = pts[:, :, :, 1].ravel()   # world y -> ih
    z_all = pts[:, :, :, 2].ravel()   # world z -> iz

    # Coordinate convention from VolumeDataset / eval CUDA:
    #   x (pts[:,0]) -> iw  (width  axis, index 2 in vol[D,H,W])
    #   y (pts[:,1]) -> ih  (height axis, index 1)
    #   z (pts[:,2]) -> iz  (depth  axis, index 0)
    iw = ((x_all + 1.0) / 2.0 * (W - 1)).clip(0, W - 1)
    ih = ((y_all + 1.0) / 2.0 * (H - 1)).clip(0, H - 1)
    iz = ((z_all + 1.0) / 2.0 * (D - 1)).clip(0, D - 1)

    coords = np.stack([iz, ih, iw])   # (3, n_steps*img_h*img_w)
    vals = map_coordinates(vol, coords, order=1, mode='constant', cval=0.0)
    vals = vals.reshape(n_steps, img_h, img_w)

    return vals.max(axis=0).astype(np.float32)   # MIP along ray


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ckpt',      default='models/z000_y000_x000/best.pth')
    parser.add_argument('--volume',    default='data/fafb/blocks/image_z0_y0_x0.tif')
    parser.add_argument('--out_dir',   default='models/z000_y000_x000/renders')
    parser.add_argument('--n_frames',  type=int,   default=36)
    parser.add_argument('--elev',      type=float, default=0.0,
                        help='camera pitch in degrees (default 0 = horizontal)')
    parser.add_argument('--fov',       type=float, default=90.0,
                        help='horizontal field of view in degrees (default 90)')
    parser.add_argument('--img_size',  type=int,   default=256,
                        help='output image resolution (square, default 256)')
    parser.add_argument('--n_steps',   type=int,   default=128,
                        help='ray-march steps per pixel (default 128)')
    parser.add_argument('--gif_fps',   type=int,   default=12)
    parser.add_argument('--cmap',      default='gray')
    args = parser.parse_args()

    root        = Path(__file__).parent.parent
    ckpt_path   = Path(args.ckpt)    if Path(args.ckpt).is_absolute()    else root / args.ckpt
    volume_path = Path(args.volume)  if Path(args.volume).is_absolute()  else root / args.volume
    out_dir     = Path(args.out_dir) if Path(args.out_dir).is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = _make_cfg()

    # ── dataset ───────────────────────────────────────────────────────────────
    vol_np  = tifffile.imread(str(volume_path)).astype('float32') / 255.
    aabb    = AABB.unit()
    dataset = VolumeDataset(torch.from_numpy(vol_np), aabb, cfg)
    D, H, W = dataset.D, dataset.H, dataset.W
    print(f"Volume   : {D}x{H}x{W}  ({volume_path.name})")

    # ── model ─────────────────────────────────────────────────────────────────
    gc = GaussianCloud.load(str(ckpt_path), aabb, device, cfg)
    print(f"Model    : {gc.N} Gaussians  ({ckpt_path.name})")

    # ── reconstruct dense volume ───────────────────────────────────────────
    print("Reconstructing volume from Gaussians ...")
    pred_vol = reconstruct_volume(gc, dataset, cfg, device)
    print(f"           shape={pred_vol.shape}  "
          f"range=[{pred_vol.min():.3f}, {pred_vol.max():.3f}]")

    # ── render frames (camera at centre, rotating around Y axis) ─────────────
    azimuths     = np.linspace(0.0, 360.0, args.n_frames, endpoint=False)
    frame_arrays = []

    print(f"Rendering {args.n_frames} frames  "
          f"(camera at centre, fov={args.fov} deg, elev={args.elev} deg, "
          f"steps={args.n_steps}) ...")
    for i, az in enumerate(azimuths):
        frame = render_frame(
            pred_vol, az,
            elev_deg=args.elev,
            fov_deg=args.fov,
            img_h=args.img_size,
            img_w=args.img_size,
            n_steps=args.n_steps,
        )
        frame_arrays.append(frame)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(frame, cmap=args.cmap, vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(f'az={az:5.1f} deg  (from centre)', fontsize=9)
        ax.axis('off')
        png_path = out_dir / f'frame_{i:04d}.png'
        fig.savefig(str(png_path), bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  [{i+1:3d}/{args.n_frames}]  az={az:6.1f} deg  "
              f"peak={frame.max():.3f}  -> {png_path.name}")

    # ── animated GIF ──────────────────────────────────────────────────────────
    try:
        from PIL import Image
        duration_ms = round(1000 / args.gif_fps)
        pil_frames  = [Image.fromarray((f * 255).clip(0, 255).astype(np.uint8))
                       for f in frame_arrays]
        gif_path = out_dir / 'orbit.gif'
        pil_frames[0].save(
            str(gif_path), save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms, loop=0,
        )
        print(f"\nGIF saved : {gif_path}  ({args.gif_fps} fps, {len(pil_frames)} frames)")
    except ImportError:
        print("\nPillow not found — GIF skipped.  pip install Pillow")

    print(f"\nDone. Frames in: {out_dir}")


if __name__ == '__main__':
    main()
