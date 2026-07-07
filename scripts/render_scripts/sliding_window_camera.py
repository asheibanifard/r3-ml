"""Sliding-window camera navigation over independently-trained per-block
GaussianClouds, stitched into one virtual global volume without merging them.

Each block (e.g. models_smoke/block_z0NN_y0NN_x0NN/best.pth) was trained on
its own local [-1,1]^3 crop. Rather than eagerly merging every block into one
huge Gaussian set, BlockGaussianCache lazily loads (and coordinate-transforms
into the shared global frame) only the blocks that overlap the current
sliding-window position, evicting any that fall out of overlap as the window
moves — a window straddling a boundary pulls Gaussians from every block it
touches (up to 8 near a shared corner).

render_sliding_block() then keeps only the Gaussians whose mean falls inside
the window and renders a 360-degree look-around from a camera fixed at the
window's centre, via one of two paths:

  method='cpu_raymarch' (default) — reconstructs a dense local cube, then
      reuses render_camera.py's CPU ray-march + MIP renderer per frame.
  method='cuda_splat' — no dense reconstruction at all. The fused CUDA
      splat_mip kernel only ever projects along a fixed world axis, so instead
      of moving the camera we rotate the *Gaussians* by -azimuth about Y (world
      rotation) and take the axis-aligned projection — mathematically
      equivalent to the camera rotating by +azimuth, and lets every frame be a
      single real GPU splat-rasterization launch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _3dgs._3dgs import AABB, GaussianCloud, _load_eval_kernel
from render_scripts.render_camera import render_frame  # camera-at-centre ray-march + MIP


def axis_transform(block_n: int, global_n: int, offset: int) -> tuple[float, float]:
    """Map a LOCAL normalised coord (a block's own [-1,1]) to the GLOBAL
    normalised coord of the stitched volume: c_global = A*c_local + B."""
    A = (block_n - 1) / (global_n - 1)
    B = (2 * offset + (block_n - 1)) / (global_n - 1) - 1
    return A, B


def voxel_to_global_coord(idx: float, n: int) -> float:
    """Same convention used everywhere else: index 0..n-1 -> [-1, 1]."""
    return 2.0 * idx / (n - 1) - 1.0


class BlockGaussianCache:
    """Lazily loads per-block GaussianClouds (transformed into the shared
    global frame) and evicts any that no longer overlap the current window.

    block_shape_fn(dz, dy, dx) -> (block_D, block_H, block_W) lets callers
    describe ragged edge blocks (e.g. smoke_data's z axis is 125 = 2x50+25,
    so the last z-block is only 25 deep); defaults to a uniform block_size^3.
    """

    def __init__(self, blocks_dir, base_zyx, grid_n, global_shape,
                 block_size=50, block_shape_fn=None,
                 aabb=None, device=None, cfg=None):
        self.blocks_dir = blocks_dir
        self.base_z, self.base_y, self.base_x = base_zyx
        self.grid_n = grid_n
        self.global_shape = global_shape  # (D, H, W)
        self.aabb = aabb if aabb is not None else AABB.unit()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg if cfg is not None else argparse.Namespace(
            scale_min_clamp=1e-6, mahal_max_clamp=20.0)

        if block_shape_fn is None:
            def block_shape_fn(dz, dy, dx):
                return block_size, block_size, block_size

        self.cache = {}  # (dz,dy,dx) -> transformed {means,log_s,quats,inten}
        self.block_voxel_bounds = {}
        for dz in range(grid_n):
            for dy in range(grid_n):
                for dx in range(grid_n):
                    bD, bH, bW = block_shape_fn(dz, dy, dx)
                    z0, y0, x0 = dz * block_size, dy * block_size, dx * block_size
                    self.block_voxel_bounds[(dz, dy, dx)] = (z0, z0 + bD, y0, y0 + bH, x0, x0 + bW)

    def _load_block(self, key):
        dz, dy, dx = key
        z, y, x = self.base_z + dz, self.base_y + dy, self.base_x + dx
        ckpt_path = f'{self.blocks_dir}/block_z{z:03d}_y{y:03d}_x{x:03d}/best.pth'
        block_gc = GaussianCloud.load(ckpt_path, self.aabb, self.device, self.cfg)

        z0, z1, y0, y1, x0, x1 = self.block_voxel_bounds[key]
        D, H, W = self.global_shape
        Ax, Bx = axis_transform(x1 - x0, W, x0)
        Ay, By = axis_transform(y1 - y0, H, y0)
        Az, Bz = axis_transform(z1 - z0, D, z0)

        means = block_gc.means.detach().clone()
        means[:, 0] = Ax * means[:, 0] + Bx
        means[:, 1] = Ay * means[:, 1] + By
        means[:, 2] = Az * means[:, 2] + Bz

        # log_scales rescale by the same per-axis factor (approximation:
        # treats the non-uniform x/y/z scaling as axis-aligned, ignoring its
        # slight interaction with rotated/anisotropic Gaussians — fine for
        # visualization, not a precise physical reconstruction)
        log_s = block_gc.log_s.detach().clone()
        log_s[:, 0] += torch.log(torch.tensor(Ax, device=self.device))
        log_s[:, 1] += torch.log(torch.tensor(Ay, device=self.device))
        log_s[:, 2] += torch.log(torch.tensor(Az, device=self.device))

        return {'means': means, 'log_s': log_s,
                'quats': block_gc.quats.detach().clone(),
                'inten': block_gc.inten.detach().clone()}

    def overlapping_blocks(self, center_zyx, block_size):
        cz, cy, cx = center_zyx
        half = block_size / 2.0
        wz0, wz1, wy0, wy1, wx0, wx1 = cz - half, cz + half, cy - half, cy + half, cx - half, cx + half
        keys = []
        for key, (z0, z1, y0, y1, x0, x1) in self.block_voxel_bounds.items():
            if wz0 < z1 and wz1 > z0 and wy0 < y1 and wy1 > y0 and wx0 < x1 and wx1 > x0:
                keys.append(key)
        return keys

    def get_gaussians_for_window(self, center_zyx, block_size):
        needed = set(self.overlapping_blocks(center_zyx, block_size))

        for key in list(self.cache.keys()):           # discard non-overlapping blocks
            if key not in needed:
                del self.cache[key]
        for key in needed:                             # load newly-overlapping blocks
            if key not in self.cache:
                self.cache[key] = self._load_block(key)

        print(f'  window overlaps blocks {sorted(needed)} '
              f'({len(self.cache)} resident in cache)')
        if not self.cache:
            return None

        gc = GaussianCloud.__new__(GaussianCloud)
        gc.aabb, gc.device = self.aabb, self.device
        gc.scale_min, gc.mahal_clamp = self.cfg.scale_min_clamp, self.cfg.mahal_max_clamp
        gc.means = torch.cat([v['means'] for v in self.cache.values()], dim=0)
        gc.log_s = torch.cat([v['log_s'] for v in self.cache.values()], dim=0)
        gc.quats = torch.cat([v['quats'] for v in self.cache.values()], dim=0)
        gc.inten = torch.cat([v['inten'] for v in self.cache.values()], dim=0)
        return gc


def filter_gaussians_in_block(gc, center_zyx, block_size, shape_zyx):
    """Keep only Gaussians whose mean lies inside the block_size^3 voxel
    window centred at center_zyx (in the parent volume's voxel grid)."""
    D, H, W = shape_zyx
    cz, cy, cx = center_zyx
    half = block_size / 2.0

    z_lo, z_hi = voxel_to_global_coord(cz - half, D), voxel_to_global_coord(cz + half, D)
    y_lo, y_hi = voxel_to_global_coord(cy - half, H), voxel_to_global_coord(cy + half, H)
    x_lo, x_hi = voxel_to_global_coord(cx - half, W), voxel_to_global_coord(cx + half, W)

    means = gc.means
    in_box = ((means[:, 2] >= z_lo) & (means[:, 2] <= z_hi) &
              (means[:, 1] >= y_lo) & (means[:, 1] <= y_hi) &
              (means[:, 0] >= x_lo) & (means[:, 0] <= x_hi))

    local_gc = GaussianCloud.__new__(GaussianCloud)
    local_gc.aabb, local_gc.device = gc.aabb, gc.device
    local_gc.scale_min, local_gc.mahal_clamp = gc.scale_min, gc.mahal_clamp
    local_gc.means = gc.means[in_box]
    local_gc.log_s = gc.log_s[in_box]
    local_gc.quats = gc.quats[in_box]
    local_gc.inten = gc.inten[in_box]

    bounds = (z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)
    return local_gc, bounds


def reconstruct_local_cube(local_gc, bounds, block_size, device, chunk_n=1024):
    """Evaluate the filtered Gaussians on a block_size^3 grid spanning exactly
    the window's world extent — this array is then a self-contained [-1,1]^3
    cube as far as render_frame() is concerned."""
    z_lo, z_hi, y_lo, y_hi, x_lo, x_hi = bounds
    zc = torch.linspace(z_lo, z_hi, block_size, device=device)
    yc = torch.linspace(y_lo, y_hi, block_size, device=device)
    xc = torch.linspace(x_lo, x_hi, block_size, device=device)
    zz, yy, xx = torch.meshgrid(zc, yc, xc, indexing='ij')
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    with torch.no_grad():
        if local_gc.means.shape[0] == 0:
            vol = torch.zeros(block_size, block_size, block_size)
        else:
            vol = local_gc.forward(pts, chunk_n=chunk_n).reshape(
                block_size, block_size, block_size).clamp(0, 1).cpu()
    return vol.numpy().astype(np.float32)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1*q2 for (...,4) quaternions [w,x,y,z]. Composes
    rotations: R(q1*q2) = R(q1) @ R(q2) (verified against quat_to_rotmat)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def recenter_and_rescale(local_gc, bounds):
    """Map the window's true world bounds to a fresh [-1,1]^3 frame centred
    on the window (means recentred+rescaled per axis; log_scales rescaled by
    the same per-axis factor — same approximation as BlockGaussianCache's
    axis_transform: treats non-uniform x/y/z scaling as axis-aligned).
    Returns new (means, log_s); quats/inten are unaffected by this step."""
    z_lo, z_hi, y_lo, y_hi, x_lo, x_hi = bounds
    device = local_gc.means.device
    lo = torch.tensor([x_lo, y_lo, z_lo], device=device)
    hi = torch.tensor([x_hi, y_hi, z_hi], device=device)
    center = (lo + hi) / 2
    scale = 2.0 / (hi - lo)  # per-axis (x,y,z)

    means = (local_gc.means - center) * scale
    log_s = local_gc.log_s + torch.log(scale).unsqueeze(0)
    return means, log_s


def render_frame_cuda_splat(means, log_s, quats, inten, azimuth_deg,
                             scale_min, mahal_clamp, img_size=256, depth_samples=32):
    """One frame via the real fused CUDA splat_mip kernel: rotate the
    (already recentred/rescaled to [-1,1]^3) Gaussians by -azimuth about Y,
    then take the axis-aligned 'looking down Z' projection — equivalent to
    the camera rotating by +azimuth, since rotating the world one way looks
    identical to rotating the camera the other way."""
    device = means.device
    theta = torch.tensor(-np.radians(azimuth_deg), device=device, dtype=means.dtype)
    q_delta = torch.stack([torch.cos(theta / 2), torch.zeros((), device=device, dtype=means.dtype),
                            torch.sin(theta / 2), torch.zeros((), device=device, dtype=means.dtype)])
    q_delta = q_delta.unsqueeze(0).expand(means.shape[0], 4)

    c, s = torch.cos(theta), torch.sin(theta)
    R_y = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=means.dtype)
    means_rot = means @ R_y.T
    quats_rot = quat_multiply(q_delta, quats)

    kernel = _load_eval_kernel()
    with torch.no_grad():
        flat = kernel.splat_mip(means_rot.contiguous(), log_s.contiguous(),
                                 quats_rot.contiguous(), inten.contiguous(),
                                 -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                                 img_size, img_size, depth_samples, 0,
                                 float(scale_min), float(mahal_clamp))
    return flat.reshape(img_size, img_size).clamp(0, 1).cpu().numpy().astype(np.float32)


def render_sliding_block(cache, shape_zyx, center_zyx, block_size=32,
                          n_frames=36, fov=90.0, img_size=256, n_steps=128,
                          out_dir=None, gif_fps=12, method='cpu_raymarch'):
    """method='cpu_raymarch': reconstruct a dense cube once, then CPU ray-march
    + MIP per frame (works on CPU or GPU, but the ray-march itself is CPU).
    method='cuda_splat': no reconstruction step; every frame is a single real
    GPU splat_mip kernel launch on the (rotated) filtered Gaussians directly.
    """
    windowed_gc = cache.get_gaussians_for_window(center_zyx, block_size)
    if windowed_gc is None:
        raise ValueError(f'No trained blocks overlap window at {center_zyx}')

    local_gc, bounds = filter_gaussians_in_block(windowed_gc, center_zyx, block_size, shape_zyx)
    print(f'Gaussians inside the {block_size}^3 window at {center_zyx}: {local_gc.means.shape[0]:,} '
          f'(of {windowed_gc.means.shape[0]:,} loaded from overlapping blocks)')

    azimuths = np.linspace(0.0, 360.0, n_frames, endpoint=False)
    frames = []
    frame_times_s = []

    if method == 'cuda_splat':
        if cache.device.type != 'cuda':
            raise RuntimeError("method='cuda_splat' requires a CUDA device")
        if local_gc.means.shape[0] == 0:
            raise ValueError(f'No Gaussians inside window at {center_zyx} — cannot splat')

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        means, log_s = recenter_and_rescale(local_gc, bounds)
        torch.cuda.synchronize()
        reconstruct_s = time.perf_counter() - t0  # one-time recentre/rescale cost

        for az in azimuths:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            frame = render_frame_cuda_splat(
                means, log_s, local_gc.quats, local_gc.inten, az,
                local_gc.scale_min, local_gc.mahal_clamp,
                img_size=img_size, depth_samples=n_steps,
            )
            torch.cuda.synchronize()
            frame_times_s.append(time.perf_counter() - t0)
            frames.append(frame)
        local_vol = frames[0]  # kept for API parity with the raymarch path
        render_desc = 'real CUDA splat_mip kernel (rotated Gaussians, no reconstruction)'
    else:
        if cache.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        local_vol = reconstruct_local_cube(local_gc, bounds, block_size, cache.device)
        if cache.device.type == 'cuda':
            torch.cuda.synchronize()
        reconstruct_s = time.perf_counter() - t0
        print(f'Local cube reconstructed: shape={local_vol.shape}  '
              f'range=[{local_vol.min():.3f}, {local_vol.max():.3f}]  '
              f'({reconstruct_s * 1000:.1f} ms, one-time cost for this window)')

        # Per-frame render: camera fixed at the window centre, CPU ray-march
        # outward through the reconstructed cube, accumulate via max (MIP)
        # along each ray. Timed per frame since this is the cost that repeats
        # every time the camera looks in a new direction.
        for az in azimuths:
            t0 = time.perf_counter()
            frame = render_frame(local_vol, az, elev_deg=0.0, fov_deg=fov,
                                  img_h=img_size, img_w=img_size, n_steps=n_steps)
            frame_times_s.append(time.perf_counter() - t0)
            frames.append(frame)
        render_desc = 'CPU ray-march (scipy map_coordinates) through a reconstructed cube'

    frame_times_s = np.array(frame_times_s)
    avg_ms = frame_times_s.mean() * 1000
    fps = 1.0 / frame_times_s.mean()
    metrics = {
        'method': method,
        'n_gaussians': int(local_gc.means.shape[0]),
        'reconstruct_ms': reconstruct_s * 1000,
        'avg_frame_ms': avg_ms,
        'min_frame_ms': frame_times_s.min() * 1000,
        'max_frame_ms': frame_times_s.max() * 1000,
        'fps': fps,
        'n_frames': n_frames,
        'img_size': img_size,
        'n_steps': n_steps,
    }
    print(f'MIP-splat render [{render_desc}]: {n_frames} frames @ {img_size}x{img_size}  '
          f'-> avg {avg_ms:.1f} ms/frame ({fps:.2f} FPS)  '
          f'[min {metrics["min_frame_ms"]:.1f} / max {metrics["max_frame_ms"]:.1f} ms]')

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        pil_frames = [Image.fromarray((f * 255).clip(0, 255).astype(np.uint8)) for f in frames]
        gif_path = out_dir / 'sliding_block_orbit.gif'
        pil_frames[0].save(str(gif_path), save_all=True, append_images=pil_frames[1:],
                            duration=round(1000 / gif_fps), loop=0)
        print(f'Saved orbit GIF -> {gif_path}')

    return frames, local_vol, metrics
