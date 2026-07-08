#!/usr/bin/env python3
"""Pure-Python tiled MIP splatting for 3D Gaussian checkpoints.

This module is a CPU/GPU-friendly reference implementation that keeps the
geometry preparation separate from the per-frame render:

* tile occupancy statistics are computed once in `prepare_tiled_mip`
* Gaussians are culled conservatively by projected footprint
* an optional per-tile top-K cap trims preview renders
* `render_tiled_mip` performs a true max-over-depth of the summed field
* coarse-to-fine depth sampling defaults to an interactive 16-sample pass

The implementation matches the repo's `(means, log_s, quats, inten)` layout
used by `GaussianCloud`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _3dgs._3dgs import AABB, GaussianCloud


def _softplus_stable(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quats.unbind(dim=-1)
    norm = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz).clamp_min(1e-12)
    qw = qw / norm
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm

    return torch.stack(
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qw * qz),
            2 * (qx * qz + qw * qy),
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qw * qx),
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ],
        dim=-1,
    ).reshape(quats.shape[:-1] + (3, 3))


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _rotate_world_y(means: torch.Tensor, quats: torch.Tensor, azimuth_deg: float) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = torch.tensor(-np.radians(float(azimuth_deg)), device=means.device, dtype=means.dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    rotation = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], device=means.device, dtype=means.dtype)
    means_rot = means @ rotation.T
    q_delta = torch.stack(
        [
            torch.cos(theta / 2),
            torch.zeros((), device=means.device, dtype=means.dtype),
            torch.sin(theta / 2),
            torch.zeros((), device=means.device, dtype=means.dtype),
        ]
    ).unsqueeze(0).expand(means.shape[0], 4)
    quats_rot = _quat_multiply(q_delta, quats)
    return means_rot, quats_rot


def _view_axes(view_axis: int) -> Tuple[int, int, int]:
    if view_axis == 0:
        return 0, 1, 2
    if view_axis == 1:
        return 0, 2, 1
    return 1, 2, 0


def _axis_bounds(view_axis: int, lo_x: float, hi_x: float, lo_y: float, hi_y: float, lo_z: float, hi_z: float) -> Tuple[float, float, float, float, float, float]:
    if view_axis == 0:
        return lo_x, hi_x, lo_y, hi_y, lo_z, hi_z
    if view_axis == 1:
        return lo_x, hi_x, lo_z, hi_z, lo_y, hi_y
    return lo_y, hi_y, lo_z, hi_z, lo_x, hi_x


def _permute_precision_for_view_axis(
    view_axis: int,
    sigma_inv_uu: torch.Tensor,
    sigma_inv_uv: torch.Tensor,
    sigma_inv_uw: torch.Tensor,
    sigma_inv_vv: torch.Tensor,
    sigma_inv_vw: torch.Tensor,
    sigma_inv_ww: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if view_axis == 0:
        return sigma_inv_uu, sigma_inv_uv, sigma_inv_uw, sigma_inv_vv, sigma_inv_vw, sigma_inv_ww
    if view_axis == 1:
        return sigma_inv_uu, sigma_inv_uw, sigma_inv_uv, sigma_inv_ww, sigma_inv_vw, sigma_inv_vv
    return sigma_inv_vv, sigma_inv_vw, sigma_inv_uv, sigma_inv_ww, sigma_inv_uw, sigma_inv_uu


@dataclass
class PreparedTileMip:
    means: torch.Tensor
    log_s: torch.Tensor
    quats: torch.Tensor
    inten: torch.Tensor
    proj_u: torch.Tensor
    proj_v: torch.Tensor
    proj_w: torch.Tensor
    sigma_inv_uu: torch.Tensor
    sigma_inv_uv: torch.Tensor
    sigma_inv_uw: torch.Tensor
    sigma_inv_vv: torch.Tensor
    sigma_inv_vw: torch.Tensor
    sigma_inv_ww: torch.Tensor
    tile_offsets: torch.Tensor
    tile_indices: torch.Tensor
    out_h: int
    out_w: int
    tile_size: int
    depth_lo: float
    depth_hi: float
    u_lo: float
    u_hi: float
    v_lo: float
    v_hi: float
    density_scale: float
    mahal_clamp: float
    max_gauss_per_tile: int
    max_tile_occupancy: int
    avg_tile_occupancy: float
    capped_tiles: int
    pruned_gaussians: int


def prepare_tiled_mip(
    means: torch.Tensor,
    log_s: torch.Tensor,
    quats: torch.Tensor,
    inten: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
    view_axis: int = 0,
    azimuth_deg: float = 0.0,
    tile_size: int = 16,
    max_gauss_per_tile: int = 0,
    density_scale: float = 1.0e-4,
    mahal_clamp: float = 20.0,
    scale_min: float = 1.0e-5,
    lo_x: float = -1.0,
    hi_x: float = 1.0,
    lo_y: float = -1.0,
    hi_y: float = 1.0,
    lo_z: float = -1.0,
    hi_z: float = 1.0,
    device: torch.device | None = None,
) -> PreparedTileMip:
    """Precompute per-tile culling and a conservative top-K candidate set."""
    if device is None:
        device = means.device

    means = means.to(device=device, dtype=torch.float32).contiguous()
    log_s = log_s.to(device=device, dtype=torch.float32).contiguous()
    quats = quats.to(device=device, dtype=torch.float32).contiguous()
    inten = inten.to(device=device, dtype=torch.float32).contiguous()

    if azimuth_deg != 0.0:
        means, quats = _rotate_world_y(means, quats, azimuth_deg)

    u_lo, u_hi, v_lo, v_hi, depth_lo, depth_hi = _axis_bounds(view_axis, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z)

    u_scale = (out_w - 1) / (u_hi - u_lo) if out_w > 1 else 1.0
    v_scale = (out_h - 1) / (v_hi - v_lo) if out_h > 1 else 1.0

    rot = _quat_to_rotmat(quats)
    scales = torch.exp(log_s).clamp_min(scale_min)
    inv_scales = 1.0 / scales

    d00 = rot[..., 0, 0] * inv_scales[..., 0]
    d01 = rot[..., 0, 1] * inv_scales[..., 1]
    d02 = rot[..., 0, 2] * inv_scales[..., 2]
    d10 = rot[..., 1, 0] * inv_scales[..., 0]
    d11 = rot[..., 1, 1] * inv_scales[..., 1]
    d12 = rot[..., 1, 2] * inv_scales[..., 2]
    d20 = rot[..., 2, 0] * inv_scales[..., 0]
    d21 = rot[..., 2, 1] * inv_scales[..., 1]
    d22 = rot[..., 2, 2] * inv_scales[..., 2]

    sigma_inv_uu_xyz = d00 * d00 + d01 * d01 + d02 * d02
    sigma_inv_uv_xyz = d00 * d10 + d01 * d11 + d02 * d12
    sigma_inv_uw_xyz = d00 * d20 + d01 * d21 + d02 * d22
    sigma_inv_vv_xyz = d10 * d10 + d11 * d11 + d12 * d12
    sigma_inv_vw_xyz = d10 * d20 + d11 * d21 + d12 * d22
    sigma_inv_ww_xyz = d20 * d20 + d21 * d21 + d22 * d22

    sigma_inv_uu, sigma_inv_uv, sigma_inv_uw, sigma_inv_vv, sigma_inv_vw, sigma_inv_ww = _permute_precision_for_view_axis(
        view_axis,
        sigma_inv_uu_xyz,
        sigma_inv_uv_xyz,
        sigma_inv_uw_xyz,
        sigma_inv_vv_xyz,
        sigma_inv_vw_xyz,
        sigma_inv_ww_xyz,
    )

    cov_uu = torch.empty_like(sigma_inv_uu)
    cov_uv = torch.empty_like(sigma_inv_uv)
    cov_vv = torch.empty_like(sigma_inv_vv)

    if view_axis == 0:
        mu_u_world = means[:, 0]
        mu_v_world = means[:, 1]
        mu_w_world = means[:, 2]
        cov_uu.copy_(rot[..., 0, 0] * rot[..., 0, 0] * scales[..., 0] ** 2 + rot[..., 0, 1] * rot[..., 0, 1] * scales[..., 1] ** 2 + rot[..., 0, 2] * rot[..., 0, 2] * scales[..., 2] ** 2)
        cov_uv.copy_(rot[..., 0, 0] * rot[..., 1, 0] * scales[..., 0] ** 2 + rot[..., 0, 1] * rot[..., 1, 1] * scales[..., 1] ** 2 + rot[..., 0, 2] * rot[..., 1, 2] * scales[..., 2] ** 2)
        cov_vv.copy_(rot[..., 1, 0] * rot[..., 1, 0] * scales[..., 0] ** 2 + rot[..., 1, 1] * rot[..., 1, 1] * scales[..., 1] ** 2 + rot[..., 1, 2] * rot[..., 1, 2] * scales[..., 2] ** 2)
    elif view_axis == 1:
        mu_u_world = means[:, 0]
        mu_v_world = means[:, 2]
        mu_w_world = means[:, 1]
        cov_uu.copy_(rot[..., 0, 0] * rot[..., 0, 0] * scales[..., 0] ** 2 + rot[..., 0, 1] * rot[..., 0, 1] * scales[..., 1] ** 2 + rot[..., 0, 2] * rot[..., 0, 2] * scales[..., 2] ** 2)
        cov_uv.copy_(rot[..., 0, 0] * rot[..., 2, 0] * scales[..., 0] ** 2 + rot[..., 0, 1] * rot[..., 2, 1] * scales[..., 1] ** 2 + rot[..., 0, 2] * rot[..., 2, 2] * scales[..., 2] ** 2)
        cov_vv.copy_(rot[..., 2, 0] * rot[..., 2, 0] * scales[..., 0] ** 2 + rot[..., 2, 1] * rot[..., 2, 1] * scales[..., 1] ** 2 + rot[..., 2, 2] * rot[..., 2, 2] * scales[..., 2] ** 2)
    else:
        mu_u_world = means[:, 1]
        mu_v_world = means[:, 2]
        mu_w_world = means[:, 0]
        cov_uu.copy_(rot[..., 1, 0] * rot[..., 1, 0] * scales[..., 0] ** 2 + rot[..., 1, 1] * rot[..., 1, 1] * scales[..., 1] ** 2 + rot[..., 1, 2] * rot[..., 1, 2] * scales[..., 2] ** 2)
        cov_uv.copy_(rot[..., 1, 0] * rot[..., 2, 0] * scales[..., 0] ** 2 + rot[..., 1, 1] * rot[..., 2, 1] * scales[..., 1] ** 2 + rot[..., 1, 2] * rot[..., 2, 2] * scales[..., 2] ** 2)
        cov_vv.copy_(rot[..., 2, 0] * rot[..., 2, 0] * scales[..., 0] ** 2 + rot[..., 2, 1] * rot[..., 2, 1] * scales[..., 1] ** 2 + rot[..., 2, 2] * rot[..., 2, 2] * scales[..., 2] ** 2)

    mu_u_px = (mu_u_world - u_lo) * u_scale
    mu_v_px = (mu_v_world - v_lo) * v_scale
    cov_uu_px = cov_uu * u_scale * u_scale
    cov_uv_px = cov_uv * u_scale * v_scale
    cov_vv_px = cov_vv * v_scale * v_scale

    tr = cov_uu_px + cov_vv_px
    disc = torch.sqrt(torch.clamp((cov_uu_px - cov_vv_px) ** 2 + 4.0 * cov_uv_px ** 2, min=0.0))
    lambda_max = 0.5 * (tr + disc)
    radius = torch.sqrt(torch.clamp(lambda_max, min=1.0e-12)) * float(np.sqrt(max(mahal_clamp, 0.0)))

    u0 = torch.floor(mu_u_px - radius).to(torch.int64)
    u1 = torch.ceil(mu_u_px + radius).to(torch.int64)
    v0 = torch.floor(mu_v_px - radius).to(torch.int64)
    v1 = torch.ceil(mu_v_px + radius).to(torch.int64)

    tiles_x = (out_w + tile_size - 1) // tile_size
    tiles_y = (out_h + tile_size - 1) // tile_size
    tile_bins: list[list[tuple[float, int]]] = [[] for _ in range(tiles_x * tiles_y)]

    intensity = _softplus_stable(inten)
    score = intensity * (1.0 + radius)
    max_occupancy = 0
    pruned_gaussians = 0
    capped_tiles = 0

    for g in range(means.shape[0]):
        if int(u1[g]) < 0 or int(v1[g]) < 0 or int(u0[g]) >= out_w or int(v0[g]) >= out_h:
            continue

        clip_u0 = max(0, int(u0[g]))
        clip_u1 = min(out_w - 1, int(u1[g]))
        clip_v0 = max(0, int(v0[g]))
        clip_v1 = min(out_h - 1, int(v1[g]))

        tx0 = clip_u0 // tile_size
        tx1 = clip_u1 // tile_size
        ty0 = clip_v0 // tile_size
        ty1 = clip_v1 // tile_size
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_bins[ty * tiles_x + tx].append((float(score[g].item()), g))

    tile_offsets = [0]
    tile_indices: list[int] = []
    for candidates in tile_bins:
        max_occupancy = max(max_occupancy, len(candidates))
        if max_gauss_per_tile > 0 and len(candidates) > max_gauss_per_tile:
            candidates.sort(key=lambda item: item[0], reverse=True)
            pruned_gaussians += len(candidates) - max_gauss_per_tile
            candidates = candidates[:max_gauss_per_tile]
            capped_tiles += 1
        candidates.sort(key=lambda item: item[0], reverse=True)
        tile_indices.extend(index for _, index in candidates)
        tile_offsets.append(len(tile_indices))

    total_tiles = max(1, len(tile_bins))
    avg_occupancy = len(tile_indices) / float(total_tiles)
    print(
        f"  splat_mip tiles: {total_tiles} tiles, avg occupancy {avg_occupancy:.1f}, "
        f"max occupancy {max_occupancy}, capped {capped_tiles} tiles, pruned {pruned_gaussians} gaussians"
    )

    return PreparedTileMip(
        means=means,
        log_s=log_s,
        quats=quats,
        inten=intensity,
        proj_u=mu_u_world,
        proj_v=mu_v_world,
        proj_w=mu_w_world,
        sigma_inv_uu=sigma_inv_uu,
        sigma_inv_uv=sigma_inv_uv,
        sigma_inv_uw=sigma_inv_uw,
        sigma_inv_vv=sigma_inv_vv,
        sigma_inv_vw=sigma_inv_vw,
        sigma_inv_ww=sigma_inv_ww,
        tile_offsets=torch.tensor(tile_offsets, dtype=torch.int32, device=device),
        tile_indices=torch.tensor(tile_indices, dtype=torch.int32, device=device),
        out_h=out_h,
        out_w=out_w,
        tile_size=tile_size,
        depth_lo=float(depth_lo),
        depth_hi=float(depth_hi),
        u_lo=float(u_lo),
        u_hi=float(u_hi),
        v_lo=float(v_lo),
        v_hi=float(v_hi),
        density_scale=float(density_scale),
        mahal_clamp=float(mahal_clamp),
        max_gauss_per_tile=int(max_gauss_per_tile),
        max_tile_occupancy=max_occupancy,
        avg_tile_occupancy=avg_occupancy,
        capped_tiles=capped_tiles,
        pruned_gaussians=pruned_gaussians,
    )


def _render_tile(
    prepared: PreparedTileMip,
    tile_id: int,
    depth_samples: int,
    coarse_to_fine: bool,
) -> torch.Tensor:
    device = prepared.means.device
    tile_size = prepared.tile_size
    tiles_x = (prepared.out_w + tile_size - 1) // tile_size
    tile_y = tile_id // tiles_x
    tile_x = tile_id % tiles_x

    y0 = tile_y * tile_size
    y1 = min(prepared.out_h, y0 + tile_size)
    x0 = tile_x * tile_size
    x1 = min(prepared.out_w, x0 + tile_size)
    if y0 >= y1 or x0 >= x1:
        return torch.empty((0, 0), device=device, dtype=torch.float32)

    idx0 = int(prepared.tile_offsets[tile_id].item())
    idx1 = int(prepared.tile_offsets[tile_id + 1].item())
    if idx0 == idx1:
        return torch.zeros((y1 - y0, x1 - x0), device=device, dtype=torch.float32)

    gauss_idx = prepared.tile_indices[idx0:idx1].long()
    u = prepared.proj_u[gauss_idx]
    v = prepared.proj_v[gauss_idx]
    w = prepared.proj_w[gauss_idx]
    a = prepared.sigma_inv_uu[gauss_idx]
    b = prepared.sigma_inv_uv[gauss_idx]
    c = prepared.sigma_inv_uw[gauss_idx]
    d = prepared.sigma_inv_vv[gauss_idx]
    e = prepared.sigma_inv_vw[gauss_idx]
    f = prepared.sigma_inv_ww[gauss_idx]
    inten = prepared.inten[gauss_idx]

    yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
    xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
    py, px = torch.meshgrid(yy, xx, indexing="ij")
    px = px.reshape(-1, 1)
    py = py.reshape(-1, 1)

    u_span = max(prepared.u_hi - prepared.u_lo, 1.0e-12)
    v_span = max(prepared.v_hi - prepared.v_lo, 1.0e-12)
    u_world = prepared.u_lo + (px / max(prepared.out_w - 1, 1)) * u_span
    v_world = prepared.v_lo + (py / max(prepared.out_h - 1, 1)) * v_span

    def _accumulate_at_depth(depth_positions: torch.Tensor) -> torch.Tensor:
        out_samples = []
        for pz in depth_positions:
            du = u_world - u.reshape(1, -1)
            dv = v_world - v.reshape(1, -1)
            dw = pz - w.reshape(1, -1)
            mah = du * (a * du + b * dv + c * dw) + dv * (b * du + d * dv + e * dw) + dw * (c * du + e * dv + f * dw)
            contrib = inten.reshape(1, -1) * torch.exp(-0.5 * mah)
            contrib = torch.where(mah >= prepared.mahal_clamp, torch.zeros_like(contrib), contrib)
            out_samples.append(contrib.sum(dim=-1))
        return torch.stack(out_samples, dim=0)

    total_depth = max(int(depth_samples), 1)
    depth_positions = torch.linspace(prepared.depth_lo, prepared.depth_hi, total_depth, device=device, dtype=torch.float32)
    depth_vals = _accumulate_at_depth(depth_positions)
    max_vals = depth_vals.max(dim=0).values

    mapped = 1.0 - torch.exp(-prepared.density_scale * torch.clamp(max_vals, min=0.0))
    return mapped.reshape(y1 - y0, x1 - x0).clamp(0.0, 1.0)


def render_tiled_mip(
    prepared: PreparedTileMip,
    *,
    depth_samples: int = 16,
    coarse_to_fine: bool = True,
    clamp_output: bool = True,
) -> torch.Tensor:
    """Render a tiled MIP image from a prepared scene."""
    out = torch.zeros((prepared.out_h, prepared.out_w), device=prepared.means.device, dtype=torch.float32)
    tiles_x = (prepared.out_w + prepared.tile_size - 1) // prepared.tile_size
    tiles_y = (prepared.out_h + prepared.tile_size - 1) // prepared.tile_size

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile_id = tile_y * tiles_x + tile_x
            tile_img = _render_tile(prepared, tile_id, depth_samples, coarse_to_fine)
            if tile_img.numel() == 0:
                continue
            y0 = tile_y * prepared.tile_size
            x0 = tile_x * prepared.tile_size
            out[y0 : y0 + tile_img.shape[0], x0 : x0 + tile_img.shape[1]] = tile_img

    return out.clamp(0.0, 1.0) if clamp_output else out


def render_checkpoint(
    ckpt_path: str | Path,
    *,
    out_h: int = 256,
    out_w: int = 256,
    view_axis: int = 0,
    azimuth_deg: float = 0.0,
    tile_size: int = 16,
    max_gauss_per_tile: int = 0,
    depth_samples: int = 16,
    density_scale: float = 1.0e-4,
    scale_min: float = 1.0e-5,
    mahal_clamp: float = 20.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, PreparedTileMip]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = argparse.Namespace(scale_min_clamp=scale_min, mahal_max_clamp=mahal_clamp)
    ckpt_path = Path(ckpt_path)
    aabb = AABB.unit()
    gc = GaussianCloud.load(str(ckpt_path), aabb, device, cfg)

    prepared = prepare_tiled_mip(
        gc.means,
        gc.log_s,
        gc.quats,
        gc.inten,
        out_h=out_h,
        out_w=out_w,
        view_axis=view_axis,
        azimuth_deg=azimuth_deg,
        tile_size=tile_size,
        max_gauss_per_tile=max_gauss_per_tile,
        density_scale=density_scale,
        mahal_clamp=mahal_clamp,
        scale_min=scale_min,
        device=device,
    )
    image = render_tiled_mip(prepared, depth_samples=depth_samples)
    return image, prepared


def _save_image(path: Path, image: torch.Tensor, cmap: str = "gray") -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(image.detach().cpu().numpy(), cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="models_smoke/block_z001_y003_x008/best.pth")
    parser.add_argument("--out", default="results/py_tiled_mip.png")
    parser.add_argument("--npy_out", default="results/py_tiled_mip.npy")
    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--out_w", type=int, default=256)
    parser.add_argument("--view_axis", type=int, default=0)
    parser.add_argument("--azimuth_deg", type=float, default=0.0)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--depth_samples", type=int, default=16)
    parser.add_argument("--max_gauss_per_tile", type=int, default=0)
    parser.add_argument("--density_scale", type=float, default=1.0e-4)
    parser.add_argument("--scale_min", type=float, default=1.0e-5)
    parser.add_argument("--mahal_clamp", type=float, default=20.0)
    parser.add_argument("--cmap", default="gray")
    args = parser.parse_args()

    if args.view_axis not in (0, 1, 2):
        raise ValueError("--view_axis must be 0, 1, or 2")

    image, prepared = render_checkpoint(
        args.ckpt,
        out_h=args.out_h,
        out_w=args.out_w,
        view_axis=args.view_axis,
        azimuth_deg=args.azimuth_deg,
        tile_size=args.tile_size,
        max_gauss_per_tile=args.max_gauss_per_tile,
        depth_samples=args.depth_samples,
        density_scale=args.density_scale,
        scale_min=args.scale_min,
        mahal_clamp=args.mahal_clamp,
    )

    out_path = Path(args.out)
    npy_path = Path(args.npy_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), image.detach().cpu().numpy())
    _save_image(out_path, image, cmap=args.cmap)

    print(f"Saved {out_path}")
    print(f"Saved {npy_path}")
    print(
        f"Tile stats: avg={prepared.avg_tile_occupancy:.1f}, max={prepared.max_tile_occupancy}, "
        f"capped_tiles={prepared.capped_tiles}, pruned={prepared.pruned_gaussians}"
    )


if __name__ == "__main__":
    main()