from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable

import numpy as np
import tifffile
import torch
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def enumerate_blocks(shape, block_h: int = 128, block_w: int = 128):
    """Generate (y0, x0, y1, x1) for each block covering H×W plane in row-major order."""
    D, H, W = shape
    blocks = []
    for y0 in range(0, H, block_h):
        for x0 in range(0, W, block_w):
            y1 = min(y0 + block_h, H)
            x1 = min(x0 + block_w, W)
            blocks.append((y0, x0, y1, x1))
    return blocks


def get_block_by_index(shape, block_idx: int, block_h: int = 128, block_w: int = 128):
    """Get (y0, x0, y1, x1) for block at given index."""
    blocks = enumerate_blocks(shape, block_h, block_w)
    if block_idx < 0 or block_idx >= len(blocks):
        raise ValueError(f"Block index {block_idx} out of range [0, {len(blocks)-1}]")
    return blocks[block_idx]


class _LogFile:
    """Thin wrapper that writes to a .log file and optionally mirrors to stdout."""

    def __init__(self, path: Path):
        self._f = open(path, 'w', buffering=1)   # line-buffered

    def write(self, line: str):
        self._f.write(line + '\n')

    def close(self):
        self._f.close()


def _visualize_middle_slices(volume_gt: torch.Tensor, gc, out_dir: Path, epoch: int, device: torch.device,
                              ext: str = "png"):
    """Save middle slices (xy, xz, yz) with GT, prediction, and difference.

    Args:
        volume_gt: Ground truth volume (D, H, W)
        gc: GaussianCloud model
        out_dir: Output directory for saving plots
        epoch: Current epoch number
        device: Torch device
        ext: Output image format/extension (e.g. "png", "pdf")
    """
    if not HAS_MATPLOTLIB:
        return

    try:
        D, H, W = volume_gt.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

        # Get reconstructed slices by evaluating Gaussian field
        from ._3dgs import AABB
        aabb = AABB.unit()

        with torch.no_grad():
            # XY slice (middle Z)
            y_coords = torch.linspace(-1, 1, H, device=device)
            x_coords = torch.linspace(-1, 1, W, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            zz = torch.full_like(xx, (2 * mid_d / (D - 1) - 1))  # normalized Z at middle
            pts_xy = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
            pred_xy = gc.forward(pts_xy, chunk_n=1024).reshape(H, W)

            # XZ slice (middle Y)
            z_coords = torch.linspace(-1, 1, D, device=device)
            x_coords = torch.linspace(-1, 1, W, device=device)
            zz_xz, xx_xz = torch.meshgrid(z_coords, x_coords, indexing='ij')
            yy_xz = torch.full_like(xx_xz, (2 * mid_h / (H - 1) - 1))
            pts_xz = torch.stack([xx_xz, yy_xz, zz_xz], dim=-1).reshape(-1, 3)
            pred_xz = gc.forward(pts_xz, chunk_n=1024).reshape(D, W)

            # YZ slice (middle X)
            z_coords = torch.linspace(-1, 1, D, device=device)
            y_coords = torch.linspace(-1, 1, H, device=device)
            zz_yz, yy_yz = torch.meshgrid(z_coords, y_coords, indexing='ij')
            xx_yz = torch.full_like(yy_yz, (2 * mid_w / (W - 1) - 1))
            pts_yz = torch.stack([xx_yz, yy_yz, zz_yz], dim=-1).reshape(-1, 3)
            pred_yz = gc.forward(pts_yz, chunk_n=1024).reshape(D, H)

        gt_xy = volume_gt[mid_d]  # (H, W)
        gt_xz = volume_gt[:, mid_h]  # (D, W)
        gt_yz = volume_gt[:, :, mid_w]  # (D, H)

        pred_xy = pred_xy.cpu().numpy()
        pred_xz = pred_xz.cpu().numpy()
        pred_yz = pred_yz.cpu().numpy()
        gt_xy = gt_xy.cpu().numpy()
        gt_xz = gt_xz.cpu().numpy()
        gt_yz = gt_yz.cpu().numpy()

        diff_xy = np.abs(pred_xy - gt_xy)
        diff_xz = np.abs(pred_xz - gt_xz)
        diff_yz = np.abs(pred_yz - gt_yz)

        # Create figure with 3 rows (xy, xz, yz) and 3 columns (GT, Pred, Diff)
        fig, axs = plt.subplots(3, 3, figsize=(12, 12), facecolor='white')

        slices = [
            (gt_xy, pred_xy, diff_xy, "XY (mid-Z)"),
            (gt_xz, pred_xz, diff_xz, "XZ (mid-Y)"),
            (gt_yz, pred_yz, diff_yz, "YZ (mid-X)"),
        ]

        for row, (gt, pred, diff, title) in enumerate(slices):
            axs[row, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axs[row, 0].set_title(f"{title} - GT")
            axs[row, 0].axis('off')

            axs[row, 1].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axs[row, 1].set_title(f"{title} - Pred")
            axs[row, 1].axis('off')

            axs[row, 2].imshow(diff, cmap='hot')
            axs[row, 2].set_title(f"{title} - |Diff|")
            axs[row, 2].axis('off')

        fig.suptitle(f"Epoch {epoch} - Middle Slices", fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = out_dir / f"slices_ep{epoch:04d}.{ext}"
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return str(out_path)
    except Exception as e:
        print(f"[warn] Visualization failed: {e}")
        return None


def _load_volume(volume_path: str) -> tuple[torch.Tensor, float, float]:
    if Path(volume_path).suffix.lower() in (".h5", ".hdf5", ".hdf"):
        import h5py
        with h5py.File(volume_path, "r") as f:
            raw = f["raw"][:]
    else:
        raw = tifffile.imread(volume_path)
    vol = raw.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return torch.from_numpy(vol), vmin, vmax


def train_impl(
    cfg: argparse.Namespace,
    *,
    aabb_cls: type,
    volume_dataset_cls: type,
    gaussian_cloud_cls: type,
    make_optimizer: Callable[[Any, argparse.Namespace], torch.optim.Optimizer],
    update_lr: Callable[[torch.optim.Optimizer, int, int, argparse.Namespace], None],
    compute_loss: Callable[..., tuple[torch.Tensor, dict]],
    evaluate_fields: Callable[..., dict],
):
    run_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    flat_out   = getattr(cfg, 'flat_out', False)
    if flat_out:
        out_dir  = Path(cfg.out)
        ckpt_dir = out_dir       # checkpoints go directly into out_dir
    else:
        # Match SIGMA's output structure: outputs/{timestamp}_3dgs_vol/
        out_dir  = Path(cfg.out) / f"{run_stamp}_3dgs_vol"
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B (optional) ────────────────────────────────────────────────────────
    wandb_run = None
    if not getattr(cfg, 'no_wandb', True):
        try:
            import wandb
            run_name  = f"{Path(cfg.out).name}/{run_stamp}"
            wandb_run = wandb.init(
                project=getattr(cfg, 'wandb_project', '3dgs-microscopy'),
                entity=getattr(cfg,  'wandb_entity',  None),
                name=run_name,
                config=vars(cfg),
                dir='.',
            )
        except Exception as e:
            print(f"[W&B] init failed ({e}), continuing without logging")

    log = _LogFile(out_dir / "train.log")
    log.write(f"# run  : {run_stamp}")
    log.write(f"# cfg  : {cfg.out}")
    log.write(f"# vol  : {cfg.volume}")
    log.write(f"# date : {datetime.now().isoformat(timespec='seconds')}")
    log.write(f"# {'epoch':>6}  {'step':>7}  {'loss':>9}  {'mse':>9}  {'reg':>9}  "
              f"{'psnr':>7}  {'vol_psnr':>8}  {'N':>7}  {'s_mean':>7}  {'elapsed':>8}")

    print(f"Loading volume: {cfg.volume}")
    volume, vmin, vmax = _load_volume(cfg.volume)
    d, h, w = volume.shape
    print(f"  Shape     : {d} × {h} × {w}  ({d*h*w/1e6:.1f} M voxels)")
    print(f"  Intensity : [{vmin:.4f}, {vmax:.4f}] → [0, 1]")

    device = torch.device(cfg.device)

    # ── Block-mode extraction ──────────────────────────────────────────────────
    block_mode = getattr(cfg, 'block_mode', False)
    block_idx = getattr(cfg, 'block', None)
    block_h = getattr(cfg, 'block_h', 128)
    block_w = getattr(cfg, 'block_w', 128)

    if block_mode and block_idx is not None:
        y0, x0, y1, x1 = get_block_by_index((d, h, w), block_idx, block_h, block_w)
        volume = volume[:, y0:y1, x0:x1]  # Extract block
        d_b, h_b, w_b = volume.shape
        print(f"\n  Block {block_idx}: [{y0}:{y1}, {x0}:{x1}]  → {d_b} × {h_b} × {w_b}")

        # Override epochs and initialization for block training
        if getattr(cfg, 'block_n_epochs', None) is not None:
            cfg.epochs = cfg.block_n_epochs
        if getattr(cfg, 'block_n_init', None) is not None:
            cfg.n_init = cfg.block_n_init
        print(f"  Block config: {cfg.epochs} epochs, {cfg.n_init} init Gaussians")

    aabb = aabb_cls.unit()
    dataset = volume_dataset_cls(volume, aabb, cfg, swc_path=cfg.swc_path)

    init_pts   = None
    init_quats = None
    if getattr(cfg, 'swc_init', True):
        if getattr(cfg, 'swc_oriented_init', False):
            pts, quats = dataset.swc_oriented_init_params()
            init_pts   = pts.to(device)
            init_quats = quats.to(device)
            print(f"  SWC init     : {init_pts.shape[0]} pts with oriented quats")
        else:
            init_pts = dataset.swc_init_points().to(device)

    # Extra Gaussians seeded inside bright interior voxels (soma fill).
    interior_n = int(getattr(cfg, 'interior_init_n', 0))
    if interior_n > 0:
        thresh  = float(getattr(cfg, 'interior_init_thresh', 0.3))
        int_pts = dataset.interior_init_points(interior_n, thresh).to(device)
        if int_pts.numel() > 0:
            if init_pts is not None and init_pts.numel() > 0:
                # identity quats for interior points
                int_q = torch.zeros(int_pts.shape[0], 4, device=device)
                int_q[:, 0] = 1.0
                init_pts   = torch.cat([init_pts, int_pts], dim=0)
                if init_quats is not None:
                    init_quats = torch.cat([init_quats, int_q], dim=0)
            else:
                init_pts = int_pts
            print(f"  Interior init: {int_pts.shape[0]} pts (thresh={thresh})")

    total_steps = cfg.epochs * cfg.steps_per_epoch
    gc = gaussian_cloud_cls(cfg.n_init, aabb, device, cfg,
                            init_pts=init_pts, init_quats=init_quats)
    optimizer = make_optimizer(gc, cfg)
    update_lr(optimizer, 0, total_steps, cfg)
    if flat_out:
        init_path  = out_dir / "init.pth"
        best_path  = out_dir / "best.pth"
        final_path = out_dir / "last.pth"
    else:
        init_path  = out_dir / f"init_{run_stamp}.pth"
        best_path  = out_dir / f"best_{run_stamp}.pth"
        final_path = out_dir / f"final_{run_stamp}.pth"
    gc.save(init_path)

    log_entries = []
    best_psnr = -float('inf')
    bad_epochs = 0
    t0 = time()
    detail_interval = max(int(cfg.eval_detail_interval), 1)

    init_desc = 'SWC' if init_pts is not None and init_pts.numel() > 0 else 'random'
    print(f"  Gaussians : {gc.N} init from {init_desc}  (max {cfg.max_gaussians})")
    print(f"  Steps     : {total_steps}  ({cfg.epochs} ep × {cfg.steps_per_epoch})")

    step = 0
    last_psnr = float('nan')
    epoch_bar = tqdm(range(cfg.epochs), desc="epoch", unit="ep", dynamic_ncols=True)
    densify_until  = getattr(cfg, 'densify_until_step',  None)
    prune_from     = getattr(cfg, 'prune_from_step',     None)
    prune_until    = getattr(cfg, 'prune_until_step',    None)
    # Growth (clone/split) and pruning are decoupled onto independent cadences
    # -- see split_and_clone()/prune_only() docstrings. prune_interval defaults
    # to densify_interval when unset, and the standalone prune phase defaults
    # to starting alongside densify (prune_from_step) unless overridden.
    prune_interval = int(getattr(cfg, 'prune_interval', None) or cfg.densify_interval)
    prune_from_active = prune_from if prune_from is not None else cfg.densify_from_step

    for epoch in epoch_bar:
        epoch_loss = torch.zeros((), device=device)
        epoch_mse  = torch.zeros((), device=device)
        epoch_reg  = torch.zeros((), device=device)

        for _ in range(cfg.steps_per_epoch):
            step += 1

            pts, gt, sample_weights = dataset.sample(cfg.batch, device, cfg=cfg)

            # Fuse SSIM crop into training forward — one kernel call instead of two.
            lambda_ssim = getattr(cfg, 'lambda_ssim', 0.0)
            ssim_start  = int(getattr(cfg, 'ssim_start_step', 0))
            fuse_ssim   = (lambda_ssim > 0.0 and step >= ssim_start)
            if fuse_ssim:
                from ._3dgs import _ssim_sample_pts
                ssim_pts, ssim_gt_flat = _ssim_sample_pts(gc.aabb, dataset, cfg, device)
                all_pts = torch.cat([pts, ssim_pts], dim=0)
            else:
                all_pts    = pts
                ssim_pts   = None
                ssim_gt_flat = None

            optimizer.zero_grad()
            all_pred = gc.forward(all_pts, chunk_n=cfg.chunk_n)

            if fuse_ssim:
                pred      = all_pred[:cfg.batch]
                ssim_pred = all_pred[cfg.batch:]
            else:
                pred      = all_pred
                ssim_pred = None

            loss, stats = compute_loss(pred, gt, gc, cfg, dataset, step=step,
                                       sample_weights=sample_weights,
                                       ssim_pred=ssim_pred,
                                       ssim_gt_flat=ssim_gt_flat)

            loss.backward()
            gc.accum_grads()

            if gc.means.grad is not None:
                torch.nn.utils.clip_grad_norm_([gc.means], max_norm=cfg.grad_clip_norm)

            optimizer.step()
            update_lr(optimizer, step, total_steps, cfg)
            gc.clamp_means()
            gc.clamp_scales(cfg.scale_max_hard, getattr(cfg, 'scale_min_hard', None))

            epoch_loss = epoch_loss + stats['loss']
            epoch_mse  = epoch_mse + stats['mse']
            epoch_reg  = epoch_reg + (stats['loss'] - stats['mse'])

            if (
                step >= cfg.densify_from_step
                and (densify_until is None or step <= densify_until)
                and step % cfg.densify_interval == 0
            ):
                n_cloned, n_split = gc.split_and_clone(cfg)
                optimizer = make_optimizer(gc, cfg)
                update_lr(optimizer, step, total_steps, cfg)
                msg = (f"  [step {step:6d}] split/clone — "
                       f"cloned {n_cloned:5d}  split {n_split:5d}  total {gc.N:6d}")
                tqdm.write(msg)
                log.write(f"# {msg.strip()}")

            if (
                step >= prune_from_active
                and (prune_until is None or step <= prune_until)
                and step % prune_interval == 0
            ):
                n_pruned = gc.prune_only(cfg)
                if n_pruned > 0:
                    optimizer = make_optimizer(gc, cfg)
                    update_lr(optimizer, step, total_steps, cfg)
                    msg = f"  [step {step:6d}] prune only — pruned {n_pruned:5d}  total {gc.N:6d}"
                    tqdm.write(msg)
                    log.write(f"# {msg.strip()}")

            if cfg.ckpt_interval > 0 and step % cfg.ckpt_interval == 0:
                gc.save(out_dir / f"ckpt_{run_stamp}_{step:07d}.pth")

        avg_loss = (epoch_loss / cfg.steps_per_epoch).item()
        avg_mse  = (epoch_mse  / cfg.steps_per_epoch).item()
        avg_reg  = (epoch_reg  / cfg.steps_per_epoch).item()
        detail_eval = ((epoch + 1) % detail_interval == 0) or (epoch == cfg.epochs - 1)
        eval_metrics = evaluate_fields(gc, dataset, cfg, detail=detail_eval)
        last_psnr = eval_metrics['psnr']
        s_mean, s_max = gc.scale_stats()
        elapsed = time() - t0
        inten_mean = gc.intensity().mean().item()

        log_entries.append({
            'epoch': epoch + 1,
            'step': step,
            'loss': round(avg_loss, 6),
            'mse': round(avg_mse, 6),
            'reg': round(avg_reg, 6),
            'psnr': round(eval_metrics['psnr'], 4),
            'vol_psnr': None if math.isnan(eval_metrics['vol_psnr']) else round(eval_metrics['vol_psnr'], 4),
            'n_gauss': gc.N,
            's_mean': round(s_mean, 5),
            's_max': round(s_max, 5),
            'inten_mean': round(inten_mean, 4),
            'elapsed_s': round(elapsed, 1),
        })

        # Save checkpoint and visualization every N epochs (controlled by ckpt_epoch_interval)
        ckpt_epoch_interval = max(int(getattr(cfg, 'ckpt_epoch_interval', 10)), 1)
        if (epoch + 1) % ckpt_epoch_interval == 0 or epoch == cfg.epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pth"
            gc.save(ckpt_path)
            tqdm.write(f"  [epoch {epoch+1:4d}] saved checkpoint → {ckpt_path.name}")

            # Save middle slices visualization
            vis_path = _visualize_middle_slices(
                dataset.vol, gc, out_dir, epoch + 1, device
            )
            if vis_path:
                tqdm.write(f"  [epoch {epoch+1:4d}] saved visualization → {Path(vis_path).name}")

        epoch_bar.set_postfix(
            loss=f'{avg_loss:.4f}',
            psnr=f'{last_psnr:.2f}',
            N=gc.N,
            t=f'{elapsed/60:.1f}m',
        )

        vol_psnr_str = ('     nan' if math.isnan(eval_metrics['vol_psnr'])
                        else f"{eval_metrics['vol_psnr']:8.3f}")
        log.write(
            f"  {epoch+1:6d}  {step:7d}  {avg_loss:9.6f}  {avg_mse:9.6f}  {avg_reg:9.6f}  "
            f"{last_psnr:7.3f}  {vol_psnr_str}  {gc.N:7d}  {s_mean:7.5f}  {elapsed/60:7.1f}m"
        )

        if wandb_run is not None:
            wandb_run.log({
                'loss':     avg_loss,
                'mse':      avg_mse,
                'reg':      avg_reg,
                'psnr':     last_psnr,
                'vol_psnr': eval_metrics['vol_psnr'] if not math.isnan(eval_metrics['vol_psnr']) else None,
                'n_gaussians': gc.N,
                's_mean':   s_mean,
                's_max':    s_max,
            }, step=step)

        # Best-checkpoint selection uses vol_psnr (exact, full-grid metric matching
        # the training distribution) rather than the continuous-sample psnr, which
        # can diverge from true reconstruction quality as Gaussians overfit to
        # exact voxel centres. Only evaluated on detail_eval epochs, since vol_psnr
        # is NaN otherwise.
        if detail_eval:
            this_vol_psnr = eval_metrics['vol_psnr']
            if this_vol_psnr > best_psnr:
                best_psnr = this_vol_psnr
                bad_epochs = 0
                gc.save(best_path)
                best_msg = (f"  ★ new best  epoch {epoch+1:4d}  "
                            f"vol_PSNR={this_vol_psnr:.2f} dB  N={gc.N}  "
                            f"t={elapsed/60:.1f} min")
                tqdm.write(best_msg)
                log.write(f"# {best_msg.strip()}")
            else:
                bad_epochs += 1
                if cfg.early_stop_patience is not None and bad_epochs >= cfg.early_stop_patience:
                    tqdm.write(
                        f"  [epoch {epoch+1:4d}] early stopping after {bad_epochs} stagnant "
                        f"detail-eval epochs (best vol_PSNR={best_psnr:.2f} dB)"
                    )
                    break

    epoch_bar.close()

    gc.save(final_path)

    with open(out_dir / "log.json", "w") as f:
        json.dump(log_entries, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    total_min = (time() - t0) / 60
    summary = f"\nDone. Best vol_PSNR = {best_psnr:.2f} dB  →  {best_path}"
    log.write(f"# finished : {datetime.now().isoformat(timespec='seconds')}")
    log.write(f"# best vol_PSNR: {best_psnr:.3f} dB")
    log.write(f"# total    : {total_min:.1f} min")
    log.write(f"# output   : {out_dir}")
    log.close()

    if wandb_run is not None:
        raw_bytes = d * h * w   # uint8 voxels
        ckpt_bytes = best_path.stat().st_size
        wandb_run.summary.update({
            'best_psnr':         best_psnr,
            'n_gaussians':       gc.N,
            'ckpt_size_bytes':   ckpt_bytes,
            'compression_ratio': raw_bytes / ckpt_bytes,
            'train_min':         round(total_min, 1),
        })
        wandb_run.finish()

    print(summary)
    return gc, log_entries
