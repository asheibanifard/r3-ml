#!/usr/bin/env python3
"""Profile Gaussian splatting kernel timings for a single block.

Usage
-----
  /venv/r3-ml/bin/python3 scripts/profile_block.py [OPTIONS]

Examples
--------
  # Default (N=5000, chunk_n=5000, batch=2048)
  /venv/r3-ml/bin/python3 scripts/profile_block.py

  # N=50000 Gaussians
  /venv/r3-ml/bin/python3 scripts/profile_block.py --n 50000 --chunk_n 50000

  # Different block
  /venv/r3-ml/bin/python3 scripts/profile_block.py --volume data/fafb/blocks/image_z0_y0_x1.tif
"""

import argparse
import sys
import time

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import tifffile

import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()

from _3dgs._3dgs import (GaussianCloud, AABB, VolumeDataset,
                          _ssim_sample_pts, make_optimizer,
                          compute_loss, gaussian_reg_loss)


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian splatting kernel profiler")
    p.add_argument("--volume",   default="data/fafb/blocks/image_z0_y0_x0.tif")
    p.add_argument("--n",        type=int,   default=5000,  help="Number of Gaussians")
    p.add_argument("--chunk_n",  type=int,   default=None,  help="Chunk size (default = N)")
    p.add_argument("--batch",    type=int,   default=2048,  help="Training sample batch size")
    p.add_argument("--eval_pts", type=int,   default=200_000, help="PSNR eval sample count")
    p.add_argument("--reps",     type=int,   default=30,    help="Timing repetitions")
    p.add_argument("--warmup",   type=int,   default=5,     help="Warmup repetitions")
    return p.parse_args()


def timeit(label, fn, n=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<56s}  {ms:6.2f} ms")
    return ms


def main():
    args = parse_args()
    chunk_n = args.chunk_n if args.chunk_n is not None else args.n

    cfg = __import__('argparse').Namespace(
        scale_min_clamp=1e-5, mahal_max_clamp=20.0, init_scale=0.05,
        init_inten=0.1, init_scale_z_factor=1.0, n_init=args.n,
        swc_path=None, chunk_n=chunk_n, eval_samples=args.eval_pts,
        ssim_crop=64, batch=args.batch, grad_sample_weight=0.0,
        lambda_ssim=0.2, lambda_scale=1e-4, lambda_scale_ceiling=1e-3,
        lambda_scale_outlier=5e-4, lambda_sparsity=1e-3, lambda_aniso=0.0,
        lambda_count=0.0, lambda_L1=0.0, lambda_coverage=0.0, lambda_grad=0.0,
        scale_max_hard=0.5, ssim_start_step=0, adam_eps=1e-15,
        lr_means=1.6e-4, lr_scales=5e-3, lr_quats=1e-3, lr_inten=1e-2,
        lr_warmup_steps=0, lr_warmup_init_factor=0.1,
        lr_means_final=1.6e-6, lr_scales_min=1e-5, lr_quats_min=1e-5,
        lr_inten_min=1e-4, total_steps=100000,
    )

    device  = torch.device('cuda')
    vol_np  = tifffile.imread(args.volume).astype('float32') / 255.
    volume  = torch.from_numpy(vol_np)
    aabb    = AABB.unit()
    dataset = VolumeDataset(volume, aabb, cfg)
    gc      = GaussianCloud(args.n, aabb, device, cfg)
    opt     = make_optimizer(gc, cfg)

    pts_train, gt, sw = dataset.sample(args.batch,    device, cfg=cfg)
    pts_eval,  _,  _  = dataset.sample(args.eval_pts, device, cfg=cfg)
    ssim_pts, ssim_gt  = _ssim_sample_pts(aabb, dataset, cfg, device)
    fused_pts          = torch.cat([pts_train, ssim_pts], dim=0)

    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"\nN={args.n}  chunk_n={chunk_n}  batch={args.batch}  "
          f"eval_pts={args.eval_pts}  free GPU: {free_gb:.1f} GB\n")

    print("━━  Kernel timings  ━━")
    t_fwd    = timeit("forward  (train batch only)",
                      lambda: gc.forward(pts_train, chunk_n=chunk_n),
                      args.reps, args.warmup)
    t_fused  = timeit("forward  (fused train+SSIM)",
                      lambda: gc.forward(fused_pts, chunk_n=chunk_n),
                      args.reps, args.warmup)
    t_bwd    = timeit("backward (fused, v2 kernel)",
                      lambda: gc.forward(fused_pts, chunk_n=chunk_n).mean().backward(),
                      args.reps, args.warmup)
    t_reg    = timeit("gaussian_reg_loss (fused reg kernel)",
                      lambda: gaussian_reg_loss(gc, cfg, dataset),
                      args.reps, args.warmup)

    print()
    print("━━  Full step  ━━")

    def full_step():
        ap = gc.forward(fused_pts, chunk_n=chunk_n)
        p, sp = ap[:args.batch], ap[args.batch:]
        loss, _ = compute_loss(p, gt, gc, cfg, dataset, step=1000,
                               ssim_pred=sp, ssim_gt_flat=ssim_gt)
        loss.backward()

    t_full_step = timeit("forward + loss + backward",
                         full_step, args.reps, args.warmup)
    t_opt  = timeit("optimizer.step()",
                    lambda: opt.step(), args.reps, args.warmup)
    t_zero = timeit("optimizer.zero_grad()",
                    lambda: opt.zero_grad(), args.reps, args.warmup)

    print()
    print("━━  Eval  ━━")
    with torch.no_grad():
        t_eval = timeit("PSNR eval",
                        lambda: gc.forward(pts_eval, chunk_n=chunk_n),
                        min(args.reps, 10), args.warmup)

    print()
    print("━━  Epoch budget (50 steps + 1 eval)  ━━")
    bwd_only = t_bwd - t_fused
    total_ms = 50 * t_full_step + 50 * t_opt + 50 * t_zero + t_eval
    rows = [
        ("forward+SSIM fused × 50", 50 * t_fused),
        ("reg + backward     × 50", 50 * (t_full_step - t_fused)),
        ("optimizer.step()   × 50", 50 * t_opt),
        ("PSNR eval          × 1",  t_eval),
    ]
    for name, ms in rows:
        print(f"  {name:<28s}  {ms:6.0f} ms  ({100*ms/total_ms:.0f}%)")
    print(f"  {'total (est)':<28s}  {total_ms:6.0f} ms  →  {1000/total_ms:.2f} ep/s")


if __name__ == "__main__":
    main()
