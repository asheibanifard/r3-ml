#!/usr/bin/env python3
"""Fit a SIREN voxel field to a volume (checkpoints only — no rendering).

Thin CLI orchestrator over src/voxel/. Each stage below calls into
exactly one package module, following:

    config.py -> io.py -> model.py -> training.py (checkpoint.py runs
    inside its loop) -> io.py

Replaces the old nvcc-compiled `siren_voxel` binary with a JIT-compiled,
modular CUDA extension (extension/{siren,render,metrics}.cu, loaded via
torch.utils.cpp_extension). Fitting and rendering are separate steps: this
script only trains and writes checkpoints + the reconstructed volume;
use rendering/reconstruct.py to render a saved checkpoint (static comparison
image, orbit frames, renderer FPS benchmark).

    output/init.pth, output/<iteration>.pth, output/best.pth, output/last.pth
    output/*.pdf (checkpoint visualization, one per .pth)
    output/ground_truth_volume.tif, output/reconstructed_volume.tif

Example:
    python3 pipeline.py --iterations 2000 --learning-rate 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# scripts/voxel/ (this file's parent's parent) — added to sys.path so the
# `from _cli_config import ...` below resolves _cli_config.py, its sibling
# one directory up (_cli_config.py itself doesn't live under representation/,
# since it's shared with rendering/reconstruct.py too).
VOXEL_SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(VOXEL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(VOXEL_SCRIPTS_DIR))

from _cli_config import parse_args_with_config
from voxel import (
    SirenVoxelField,
    VoxelFieldConfig,
    load_volume,
    save_volume,
    train_impl,
)
from voxel.extension.loader import load_voxel_extension


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with an optional YAML config file.

    Priority (highest to lowest): explicit CLI flags, YAML config values,
    argparse defaults. Two-pass strategy: parse_known_args -> load YAML ->
    set_defaults -> full parse (mirrors gaussian_volume/_3dgs.py).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config file (e.g. configs/siren.yml); CLI flags override YAML values",
    )
    parser.add_argument(
        "--volume", type=Path, default=None,
        help=(
            "Real cubic volume to fit (e.g. a processed_data/*.tif EM block), "
            "normalized to [0,1] by its own min/max. grid_size is inferred "
            "from this file and --grid-size is ignored. Omit to fit the "
            "synthetic sphere/torus/blob demo volume instead."
        ),
    )
    parser.add_argument(
        "--save-volume", type=Path, default=None,
        help="Path to save the final reconstructed volume as a .tif",
    )
    parser.add_argument("--iterations", type=int, default=2000, help="SIREN training iterations")
    parser.add_argument(
        "--learning-rate", type=float, default=1.0e-3,
        help="Base Adam learning rate (schedule start; see --lr-min-ratio/--lr-warmup-steps)",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=1.0,
        help=(
            "Cosine-anneal the learning rate down to this fraction of "
            "--learning-rate by the final iteration. 1.0 (default) disables "
            "decay — a constant learning rate throughout."
        ),
    )
    parser.add_argument(
        "--lr-warmup-steps", type=int, default=0,
        help="Linearly warm up the learning rate over this many initial iterations; 0 disables warmup",
    )
    parser.add_argument(
        "--lr-warmup-init-factor", type=float, default=0.1,
        help="Learning rate at iteration 1, as a fraction of --learning-rate, when warmup is enabled",
    )
    parser.add_argument("--grid-size", type=int, default=512, help="Ignored if --volume is given")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument(
        "--first-omega-0", type=float, default=30.0,
        help="SIREN first-layer frequency (paper default 30.0); higher values fit finer detail",
    )
    parser.add_argument(
        "--hidden-omega-0", type=float, default=1.0,
        help="SIREN hidden-layer frequency (paper default 1.0)",
    )
    parser.add_argument("--train-batch", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("output"))

    # Rendering-only keys accepted (but unused) purely so the same
    # configs/siren.yml used here can also be passed to reconstruct.py (and
    # vice versa) without --config failing on "unknown keys".
    parser.add_argument("--image-width", type=int, default=1024, help="Unused here; kept for a shared config")
    parser.add_argument("--image-height", type=int, default=1024, help="Unused here; kept for a shared config")
    parser.add_argument("--dvr-steps", type=int, default=384, help="Unused here; kept for a shared config")
    parser.add_argument("--density-scale", type=float, default=8.0, help="Unused here; kept for a shared config")
    parser.add_argument("--orbit-frames", type=int, default=120, help="Unused here; kept for a shared config")
    parser.add_argument("--benchmark-frames", type=int, default=1000, help="Unused here; kept for a shared config")

    return parse_args_with_config(parser, path_dests=("volume", "save_volume", "out"))


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the voxel rasteriser requires an NVIDIA GPU.")

    device = torch.device("cuda")

    # --------------------------------------------------------------
    # io.py: load the target volume, or fall back to the synthetic
    # sphere/torus/blob demo volume (created below, once the extension
    # is loaded)
    # --------------------------------------------------------------
    if args.volume is not None:
        ground_truth_volume = load_volume(args.volume, device=device)
        grid_size = ground_truth_volume.shape[0]
    else:
        ground_truth_volume = None
        grid_size = args.grid_size

    # --------------------------------------------------------------
    # config.py: VoxelFieldConfig drives both the CUDA extension's
    # compile-time architecture (hidden_size/hidden_layers) and every
    # runtime kernel argument
    # --------------------------------------------------------------
    config = VoxelFieldConfig(
        grid_size=grid_size,
        image_width=args.image_width,
        image_height=args.image_height,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        train_batch=args.train_batch,
        dvr_steps=args.dvr_steps,
        density_scale=args.density_scale,
        seed=args.seed,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
    )

    extension = load_voxel_extension(config.hidden_size, config.hidden_layers)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if ground_truth_volume is None:
        ground_truth_volume = extension.create_ground_truth_volume(config.grid_size, device)

    # --------------------------------------------------------------
    # model.py: SirenVoxelField owns the flat parameter buffer and the
    # CUDA training-step/reconstruct calls
    # --------------------------------------------------------------
    field = SirenVoxelField(config, device=device)

    # --------------------------------------------------------------
    # training.py: the training loop (checkpoint.py's save/load runs
    # once per checkpoint inside this loop)
    # --------------------------------------------------------------
    summary = train_impl(
        field,
        ground_truth_volume,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        out_dir=out_dir,
        checkpoint_every=args.checkpoint_every,
        lr_min_ratio=args.lr_min_ratio,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_warmup_init_factor=args.lr_warmup_init_factor,
    )

    # --------------------------------------------------------------
    # io.py: save the final reconstructed volume
    # --------------------------------------------------------------
    reconstructed_volume = field.reconstruct()
    save_volume(out_dir / "ground_truth_volume.tif", ground_truth_volume)
    save_volume(out_dir / "reconstructed_volume.tif", reconstructed_volume)
    if args.save_volume is not None:
        save_volume(args.save_volume, reconstructed_volume)

    print(f"\nBest volume PSNR during training = {summary['best_volume_psnr']:.3f} dB")
    print(f"Final learning rate = {summary['final_learning_rate']:.2e}")

    print("\nOutputs")
    print(f"  {out_dir}/best.pth (+ .pdf), {out_dir}/last.pth (+ .pdf)")
    print(f"  {out_dir}/ground_truth_volume.tif")
    print(f"  {out_dir}/reconstructed_volume.tif")
    if args.save_volume is not None:
        print(f"  {args.save_volume}")
    print(
        "Run: python3 reconstruct.py "
        f"--checkpoint {out_dir}/best.pth --grid-size {config.grid_size} --out {out_dir}"
    )


if __name__ == "__main__":
    main()
