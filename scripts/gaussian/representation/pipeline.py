# scripts/gaussian/representation/pipeline.py
#
# Thin CLI orchestrator over src/gaussian_volume/. Each stage below calls
# into exactly one package module, following:
#
#   config.py -> io.py -> initialization.py -> model.py -> training.py
#   -> reconstruction.py (via model.forward) -> checkpoint.py -> io.py

# Enables postponed evaluation of type hints (so `Path | None`-style
# annotations work without needing every name imported at module load time).
from __future__ import annotations

# Command-line argument parsing.
import argparse
# Used to read raw CLI args (sys.argv) when parse_arguments isn't given argv explicitly.
import sys
# Wall-clock timing for the per-stage debug instrumentation below.
import time
# Filesystem path handling for CLI arguments (input/output locations).
from pathlib import Path
# Generic type hint used for the loosely-typed parsed-TOML dict.
from typing import Any

# Core tensor library — device handling, seeding, and no_grad/inference_mode context managers.
import torch

# Public package API: the trainable model plus the Gaussian-initialization,
# volume-loading/saving, and checkpoint-saving entry points.
from gaussian_volume import (
    GaussianVolumeModel,
    initialize_gaussians,
    load_volume,
    save_gaussian_checkpoint,
    save_volume,
)
# config.py: TOML loading/merging and the per-run logger.
from gaussian_volume.config import (
    TrainingLogger,
    config_defaults,
    config_section,
    load_config,
    resolve_log_path,
)
# training.py: the optimizer factory, densify/prune config builders, PSNR
# metric, checkpoint-visualization helper, and the training loop itself.
from gaussian_volume.training import (
    CHECKPOINT_FIGURE_DPI,
    build_densification_config,
    build_pruning_config,
    compute_psnr,
    create_optimizer,
    save_reconstruction_pdf,
    train_model,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for Gaussian volume reconstruction.

    A --config TOML file, if given, supplies defaults for the parameters
    listed in gaussian_volume.config._CONFIG_TO_ARG; any of those parameters
    also passed explicitly on the command line override the config value.
    """
    # Default to the process's real argv (minus the script name) when the
    # caller doesn't supply one explicitly — lets tests pass a custom argv.
    if argv is None:
        argv = sys.argv[1:]

    # First pass: pull out --config only, so its values can seed the real
    # parser's defaults before the full set of arguments is parsed.
    config_parser = argparse.ArgumentParser(add_help=False)

    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML experiment config file.",
    )

    # parse_known_args tolerates every other flag (they belong to the real
    # parser below); remaining_argv carries them forward untouched.
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    # The real parser: reuses config_parser as a "parent" so --config is
    # still accepted (and shows up in --help) even though it was already
    # consumed above.
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Fit anisotropic 3D Gaussians directly to a target voxel volume.",
    )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    # Positional and optional (nargs="?"): can be omitted if [input].path is
    # set in the config file instead (checked later in validate_arguments).
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to the target volume. Supported formats: "
            ".npy, .npz, .h5, .hdf5, .tif, .tiff, .pt, and .pth. "
            "May be omitted if --config supplies [input].path."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="raw",
        help="Dataset name for HDF5 or NPZ files. Default: raw.",
    )

    # ------------------------------------------------------------------
    # Gaussian representation
    # ------------------------------------------------------------------

    # Initial Gaussian count — initialize_gaussians samples exactly this
    # many centres; densification may grow the population from here.
    parser.add_argument(
        "--gaussians",
        type=int,
        default=4096,
        help="Number of trainable 3D Gaussians. Default: 4096.",
    )

    # Physical-space starting standard deviation for every Gaussian (before
    # the model's softplus reparameterization is applied internally).
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=2.0,
        help="Initial Gaussian standard deviation in voxel units. Default: 2.0.",
    )

    # Physical-space starting confidence (opacity-like weight multiplier).
    parser.add_argument(
        "--initial-confidence",
        type=float,
        default=1.0,
        help="Initial Gaussian confidence. Default: 1.0.",
    )

    # Hard Mahalanobis-distance cutoff (in standard deviations) beyond which
    # a Gaussian contributes nothing to a voxel — bounds the CUDA kernel's
    # per-voxel work.
    parser.add_argument(
        "--cutoff-sigma",
        type=float,
        default=4.0,
        help="Ignore contributions beyond this Mahalanobis radius. Default: 4.0.",
    )

    # Added to the denominator of the normalized reconstruction
    # (sum_i w_i + epsilon) to avoid divide-by-zero where no Gaussian reaches
    # a voxel.
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Numerical stability value for normalized reconstruction. Default: 1e-8.",
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    parser.add_argument(
        "--iterations",
        type=int,
        default=3000,
        help="Number of optimization iterations. Default: 3000.",
    )

    # Each Gaussian parameter type gets its own Adam learning rate, since
    # they live on very different natural scales (positions vs. rotations
    # vs. a bounded confidence/feature range).
    parser.add_argument(
        "--mean-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Gaussian centres. Default: 1e-3.",
    )

    parser.add_argument(
        "--scale-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Gaussian scales. Default: 1e-3.",
    )

    parser.add_argument(
        "--rotation-lr",
        type=float,
        default=5e-4,
        help="Learning rate for quaternion rotations. Default: 5e-4.",
    )

    parser.add_argument(
        "--confidence-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Gaussian confidences. Default: 1e-3.",
    )

    parser.add_argument(
        "--feature-lr",
        type=float,
        default=1e-2,
        help="Learning rate for Gaussian intensity features. Default: 1e-2.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay. Default: 0.",
    )

    # Seeds both torch's CPU and (if available) CUDA RNGs, so a run is
    # reproducible end-to-end (initialization sampling, jitter, densify/prune
    # tie-breaks).
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device. Default: cuda.",
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    # Directory that will receive init.pth / best.pth / last.pth plus one
    # numbered <step>.pth snapshot per --checkpoint-every interval.
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help=(
            "Output directory for checkpoints: init.pth, last.pth, best.pth, "
            "and one <step>.pth snapshot every --checkpoint-every iterations. "
            "Default: outputs."
        ),
    )

    parser.add_argument(
        "--save-volume",
        type=Path,
        default=Path("outputs/reconstruction.tif"),
        help="Path for the final reconstructed volume. Default: outputs/reconstruction.tif.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save an intermediate checkpoint every N iterations. Use 0 to disable. Default: 500.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print training metrics every N iterations. Default: 500.",
    )

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    # Off by default: the extra per-stage diagnostics below (tensor
    # shapes/dtypes/devices, memory usage, file sizes, per-stage timing) are
    # verbose and only meant for troubleshooting, not routine runs.
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostic information at each pipeline stage.",
    )

    # If a --config file was given, load it and use its values as the
    # defaults for every matching argument above. Any of those arguments
    # also present on the command line still win, since parse_args() below
    # only falls back to a dest's default when it wasn't supplied.
    config: dict[str, Any] = {}

    if config_args.config is not None:
        # Read and parse the TOML file into a plain nested dict.
        config = load_config(config_args.config)
        # Flatten the subset of that dict that maps onto CLI flags (per
        # _CONFIG_TO_ARG) and install them as this parser's new defaults.
        parser.set_defaults(**config_defaults(config))

    # Now do the real parse: config-file values act as defaults, explicit
    # CLI flags in remaining_argv override them.
    args = parser.parse_args(remaining_argv)
    # Stash the config path itself on the namespace (parse_args didn't see
    # this flag directly since config_parser consumed it in the first pass).
    args.config = config_args.config

    # The raw parsed TOML, kept alongside the flattened argparse defaults so
    # deeper/non-CLI knobs (model hyperparameters, initialization details,
    # densification, pruning) can be read directly via config_section().
    args.config_data = config

    return args


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments before loading data.
    """
    # The positional `input` may have been omitted entirely if the config
    # file was also missing an [input].path entry — fail fast rather than
    # letting a later stage raise a more confusing error.
    if args.input is None:
        raise ValueError(
            "No input volume given: pass it positionally or set "
            "[input].path in the --config file."
        )

    if not args.input.exists():
        raise FileNotFoundError(f"Input volume does not exist: {args.input}")

    if args.gaussians <= 0:
        raise ValueError(f"--gaussians must be positive, got {args.gaussians}.")

    if args.initial_scale <= 0:
        raise ValueError(f"--initial-scale must be positive, got {args.initial_scale}.")

    if args.initial_confidence <= 0:
        raise ValueError(
            f"--initial-confidence must be positive, got {args.initial_confidence}."
        )

    if args.cutoff_sigma <= 0:
        raise ValueError(f"--cutoff-sigma must be positive, got {args.cutoff_sigma}.")

    if args.epsilon <= 0:
        raise ValueError(f"--epsilon must be positive, got {args.epsilon}.")

    if args.iterations <= 0:
        raise ValueError(f"--iterations must be positive, got {args.iterations}.")

    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")

    # Unlike the others, 0 is a legitimate value here (disables periodic
    # checkpointing entirely), so only negative values are rejected.
    if args.checkpoint_every < 0:
        raise ValueError(f"--checkpoint-every cannot be negative, got {args.checkpoint_every}.")

    # Validate every per-parameter-group learning rate the same way, via one
    # loop, instead of five near-identical if-statements.
    learning_rates = {
        "--mean-lr": args.mean_lr,
        "--scale-lr": args.scale_lr,
        "--rotation-lr": args.rotation_lr,
        "--confidence-lr": args.confidence_lr,
        "--feature-lr": args.feature_lr,
    }

    for name, value in learning_rates.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay cannot be negative, got {args.weight_decay}.")

    # Fail before any GPU work starts if CUDA was requested but isn't
    # actually present — a clearer error than whatever the first CUDA call
    # would raise deep inside the extension.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)

    # Only touch the CUDA RNG state if CUDA is actually available, so this
    # function also works unmodified on CPU-only runs.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_bytes(num_bytes: float) -> str:
    """
    Human-readable byte count for debug logging (e.g. tensor/file sizes).
    """
    value = float(num_bytes)

    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} TB"


def _tensor_debug_info(tensor: torch.Tensor) -> str:
    """
    One-line shape/dtype/device/memory summary of a tensor, for debug logs.
    """
    size_bytes = tensor.numel() * tensor.element_size()

    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} size={_format_bytes(size_bytes)}"
    )


def _cuda_memory_debug_info(device: torch.device) -> str:
    """
    Currently allocated/reserved CUDA memory, for debug logs. Empty string
    on CPU-only runs, since there is nothing meaningful to report.
    """
    if device.type != "cuda":
        return ""

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    return (
        f"cuda_allocated={_format_bytes(allocated)} "
        f"cuda_reserved={_format_bytes(reserved)}"
    )


def _file_size_debug_info(path: Path) -> str:
    """
    On-disk size of a just-written file, for debug logs.
    """
    if not path.exists():
        return "file not found"

    return _format_bytes(path.stat().st_size)


def main() -> None:
    # Parse CLI + config-file arguments, then fail fast on anything invalid
    # before touching the GPU or the filesystem beyond existence checks.
    args = parse_arguments()
    validate_arguments(args)
    set_random_seed(args.seed)

    device = torch.device(args.device)

    # Per-block log file (e.g. logs/<block_name>.log), derived from the
    # input volume's parent directory name; TrainingLogger mirrors every
    # message to both stdout and this file.
    log_path = resolve_log_path(args.config_data, args.input)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = TrainingLogger(log_path)

    # Debug-only diagnostics, gated behind --debug so routine runs stay at
    # the normal, terse log verbosity. `stage_timer` is a tiny stopwatch:
    # each call logs the elapsed time since the *previous* call and resets.
    debug_enabled = args.debug
    _stage_clock = {"last": time.perf_counter()}

    def dbg(message: str) -> None:
        if debug_enabled:
            log(f"[DEBUG] {message}")

    def stage_timer(stage_name: str) -> None:
        if debug_enabled:
            now = time.perf_counter()
            elapsed = now - _stage_clock["last"]
            log(f"[DEBUG] stage '{stage_name}' took {elapsed:.3f}s")
            _stage_clock["last"] = now

    dbg(f"parsed arguments: {vars(args)}")

    log("3D Gaussian volume reconstruction")
    log(f"Input:       {args.input}")
    log(f"Dataset:     {args.dataset}")
    log(f"Gaussians:   {args.gaussians}")
    log(f"Iterations:  {args.iterations}")
    log(f"Device:      {device}")
    log(f"Output dir:  {args.output}")
    log(f"Volume:      {args.save_volume}")
    log(f"Log:         {log_path}")

    # --------------------------------------------------------------
    # io.py: load the target TIFF or NumPy volume
    # --------------------------------------------------------------

    # Deeper, non-CLI [input] knobs (e.g. explicit normalization bounds) —
    # config_section returns {} if the section is absent, so .get() below
    # always has something to call.
    input_section = config_section(args.config_data, "input")

    # target: normalized [0,1] float32 tensor on `device`.
    # metadata: dict including "normalization" {minimum, maximum} — needed
    # later both to store in checkpoints and to invert normalization when
    # reconstructing.
    target, metadata = load_volume(
        args.input,
        dataset=args.dataset,
        device=device,
        normalize=True,
        normalization_minimum=input_section.get("normalization_minimum"),
        normalization_maximum=input_section.get("normalization_maximum"),
    )

    volume_shape = tuple(target.shape)

    log("Target volume")
    log(f"  Shape: {volume_shape}")
    log(f"  Min:   {target.min().item():.6f}")
    log(f"  Max:   {target.max().item():.6f}")
    log(f"  Mean:  {target.mean().item():.6f}")

    dbg(f"target tensor: {_tensor_debug_info(target)}")
    dbg(f"volume metadata: {metadata}")
    dbg(_cuda_memory_debug_info(device))
    stage_timer("load_volume")

    # --------------------------------------------------------------
    # initialization.py: create the initial Gaussian parameters
    # --------------------------------------------------------------

    init_section = config_section(args.config_data, "initialization")

    # Samples num_gaussians centres from `target` (by default weighted
    # toward high-intensity voxels), assigns each the same starting
    # scale/confidence, an identity rotation, and a feature value sampled
    # from the volume at that centre.
    initialization = initialize_gaussians(
        volume=target,
        num_gaussians=args.gaussians,
        strategy=init_section.get("strategy", "intensity_weighted"),
        initial_scale=args.initial_scale,
        initial_confidence=args.initial_confidence,
        uniform_sample_fraction=float(init_section.get("uniform_sample_fraction", 0.3)),
        intensity_power=float(init_section.get("intensity_power", 1.0)),
        minimum_sampling_weight=float(init_section.get("minimum_sampling_weight", 1e-6)),
        jitter=float(init_section.get("jitter", 0.25)),
        feature_sampling=init_section.get("feature_sampling", "trilinear"),
        generator=None,
    )

    dbg(f"initialization.strategy: {init_section.get('strategy', 'intensity_weighted')}")
    dbg(f"initialization.means: {_tensor_debug_info(initialization.means)}")
    dbg(f"initialization.scales: {_tensor_debug_info(initialization.scales)}")
    dbg(f"initialization.quaternions: {_tensor_debug_info(initialization.quaternions)}")
    dbg(f"initialization.confidences: {_tensor_debug_info(initialization.confidences)}")
    dbg(f"initialization.features: {_tensor_debug_info(initialization.features)}")
    stage_timer("initialize_gaussians")

    # --------------------------------------------------------------
    # model.py: stores trainable means, scales, rotations, features,
    # and confidences
    # --------------------------------------------------------------

    model_section = config_section(args.config_data, "model")

    # Kept as its own dict (rather than passed inline) so the exact same
    # values can also be written into every checkpoint's "extra" metadata —
    # required to reconstruct an identical model later without the config
    # file (see scripts/gaussian/reconstruct.py's build_model()).
    model_config = {
        "minimum_scale": float(model_section.get("minimum_scale", 1e-3)),
        "minimum_confidence": float(model_section.get("minimum_confidence", 1e-6)),
        "constrain_features": bool(model_section.get("constrain_features", True)),
        "feature_min": float(model_section.get("feature_min", 0.0)),
        "feature_max": float(model_section.get("feature_max", 1.0)),
    }

    # GaussianVolumeModel's constructor takes physical-space values (the
    # same space initialize_gaussians produced above) and inverse-transforms
    # them into raw, optimizable parameters internally.
    model = GaussianVolumeModel(
        means=initialization.means,
        scales=initialization.scales,
        quaternions=initialization.quaternions,
        confidences=initialization.confidences,
        features=initialization.features,
        **model_config,
    ).to(device)

    log(str(model))

    dbg(f"model.means: {_tensor_debug_info(model.means)}")
    dbg(f"model total parameters: {sum(p.numel() for p in model.parameters())}")
    dbg(_cuda_memory_debug_info(device))
    stage_timer("build_model")

    # --------------------------------------------------------------
    # training.py: build the optimizer that will drive the
    # optimization loop
    # --------------------------------------------------------------

    # Adam with one parameter group (and learning rate) per Gaussian
    # parameter type.
    optimizer = create_optimizer(
        model,
        mean_lr=args.mean_lr,
        scale_lr=args.scale_lr,
        rotation_lr=args.rotation_lr,
        confidence_lr=args.confidence_lr,
        feature_lr=args.feature_lr,
        weight_decay=args.weight_decay,
    )

    if debug_enabled:
        for group in optimizer.param_groups:
            dbg(
                f"optimizer group '{group.get('name')}': lr={group['lr']} "
                f"num_params={sum(p.numel() for p in group['params'])}"
            )
    stage_timer("create_optimizer")

    # Ensure both output locations exist before anything tries to write into them.
    args.output.mkdir(parents=True, exist_ok=True)
    args.save_volume.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # checkpoint.py: save the initial (pre-training) state
    # --------------------------------------------------------------

    init_path = args.output / "init.pth"

    save_gaussian_checkpoint(
        init_path,
        model=model,
        optimizer=optimizer,
        step=0,
        volume_shape=volume_shape,
        normalization=metadata.get("normalization"),
        extra={
            "input_path": str(args.input.resolve()),
            "dataset": args.dataset,
            "cutoff_sigma": args.cutoff_sigma,
            "epsilon": args.epsilon,
            "model_config": model_config,
        },
    )

    dbg(f"wrote {init_path} ({_file_size_debug_info(init_path)})")

    # Render a "before training" reconstruction so init.pth has a companion
    # PDF just like every later checkpoint — no gradient needed for a
    # visualization-only forward pass.
    with torch.no_grad():
        initial_prediction = model(
            shape=volume_shape, cutoff_sigma=args.cutoff_sigma, epsilon=args.epsilon
        )

    dbg(
        f"initial_prediction: {_tensor_debug_info(initial_prediction)} "
        f"min={initial_prediction.min().item():.6f} "
        f"max={initial_prediction.max().item():.6f}"
    )

    save_reconstruction_pdf(
        initial_prediction, target, init_path.with_suffix(".pdf"), dpi=CHECKPOINT_FIGURE_DPI
    )

    dbg(f"wrote {init_path.with_suffix('.pdf')} ({_file_size_debug_info(init_path.with_suffix('.pdf'))})")
    stage_timer("save_init_checkpoint")

    # --------------------------------------------------------------
    # training.py: starts the optimization loop (reconstruction.py /
    # extension / losses.py / densification.py / pruning.py /
    # checkpoint.py all run inside this loop, once per step)
    # --------------------------------------------------------------

    # L1/SSIM loss's data_range (max possible voxel value — 1.0 since the
    # volume was normalized above).
    data_range = float(config_section(args.config_data, "loss").get("data_range", 1.0))

    # How far outside the volume a Gaussian's mean may drift before being
    # clamped back in, each step (0.0 = clamp exactly to the volume bounds).
    mean_clamp_margin = float(
        config_section(args.config_data, "training").get("mean_clamp_margin", 0.0)
    )

    densification_config = build_densification_config(args.config_data)
    pruning_config = build_pruning_config(args.config_data)

    dbg(f"data_range={data_range} mean_clamp_margin={mean_clamp_margin}")
    dbg(f"densification_config: {densification_config}")
    dbg(f"pruning_config: {pruning_config}")

    # The training loop may replace `model`/`optimizer` internally whenever
    # densify/prune changes the Gaussian count (a fresh model needs a fresh
    # Adam, since old momentum buffers don't match the new parameter
    # shapes), so the returned objects — not the ones passed in — are the
    # ones to keep using afterward.
    model, optimizer, history = train_model(
        model=model,
        target=target,
        optimizer=optimizer,
        metadata=metadata,
        data_range=data_range,
        log=log,
        iterations=args.iterations,
        cutoff_sigma=args.cutoff_sigma,
        epsilon=args.epsilon,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output,
        input_path=args.input,
        dataset=args.dataset,
        mean_lr=args.mean_lr,
        scale_lr=args.scale_lr,
        rotation_lr=args.rotation_lr,
        confidence_lr=args.confidence_lr,
        feature_lr=args.feature_lr,
        weight_decay=args.weight_decay,
        densification_config=densification_config,
        pruning_config=pruning_config,
        mean_clamp_margin=mean_clamp_margin,
    )

    dbg(f"post-training gaussian count: {model.num_gaussians}")
    dbg(f"history length: {len(history)}")
    dbg(_cuda_memory_debug_info(device))
    stage_timer("train_model")

    # --------------------------------------------------------------
    # reconstruction.py (via model.forward): final reconstruction
    # --------------------------------------------------------------

    # eval() + inference_mode(): no dropout/batchnorm-style behavior differs
    # here, but this documents "we are done training" and additionally
    # disables autograd bookkeeping for the final forward pass.
    model.eval()

    with torch.inference_mode():
        final_prediction = model(
            shape=volume_shape, cutoff_sigma=args.cutoff_sigma, epsilon=args.epsilon
        )
        # The raw reconstruction can slightly exceed [0,1] (weighted average
        # of features that are themselves bounded, but not perfectly so at
        # the numeric margins); clamp before computing PSNR/saving.
        final_prediction = final_prediction.clamp(0.0, 1.0)

        final_psnr = compute_psnr(final_prediction, target, data_range=data_range, eps=1e-12)

    dbg(f"final_prediction: {_tensor_debug_info(final_prediction)} psnr={final_psnr.item():.3f}dB")
    stage_timer("final_reconstruction")

    # --------------------------------------------------------------
    # checkpoint.py: save the final Gaussian parameters
    # --------------------------------------------------------------

    last_path = args.output / "last.pth"

    save_gaussian_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        step=args.iterations,
        volume_shape=volume_shape,
        normalization=metadata.get("normalization"),
        extra={
            "input_path": str(args.input.resolve()),
            "dataset": args.dataset,
            "cutoff_sigma": args.cutoff_sigma,
            "epsilon": args.epsilon,
            # Full per-log-step metric history, so a later analysis doesn't
            # need the (possibly rotated/deleted) log file to see the curve.
            "history": history,
            "model_config": model_config,
        },
    )

    dbg(f"wrote {last_path} ({_file_size_debug_info(last_path)})")

    save_reconstruction_pdf(
        final_prediction, target, last_path.with_suffix(".pdf"), dpi=CHECKPOINT_FIGURE_DPI
    )

    dbg(f"wrote {last_path.with_suffix('.pdf')} ({_file_size_debug_info(last_path.with_suffix('.pdf'))})")
    stage_timer("save_last_checkpoint")

    # --------------------------------------------------------------
    # io.py: save the final reconstructed volume
    # --------------------------------------------------------------

    save_volume(
        final_prediction,
        args.save_volume,
        dataset="reconstruction",
        metadata={
            "source": str(args.input.resolve()),
            "checkpoint": str(last_path.resolve()),
            "volume_shape": volume_shape,
            "num_gaussians": model.num_gaussians,
            "iterations": args.iterations,
            "psnr": float(final_psnr.item()),
            "normalized": True,
        },
    )

    dbg(f"wrote {args.save_volume} ({_file_size_debug_info(args.save_volume)})")
    stage_timer("save_volume")

    log("Training complete")
    log(f"  Final loss:       {history[-1]['loss']:.6f}")
    log(f"  Final PSNR:       {final_psnr.item():.3f} dB")
    log(f"  Gaussian count:   {model.num_gaussians}")
    log(f"  Checkpoint:       {last_path}")
    log(f"  Reconstruction:   {args.save_volume}")

    log.close()


if __name__ == "__main__":
    main()
