# gaussian_volume/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

# Maps (toml section, toml key) -> argparse dest name, for every config value
# that has a corresponding CLI parameter in
# scripts/gaussian/representation/pipeline.py or
# scripts/gaussian/rendering/rasterise.py. Config values fill in the
# parameter defaults; explicit CLI flags always take precedence.
_CONFIG_TO_ARG: dict[tuple[str, str], str] = {
    ("experiment", "seed"): "seed",
    ("input", "path"): "input",
    ("input", "dataset"): "dataset",
    ("runtime", "device"): "device",
    ("model", "n_init"): "gaussians",
    ("model", "initial_scale"): "initial_scale",
    ("model", "initial_confidence"): "initial_confidence",
    ("model", "cutoff_sigma"): "cutoff_sigma",
    ("model", "epsilon"): "epsilon",
    ("training", "iterations"): "iterations",
    ("optimizer", "mean_lr"): "mean_lr",
    ("optimizer", "scale_lr"): "scale_lr",
    ("optimizer", "rotation_lr"): "rotation_lr",
    ("optimizer", "confidence_lr"): "confidence_lr",
    ("optimizer", "feature_lr"): "feature_lr",
    ("optimizer", "weight_decay"): "weight_decay",
    ("output", "checkpoint"): "output",
    ("output", "reconstruction"): "save_volume",
    ("logging", "checkpoint_every"): "checkpoint_every",
    ("logging", "log_every"): "log_every",
    # rasterise.py: independent MIP-projection output resolution, decoupled
    # from the volume's own voxel-grid dimensions (None/unset -> each view
    # matches the volume's own shape, no downsampling).
    ("rasterisation", "screen_width"): "screen_width",
    ("rasterisation", "screen_height"): "screen_height",
    # rasterise.py: ground-truth direct volume rendering (DVR) parameters.
    ("rasterisation", "dvr_density_scale"): "dvr_density_scale",
    ("rasterisation", "dvr_step_size"): "dvr_step_size",
    # rasterise.py: shared auto-exposure target for both the rasterised
    # MIPs and the ground-truth DVR (when their density_scale is unset).
    ("rasterisation", "target_opacity"): "target_opacity",
    ("rasterisation", "reference_density_scale"): "reference_density_scale",
}

# Destinations that hold filesystem paths in the config, and so need
# str -> Path conversion before being used as argparse defaults.
_PATH_ARG_DESTS = {"input", "output", "save_volume"}


def load_config(path: Path) -> dict[str, Any]:
    """
    Load a TOML experiment config file.
    """
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def config_section(config_data: dict[str, Any], section: str) -> dict[str, Any]:
    """
    Fetch a TOML table by name, or an empty dict if absent/malformed.

    Used to read the deeper, non-CLI config knobs (model hyperparameters,
    initialization details, densification/pruning) directly, rather than
    through an argparse flag for each one.
    """
    value = config_data.get(section)

    return value if isinstance(value, dict) else {}


def config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a parsed TOML config into argparse-dest-keyed defaults.

    Only keys present in _CONFIG_TO_ARG and actually found in the config
    are included, so unrelated/unimplemented config sections (e.g.
    densification, regularization) are silently ignored here.
    """
    defaults: dict[str, Any] = {}

    for (section, key), dest in _CONFIG_TO_ARG.items():
        section_values = config.get(section)

        if not isinstance(section_values, dict) or key not in section_values:
            continue

        value = section_values[key]

        if dest in _PATH_ARG_DESTS:
            value = Path(value)

        defaults[dest] = value

    return defaults


class TrainingLogger:
    """
    Writes every message to stdout and, if a path was given, appends it to
    a log file too — so a run's console output and its on-disk record are
    always identical, without every call site managing a file handle.
    """

    def __init__(self, path: Path | None) -> None:
        self._file = (
            path.open("a", encoding="utf-8") if path is not None else None
        )

    def __call__(self, message: str = "") -> None:
        print(message)

        if self._file is not None:
            self._file.write(message + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def resolve_log_path(config_data: dict[str, Any], input_path: Path) -> Path:
    """
    Build `<logging.log_dir>/<block_name>.log`.

    `block_name` is the input volume's parent directory name (e.g.
    `processed_data/complex/image_z14_y11_x17.tif` -> "complex"), so every
    block under processed_data/ gets its own named log file.
    """
    log_dir = Path(
        config_section(config_data, "logging").get("log_dir", "logs")
    )

    block_name = input_path.resolve().parent.name

    return log_dir / f"{block_name}.log"
