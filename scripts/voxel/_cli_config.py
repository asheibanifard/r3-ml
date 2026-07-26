# scripts/voxel/_cli_config.py
"""Shared YAML-config-as-argparse-defaults helper for train.py/reconstruct.py.

Priority (highest to lowest): explicit CLI flags, YAML config values,
argparse defaults. Two-pass strategy: parse_known_args -> load YAML ->
set_defaults -> full parse (mirrors gaussian_volume/_3dgs.py's
_load_yaml_config), so both scripts read the exact same
configs/siren.yml — the SIREN architecture (hidden_size/hidden_layers)
and seed that initialize a checkpoint's parameters must match between
whatever trained it and whatever later reconstructs from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml_config(config_path: Path, parser: argparse.ArgumentParser) -> dict:
    """Load and validate a YAML config file against the known CLI arguments.

    Unknown keys in the YAML are almost always typos, so fail loudly rather
    than silently ignoring a parameter the caller thinks they've set.
    """
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        parser.error(f"--config must point to a YAML mapping: {config_path}")

    valid_keys = {action.dest for action in parser._actions if action.dest != "help"}
    unknown_keys = sorted(set(data) - valid_keys)
    if unknown_keys:
        parser.error(f"Unknown keys in config file {config_path}: {', '.join(unknown_keys)}")

    return data


def parse_args_with_config(
    parser: argparse.ArgumentParser, *, path_dests: tuple[str, ...]
) -> argparse.Namespace:
    """Two-pass parse: --config sets defaults, then a full parse applies CLI overrides.

    path_dests lists dests that use `type=Path` — YAML-provided defaults
    arrive as plain str (argparse's `type=` conversion only runs on
    CLI-supplied values, not on set_defaults() values), so they're
    re-wrapped in Path(...) here. Path(...) is idempotent, so this is safe
    regardless of whether the value came from YAML, CLI, or a hardcoded
    default.
    """
    pre, _ = parser.parse_known_args()
    if pre.config is not None:
        parser.set_defaults(**load_yaml_config(pre.config, parser))

    args = parser.parse_args()

    for path_dest in path_dests:
        value = getattr(args, path_dest, None)
        if value is not None:
            setattr(args, path_dest, Path(value))

    return args
