#!/usr/bin/env python3
"""Alias for pipeline.py — fit a SIREN voxel field (checkpoints only, no rendering).

pipeline.py holds the real CLI implementation (config.py -> io.py ->
model.py -> training.py -> io.py); this is a thin alias so `train.py` keeps
working as an entry point (e.g. for train_processed_data.sh) without
duplicating any logic.

Example:
    python3 train.py --iterations 2000 --learning-rate 0.001
"""

from __future__ import annotations

from pipeline import main

if __name__ == "__main__":
    main()
