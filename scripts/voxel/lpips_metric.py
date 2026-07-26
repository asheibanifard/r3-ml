#!/usr/bin/env python3
"""LPIPS evaluation only.
LPIPS requires a pretrained deep network, so it is deliberately kept outside
of the dependency-free CUDA renderer. Images are converted from [0,1] to
LPIPS's expected [-1,1] range.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import lpips

if len(sys.argv) != 3:
    raise SystemExit("usage: python3 lpips_metric.py gt_projection.pgm pred_projection.pgm")

def load(path: str) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

loss_fn = lpips.LPIPS(net="alex")
with torch.no_grad():
    value = loss_fn(load(sys.argv[1]), load(sys.argv[2])).item()
print(f"LPIPS(AlexNet) = {value:.6f}")
