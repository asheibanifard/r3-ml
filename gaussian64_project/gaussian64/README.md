# Gaussian64

Train a normalized 3D Gaussian field to reconstruct a synthetic `64×64×64`
volume, then reconstruct the dense volume with a JIT-compiled CUDA kernel.

The fitted field is

```text
             sum_i confidence_i * G_i(x) * feature_i
V(x) = ---------------------------------------------------
             sum_i confidence_i * G_i(x) + epsilon
```

The displayed MIPs are computed from the reconstructed dense field:

```text
MIP_xy = max_z V(x,y,z)
```

They are therefore MIPs of the fitted normalized volume—not maxima over
individual Gaussian ellipses.

## Install and run

```bash
cd gaussian64
python -m pip install -r requirements.txt
python train.py --config configs/synthetic64.toml
python reconstruct.py \
  --config configs/synthetic64.toml \
  --checkpoint outputs/synthetic64/model.pt
```

The first reconstruction call JIT-compiles the CUDA extension. Build products
are cached by PyTorch. A CUDA-capable PyTorch installation, NVIDIA driver, and
CUDA compiler (`nvcc`) are required for the JIT path.

For a quick CPU/PyTorch smoke test:

```bash
python reconstruct.py \
  --config configs/synthetic64.toml \
  --checkpoint outputs/synthetic64/model.pt \
  --backend torch
```

Outputs:

- `target.tif`: synthetic target volume
- `reconstruction.tif`: Gaussian reconstruction
- `error.tif`: absolute error
- `target_mip_*.png` and `reconstruction_mip_*.png`
- `metrics.json`

## Structure

```text
gaussian64/
├── configs/synthetic64.toml
├── csrc/bindings.cpp
├── csrc/normalized_rasterize.cu
├── gaussian64/config.py
├── gaussian64/io.py
├── gaussian64/jit.py
├── gaussian64/losses.py
├── gaussian64/model.py
├── gaussian64/synthetic.py
├── reconstruct.py
└── train.py
```
