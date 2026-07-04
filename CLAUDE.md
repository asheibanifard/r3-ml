# Project Report: 3D Gaussian Splatting for Volumetric EM Microscopy

## Overview

This project fits anisotropic 3-D Gaussian mixtures to blocks of electron-microscopy (EM) brain volume data using a hand-written CUDA C++ training kernel. The goal is to represent each spatial block of the FAFB (Full Adult Fly Brain) EM volume as a compact set of 3D Gaussians that can be stored, transmitted, and re-rendered far more efficiently than the raw voxel grid.

---

## Dataset

| Property | Value |
|---|---|
| Source | FAFB v14 — `data/fafb/fafb_v14_ffn1_z2000-6096.h5` |
| Pre-extracted blocks | `data/fafb/blocks/` |
| Block naming | `image_z{Z}_y{Y}_x{X}.tif` + `segment_z{Z}_y{Y}_x{X}.tif` |
| Grid dimensions | 64 × 64 × 64 blocks (Z × Y × X) |
| Total blocks | 262,144 |
| Block shape | 64 × 64 × 64 voxels (uint8) |
| Intensity range | [0, 255] → normalised to [0, 1] for training |

---

## Model: GaussianCloud

Each block is represented by a mixture of anisotropic 3-D Gaussians:

```
f(x) = Σ_k  softplus(inten_k) · exp(−½ (x−μ_k)ᵀ Σ_k⁻¹ (x−μ_k))
```

where the covariance is parameterised as:

```
Σ_k = R_k · diag(s_k²) · R_kᵀ
```

**Learnable parameters per Gaussian:**

| Parameter | Shape | Description |
|---|---|---|
| `means` | (N, 3) | Gaussian centres in [-1, 1]³ |
| `log_scales` | (N, 3) | Log per-axis standard deviations |
| `quats` | (N, 4) | Rotation quaternions [w, x, y, z] |
| `inten` | (N,) | Raw intensity; v_k = softplus(inten_k) |

**Adaptive density control** (clone / split / prune) runs every `densify_interval` steps starting at `densify_from_step`, growing from `n_init` up to `max_gaussians` per block. Each event reports pruned/cloned/split counts separately. Clone duplicates a small, high-gradient Gaussian with a ≈1σ offset; split replaces a large, high-gradient Gaussian with 2 shrunk daughters (parent removed). After `densify_until_step`, growth stops; an optional **prune-only** phase (`prune_from_step` → `prune_until_step`) continues removing dead/dim Gaussians without clone/split, for a final cleanup window once the population is fixed.

---

## CUDA C++ Kernel

The core computation lives in two fused CUDA kernels compiled JIT via `torch.utils.cpp_extension`:

| File | Purpose |
|---|---|
| `scripts/_3dgs/3dgs_cuda.cu` | Forward + backward pass for training |
| `scripts/_3dgs/3dgs_eval_cuda.cu` | Volume reconstruction + MIP splatting for inference |

### Forward kernel (`gaussian_forward_kernel`)
- **Tiled shared-memory** layout: 256 threads × 256 Gaussian tile
- One thread per sample point; Gaussians loaded cooperatively into shared memory
- Shared memory per block: ~12 KB (well within 48 KB limit)

### Backward kernel (`gaussian_backward_kernel`)
- **Transposed layout**: one thread per Gaussian, loops over all M sample points
- Eliminates all `atomicAdd` contention from the naïve M-thread design
- Gradient through quaternion normalisation and Rodrigues formula handled analytically

### Enabling the kernel
Pass `--use_kernel` to the training script. Without it, a pure-PyTorch fallback is used.

---

## Training

### Script

```
scripts/_3dgs/_3dgs.py
```

Run a single block:

```bash
/venv/r3-ml/bin/python3 scripts/_3dgs/_3dgs.py \
  --volume data/fafb/blocks/image_z0_y0_x0.tif \
  --use_kernel \
  --flat_out \
  --no_swc_init \
  --no_wandb \
  --out models/z000_y000_x000
```

### Loss function

```
L = L1(pred, gt)
  + λ_ssim   · SSIM(random 64×64 Z-crop)
  + λ_scale  · mean(s_max²)
  + λ_ceiling· mean(relu(s_max − cap))
  + λ_outlier· mean(relu(s_max − median − 3·MAD))
  + λ_sparsity·mean(v_k · (1 − GT(μ_k)))
  + λ_aniso  · mean(s_min²)
  + λ_count  · mean(sigmoid(raw_inten))
  + λ_L1     · mean(softplus(raw_inten))
  + λ_coverage·mean(−log(s_max / s_ref))
```

### Optimizer

Adam with per-parameter-group learning rates, rebuilt whenever N changes (densification invalidates Adam's momentum buffers):

| Group | Initial LR | Schedule |
|---|---|---|
| `means` | 1.6×10⁻⁴ | **Flat** until `densify_until_step`, then exponential decay (cosine ease-in over `lr_means_decay_ease_steps`) → `lr_means_final` |
| `log_scales` | 5×10⁻³ | Cosine annealing |
| `quats` | 1×10⁻³ | Cosine annealing |
| `inten` | 1×10⁻² | Cosine annealing |

Linear warmup over first `lr_warmup_steps` steps (from `lr_warmup_init_factor` of initial LR).

`means` is held flat through the whole densification phase (population still being shaped by clone/split, so it needs full step size to chase moving targets) and only starts decaying once `densify_until_step` is reached — avoids the means optimizer freezing (LR ≪ voxel spacing) before structure is resolved.

---

## Batch Training: All Blocks

### Script

```
scripts/train_all_blocks.py
```

Trains all 262,144 blocks sequentially using the fused CUDA kernel. Fully resumable — blocks with an existing `last.pth` are skipped.

**Launch command:**

```bash
nohup /venv/r3-ml/bin/python3 scripts/train_all_blocks.py \
  --blocks_dir data/fafb/blocks \
  --models_dir models \
  --logs_dir   logs/3dgs/blocks \
  --epochs     2000 \
  --steps      50 \
  --n_init     1000 \
  --max_gauss  5000 \
  --batch      2048 \
  --chunk_n    1024 \
  --ckpt_epoch_interval 100 \
  >> logs/train_all_blocks.log 2>&1 &
```

To resume from a specific block index: add `--start N`.

### Output layout

```
models/
  z000_y000_x000/
    init.pth           ← Gaussian cloud at initialisation
    best.pth           ← checkpoint with highest vol_PSNR (exact, full-voxel-grid metric — matches the training distribution; not the noisier continuous-sample psnr)
    last.pth           ← final checkpoint (marks block as done)
    epoch_0100.pth     ← periodic snapshot every 100 epochs
    epoch_0200.pth
    train.log          ← per-epoch loss / PSNR / N / elapsed
    log.json           ← same data as structured JSON
    config.json        ← all hyperparameters

logs/3dgs/blocks/
  z000_y000_x001.log  ← full stdout/stderr of each training run
  training_log.jsonl  ← master log: one JSON line per completed block
logs/
  train_all_blocks.log ← batch-level progress (stdout of the batch script)
```

### Observed training performance (block z000_y000_x001)

| Epoch | PSNR (dB) | N Gaussians | Notes |
|---|---|---|---|
| 1 | 4.42 | 1000 | random init |
| 3 | 10.94 | 1000 | rapid early improvement |
| 10 | 17.09 | 1000 | pre-densification |
| 11 | 8.56 | 2000 | first densify (N doubles) |
| 28 | 19.47 | 5000 | at max capacity |
| 51 | 22.13 | 5000 | stable refinement |
| 91 | 22.71 | 5000 | near convergence |
| ~200 | ~22–23 | 5000 | expected final |

**Throughput:** ~1.8 epochs/s → ~1100 s (~18 min) per block at 2000 epochs.

---

## Environment

| Component | Detail |
|---|---|
| GPU | NVIDIA RTX 5000 Ada Generation (32 GB) |
| CUDA | 12.9 (`nvcc` release 12.9.86) |
| Python | 3.11 (`/venv/r3-ml`) |
| PyTorch | JIT extension via `torch.utils.cpp_extension` |
| Key packages | `tifffile`, `tqdm`, `numpy`, `matplotlib`, `yaml` |

### CUDA dev headers (required for kernel compilation)

The following packages must be installed for JIT compilation to succeed:

```bash
apt-get install -y \
  libcusparse-dev-12-9 \
  libcublas-dev-12-9 \
  libcurand-dev-12-9 \
  libcufft-dev-12-9 \
  libcusolver-dev-12-9 \
  libcudnn9-dev-cuda-12
```

The compiled `.so` is cached in `~/.cache/torch_extensions/` and reused on subsequent runs.

---

## Key Source Files

| File | Role |
|---|---|
| `scripts/_3dgs/_3dgs.py` | Model (`GaussianCloud`), dataset, loss, optimizer, CLI |
| `scripts/_3dgs/_3dgs_training.py` | Training loop (`train_impl`) |
| `scripts/_3dgs/3dgs_cuda.cu` | Fused CUDA forward + backward kernel |
| `scripts/_3dgs/3dgs_eval_cuda.cu` | Inference kernels (volume reconstruct, MIP splat) |
| `scripts/train_all_blocks.py` | Batch runner for all 262,144 blocks |
| `data/fafb/blocks/` | Pre-extracted 64³ block TIFs |
| `models/` | Per-block checkpoint output |
| `logs/3dgs/blocks/` | Per-block training logs |
| `configs/train_single_block.yml` | Tuned single-block config (densification schedule, LR, loss weights) |
| `scripts/siren/siren.py` | SIREN model (`SIRENField`), training loop, CLI |
| `scripts/siren/siren_cuda.cu` | Hand-written CUDA forward + backward kernel for the MLP |

---

## Baseline: SIREN (Pure-CUDA MLP)

A second, much simpler model fits the same 64³ blocks for direct comparison
against the Gaussian-mixture model: Sitzmann et al., *"Implicit Neural
Representations with Periodic Activation Functions"* (NeurIPS 2020).

```
f(x) = sigmoid( W_L sin(w0 (... sin(w0 (W_0 x + b_0)) ...)) + b_L )
```

Default architecture (paper default): 3-D input → 4 sine layers × 256 units
→ 1 linear output → sigmoid (predictions guaranteed in [0,1], no clamping
needed). `omega_0 = 30` for every layer; weight init follows the paper's
Sec. 3.2 scheme (`U(-1/n, 1/n)` for the first layer, `U(-√(6/n)/ω₀,
√(6/n)/ω₀)` for every later layer including the output layer).

**Kernel:** every linear layer's matmul, bias-add, and the sine/sigmoid
activations + their analytic derivatives are hand-written CUDA kernels
(`scripts/siren/siren_cuda.cu`) — a tiled shared-memory GEMM (no cuBLAS) plus
elementwise kernels, JIT-compiled the same way as `3dgs_cuda.cu`. Forward
saves the pre-activations and per-layer inputs needed for backward, so
backward never re-runs the forward pass. Verified against finite-difference
gradients (worst-case 0.6% relative error across all layers).

**Data/eval reuse:** imports `VolumeDataset`, `AABB`, `evaluate_fields`,
`psnr_on_samples`, `vol_psnr`, `_load_volume`, `_visualize_middle_slices`
directly from the 3DGS module — same sampling distribution (importance-
weighted exact voxel centres), same PSNR definitions, same output layout
(`init.pth` / `best.pth` / `last.pth` / `train.log` / `log.json` /
`config.json`), so the two models are directly comparable on identical blocks.
`best.pth` selection uses `vol_psnr` (exact full-grid metric), same rationale
as the 3DGS pipeline.

```bash
/venv/r3-ml/bin/python3 scripts/siren/siren.py \
  --volume data/fafb/blocks/image_z0_y0_x0.tif \
  --flat_out \
  --out models_siren/z000_y000_x000
```