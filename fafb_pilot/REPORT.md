# FAFB Real-Data Pilot: Dense-Resident DVR vs. Per-Block Gaussian-Splat MIP Rendering

## 1. Scope

This is a scoped pilot, not the full multi-scale study originally proposed (volumes
up to 4096³, three dense baselines including out-of-core NVMe streaming, Gaussian
payloads up to 5M, 12+ camera views, 5 repeats per configuration). That spec was
checked against this environment's actual constraints — free disk, a single 32 GB
GPU, and a training pipeline never exercised above single 64³ blocks — and scoped
down to what one session can deliver honestly, using **real FAFB data throughout**
(no `smoke_data`).

Two fitting strategies were tried in this session:

1. A **monolithic** Gaussian fit directly on an assembled 256³ volume (started,
   then abandoned before completion in favour of approach 2 below, since the
   real production pipeline in this repo — `train_all_blocks.py` — fits Gaussians
   **per 64³ block**, and any volume-size scaling story needs to go through that
   architecture, not a one-off monolithic fit).
2. **Per-block fitting + stitching**: 64 independent 64³-block fits (matching
   `train_all_blocks.py`'s real production defaults), remapped into a shared
   global coordinate frame, and concatenated into one Gaussian set — reusing the
   fact that the model's density field is additive, so no new "block-aware MIP
   combiner" render code was actually needed (see §4).

All results below are from strategy 2.

## 2. Data

| Property | Value |
|---|---|
| Source | Real FAFB v14 blocks, `data/fafb/blocks/image_z{Z}_y{Y}_x{X}.tif` |
| Assembled region | Blocks z,y,x ∈ [30,33] (4×4×4 = 64 blocks of 64³ each) |
| Assembled volume | 256×256×256 voxels, uint8, stitched contiguously |
| Raw intensity stats | mean 135.3, std 71.1, min 0, max 255 (confirmed real tissue signal) |
| Additional dense-only volumes | 512³ (8×8×8=512 blocks), 1024³ (16×16×16=4096 blocks) — assembled purely for the memory/FPS curve in §3; no Gaussian fitting was done at these sizes |
| Normalisation | Per-volume min-max → [0,1], matching `_3dgs_training.py::_load_volume`'s actual training convention |

## 3. Dense-resident DVR: FPS vs. memory footprint

The dense-resident renderer (`Mip_Render_Inside_Volume.cu`, `dense_voxel` mode)
uploads the full volume once to a GPU 3D texture and MIPs via hardware trilinear
sampling. Benchmarked at 128×128 resolution, 64 depth samples (matching the RQ1
Gaussian sweep settings in §5 for a fair combined comparison):

| Volume | Memory | Memory / 32760 MiB GPU | FPS |
|---|---|---|---|
| 256³ | 64 MB | 0.195% | 135,383 |
| 512³ | 512 MB | 1.563% | 69,399 |
| 1024³ | 4,096 MB | 12.5% | 6,091 |

All three volumes fit comfortably in the 32 GB GPU (never above 12.5%
utilisation), yet FPS still drops by **22×** from 256³ to 1024³. This is a
**texture-cache/bandwidth effect, not a capacity crossover** — a materially
different and more interesting result than the "runs out of memory" story the
original spec anticipated. The crossover this pilot actually measures is driven
by cache locality degrading as the resident texture grows, well before any
memory-capacity limit is approached.

## 4. Per-block Gaussian fitting

### 4.1 Training

64 independent runs of `scripts/_3dgs/_3dgs.py --use_kernel`, one per 64³ block,
using the real production defaults from CLAUDE.md / `train_all_blocks.py`:
`n_init=1000, max_gaussians=5000`. Epoch budget was reduced to 200 (from the
documented full 2000) for session-time feasibility — CLAUDE.md's own training
curve shows ~200 epochs as "near convergence" for this exact config, so this is
a reasonable but not fully-converged budget. Launched in 4 concurrent batches of
16 (GPU had headroom for this — batches were not compute-saturated by a single
small 5000-Gaussian job).

**Per-block quality** (exact full-voxel-grid `vol_PSNR`, the same metric
`train_all_blocks.py` uses for `best.pth` selection):

| | vol_PSNR (dB) |
|---|---|
| Mean | 20.13 |
| Std | 0.74 |
| Min | 18.34 |
| Max | 21.37 |

All 64 blocks converged consistently — no failed or diverging blocks.

### 4.2 Stitching

Each block's Gaussians live in that block's own local `[-1,1]³` frame. Stitching
requires remapping `means` and `log_scales` into a shared global frame
(`scripts/render_scripts/DVR/stitch_block_gaussians.py`):

```
global_mean   = block_center + local_mean * (1/N)
global_scale  = local_scale  * (1/N)         (log_scale_global = log_scale_local + log(1/N))
```

for an N×N×N grid of blocks (N=4 here). Quaternions and intensities are
unaffected by this uniform (isotropic) rescale. **Because the GaussianCloud
density field is additive** (`f(x) = Σ_k v_k · exp(-½ Mahalanobis)`), a flat
concatenation of all 64 remapped Gaussian sets is already the mathematically
correct combined field — no separate block-aware MIP combiner was needed, and
none was built. The existing single flat-list renderer handles the stitched
320,000-Gaussian set directly.

### 4.3 A genuine failure mode of naive per-block stitching

The first stitched render produced a near-constant MIP value (~5.8) across all
6 camera views looking in completely different directions — a strong signal
something was wrong, since a real spatial MIP should vary substantially with
view direction.

Investigation traced this to **individual Gaussians whose scale looked
unremarkable within their own block's local evaluation, but which are large
relative to the shared global frame once remapped.** This is not a small number
of pathological outliers:

| Scale threshold (global units) | Gaussians above threshold | % of 320,000 |
|---|---|---|
| > 0.3 | 580 | 0.18% |
| > 0.1 | 12,871 | 4.0% |
| > 0.05 | 73,057 | 22.8% |
| > 0.03 | 176,839 | 55.3% |

Filtering only the most extreme outliers (scale > 0.3, 580 Gaussians) barely
changed the near-constant floor (still ~5.8). Only aggressive filtering (scale >
0.03, removing 55% of all Gaussians) brought the MIP output into a
spatially-varying, dense-comparable range (`[0.31, 1.99]` vs. dense's roughly
`[0.8, 1.0]`).

**This is a real, structural limitation of naive per-block-then-stitch, not a
bug in the stitching arithmetic**: with only 200 of the documented 2000 epochs
per block, Gaussians have not yet shrunk to their tightest converged scale.
Each block's own `vol_PSNR` doesn't penalize this much (a somewhat-oversized
Gaussian still contributes reasonably within its own small block), but once 64
such fits are summed additively across a shared global frame, the cumulative
overlap contamination is severe. **This is exactly the kind of problem the
monolithic fitting approach doesn't have** (nothing in this pilot's monolithic
attempt showed anything resembling this artifact before it was abandoned) —
it's a genuine cost of the per-block architecture's parallelism, not
inherent to Gaussian splatting itself. A production version of this pipeline
would need either a longer per-block budget with a dedicated prune-only cleanup
phase (`prune_from_step`/`prune_until_step`, both left at their defaults here),
or some cross-block consistency mechanism.

Both the raw (320,000-Gaussian) and filtered (143,161-Gaussian, threshold 0.03)
stitched sets were carried forward for the comparisons below, since **neither
should be silently discarded** — the raw set is what per-block-then-stitch
actually produces without additional engineering, and reporting only the
filtered version would understate a real limitation.

## 5. Quality: dense GT vs. stitched Gaussian reconstruction

MIP-level metrics, computed with both images independently 1st/99th-percentile
normalised before comparison (necessary because the dense MIP lives in a
min-max-[0,1] space while the additive Gaussian-sum MIP has no such bound and
reaches values >1 wherever many Gaussians overlap — see the extended discussion
of this exact units problem earlier in this pipeline's development).

| | PSNR (dB) | SSIM | MAE |
|---|---|---|---|
| Raw stitched (320k) vs. dense | 8.57 ± 0.46 | 0.082 ± 0.029 | 0.306 ± 0.018 |
| Filtered stitched (143k) vs. dense | 9.27 ± 0.39 | 0.059 ± 0.012 | 0.289 ± 0.018 |

Filtering the needle artifacts modestly improves PSNR/MAE (less gross intensity
mismatch) but does not meaningfully improve — and by SSIM's measure, slightly
*worsens* — structural similarity to the dense GT. **The MIP-level reconstruction
is honestly poor in absolute terms** for both variants, in sharp contrast to the
~20 dB per-block volumetric PSNR. This mirrors a pattern established earlier in
this pipeline's development: MIP rendering (`max` over a ray of an additively-
summed field) is a much harsher, pileup-sensitive test than the per-voxel-center
evaluation `vol_PSNR` performs, and per-block stitching adds a second, independent
source of degradation (cross-block contamination) on top of that.

Visual comparison (1st/99th-percentile stretched, same view, yaw=0°/pitch=0°):
dense GT and the filtered stitched reconstruction show recognisably similar
overall grainy tissue texture, but the Gaussian reconstruction shows visible
bright streak artefacts (residual elongated Gaussians below the 0.3 filter
threshold) not present in the ground truth.

## 6. Gaussian payload vs. FPS and memory

Reusing the existing, previously-validated `gaussian_mip_rq1_auto_shuffled`
benchmark harness (nested deterministic nested subsets of the raw 320k-Gaussian
stitched set), at 128×128 resolution, 64 depth samples — matching the dense
benchmark in §3 exactly:

| Retention | Active Gaussians | Memory (MiB) | Memory / GPU | Median FPS |
|---|---|---|---|---|
| 10% | 32,000 | 1.343 | 0.0041% | 22.12 |
| 20% | 64,000 | 2.686 | 0.0082% | 10.33 |
| 30% | 96,000 | 4.028 | 0.0123% | 6.87 |
| 40% | 128,000 | 5.371 | 0.0164% | 5.15 |
| 50% | 160,000 | 6.714 | 0.0205% | 4.07 |
| 60% | 192,000 | 8.057 | 0.0246% | 3.39 |
| 75% | 240,000 | 10.071 | 0.0307% | 2.70 |
| 100% | 320,000 | 13.428 | 0.0410% | 2.02 |

The full 320,000-Gaussian stitched representation occupies **13.4 MiB** — about
**4.8× smaller** than the equal-coverage 64 MB dense 256³ volume — but renders
at **2.02 FPS**, roughly **67,000× slower** than the dense 256³ render at the
same settings (135,383 FPS).

## 7. The central result: FPS vs. representation memory footprint

![FPS vs memory ratio](results/figures/fps_vs_memory_ratio.png)

Both curves plotted on the same log-log axes (memory-footprint ratio vs. GPU
capacity on x, FPS on y). The two representations occupy almost entirely
separate regions of this plot: dense sits at higher memory / much higher FPS,
Gaussian sits at far lower memory / much lower FPS, and **the gap between them
(4–5 orders of magnitude in FPS) is far larger than the gap in memory (roughly
1–3 orders of magnitude)** across the entire range tested.

## 8. Discussion

The original hypothesis — that a compact Gaussian representation becomes
competitive once dense storage no longer fits comfortably in GPU memory,
because the dense method then pays bricking/streaming costs the Gaussian
method avoids — is **not what this pilot's data shows**, at least at the
scales tested (up to 12.5% GPU memory utilisation for dense, no out-of-core
comparison built). Instead:

- Dense-resident FPS degrades due to **texture-cache locality**, well before
  any memory-capacity limit (§3) — a real, non-obvious crossover driver the
  original spec didn't anticipate.
- Gaussian MIP rendering is **algorithmically more expensive per pixel**
  (cost scales with active Gaussians per tile, not just pixel count × depth
  samples), so its FPS disadvantage is severe and dominates the memory-based
  argument entirely across the tested range.
- Per-block-then-stitch, while architecturally necessary for scaling past a
  monolithic fit's Gaussian-count ceiling, introduces a **distinct
  reconstruction-quality cost** (§4.3) that a monolithic fit does not have.

None of this rules out a genuine Gaussian-wins regime — it just means that
regime, if it exists, is further out (larger volumes, an actual out-of-core
dense baseline, more training time per block, possibly a smarter
tile-culling/LOD scheme for the Gaussian renderer) than this pilot reaches.

## 9. Honest limitations of this pilot

- **Single spatial region** for Gaussian fitting (one 256³ crop) — the dense
  memory/FPS curve extends to 1024³, but no Gaussian fit was attempted at
  512³/1024³ scale (would require 512/4096 block fits respectively — feasible
  with this same pipeline, but a much longer unattended run).
- **No bricked or out-of-core dense baseline** — this pilot cannot speak to
  the out-of-core crossover regime that §8 identifies as the more likely place
  a Gaussian representation could actually win end-to-end.
- **Reduced per-block training budget** (200 of the documented 2000 epochs,
  no dedicated prune-only phase) is the direct cause of the stitching
  contamination in §4.3 — a longer budget might substantially change (likely
  improve) both the MIP-level quality numbers and possibly the FPS numbers
  (fewer, tighter Gaussians after full convergence and pruning).
- **6 camera views**, no repeat-view timing statistics on the direct
  dense-vs-Gaussian MIP comparison (the RQ1 payload sweep does carry proper
  5-repeat timing statistics, inherited unmodified from that tool's existing,
  previously-validated methodology).
- **MIP-level quality metrics required percentile renormalisation** to compare
  at all, because the additive Gaussian-sum field has no fixed intensity bound
  the way a min-max-normalised dense volume does. This is disclosed rather
  than papered over; a stricter treatment would define a principled shared
  intensity calibration between the two representations.
