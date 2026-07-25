# SIREN voxel reconstruction and CUDA voxel rasterization

This project performs the complete experiment:

1. Create a synthetic `64 x 64 x 64` scalar voxel volume.
2. Fit a continuous SIREN function `F_theta(x,y,z)` to voxel-centre samples.
3. Query the trained SIREN at all voxel centres to reconstruct the volume.
4. Render the original volume using trilinear direct volume ray marching.
5. Render the reconstructed volume using front-to-back 3D-DDA voxel traversal.
6. Save GT, prediction, difference, comparison panels, orbit frames, and videos.
7. Report MSE, PSNR, SSIM, and separate kernel-only FPS for both renderers.
8. Compute LPIPS with the supplied Python evaluator.

The dense DDA renderer is a compact teaching analogue of SVRaster's sparse sorted-voxel renderer. It preserves ray-box intersection, exact front-to-back cell order, Beer-Lambert opacity, and transmittance compositing. It intentionally omits adaptive octrees, tile duplication, Morton ordering, radix sorting, and differentiable backward rasterization.

## Build

RTX 40-series:

```bash
make ARCH=89
```

Other common architectures:

- `75`: RTX 20-series
- `80`: A100
- `86`: RTX 30-series
- `89`: RTX 40-series
- `90`: H100

Direct command:

```bash
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_89 main.cu -o siren_voxel
```

## Run

```bash
./siren_voxel 2000 0.001 120 1000
```

Arguments:

```text
1. SIREN training iterations
2. Adam learning rate
3. Number of orbit views; use 0 to disable orbit output
4. Number of timed frames per renderer
```

A quick smoke test:

```bash
./siren_voxel 100 0.001 12 100
```

A higher-quality fit:

```bash
./siren_voxel 5000 0.0005 120 2000
```

## FPS meaning

The code reports two independent GPU timings:

- `GT direct volume renderer`: 384 trilinear ray-marching samples per intersecting ray.
- `Pred voxel rasterizer`: exact dense-grid DDA traversal of reconstructed voxels.

FPS excludes:

- SIREN training
- SIREN full-grid reconstruction
- CPU/GPU copies
- metric evaluation
- image and video writing
- CUDA startup

The benchmark uses 100 warm-up frames and CUDA events around repeated kernel launches.

## Outputs

Static outputs:

```text
output/gt_projection.pgm
output/pred_projection.pgm
output/diff.pgm
output/comparison.ppm
```

Orbit frames:

```text
output/orbit_gt/
output/orbit_pred/
output/orbit_diff/
output/orbit_comparison/
```

## Visualization and videos

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
sudo apt install ffmpeg
```

Create a labeled static PNG and four orbit videos:

```bash
python3 visualize.py --make-video --show
```

Generated files include:

```text
output/comparison.png
output/gt_orbit.mp4
output/pred_orbit.mp4
output/diff_orbit.mp4
output/comparison_orbit.mp4
```

## LPIPS

```bash
python3 lpips_metric.py \
  output/gt_projection.pgm \
  output/pred_projection.pgm
```

LPIPS uses a pretrained AlexNet feature network and maps image values from `[0,1]` to `[-1,1]`.

## Main equations

SIREN reconstruction:

```text
V_hat[i,j,k] = F_theta(x_i, y_j, z_k)
```

Beer-Lambert alpha for a ray segment of length `delta`:

```text
alpha = 1 - exp(-sigma * delta)
```

Front-to-back compositing:

```text
C <- C + T * alpha
T <- T * (1 - alpha)
```

Image comparison:

```text
difference = abs(GT - prediction)
PSNR = -10 log10(MSE)
```

## Important interpretation

The GT renderer samples a trilinearly interpolated continuous field. The predicted renderer treats each reconstructed voxel as piecewise constant. Therefore, the final image error contains both SIREN reconstruction error and renderer discretization error.
