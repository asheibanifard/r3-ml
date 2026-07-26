# SIREN voxel reconstruction and CUDA voxel rasterization

This project performs the complete experiment, split into two separate
steps — training (`train.py`) and rendering (`reconstruct.py`) — so a fit
can be batch-run over many blocks without paying for rendering/benchmark
work it doesn't need, and any saved checkpoint can be re-rendered later
without retraining:

1. **pipeline.py**: the real CLI implementation — create a synthetic
   `64 x 64 x 64` scalar voxel volume (or load a real one via `--volume`),
   fit a continuous SIREN function `F_theta(x,y,z)` to voxel-centre
   samples, and save checkpoints + the raw reconstructed volume. No
   rendering happens here (mirrors `gaussian_volume/pipeline.py`'s
   convention of being training-only). `train.py` is a thin alias for it.
2. **reconstruct.py**: load a saved checkpoint, query the SIREN at all
   voxel centres, then render both the original volume (trilinear direct
   volume ray marching) and the reconstruction (front-to-back 3D-DDA voxel
   traversal). Every checkpoint gets its own self-contained folder with the
   `.pth`, a 3D axial/coronal/sagittal slice comparison (`slices.pdf`), a
   rendered GT/prediction/difference comparison annotated with
   PSNR/SSIM/LPIPS/FPS (`rendered.pdf`), the underlying PGM/PPM images, and
   optional orbit frames.

`train_processed_data.sh` batch-runs `train.py` (i.e. `pipeline.py`) over
every block in `processed_data/`, skipping rendering entirely since it's
about fitting every block, not producing per-block demo videos.

The dense DDA renderer is a compact teaching analogue of SVRaster's sparse sorted-voxel renderer. It preserves ray-box intersection, exact front-to-back cell order, Beer-Lambert opacity, and transmittance compositing. It intentionally omits adaptive octrees, tile duplication, Morton ordering, radix sorting, and differentiable backward rasterization.

## Package layout

This used to be a single `nvcc`-compiled `main.cu` binary. It is now a
modular, JIT-compiled CUDA extension, mirroring `src/gaussian_volume`:

```text
src/voxel/
  config.py            VoxelFieldConfig (grid size, image size, SIREN
                        architecture, first/hidden omega_0, train batch,
                        DVR steps, density scale)
  model.py              SirenVoxelField: parameter buffer + training_step/
                        reconstruct, backed by the CUDA extension
  training.py           train_impl: the checkpointing training loop
  lr_schedule.py        cosine_lr_schedule: linear warmup + cosine annealing
                        for the single flat learning rate this CUDA
                        training loop uses (see configs/siren.yml's
                        lr_min_ratio/lr_warmup_steps/lr_warmup_init_factor)
  checkpoint.py         save/load the flat parameter buffer as a torch
                        .pth checkpoint (init.pth/best.pth/last.pth/
                        <iteration>.pth), matching gaussian_volume's convention
  io.py                 load_volume / save_volume (.tif) / save_pgm /
                        save_comparison_ppm
  rendering.py          make_camera + render_direct_volume /
                        render_voxel_rasterizer / evaluate_images /
                        compute_volume_psnr wrappers over the extension
  extension/
    common.cuh          Vec3/camera math, SIREN parameter layout
    siren.cu            init / fused forward+backward training step / Adam
                        update / full-grid reconstruction kernels
    render.cu           synthetic ground-truth volume + both renderers
    metrics.cu          MSE/PSNR/SSIM kernels
    bindings.cpp        pybind11 module definition
    loader.py           JIT-compiles the above via
                        torch.utils.cpp_extension.load(), cached per
                        (hidden_size, hidden_layers) pair
```

`hidden_size`/`hidden_layers` are compile-time for the extension (the
per-thread SIREN activation buffer needs a fixed size), so they are passed
as `-DHIDDEN_SIZE=...`/`-DHIDDEN_LAYERS=...` at JIT-compile time and a new
combination triggers a one-time recompile. Every other shape — grid size,
image dimensions, train batch, DVR steps, density scale — is a plain
runtime kernel argument.

No Makefile/`nvcc` step is needed anymore; the extension compiles the
first time it's loaded and is cached by PyTorch under
`~/.cache/torch_extensions/`.

## Run

Train (writes checkpoints + raw volumes only):

```bash
python3 train.py --iterations 2000 --learning-rate 0.001
```

Every `main.cu` CLI knob is now a flag (see `train.py --help`), plus the
previously-hardcoded grid size / SIREN architecture / train batch, e.g.
`--grid-size`, `--hidden-size`, `--hidden-layers`.

A quick smoke test:

```bash
python3 train.py --iterations 100
```

A higher-quality fit:

```bash
python3 train.py --iterations 5000 --learning-rate 0.0005
```

Render + evaluate a saved checkpoint into `output/best/` (this is where
`--orbit-frames`, `--benchmark-frames`, image size, DVR steps, and density
scale apply):

```bash
python3 reconstruct.py --checkpoint output/best.pth --out output \
  --orbit-frames 120 --benchmark-frames 1000
```

`train.py` is a thin alias for `pipeline.py` — use either name
interchangeably; neither renders.

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

`train.py` writes `init.pth`/`best.pth`/`last.pth`/`<iteration>.pth` (+ a
matching `.pdf` slice-comparison per checkpoint), plus
`ground_truth_volume.tif` and `reconstructed_volume.tif`, directly
under `--out`.

`reconstruct.py` writes everything below into its own subfolder per
checkpoint, `<out>/<checkpoint-stem>/` (e.g. `--checkpoint output/best.pth`
-> `output/best/`), so re-rendering different checkpoints never collides:

```text
output/best/best.pth                  copy of the checkpoint
output/best/slices.pdf                axial/coronal/sagittal reconstruction-
                                       vs-ground-truth-vs-diff grid (3D, no
                                       camera — same as train.py's per-
                                       checkpoint .pdf)
output/best/rendered.pdf              GT | prediction | difference *rendered*
                                       comparison, annotated with PSNR/SSIM/
                                       LPIPS and both renderers' FPS
output/best/gt_projection.pgm
output/best/pred_projection.pgm
output/best/diff.pgm
output/best/comparison.ppm
output/best/reconstructed_volume.tif
```

Orbit frames (if `--orbit-frames > 0`):

```text
output/best/orbit_gt/
output/best/orbit_pred/
output/best/orbit_diff/
output/best/orbit_comparison/
```

## Visualization and videos

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
sudo apt install ffmpeg
```

Create a labeled static PNG and four orbit videos from a checkpoint's folder:

```bash
python3 visualize.py --output-dir output/best --make-video --show
```

Generated files include:

```text
output/best/comparison.png
output/best/gt_orbit.mp4
output/best/pred_orbit.mp4
output/best/diff_orbit.mp4
output/best/comparison_orbit.mp4
```

## LPIPS

`reconstruct.py` already computes LPIPS inline (see `rendered.pdf`'s
annotation and its printed `LPIPS = ...` line) using a pretrained AlexNet
feature network (`--lpips-net` selects `alex`/`vgg`/`squeeze`). To compute
it standalone against any two images instead:

```bash
python3 lpips_metric.py \
  output/best/gt_projection.pgm \
  output/best/pred_projection.pgm
```

LPIPS maps image values from `[0,1]` to `[-1,1]`.

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
