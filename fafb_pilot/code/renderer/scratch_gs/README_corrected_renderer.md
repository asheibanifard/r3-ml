# Corrected Gaussian Rasterizer Benchmark and Evaluation

This package contains:

- `gaussian_splat_scratch_corrected.cu` — the original standalone training and rendering pipeline, with a corrected and reorganised benchmark/export section.
- `render_outputs_corrected.py` — robust frame pairing, quality metrics, videos, plots and CSV summaries.

The mathematical training and rasterization implementation from the supplied CUDA file is retained. The benchmark and evaluation code are rewritten because those were the source of the misleading FPS table.

---

## 1. What is being compared?

The program produces three rendering paths.

### GT DVR

The ground-truth dense voxel grid is rendered directly using fixed-step maximum-intensity-projection DVR.

For a camera ray

\[
\mathbf{x}(t)=\mathbf{o}+t\mathbf{d},
\]

with box-entry and box-exit distances \(t_0,t_1\), the renderer takes `DVR_SAMPLES` midpoint samples:

\[
t_s=t_0+\left(s+\tfrac12\right)\frac{t_1-t_0}{S},
\qquad s=0,\ldots,S-1.
\]

The output pixel is

\[
I_{\mathrm{GT}}(p)=\max_s V(\mathbf{x}(t_s)),
\]

where \(V\) is the voxel grid sampled by trilinear interpolation.

### Baked + DVR

The fitted Gaussian mixture is first evaluated on the regular voxel grid to create a reconstructed or **baked** volume. That volume is passed through exactly the same DVR kernel:

\[
I_{\mathrm{baked}}(p)=\max_s \widehat V(\mathbf{x}(t_s)).
\]

Because GT DVR and Baked+DVR use the same renderer, their difference mainly measures representation and baking error rather than renderer differences.

### Live Gaussian rasterizer

Each anisotropic Gaussian has density

\[
g_k(\mathbf{x})=a_k\exp\left[-\frac12
(\mathbf{x}-\boldsymbol\mu_k)^T
\boldsymbol\Sigma_k^{-1}
(\mathbf{x}-\boldsymbol\mu_k)\right].
\]

The covariance is parameterised as

\[
\boldsymbol\Sigma_k=
\mathbf R_k\operatorname{diag}(s_{kx}^2,s_{ky}^2,s_{kz}^2)\mathbf R_k^T.
\]

Along a ray, the desired continuous MIP quantity is conceptually

\[
I_{\mathrm{GS}}(p)=
\max_t\sum_k g_k(\mathbf{o}+t\mathbf d).
\]

The implementation approximates depth using `RENDER_BINS`. Contributions are summed within each bin and the maximum bin value is selected:

\[
I_{\mathrm{GS}}(p)\approx
\max_b\sum_k g_k(\mathbf{o}+t_b\mathbf d).
\]

This ordering is important. Computing \(\sum_k\max_t g_k\) or \(\max_k\max_t g_k\) would be a different and generally incorrect MIP quantity when Gaussians overlap.

The tile-based implementation performs:

1. Project each Gaussian to a conservative screen bounding box.
2. Append its index to every overlapping tile.
3. Assign one CUDA block to each tile and one thread to each pixel.
4. Stream tile Gaussians through shared memory.
5. Maintain private depth-bin accumulators per pixel.
6. Return the maximum accumulated bin.

---

## 2. Why the previous FPS values were misleading

The previous Python script selected:

```text
dvr_fps
baked_fps
rasterizer_fps
```

Those values came from a host wall-clock interval around a single kernel launch and `cudaDeviceSynchronize()`. For very short kernels, the fixed CPU launch/synchronisation overhead can dominate the real GPU work. That can produce counter-intuitive values such as 1024×1024 apparently rendering faster than 64×64.

The corrected CUDA file does the following:

1. Performs explicit warm-up rounds.
2. Uses all camera yaw angles during warm-up and measurement.
3. Measures many renders inside one CUDA-event interval.
4. Uses CUDA events, which timestamp work on the GPU timeline.
5. Keeps device-to-host copies and disk writes outside the benchmark.
6. Measures the Gaussian method end-to-end, including tile-list clearing, tile-list construction and tile rendering.

For each method:

\[
\mathrm{FPS}=
\frac{N_{\mathrm{measured\ renders}}}
{T_{\mathrm{GPU}}/1000},
\]

where \(T_{\mathrm{GPU}}\) is the CUDA-event elapsed time in milliseconds.

The summary file labels this timing as:

```text
timing_kind gpu_events_repeated
```

The Python evaluator requires the GPU keys:

```text
dvr_gpu_fps
baked_gpu_fps
rasterizer_gpu_fps
```

---

## 3. Expected performance trend

The DVR kernel uses one thread per output pixel and exactly `DVR_SAMPLES` ray samples for every ray that intersects the box. Its approximate arithmetic work is

\[
O(WHS),
\]

where \(W\times H\) is screen resolution and \(S\) is the number of ray samples.

Small images may underutilise the GPU, so FPS does not have to decrease exactly in inverse proportion to pixel count. Nevertheless, once the GPU is sufficiently occupied, larger resolutions should require more time and should not systematically become dramatically faster.

The Gaussian renderer has approximate work

\[
O\left(\sum_{p} K_p B_p\right),
\]

where \(K_p\) is the number of candidate Gaussians in the pixel's tile and \(B_p\) is the number of relevant depth bins evaluated for those Gaussians. Its scaling therefore depends on both screen resolution and projected Gaussian coverage.

---

## 4. Quality metrics

Both Baked+DVR and live Gaussian rasterization are compared against GT DVR.

### Mean squared error

\[
\mathrm{MSE}=\frac1N\sum_{i=1}^N(x_i-y_i)^2.
\]

### PSNR

The frame values are physically defined in `[0,1]`, so `MAX=1`:

\[
\mathrm{PSNR}=10\log_{10}\left(\frac{1}{\mathrm{MSE}}\right).
\]

Higher is better. Identical images have infinite PSNR.

The script does not independently normalise each frame, because doing so would change the underlying intensity meaning and artificially improve the metric.

### SSIM

SSIM compares local luminance, contrast and structure. The script calls `skimage.metrics.structural_similarity` with `data_range=1.0`, matching the known image range.

Higher is better; 1 is perfect.

### LPIPS

LPIPS measures perceptual distance using an AlexNet-based feature network. The grayscale frame is repeated across three channels and mapped from `[0,1]` to `[-1,1]`:

\[
x_{\mathrm{LPIPS}}=2x-1.
\]

Lower is better. Repeating the channel satisfies the network input interface without introducing colour information.

---

## 5. Difference videos

Raw absolute error is

\[
D_i=|x_i-y_i|.
\]

For visibility only, each difference video uses one global scale equal to the 99.5th percentile of all error pixels in that sequence:

\[
D_i^{\mathrm{display}}=
\operatorname{clip}\left(\frac{D_i}{q_{99.5}},0,1\right).
\]

This prevents one extreme pixel from making the entire video appear black. This scaling is never used for PSNR, SSIM or LPIPS.

---

## 6. Compile and run

### CUDA compilation

From the directory containing the files:

```bash
nvcc -O3 -std=c++17 \
  gaussian_splat_scratch_corrected.cu \
  -o gaussian_splat_scratch_corrected
```

For a specific GPU architecture, add the matching architecture flag. Example for Ada GPUs such as an RTX 4060:

```bash
nvcc -O3 -std=c++17 -arch=sm_89 \
  gaussian_splat_scratch_corrected.cu \
  -o gaussian_splat_scratch_corrected
```

### Renderer arguments

```text
./gaussian_splat_scratch_corrected OUTPUT_DIR WIDTH HEIGHT CHECKPOINT
```

Example:

```bash
./gaussian_splat_scratch_corrected \
  frames_512 \
  512 512 \
  checkpoint.bin
```

If the checkpoint exists and matches the compiled `N_GAUSSIANS`, training is skipped. Otherwise the program performs its self-test and training before rendering.

For separate screen-size experiments, use separate output directories so old frames and summaries cannot be mixed:

```bash
./gaussian_splat_scratch_corrected frames_64   64   64   checkpoint.bin
./gaussian_splat_scratch_corrected frames_128  128  128  checkpoint.bin
./gaussian_splat_scratch_corrected frames_256  256  256  checkpoint.bin
./gaussian_splat_scratch_corrected frames_512  512  512  checkpoint.bin
./gaussian_splat_scratch_corrected frames_1024 1024 1024 checkpoint.bin
```

### Python dependencies

```bash
pip install numpy matplotlib scikit-image torch lpips
```

`ffmpeg` must also be installed and available on `PATH`.

### Evaluation

```bash
python render_outputs_corrected.py \
  --frames_dir frames_512 \
  --out_dir results_512 \
  --video_fps 24 \
  --lpips_device auto
```

---

## 7. Generated outputs

The evaluator writes:

```text
gt.mp4
baked.mp4
reconstruction.mp4
baked_difference_scaled.mp4
raster_difference_scaled.mp4
frame_comparisons/*.png
psnr_over_frames.png
ssim_over_frames.png
lpips_over_frames.png
metrics_summary.csv
metrics_per_frame.csv
metrics_summary.png
```

`metrics_summary.csv` is the main table for papers or further analysis. It contains GPU-only FPS and mean quality metrics.

---

## 8. Interpretation rules

1. Use `*_gpu_fps`, not single-launch host wall-clock FPS, for algorithmic throughput.
2. State that Gaussian FPS includes dynamic tile-list rebuilding.
3. Compare Baked+DVR quality to GT to quantify representation/baking error.
4. Compare live rasterizer quality to GT to quantify the combined representation and rasterization approximation error.
5. Keep `DVR_SAMPLES`, `RENDER_BINS`, volume resolution, Gaussian count, camera path and GPU fixed when comparing screen resolutions.
6. Report the GPU model, CUDA version, compilation flags and benchmark repetition count.
7. Run each resolution more than once if the final numbers are publication-critical, then report mean and standard deviation across independent program runs.
