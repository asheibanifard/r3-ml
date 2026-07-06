# Roadmap: From Raw EM Volume to Masked Neuron Blocks

The story of how a single 68.7 GB electron-microscopy volume becomes a
focused set of image blocks containing only the five largest neurons/segments
in the dataset — five steps, each with its own standalone `.py` script.

---

## Step 1 — Read the raw volume

**File:** [`scripts/data_scripts/step1_read_h5_volume.py`](scripts/data_scripts/step1_read_h5_volume.py)

```bash
/venv/r3-ml/bin/python3 scripts/data_scripts/step1_read_h5_volume.py \
  --h5_path data/fafb/fafb_v14_ffn1_z2000-6096.h5 \
  --slice_index 15 \
  --out results/h5_overlay_slice.png
```

The story begins at `data/fafb/fafb_v14_ffn1_z2000-6096.h5`, a single HDF5
file holding two co-registered 4096×4096×4096 `uint8` datasets:

- `image` — the raw EM grayscale volume
- `annotations` — the FFN1 instance-segmentation labels (same shape, `uint64`
  ids, `0` = background)

This step just opens the file with `h5py` and pulls a single 2D slice from
each dataset to sanity-check that image and segmentation line up, saving a
grayscale image with a randomly-palette-colored segmentation overlay.

At 4096³ voxels, the full volume is far too large to load or process at
once — every step after this one works on small, extractable pieces of it.

---

## Step 2 — Break the volume into blocks

**File:** [`scripts/data_scripts/step2_extract_blocks.py`](scripts/data_scripts/step2_extract_blocks.py) (`save_blocks_to_tif`)

```bash
/venv/r3-ml/bin/python3 scripts/data_scripts/step2_extract_blocks.py \
  --h5_path data/fafb/fafb_v14_ffn1_z2000-6096.h5 \
  --output_dir data/fafb/blocks \
  --block_size 64
```

The 4096³ volume is chopped into a **64×64×64 grid of 64³-voxel blocks**
(262,144 blocks total), each written out as a pair of TIFs:

```
data/fafb/blocks/image_z{Z}_y{Y}_x{X}.tif      # raw intensity, uint8
data/fafb/blocks/segment_z{Z}_y{Y}_x{X}.tif    # segmentation ids, uint64
```

`{Z}`, `{Y}`, `{X}` are **block indices** (`0`–`63`), not voxel offsets —
`save_blocks_to_tif` computes the voxel slice (`z = zi * 64`, etc.) internally
and names the file by block index directly. The function is resumable: it
skips any `(zi, yi, xi)` pair whose files already exist, so re-running it
after an interruption picks up where it left off.

This is what turns one unmanageable 68.7 GB file into 262,144 small,
independently-readable files — every later step operates block-by-block
instead of touching the H5 file again.

---

## Step 3 — Check the segment ids

**File:** [`scripts/data_scripts/step3_inspect_segment_ids.py`](scripts/data_scripts/step3_inspect_segment_ids.py)

```bash
/venv/r3-ml/bin/python3 scripts/data_scripts/step3_inspect_segment_ids.py \
  --blocks_dir data/fafb/blocks \
  --base_z 0 --base_y 0 --base_x 0 \
  --out_dir results/segment_id_inspection
```

Before trusting any one neuron id, this step stitches a handful of
neighbouring blocks into a small 128³ volume (`stitch_blocks`) and inspects
what's actually inside a `segment_*.tif`:

- ~197 distinct ids can show up in just a 128³ crop
- ids are large, sparse `uint64` values (background is `0`)
- naively plotting raw ids with `imshow(cmap='gray')` is misleading — matplotlib
  autoscales linearly between 0 and the max id, so only the single largest id
  reads as white and everything else collapses toward black

This step also introduces `depth_coded_mip` — coloring a MIP projection by
the depth of the first non-background voxel — so that flattening a 3D
segmentation crop to 2D doesn't erase all sense of neurite branching/depth.
It saves two PNGs to `results/segment_id_inspection/`: a depth-coded MIP for
the single dominant local id, and one for the top few ids together (hue = id,
brightness = depth).

This is the "look before you leap" step: it establishes that segment ids
need careful handling before picking any of them for downstream use.

---

## Step 4 — Select the top five high-voxel-count ids

**File:** [`scripts/data_scripts/step4_find_top5_neuron_ids.py`](scripts/data_scripts/step4_find_top5_neuron_ids.py)
**Output:** [`results/top5_neuron_ids.json`](results/top5_neuron_ids.json)

```bash
/venv/r3-ml/bin/python3 scripts/data_scripts/step4_find_top5_neuron_ids.py \
  --blocks_dir data/fafb/blocks \
  --top_n 5 \
  --out results/top5_neuron_ids.json \
  --progress_every 5000
```

Now the check from Step 3 is scaled up to the *entire* dataset: every one of
the 262,144 `segment_*.tif` blocks is crawled, and a running voxel-count
tally (`collections.Counter`) is kept per id (background `0` excluded). The
five ids with the highest total voxel count across the whole volume are kept:

```json
[
  {"id": 4341159055, "voxel_count": 410359045},
  {"id": 3293809532, "voxel_count": 290248353},
  {"id": 6293835881, "voxel_count": 194730958},
  {"id": 4869512842, "voxel_count": 135333179},
  {"id": 1215218846, "voxel_count": 104870354}
]
```

These turned out to be large, sprawling structures spanning many blocks —
not the same id that happened to dominate any single small crop from Step 3.

*(A related, optional script,* [`scripts/data_scripts/find_neuron_voxels.py`](scripts/data_scripts/find_neuron_voxels.py)*,
can dump the full sparse `[block_id, z, y, x]` coordinate list for one id or
this whole top-5 set — used earlier to trace a single neuron's footprint —
but it's not required for the next step, which recomputes the mask directly.)*

---

## Step 5 — Make the masked blocks

**File:** [`scripts/data_scripts/step5_mask_top5_neuron_blocks.py`](scripts/data_scripts/step5_mask_top5_neuron_blocks.py)
**Output:** [`results/top5_masked_blocks/`](results/top5_masked_blocks/)

```bash
/venv/r3-ml/bin/python3 scripts/data_scripts/step5_mask_top5_neuron_blocks.py \
  --blocks_dir data/fafb/blocks \
  --target_ids_file results/top5_neuron_ids.json \
  --out_dir results/top5_masked_blocks \
  --progress_every 5000
```

The final step walks all 262,144 blocks once more. For each block:

1. Load `segment_z{Z}_y{Y}_x{X}.tif` and compute a binary mask —
   `1` where the voxel's id is one of the top-5 ids from Step 4, `0` otherwise.
2. **Skip the block entirely if the mask is all zero** (none of the top-5
   neurons appear in it) — no point writing an all-black file.
3. Otherwise load the matching `image_z{Z}_y{Y}_x{X}.tif` and apply the mask:
   `masked = where(mask, image, 0)` — voxels belonging to a top-5 neuron keep
   their original grayscale intensity, everything else becomes `0`.
4. Write the result to `results/top5_masked_blocks/image_z{Z}_y{Y}_x{X}.tif`.

Of the 262,144 blocks, **24,509** contain at least one top-5 voxel and get a
masked file written (~6.3 GB total) — the end result: a focused view of the
raw EM data containing *only* the five largest neurons, with everything else
zeroed out.

---

## Pipeline at a glance

| Step | What happens | File | Output |
|---|---|---|---|
| 1 | Read the raw H5 volume | `scripts/data_scripts/step1_read_h5_volume.py` | `results/h5_overlay_slice.png` |
| 2 | Break into 64³ blocks | `scripts/data_scripts/step2_extract_blocks.py` | `data/fafb/blocks/*.tif` |
| 3 | Check segment ids (small-scale) | `scripts/data_scripts/step3_inspect_segment_ids.py` | `results/segment_id_inspection/*.png` |
| 4 | Select top-5 ids (whole dataset) | `scripts/data_scripts/step4_find_top5_neuron_ids.py` | `results/top5_neuron_ids.json` |
| 5 | Make masked blocks | `scripts/data_scripts/step5_mask_top5_neuron_blocks.py` | `results/top5_masked_blocks/*.tif` |

The original exploratory work for steps 1, 3, and part of 4 lives in
[`notebooks/read_data.ipynb`](notebooks/read_data.ipynb) and
[`notebooks/model_design.ipynb`](notebooks/model_design.ipynb) — the scripts
above are the same logic extracted into standalone, runnable files.
