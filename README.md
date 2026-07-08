## Steps so far

- Fit 3D Gaussian mixtures / SIREN to individual FAFB EM blocks (CUDA kernels
  in `scripts/_3dgs/`, `scripts/siren/`); batch runner for all 262,144 blocks
  in `scripts/train_scripts/train_all_blocks.py`.
- Explored `segment_*.tif` instance-segmentation blocks in
  `notebooks/model_design.ipynb`: stitched neighbouring blocks into full
  2×2(×2) volumes, inspected neuron/segment ids, and built depth-coded MIPs
  so 3D branching survives a flattened 2D projection.
- Crawled all 262,144 segment blocks to find the top 5 neuron ids globally
  by voxel count (`results/top5_neuron_ids.json`), then wrote
  `scripts/data_scripts/find_neuron_voxels.py` to dump local voxel coordinates
  (`[block_id, z, y, x]`) for a given id or set of ids — see the "Neuron
  Segmentation Exploration" section in `CLAUDE.md` for details and file
  sizes.

nohup /venv/r3-ml/bin/python3 scripts/_3dgs/_3dgs.py \
  --volume data/fafb/blocks/image_z0_y0_x0.tif \
  --use_kernel --flat_out --no_swc_init --no_wandb \
  --epochs 2000 --n_init 1000 --max_gaussians 50000 \
  --batch 2048 --chunk_n 50000 --ckpt_interval 0 \
  --out models/z000_y000_x000 \
  > /tmp/single_block.log 2>&1 &


nohup /venv/r3-ml/bin/python3 scripts/data_scripts/find_neuron_voxels.py \
  --blocks_dir data/fafb/blocks \
  --target_ids_file results/top5_neuron_ids.json \
  --block_name_prefix image \
  --out results/top5_binary_voxels.json \
  --progress_every 5000 \
  > logs/find_top5_neuron_voxels.log 2>&1 &
echo "launched pid $!"

nohup /venv/r3-ml/bin/python3 scripts/data_scripts/upload_to_hf.py \
  --repo_id Arminshfard/fafb-em-blocks \
  --folder_path data/smoke_data \
  --path_in_repo blocks \
  --repo_type dataset \
  > logs/upload_to_hf.log 2>&1 &
echo "launched pid $!"

nohup /venv/r3-ml/bin/python3 scripts/data_scripts/step5_mask_top5_neuron_blocks.py \
  --blocks_dir data/fafb/blocks \
  --target_ids_file results/top5_neuron_ids.json \
  --out_dir results/top5_masked_blocks \
  --progress_every 5000 \
  > logs/mask_top5_neuron_blocks.log 2>&1 &
echo "launched pid $!"