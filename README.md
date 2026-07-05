nohup /venv/r3-ml/bin/python3 scripts/_3dgs/_3dgs.py \
  --volume data/fafb/blocks/image_z0_y0_x0.tif \
  --use_kernel --flat_out --no_swc_init --no_wandb \
  --epochs 2000 --n_init 1000 --max_gaussians 50000 \
  --batch 2048 --chunk_n 50000 --ckpt_interval 0 \
  --out models/z000_y000_x000 \
  > /tmp/single_block.log 2>&1 &


nohup /venv/r3-ml/bin/python3 scripts/find_neuron_voxels.py \
  --blocks_dir data/fafb/blocks \
  --target_ids_file results/top5_neuron_ids.json \
  --block_name_prefix image \
  --out results/top5_binary_voxels.json \
  --progress_every 5000 \
  > logs/find_top5_neuron_voxels.log 2>&1 &
echo "launched pid $!"