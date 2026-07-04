nohup /venv/r3-ml/bin/python3 scripts/_3dgs/_3dgs.py \
  --volume data/fafb/blocks/image_z0_y0_x0.tif \
  --use_kernel --flat_out --no_swc_init --no_wandb \
  --epochs 2000 --n_init 1000 --max_gaussians 50000 \
  --batch 2048 --chunk_n 50000 --ckpt_interval 0 \
  --out models/z000_y000_x000 \
  > /tmp/single_block.log 2>&1 &