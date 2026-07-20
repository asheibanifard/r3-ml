# Step by Step load the best.pth of the blocks and attach the selected blocks gaussians
# 1. Load the best.pth of the blocks
from glob import glob
import torch
import os
from tqdm import tqdm
model_paths='/root/project/fafb_pilot/models/blocks_v18/b_*/best.pth'
paths = glob(model_paths)
# Each block's name comes from ITS OWN path, not a separately sorted
# os.listdir() zipped by position -- glob() and sorted(os.listdir()) are not
# guaranteed to be in the same order, so the old zip-by-index approach
# silently paired each checkpoint's tensors with the WRONG block name (e.g.
# "b_211" held b_213's Gaussians). Deriving the name from the path itself
# makes the pairing correct by construction.
json_gaussians = {}
for path in tqdm(paths):
    bname = os.path.basename(os.path.dirname(path))
    model = torch.load(path, map_location='cpu', weights_only=False)
    for key in model.keys():
        if isinstance(model[key], torch.Tensor):
            model[key] = model[key].detach().cpu().numpy().tolist()
    json_gaussians[bname] = model
# save the json file of json_gaussians
import json
with open('/root/project/fafb_pilot/code/data/gaussians.json', 'w') as f:
    json.dump(json_gaussians, f)