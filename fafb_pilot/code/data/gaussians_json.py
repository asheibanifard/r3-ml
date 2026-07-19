# Step by Step load the best.pth of the blocks and attach the selected blocks gaussians
# 1. Load the best.pth of the blocks
from glob import glob
import torch
import os
from tqdm import tqdm
model_paths='/root/project/fafb_pilot/models/blocks_v18/b_*/best.pth'
models = [torch.load(path) for path in glob(model_paths)]
# TypeError: Object of type Tensor is not JSON serializable
for i in tqdm(range(len(models))):
    for key in models[i].keys():
        if isinstance(models[i][key], torch.Tensor):
            models[i][key] = models[i][key].detach().cpu().numpy().tolist()
b_names = sorted(os.listdir('/root/project/fafb_pilot/models/blocks_v18/'))
# detach the model from tensor to numpy array
json_gaussians = {b_names[i]: model for i, model in enumerate(models)}
# save the json file of json_gaussians
import json
with open('/root/project/fafb_pilot/code/data/gaussians.json', 'w') as f:
    json.dump(json_gaussians, f)