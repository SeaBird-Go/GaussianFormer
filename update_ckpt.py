'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-02-11 18:50:40
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import torch

ckpt_path = "out/nuscenes_gs25600_solid_pretrain_rgb_depth/epoch_20.pth"
ckpt_path = "nuscenes_gs25600_solid_pretrain_rgb_depth_e20.pth"
# ckpt_path = "out/nuscenes_gs25600_solid_quarter/epoch_20.pth"
ckpt_path = "./ckpts/nuscenes_gs25600_solid.pth"
ckpt_path = "out/nuscenes_gs25600_solid_pretrain_rgb_depth_w_extra_head/epoch_20.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
try:
    state_dict = ckpt["state_dict"]
except:
    state_dict = ckpt


def remove_keys(state_dict):
    # # 删除指定的 key  
    keys_to_remove = ["lifter.anchor", "lifter.instance_feature"]  
    for key in keys_to_remove:  
        if key in state_dict:  
            del state_dict[key]  
            print(f"Removed key: {key}")  
        else:  
            print(f"Key not found: {key}")
    return state_dict

# print(state_dict['lifter.instance_feature'])
# print(state_dict['lifter.anchor'])
# print(state_dict['encoder.layers.21.layers.11.scale'])

# dump the state_dict to a text file
# with open("nuscenes_gs25600_solid_finetune.txt", "w") as f:
#     for k, v in state_dict.items():
#         f.write(f"{k}: {v.size()}\n")

remove_keys(state_dict)

# 保存修改后的 checkpoint  
new_checkpoint_path = "out/nuscenes_gs25600_solid_pretrain_rgb_depth_w_extra_head/nuscenes_gs25600_solid_pretrain_rgb_depth_w_extra_head_wo_anchor.pth"  # 替换为保存的新文件路径  
torch.save(ckpt, new_checkpoint_path)  
print(f"Modified checkpoint saved to: {new_checkpoint_path}")