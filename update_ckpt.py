'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-02-11 18:50:40
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from collections import OrderedDict
import numpy as np
import pickle
import torch


def dump_state_dict_keys(state_dict, file_path):
    # dump the state_dict to a text file
    with open(file_path, "w") as f:
        for k, v in state_dict.items():
            f.write(f"{k}: {v.size()}\n")


def remove_keys(state_dict, keys_to_remove):
    # # 删除指定的 key  
    for key in keys_to_remove:  
        if key in state_dict:  
            del state_dict[key]  
            print(f"Removed key: {key}")  
        else:  
            print(f"Key not found: {key}")


def remove_keys_with_prefix(state_dict, prefix_to_remove):
    new_state_dict = dict()
    keys_removed = []
    for key, value in state_dict.items():
        if any([key.startswith(_prefix) for _prefix in prefix_to_remove]):
            keys_removed.append(key)
        else:
            new_state_dict[key] = value
    return new_state_dict


def add_prefix_to_keys(state_dict, prefix_to_add):
    new_state_dict = dict()
    for key, value in state_dict.items():
        new_key = prefix_to_add + key
        new_state_dict[new_key] = value
    return new_state_dict


def revise_ckpts(state_dict, revise_keys):
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {k.replace(p, r): v
             for k, v in state_dict.items()})
    
    return state_dict


def main_remove_query_weights(dir_name):
    ckpt_path = f"out/pretrain/{dir_name}/epoch_20.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    keys_to_remove = ["lifter.anchor", "lifter.instance_feature"]
    remove_keys(state_dict, keys_to_remove)

    # 保存修改后的 checkpoint  
    new_checkpoint_path = f"out/pretrain/{dir_name}/{dir_name}_wo_anchor.pth"  # 替换为保存的新文件路径  
    torch.save(ckpt, new_checkpoint_path)  
    print(f"Modified checkpoint saved to: {new_checkpoint_path}")


def main_construct_ssp_finetune_weight():
    # 1. Load the pretrained SSP model
    dir_name = "nuscenes_gs25600_solid_pretrain_depth_only"
    ssp_ckpt_path = "out/pretrain/nuscenes_gs25600_solid_pretrain_depth_only/epoch_20.pth"
    origin_ckpt = torch.load(ssp_ckpt_path, map_location="cpu")
    
    flag = False
    if "state_dict" in origin_ckpt:
        ssp_state_dict = origin_ckpt["state_dict"]
        flag = True
    else:
        ssp_state_dict = origin_ckpt
    
    # 2. Construct the finetune weight for ssp model
    ssp_model_dict = add_prefix_to_keys(ssp_state_dict, "ssp_model.")

    # 3. Construct the finetune weight for the gaussianformer image backbone
    prefix_to_remove = ['lifter', 'encoder', 'head']
    ssp_state_dict_backbone = remove_keys_with_prefix(ssp_state_dict, prefix_to_remove)

    final_state_dict = dict()
    final_state_dict.update(ssp_model_dict)
    final_state_dict.update(ssp_state_dict_backbone)
    
    # 保存修改后的 checkpoint  
    final_checkpoint_path = f"out/pretrain/{dir_name}/{dir_name}_ssp_model.pth"  # 替换为保存的新文件路径
    if flag:
        origin_ckpt['state_dict'] = final_state_dict
    else:
        origin_ckpt = final_state_dict
    torch.save(origin_ckpt, final_checkpoint_path)
    print(f"Modified checkpoint saved to: {final_checkpoint_path}")

def main_dump_state_dict_keys(ckpt_path, save_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    try:
        state_dict = ckpt["state_dict"]
    except:
        state_dict = ckpt
    
    dump_state_dict_keys(state_dict, save_path)


if __name__ == "__main__":
    ckpt_path = "ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"
    main_dump_state_dict_keys(ckpt_path, save_path="./ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.txt")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    flag = False
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        flag = True
    else:
        state_dict = ckpt
    
    new_state_dict = revise_ckpts(state_dict, [("backbone", "img_backbone")])

    save_path = "./ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960_new.txt"
    dump_state_dict_keys(new_state_dict, save_path)

    if flag:
        ckpt['state_dict'] = new_state_dict
    else:
        ckpt = new_state_dict
    
    # 保存修改后的 checkpoint
    new_checkpoint_path = f"ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960_new.pth"  # 替换为保存的新文件路径
    torch.save(ckpt, new_checkpoint_path)
    
    exit(0)

    dir_name = "nuscenes_gs25600_solid_pretrain_rgb_only"
    main_remove_query_weights(dir_name)
    exit(0)
    # main_construct_ssp_finetune_weight()
    # exit(0)
    dir_name = "nuscenes_gs25600_solid_pretrain_rgb_depth_quarter_fix"
    ckpt_path = f"out/pretrain/{dir_name}/epoch_20.pth"
    # ckpt_path = "out/pretrain/nuscenes_gs25600_solid_pretrain_depth_only/nuscenes_gs25600_solid_pretrain_depth_only_ssp_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    try:
        state_dict = ckpt["state_dict"]
    except:
        state_dict = ckpt

    # dump_state_dict_keys(state_dict, "nuscenes_gs25600_solid_pretrain_depth_only.txt")
    
    prefix_to_remove = ['lifter', 'encoder', 'head']
    new_state_dict = remove_keys_with_prefix(state_dict, prefix_to_remove)
    try:
        ckpt['state_dict'] = new_state_dict
    except:
        ckpt = new_state_dict
    
    # 保存修改后的 checkpoint  
    new_checkpoint_path = f"out/pretrain/{dir_name}/{dir_name}_backbone_only.pth"  # 替换为保存的新文件路径  
    torch.save(ckpt, new_checkpoint_path)  
    print(f"Modified checkpoint saved to: {new_checkpoint_path}")
