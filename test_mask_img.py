'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-03-19 15:00:21
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import mmcv
import torch


def mask_image(image, patch_size=16, mask_ratio=0.75):  
    """  
    随机遮掩图像的一部分。  
    Args:  
        image: 输入图像，形状为 (B, C, H, W)。  
        patch_size: 每个 patch 的大小。  
        mask_ratio: 遮掩比例。  
    Returns:  
        masked_image: 遮掩后的图像。  
        mask: 遮掩的二值掩码，形状为 (B, H // patch_size, W // patch_size)。  
    """  
    B, C, H, W = image.shape  
    assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"  
    num_patches_h = H // patch_size  
    num_patches_w = W // patch_size  
    num_patches = num_patches_h * num_patches_w  
    num_masked = int(mask_ratio * num_patches)  

    # 生成遮掩掩码  
    mask = torch.ones(B, num_patches_h, num_patches_w)  
    for i in range(B):  
        idx = torch.randperm(num_patches)[:num_masked]  
        mask.view(B, -1)[i, idx] = 0  

    # 将掩码应用到图像  
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)  # 放大到图像大小  
    masked_image = image * mask.unsqueeze(1)  # 遮掩图像  
    return masked_image, mask 


# 图像切块与遮掩函数  
def patchify(image, patch_size=16):  
    """  
    将图像分割成小块(patch)。  
    Args:  
        image: 输入图像，形状为 (B, C, H, W)。  
        patch_size: 每个 patch 的大小。  
    Returns:  
        patches: 分割后的图像块，形状为 (B, num_patches, patch_size*patch_size*C)。  
    """  
    B, C, H, W = image.shape  
    assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"  
    num_patches_h = H // patch_size  
    num_patches_w = W // patch_size  
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_size * patch_size)  
    return patches  

def unpatchify(patches, patch_size=16, image_size=(224, 224)):  
    """  
    将图像块(patch)还原为完整图像。  
    Args:  
        patches: 图像块，形状为 (B, num_patches, patch_size*patch_size*C)。  
        patch_size: 每个 patch 的大小。  
        image_size: 原始图像大小 (H, W)。  
    Returns:  
        image: 还原后的图像，形状为 (B, C, H, W)。  
    """  
    B, num_patches, patch_dim = patches.shape  
    H, W = image_size  
    C = patch_dim // (patch_size * patch_size)  
    num_patches_h = H // patch_size  
    num_patches_w = W // patch_size  
    patches = patches.reshape(B, num_patches_h, num_patches_w, C, patch_size, patch_size)  
    patches = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)  
    return patches  

def mask_patches(patches, mask_ratio=0.75):  
    """  
    随机遮掩一部分图像块。  
    Args:  
        patches: 输入图像块，形状为 (B, num_patches, patch_dim)。  
        mask_ratio: 遮掩比例。  
    Returns:  
        masked_patches: 遮掩后的图像块。  
        mask: 遮掩的二值掩码，形状为 (B, num_patches)。  
    """  
    B, num_patches, _ = patches.shape  
    num_masked = int(mask_ratio * num_patches)  
    mask = torch.ones(B, num_patches)  
    for i in range(B):  
        idx = torch.randperm(num_patches)[:num_masked]  
        mask[i, idx] = 0  
    mask = mask.unsqueeze(-1)  # (B, num_patches, 1)  
    masked_patches = patches * mask  
    return masked_patches, mask


if __name__ == "__main__":
    img_fp = "data/nuscenes/samples/CAM_FRONT/n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514864956220368.jpg"
    image = mmcv.imread(img_fp, "unchanged")
    image = image.astype(np.float32)
    mmcv.imwrite(image, "image.jpg")
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    print(image.shape)

    masked_image, mask = mask_image(image, patch_size=20, mask_ratio=0.25)
    masked_image = masked_image.squeeze(0).permute(1, 2, 0).numpy()
    mmcv.imwrite(masked_image, "masked.jpg")