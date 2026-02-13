'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-03-19 15:11:41
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
import torch

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torchvision.models import resnet50  
import numpy as np  

# 图像切块与遮掩函数  
def patchify(image, patch_size=16):  
    """  
    将图像分割成小块（patch）。  
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
    将图像块（patch）还原为完整图像。  
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

# 解码器  
class SimpleDecoder(nn.Module):  
    def __init__(self, in_channels, patch_dim, num_patches):  
        super(SimpleDecoder, self).__init__()  
        self.decoder = nn.Sequential(  
            nn.Linear(in_channels, 512),  
            nn.ReLU(),  
            nn.Linear(512, patch_dim)  
        )  
        self.num_patches = num_patches  

    def forward(self, x):  
        return self.decoder(x)  

# MAE 模型  
class MAE(nn.Module):  
    def __init__(self, encoder, decoder, patch_size=16, image_size=(224, 224)):  
        super(MAE, self).__init__()  
        self.encoder = encoder  
        self.decoder = decoder  
        self.patch_size = patch_size  
        self.image_size = image_size  

    def forward(self, x):  
        # 图像切块  
        patches = patchify(x, self.patch_size)  
        # 随机遮掩  
        masked_patches, mask = mask_patches(patches)  
        print("=========", masked_patches.shape, mask.shape)
        # 编码器提取特征  
        features = self.encoder(masked_patches)  
        print("=========", features.shape)
        # 解码器重建图像块  
        reconstructed_patches = self.decoder(features)  
        # 还原图像  
        reconstructed_image = unpatchify(reconstructed_patches, self.patch_size, self.image_size)  
        return reconstructed_image, mask  

# 主函数  
if __name__ == "__main__":  
    # 加载 ResNet 编码器  
    resnet = resnet50(pretrained=False)  
    encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉全连接层和池化层  

    # 解码器  
    patch_size = 16  
    image_size = (224, 224)  
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)  
    patch_dim = 3 * patch_size * patch_size  
    decoder = SimpleDecoder(in_channels=2048, patch_dim=patch_dim, num_patches=num_patches)  

    # MAE 模型  
    mae_model = MAE(encoder, decoder, patch_size=patch_size, image_size=image_size)  

    # 输入图像  
    batch_size = 4  
    image = torch.rand(batch_size, 3, 224, 224)  # 随机生成图像  

    # 前向传播  
    reconstructed_image, mask = mae_model(image)  

    print(f"Reconstructed Image Shape: {reconstructed_image.shape}")  