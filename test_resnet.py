'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-03-19 14:59:32
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
from torchvision.transforms import ToTensor  
import numpy as np  

# 图像遮掩函数  
def mask_image(image, mask_ratio=0.75):  
    """  
    随机遮掩图像的一部分。  
    Args:  
        image: 输入图像，形状为 (B, C, H, W)。  
        mask_ratio: 遮掩比例。  
    Returns:  
        masked_image: 遮掩后的图像。  
        mask: 遮掩的二值掩码。  
    """  
    B, C, H, W = image.shape  
    num_patches = H * W  
    num_masked = int(mask_ratio * num_patches)  

    # 随机生成遮掩位置  
    mask = torch.ones(B, H, W)  
    for i in range(B):  
        idx = torch.randperm(num_patches)[:num_masked]  
        mask.view(B, -1)[i, idx] = 0  

    mask = mask.unsqueeze(1)  # (B, 1, H, W)  
    masked_image = image * mask  # 遮掩图像  
    return masked_image, mask  

# 解码器  
class SimpleDecoder(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(SimpleDecoder, self).__init__()  
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(in_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),  
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),  
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  
        )  

    def forward(self, x):  
        return self.decoder(x)  

# MAE 模型  
class MAE(nn.Module):  
    def __init__(self, encoder, decoder):  
        super(MAE, self).__init__()  
        self.encoder = encoder  
        self.decoder = decoder  

    def forward(self, x, mask):  
        # 编码器提取特征  
        features = self.encoder(x)  
        # 解码器重建图像  
        reconstructed = self.decoder(features)  
        # 仅计算被遮掩部分的重建误差  
        loss = F.mse_loss(reconstructed * mask, x * mask)  
        return loss, reconstructed  

# 主函数  
if __name__ == "__main__":  
    # 加载 ResNet 编码器  
    resnet = resnet50(pretrained=False)  
    encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉全连接层和池化层  

    # 解码器  
    decoder = SimpleDecoder(in_channels=2048, out_channels=3)  

    # MAE 模型  
    mae_model = MAE(encoder, decoder)  

    # 输入图像  
    batch_size = 4  
    image = torch.rand(batch_size, 3, 224, 224)  # 随机生成图像  

    # 遮掩图像  
    masked_image, mask = mask_image(image, mask_ratio=0.75)  

    # 前向传播  
    loss, reconstructed = mae_model(masked_image, mask)  

    print(f"Reconstruction Loss: {loss.item()}")  