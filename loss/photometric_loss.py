'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-01-17 10:44:25
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class PhotometricLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        input_dict=None
    ):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred_rgb': 'pred_rgb',
                'gt_rgb': 'gt_rgb',
            }
        else:
            self.input_dict = input_dict

        self.loss_func = self.loss_rgb

    def loss_rgb(self, pred_rgb, gt_rgb):
        loss = nn.functional.l1_loss(pred_rgb, gt_rgb)
        return loss
