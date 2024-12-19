# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Masking utils
# --------------------------------------------------------

import torch
import torch.nn as nn    
    
class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        argsort = torch.argsort(noise, dim=1) 
        return argsort < self.num_mask


class AttentionMask(nn.Module): # Mask Align & Ref: https://github.com/OpenDriveLab/maskalign/blob/master/models_pretrain.py
    """
    attention masking
    """
    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)

    def forward(self, image, importance):
        N, C, H, W = image.size()
        len_keep = int(H * W * (1 - mask_ratio))
        flatten_importance = importance.view(N, -1)
        _, sort_indices = torch.sort(flatten_importance, dim=1)
        attention_mask = torch.ones(N, H * W, device=image.device)
        attention_mask.scatter_(1, sort_indices[:, :len_keep], 0)
        attention_mask = attention_mask.view(N, 1, H, W)
        return attention_mask