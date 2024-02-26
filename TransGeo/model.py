# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os

from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import math
from torch.nn.functional import interpolate

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, crop=False, save=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.save = save
        self.crop = crop
        self.crop_rate = 0.64 # keep rate: 0.53 352, 0.64 320, 0.79 288

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:,0], x[:,1]

    def forward(self, x, atten=None, indexes=None, vlad=False):

        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        x=(x+x_dist)/2

        return x

@register_model
def deit_small_distilled_patch16_224(pretrained=True, mode='satellite', img_size=(640,640), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained and mode=='satellite':
        checkpoint=torch.load("./model/weights/checkpoint.pth.tar",map_location="cpu")
        
        # Original state_dict
        state_dict = checkpoint['state_dict']

        # Remove keys that start with 'module.query_net' and rename keys that start with 'module.reference_net.'
        new_chechpoint = {k.replace('module.reference_net.', ''): v for k, v in state_dict.items() if
                          not k.startswith('module.query_net')}

        checkpoint['state_dict']=new_chechpoint


        # for key in checkpoint["model"]:
        #     print(key)
        weight = checkpoint["state_dict"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["state_dict"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"],strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"],strict=False)

        if pretrained and mode=='query':

            checkpoint = torch.load(
                "./model/weights/checkpoint.pth.tar",map_location="cpu")

            # Original state_dict
            state_dict = checkpoint['state_dict']

            # Remove keys that start with 'module.query_net' and rename keys that start with 'module.reference_net.'
            new_chechpoint = {k.replace('module.query_net.', ''): v for k, v in state_dict.items() if
                              not k.startswith('module.reference_net')}

            checkpoint['state_dict'] = new_chechpoint

            weight = checkpoint["state_dict"]['pos_embed']
            ori_size = int((802 - 1) ** 0.5)  # Should be 28

            class_token = weight[:, :1, :]  # Extract the class token
            pos_embed_old = weight[:, 1:, :]  # Remove the class token

            # Reshape to [1, ori_size, ori_size, 384]
            pos_embed_old = pos_embed_old.reshape(1, ori_size, ori_size, 384)

            # Resize to the new grid size [120, 68]
            new_grid_size = (120, 68)
            pos_embed_new = interpolate(pos_embed_old.permute(0, 3, 1, 2), size=new_grid_size, mode='bilinear',
                                        align_corners=False)

            # Flatten and concatenate the class token back
            pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed_new = torch.cat([class_token, pos_embed_new], dim=1)  # Should now be [1, 8161, 384]

            # Replace the original pos_embed
            checkpoint["state_dict"]['pos_embed'] = pos_embed_new

            model.load_state_dict(checkpoint)


    return model
