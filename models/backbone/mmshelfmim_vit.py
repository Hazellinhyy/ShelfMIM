# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet.registry import MODELS

from models.backbone.vit import vit_base_patch16


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(32, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


@MODELS.register_module()
class MMShelfMIMViTWithSimpleFPN(BaseModule):

    def __init__(
        self,
        out_channels: int = 256,
        vit_drop_path_rate: float = 0.1,
        vit_use_cls_token: bool = True,
        vit_pos_base_grid_hw: Tuple[int, int] = (14, 14),
        patch_size: int = 16,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.vit = vit_base_patch16(
            drop_path_rate=vit_drop_path_rate,
            use_cls_token=vit_use_cls_token,
            pos_base_grid_hw=vit_pos_base_grid_hw,
        )

        embed_dim = self.vit.embed_dim

        self.proj = _ConvGNAct(embed_dim, out_channels, 1, 1, 0)
        self.smooth3 = _ConvGNAct(out_channels, out_channels, 3, 1, 1)
        self.smooth2 = _ConvGNAct(out_channels, out_channels, 3, 1, 1)
        self.down5 = _ConvGNAct(out_channels, out_channels, 3, 2, 1)

        self.out_channels = out_channels

    def forward(self, x):

        out = self.vit(x)
        patch_tokens = out["patch_tokens"]
        Hp, Wp = out["grid_hw"]

        B, N, D = patch_tokens.shape
        feat16 = patch_tokens.transpose(1, 2).reshape(B, D, Hp, Wp)

        res4 = self.proj(feat16)
        res3 = self.smooth3(F.interpolate(res4, scale_factor=2.0))
        res2 = self.smooth2(F.interpolate(res3, scale_factor=2.0))
        res5 = self.down5(res4)

        return [res2, res3, res4, res5]
