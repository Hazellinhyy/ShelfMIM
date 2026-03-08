# # -*- coding: utf-8 -*-
# """
# ViT-B/16 backbone for ShelfMIM:
# - Outputs patch tokens F: (B, N, D)
# - Outputs global vector u: average pool of patch tokens (B, D)
# - Learnable absolute pos embed with runtime interpolation for arbitrary image sizes
# """
#
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, Any
#
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .position_embed import interpolate_pos_embed
#
#
# # -----------------------------
# # helpers
# # -----------------------------
#
# def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
#     # Truncated normal init (approx). Good enough for ViT-style init.
#     # Avoids extra deps.
#     with torch.no_grad():
#         size = tensor.shape
#         tmp = tensor.new_empty(size + (4,)).normal_()
#         valid = (tmp < 2) & (tmp > -2)
#         ind = valid.max(-1, keepdim=True)[1]
#         tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#         tensor.data.mul_(std).add_(mean)
#     return tensor
#
#
# def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
#     """
#     Stochastic depth.
#     """
#     if drop_prob == 0.0 or not training:
#         return x
#     keep_prob = 1.0 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()
#     return x.div(keep_prob) * random_tensor
#
#
# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = float(drop_prob)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return drop_path(x, self.drop_prob, self.training)
#
#
# # -----------------------------
# # ViT building blocks
# # -----------------------------
#
# class PatchEmbed(nn.Module):
#     """
#     2D image to patch embeddings with Conv2d.
#     """
#     def __init__(self, in_chans: int = 3, embed_dim: int = 768, patch_size: int = 16):
#         super().__init__()
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
#         """
#         x: (B, C, H, W)
#         returns:
#           tokens: (B, N, D)
#           grid_hw: (H_p, W_p)
#         """
#         B, C, H, W = x.shape
#         ps = self.patch_size
#         if H % ps != 0 or W % ps != 0:
#             raise ValueError(f"Input H,W must be divisible by patch_size={ps}. Got {(H, W)}")
#
#         x = self.proj(x)  # (B, D, H_p, W_p)
#         H_p, W_p = x.shape[2], x.shape[3]
#         x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
#         return x, (H_p, W_p)
#
#
# class MLP(nn.Module):
#     def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, in_dim)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 12,
#         qkv_bias: bool = True,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#     ):
#         super().__init__()
#         if dim % num_heads != 0:
#             raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
#
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x)  # (B, N, 3C)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         out = attn @ v  # (B, heads, N, head_dim)
#         out = out.transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#         return out
#
#
# class Block(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         drop: float = 0.0,
#         attn_drop: float = 0.0,
#         drop_path_prob: float = 0.0,
#     ):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim, eps=1e-6)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
#         self.norm2 = nn.LayerNorm(dim, eps=1e-6)
#         hidden = int(dim * mlp_ratio)
#         self.mlp = MLP(in_dim=dim, hidden_dim=hidden, drop=drop)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# # -----------------------------
# # ViT backbone
# # -----------------------------
#
# class VisionTransformer(nn.Module):
#     """
#     ViT backbone that returns patch tokens and global pooled vector.
#     Global vector u = mean(patch_tokens), aligned with paper definition.
#     """
#     def __init__(
#         self,
#         patch_size: int = 16,
#         in_chans: int = 3,
#         embed_dim: int = 768,
#         depth: int = 12,
#         num_heads: int = 12,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         drop_rate: float = 0.0,
#         attn_drop_rate: float = 0.0,
#         drop_path_rate: float = 0.1,
#         use_cls_token: bool = True,
#         pos_base_grid_hw: Tuple[int, int] = (14, 14),  # base for 224/16
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.use_cls_token = use_cls_token
#         self.pos_base_grid_hw = pos_base_grid_hw
#
#         self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
#
#         num_patches_base = pos_base_grid_hw[0] * pos_base_grid_hw[1]
#         self.num_extra_tokens = 1 if use_cls_token else 0
#
#         if use_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         else:
#             self.cls_token = None
#
#         # Learnable absolute pos embed at base grid; interpolate at runtime
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_extra_tokens + num_patches_base, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth decay
#         dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path_prob=dpr[i],
#             )
#             for i in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         if self.cls_token is not None:
#             _trunc_normal_(self.cls_token, std=0.02)
#         _trunc_normal_(self.pos_embed, std=0.02)
#
#         # init linear/conv
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 _trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Conv2d):
#                 # fan_out init
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out")
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#
#     def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
#         """
#         Returns:
#           F: patch tokens (B, N, D)
#           u: global pooled vector (B, D) = mean over patch tokens
#           grid_hw: (H_p, W_p)
#         """
#         tokens, grid_hw = self.patch_embed(x)  # (B,N,D)
#         B, N, D = tokens.shape
#
#         if self.use_cls_token:
#             cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
#             tokens_all = torch.cat([cls, tokens], dim=1)  # (B,1+N,D)
#         else:
#             tokens_all = tokens  # (B,N,D)
#
#         # interpolate pos embed to current grid
#         pos = interpolate_pos_embed(
#             self.pos_embed,
#             old_grid_hw=self.pos_base_grid_hw,
#             new_grid_hw=grid_hw,
#             num_extra_tokens=self.num_extra_tokens,
#         )
#         if pos.shape[1] != tokens_all.shape[1]:
#             raise RuntimeError(f"pos shape {pos.shape} mismatch tokens {tokens_all.shape}")
#
#         x_all = tokens_all + pos
#         x_all = self.pos_drop(x_all)
#
#         for blk in self.blocks:
#             x_all = blk(x_all)
#         x_all = self.norm(x_all)
#
#         if self.use_cls_token:
#             patch_tokens = x_all[:, 1:, :]  # (B,N,D)
#         else:
#             patch_tokens = x_all
#
#         # Paper requirement: global vector by average pooling over patch tokens
#         global_u = patch_tokens.mean(dim=1)  # (B,D)
#
#         return patch_tokens, global_u, grid_hw
#
#     def forward(self, x: torch.Tensor) -> Dict[str, Any]:
#         F_tokens, u, grid_hw = self.forward_features(x)
#         return {
#             "patch_tokens": F_tokens,   # (B,N,D)
#             "global_u": u,              # (B,D)
#             "grid_hw": grid_hw,         # (H_p, W_p)
#         }
#
#
# # -----------------------------
# # Factory: ViT-B/16
# # -----------------------------
#
# def vit_base_patch16(**kwargs) -> VisionTransformer:
#     """
#     ViT-B/16 as used in the paper backbone:
#       embed_dim=768, depth=12, heads=12, patch_size=16
#     """
#     return VisionTransformer(
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         **kwargs,
#     )
# -*- coding: utf-8 -*-
"""
ViT-B/16 backbone for ShelfMIM
WITH GRADIENT CHECKPOINT (OOM FIX VERSION)
"""

from __future__ import annotations
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .position_embed import interpolate_pos_embed


# -----------------------------
# helpers
# -----------------------------

def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------
# Patch Embedding
# -----------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            raise ValueError(f"H,W must be divisible by patch_size={ps}")

        x = self.proj(x)
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x, (H_p, W_p)


# -----------------------------
# MLP
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# -----------------------------
# Attention
# -----------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# -----------------------------
# Transformer Block
# -----------------------------

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path_prob=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------
# Vision Transformer
# -----------------------------

class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        use_cls_token=True,
        pos_base_grid_hw=(14, 14),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, patch_size=patch_size)
        self.use_cls_token = use_cls_token
        self.embed_dim = embed_dim
        self.pos_base_grid_hw = pos_base_grid_hw

        num_patches_base = pos_base_grid_hw[0] * pos_base_grid_hw[1]
        self.num_extra_tokens = 1 if use_cls_token else 0

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_extra_tokens + num_patches_base, embed_dim)
        )

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, drop_path_prob=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.cls_token is not None:
            _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)

    # 🔥🔥🔥 核心修改在这里 🔥🔥🔥
    def forward_features(self, x):
        tokens, grid_hw = self.patch_embed(x)
        B, N, D = tokens.shape

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens_all = torch.cat([cls, tokens], dim=1)
        else:
            tokens_all = tokens

        pos = interpolate_pos_embed(
            self.pos_embed,
            old_grid_hw=self.pos_base_grid_hw,
            new_grid_hw=grid_hw,
            num_extra_tokens=self.num_extra_tokens,
        )

        x_all = tokens_all + pos

        for blk in self.blocks:
            if self.training:
                x_all = checkpoint.checkpoint(blk, x_all, use_reentrant=False)
            else:
                x_all = blk(x_all)

        x_all = self.norm(x_all)

        if self.use_cls_token:
            patch_tokens = x_all[:, 1:, :]
        else:
            patch_tokens = x_all

        global_u = patch_tokens.mean(dim=1)
        return patch_tokens, global_u, grid_hw

    def forward(self, x):
        F_tokens, u, grid_hw = self.forward_features(x)
        return {
            "patch_tokens": F_tokens,
            "global_u": u,
            "grid_hw": grid_hw,
        }


# -----------------------------
# Factory
# -----------------------------

def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
