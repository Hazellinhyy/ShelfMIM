# -*- coding: utf-8 -*-
"""
Position embedding utilities for ViT:
- Learnable absolute pos embed interpolation (for arbitrary HxW)
- (Optional) 2D sine-cosine pos embed generation
- patchify / unpatchify helpers (useful for MIM-style reconstruction)
"""

from __future__ import annotations
from typing import Tuple, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F


def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    imgs: (B, C, H, W), H and W must be divisible by patch_size
    returns: (B, N, patch_size*patch_size*C), where N = (H/ps)*(W/ps)
    """
    if imgs.dim() != 4:
        raise ValueError(f"imgs must be 4D (B,C,H,W), got {imgs.shape}")

    B, C, H, W = imgs.shape
    ps = patch_size
    if H % ps != 0 or W % ps != 0:
        raise ValueError(f"H,W must be divisible by patch_size={ps}, got H={H}, W={W}")

    h = H // ps
    w = W // ps
    # (B, C, h, ps, w, ps) -> (B, h, w, ps, ps, C) -> (B, h*w, ps*ps*C)
    x = imgs.reshape(B, C, h, ps, w, ps)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    patches = x.reshape(B, h * w, ps * ps * C)
    return patches


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    img_hw: Tuple[int, int],
    channels: int = 3,
) -> torch.Tensor:
    """
    patches: (B, N, ps*ps*C)
    img_hw: (H, W) - must be divisible by patch_size
    returns: (B, C, H, W)
    """
    if patches.dim() != 3:
        raise ValueError(f"patches must be 3D (B,N,dim), got {patches.shape}")
    B, N, dim = patches.shape
    H, W = img_hw
    ps = patch_size
    if H % ps != 0 or W % ps != 0:
        raise ValueError(f"H,W must be divisible by patch_size={ps}, got H={H}, W={W}")

    h = H // ps
    w = W // ps
    if N != h * w:
        raise ValueError(f"N mismatch: N={N}, expected h*w={h*w} for img_hw={img_hw} and ps={ps}")
    if dim != ps * ps * channels:
        raise ValueError(f"dim mismatch: dim={dim}, expected ps*ps*C={ps*ps*channels}")

    x = patches.reshape(B, h, w, ps, ps, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    imgs = x.reshape(B, channels, H, W)
    return imgs


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_hw: Tuple[int, int],
    cls_token: bool = True,
) -> torch.Tensor:
    """
    Create 2D sine-cosine positional embeddings.

    Returns:
      pos_embed: (1, N(+1), D)
    """
    grid_h, grid_w = grid_hw
    grid_y = np.arange(grid_h, dtype=np.float32)
    grid_x = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y)  # (2, H, W) but order is (x, y)
    grid = np.stack(grid, axis=0)       # (2, H, W)
    grid = grid.reshape(2, 1, grid_h, grid_w)

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D)
    if cls_token:
        cls = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls, pos_embed], axis=0)

    pos_embed = torch.from_numpy(pos_embed).unsqueeze(0)  # (1, N, D)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sincos pos embed.")
    # half for x, half for y
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))  # y
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))  # x
    return np.concatenate([emb_w, emb_h], axis=1)


def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@torch.no_grad()
def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    old_grid_hw: Tuple[int, int],
    new_grid_hw: Tuple[int, int],
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    """
    Interpolate learned absolute position embeddings from old grid to new grid.
    pos_embed: (1, num_extra_tokens + old_H*old_W, D)
    returns : (1, num_extra_tokens + new_H*new_W, D)
    """
    if pos_embed.dim() != 3 or pos_embed.size(0) != 1:
        raise ValueError(f"pos_embed must be (1, N, D), got {pos_embed.shape}")

    old_h, old_w = old_grid_hw
    new_h, new_w = new_grid_hw

    if old_h == new_h and old_w == new_w:
        return pos_embed

    extra = pos_embed[:, :num_extra_tokens, :]
    pos = pos_embed[:, num_extra_tokens:, :]  # (1, old_h*old_w, D)
    D = pos.size(-1)

    pos = pos.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)  # (1, D, old_h, old_w)
    pos = F.interpolate(pos, size=(new_h, new_w), mode="bicubic", align_corners=False)
    pos = pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, D)

    return torch.cat([extra, pos], dim=1)
