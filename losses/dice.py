# -*- coding: utf-8 -*-
"""
Soft Dice utilities (unit-test friendly).

Supports:
- masks: (B,K,N) or (B,K,H,W)
- valid_mask: (B,N) or (B,1,H,W) or (B,H,W)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch


def _to_bk_hw(m: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Convert masks to (B,K,H,W).
    """
    if m.dim() == 4:
        return m
    if m.dim() == 3:
        if grid_hw is None:
            raise ValueError("grid_hw is required when masks are (B,K,N)")
        B, K, N = m.shape
        H, W = grid_hw
        if N != H * W:
            raise ValueError(f"N={N} != H*W={H*W}")
        return m.reshape(B, K, H, W)
    raise ValueError(f"masks must be (B,K,N) or (B,K,H,W), got {m.shape}")


def _to_valid_b1_hw(
    valid_mask: Optional[torch.Tensor],
    grid_hw: Tuple[int, int],
    device,
    dtype,
) -> Optional[torch.Tensor]:
    """
    Return valid mask as (B,1,H,W) float in {0,1}.
    Accepts:
      - None
      - (B,N)
      - (B,1,H,W)
      - (B,H,W)
    """
    if valid_mask is None:
        return None
    H, W = grid_hw
    if valid_mask.dim() == 4:
        if valid_mask.shape[1] != 1:
            raise ValueError("valid_mask (B,1,H,W) expected")
        return valid_mask.to(device=device, dtype=dtype)
    if valid_mask.dim() == 3:
        B, h, w = valid_mask.shape
        if (h, w) != (H, W):
            raise ValueError(f"valid_mask hw {(h,w)} != {(H,W)}")
        return valid_mask.unsqueeze(1).to(device=device, dtype=dtype)
    if valid_mask.dim() == 2:
        B, N = valid_mask.shape
        if N != H * W:
            raise ValueError(f"valid_mask N {N} != H*W {H*W}")
        return valid_mask.reshape(B, 1, H, W).to(device=device, dtype=dtype)
    raise ValueError(f"valid_mask dims not supported: {valid_mask.shape}")


def dice_score(
    masks1: torch.Tensor,
    masks2: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]] = None,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns per-(B,K) dice: (B,K)
    """
    m1 = _to_bk_hw(masks1, grid_hw=grid_hw)
    m2 = _to_bk_hw(masks2, grid_hw=grid_hw)

    if m1.shape != m2.shape:
        raise ValueError(f"shape mismatch: {m1.shape} vs {m2.shape}")

    B, K, H, W = m1.shape
    vm = _to_valid_b1_hw(valid_mask, (H, W), device=m1.device, dtype=m1.dtype)
    if vm is not None:
        m1 = m1 * vm
        m2 = m2 * vm

    # flatten spatial
    m1f = m1.reshape(B, K, -1)
    m2f = m2.reshape(B, K, -1)

    inter = (m1f * m2f).sum(dim=-1)
    denom = m1f.sum(dim=-1) + m2f.sum(dim=-1)
    dice = (2.0 * inter) / (denom + eps)
    return dice


def dice_loss(
    masks1: torch.Tensor,
    masks2: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]] = None,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Dice loss = 1 - dice.
    reduction: "mean" | "none"
    """
    d = dice_score(masks1, masks2, grid_hw=grid_hw, valid_mask=valid_mask, eps=eps)  # (B,K)
    loss = 1.0 - d
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
