# -*- coding: utf-8 -*-
"""
Collate for SSL two-view training:
- stacks img_a/img_b
- stacks homography H_b2a
- keeps meta/pad info
"""

from __future__ import annotations
from typing import Any, Dict, List
import torch


def collate_ssl_two_view(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch item keys expected:
      img_a: [3,H,W] float
      img_b: [3,H,W] float
      H_b2a: [3,3] float
      valid_mask_a: [H,W] uint8 (0/1)
      valid_mask_b: [H,W] uint8 (0/1)
      meta: dict (orig_size, resized_size, pad_right, pad_bottom, ...)
      path/index
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    img_a = torch.stack([b["img_a"] for b in batch], dim=0)  # [B,3,H,W]
    img_b = torch.stack([b["img_b"] for b in batch], dim=0)  # [B,3,H,W]

    H_b2a = torch.stack([b["H_b2a"] for b in batch], dim=0)  # [B,3,3]
    H_a2b = torch.stack([b["H_a2b"] for b in batch], dim=0)  # [B,3,3]

    # valid masks: convert to float mask (useful for loss masking)
    vm_a = torch.stack([b["valid_mask_a"] for b in batch], dim=0).unsqueeze(1).float()  # [B,1,H,W]
    vm_b = torch.stack([b["valid_mask_b"] for b in batch], dim=0).unsqueeze(1).float()  # [B,1,H,W]

    metas = [b.get("meta", {}) for b in batch]
    paths = [b.get("path", "") for b in batch]
    indices = torch.tensor([int(b.get("index", i)) for i, b in enumerate(batch)], dtype=torch.long)

    # convenience: padding info tensor (pad_right, pad_bottom)
    pad_info = torch.tensor(
        [[int(m.get("pad_right", 0)), int(m.get("pad_bottom", 0))] for m in metas],
        dtype=torch.long,
    )  # [B,2]

    return {
        "img_a": img_a,
        "img_b": img_b,
        "H_b2a": H_b2a,
        "H_a2b": H_a2b,
        "valid_mask_a": vm_a,
        "valid_mask_b": vm_b,
        "pad_info": pad_info,
        "meta": metas,
        "path": paths,
        "index": indices,
    }
