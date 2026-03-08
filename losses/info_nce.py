# -*- coding: utf-8 -*-
"""
Slot InfoNCE loss (Branch I):
- positives: (Z_a[b,k], Z_b[b, pi[b,k]])
- negatives: other slots in opposite view
- symmetric: A->B and B->A
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def invert_permutation(pi: torch.Tensor) -> torch.Tensor:
    """
    pi: (B,K) where pi[b,k]=j
    returns inv: (B,K) where inv[b,j]=k
    """
    if pi.dim() != 2:
        raise ValueError(f"pi must be (B,K), got {pi.shape}")
    B, K = pi.shape
    inv = torch.empty_like(pi)
    idx = torch.arange(K, device=pi.device).unsqueeze(0).expand(B, K)
    inv.scatter_(dim=1, index=pi, src=idx)
    return inv


def slot_info_nce(
    z_a: torch.Tensor,        # (B,K,D)
    z_b: torch.Tensor,        # (B,K,D)
    pi: torch.Tensor,         # (B,K)
    temperature: float = 0.2,
    symmetric: bool = True,
    valid_slots: Optional[torch.Tensor] = None,  # (B,K) 1/0 to ignore some slots
) -> torch.Tensor:
    """
    Returns scalar loss.
    """
    if z_a.dim() != 3 or z_b.dim() != 3:
        raise ValueError("z_a/z_b must be (B,K,D)")
    if z_a.shape != z_b.shape:
        raise ValueError(f"z_a shape {z_a.shape} != z_b shape {z_b.shape}")
    B, K, D = z_a.shape
    if pi.shape != (B, K):
        raise ValueError(f"pi shape {pi.shape} != {(B,K)}")

    za = F.normalize(z_a, dim=-1)
    zb = F.normalize(z_b, dim=-1)

    # sim[b,i,j] = cos(za[b,i], zb[b,j])
    sim = torch.einsum("bkd,bjd->bkj", za, zb)  # (B,K,K)

    # A -> B
    logits_ab = sim / float(temperature)        # (B,K,K)
    labels_ab = pi                              # (B,K)
    loss_ab = F.cross_entropy(
        logits_ab.reshape(B * K, K),
        labels_ab.reshape(B * K),
        reduction="none",
    )  # (B*K,)

    if valid_slots is not None:
        if valid_slots.shape != (B, K):
            raise ValueError(f"valid_slots shape {valid_slots.shape} != {(B,K)}")
        w = valid_slots.reshape(B * K).float()
        loss_ab = (loss_ab * w).sum() / w.sum().clamp_min(1e-6)
    else:
        loss_ab = loss_ab.mean()

    if not symmetric:
        return loss_ab

    # B -> A
    inv_pi = invert_permutation(pi)             # (B,K)
    logits_ba = sim.transpose(1, 2) / float(temperature)  # (B,K,K)
    labels_ba = inv_pi
    loss_ba = F.cross_entropy(
        logits_ba.reshape(B * K, K),
        labels_ba.reshape(B * K),
        reduction="none",
    )
    if valid_slots is not None:
        w = valid_slots.reshape(B * K).float()
        loss_ba = (loss_ba * w).sum() / w.sum().clamp_min(1e-6)
    else:
        loss_ba = loss_ba.mean()

    return 0.5 * (loss_ab + loss_ba)
