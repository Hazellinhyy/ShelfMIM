# -*- coding: utf-8 -*-
"""
Gumbel-TopK + Straight-Through sampling for k-hot masks.

Given logits a (B,N), sample a k-hot mask m (B,N) with:
- stochasticity via Gumbel noise
- hard selection via topk
- straight-through gradient via a soft relaxation

Returned:
  m_hard: (B,N) float {0,1}
  m_soft: (B,N) float in [0,1], used for gradients
  topk_idx: (B,k) indices
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F


def _sample_gumbel(shape, device, dtype, eps: float = 1e-6) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=dtype).clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(u))


def gumbel_topk_st(
    logits: torch.Tensor,          # (B,N)
    k: int,
    tau: float = 1.0,
    hard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B,N), got {logits.shape}")
    B, N = logits.shape
    if k <= 0 or k > N:
        raise ValueError(f"k must be in [1,N], got k={k}, N={N}")

    g = _sample_gumbel(logits.shape, logits.device, logits.dtype)
    y = (logits + g) / float(tau)  # (B,N)

    # soft selection distribution
    p = F.softmax(y, dim=-1)  # (B,N)

    # choose top-k (hard) - NON-inplace scatter for stability
    topk_idx = torch.topk(y, k=k, dim=-1).indices  # (B,k)
    m_hard = torch.zeros_like(logits).scatter(-1, topk_idx, 1.0)

    # soft mask for gradient: scale probabilities to expected k selected
    m_soft = (p * float(k)).clamp(0.0, 1.0)

    if hard:
        # straight-through: forward uses hard, backward uses soft
        m = m_hard + (m_soft - m_soft.detach())
    else:
        m = m_soft

    return m, m_soft, topk_idx


def k_from_ratio(N: int, ratio: float) -> int:
    k = int(round(float(N) * float(ratio)))
    return max(1, min(N, k))
