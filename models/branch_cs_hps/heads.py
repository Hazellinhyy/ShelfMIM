# -*- coding: utf-8 -*-
"""
Heads for content/style factorization (Branch III).
From pooled token g_v (B,D) produce:
  c_v: content code (B,Cd)
  s_v: style code   (B,Sd)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else in_dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContentStyleHeads(nn.Module):
    """
    Two heads:
      c_v = f_c(g_v)
      s_v = f_s(g_v)
    """
    def __init__(
        self,
        in_dim: int,
        content_dim: int = 256,
        style_dim: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.content_head = MLPHead(in_dim, content_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.style_head = MLPHead(in_dim, style_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pooled: (B,D)
        returns: c (B,Cd), s (B,Sd)
        """
        c = self.content_head(pooled)
        s = self.style_head(pooled)
        if self.l2_normalize:
            c = F.normalize(c, dim=-1)
            s = F.normalize(s, dim=-1)
        return c, s
