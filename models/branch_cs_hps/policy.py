# -*- coding: utf-8 -*-
"""
Hard patch sampling policy network (Branch III).
Input: stop-grad patch tokens F (B,N,D)
Output: importance logits a (B,N)

We mask invalid padded patches by setting logits to a very negative value.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchImportancePolicy(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,                 # (B,N,D)
        valid_mask: Optional[torch.Tensor] = None,  # (B,N) 1 for valid
        detach_input: bool = True,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B,N,D), got {tokens.shape}")
        x = tokens.detach() if detach_input else tokens
        logits = self.net(x).squeeze(-1)  # (B,N)

        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError(f"valid_mask shape {valid_mask.shape} != logits shape {logits.shape}")
            # invalidate padded patches so they won't be sampled
            logits = logits + (valid_mask.to(dtype=logits.dtype) - 1.0) * 1e9

        return logits
