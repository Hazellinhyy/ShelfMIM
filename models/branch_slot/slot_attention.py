# -*- coding: utf-8 -*-
"""
Slot Attention (stable masks)

Return:
  slots: (B, K, D)
  masks: (B, K, N)  IMPORTANT: masks are softmax over slots per patch:
                      for each patch n: sum_k masks[b,k,n] = 1

Updates use attn normalized over patches per slot (as in paper),
but we DO NOT return that normalized-by-patches attn as masks,
otherwise Dice/Hungarian/Warp supervision becomes unstable.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        mlp_hidden_size: Optional[int] = None,
        epsilon: float = 1e-6,
        slot_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.iters = int(iters)
        self.eps = float(epsilon)
        self.slot_temperature = float(max(slot_temperature, 1e-4))

        hidden = int(mlp_hidden_size) if mlp_hidden_size is not None else 4 * self.dim

        self.norm_inputs = nn.LayerNorm(self.dim, eps=1e-6)
        self.project_k = nn.Linear(self.dim, self.dim)
        self.project_v = nn.Linear(self.dim, self.dim)

        self.norm_slots = nn.LayerNorm(self.dim, eps=1e-6)
        self.project_q = nn.Linear(self.dim, self.dim)

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, self.dim))

        self.gru = nn.GRUCell(self.dim, self.dim)
        self.norm_mlp = nn.LayerNorm(self.dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim),
        )

        self.scale = self.dim ** -0.5
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.slot_mu, std=0.02)
        nn.init.normal_(self.slot_logsigma, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,                # (B,N,D)
        valid_mask: Optional[torch.Tensor] = None,  # (B,N) 1=valid
        init_slots: Optional[torch.Tensor] = None,  # (B,K,D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B,N,D), got {tokens.shape}")
        B, N, D = tokens.shape
        if D != self.dim:
            raise ValueError(f"tokens dim D={D} != slot dim {self.dim}")

        x = self.norm_inputs(tokens)
        k = self.project_k(x)  # (B,N,D)
        v = self.project_v(x)  # (B,N,D)

        if init_slots is not None:
            if init_slots.shape != (B, self.num_slots, self.dim):
                raise ValueError(f"init_slots must be (B,K,D), got {init_slots.shape}")
            slots = init_slots
        else:
            mu = self.slot_mu.expand(B, self.num_slots, -1)
            sigma = self.slot_logsigma.exp().expand(B, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)

        if valid_mask is not None:
            if valid_mask.shape != (B, N):
                raise ValueError(f"valid_mask must be (B,N), got {valid_mask.shape}")
            vm = valid_mask.to(dtype=tokens.dtype)
        else:
            vm = None

        masks = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots_prev))  # (B,K,D)

            logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale  # (B,K,N)
            if vm is not None:
                logits = logits + (vm[:, None, :] - 1.0) * 1e9

            # ✅ supervision masks: softmax over slots (sum_k=1 per patch)
            masks = F.softmax(logits / self.slot_temperature, dim=1)  # (B,K,N)

            # ✅ update weights: normalize over patches per slot
            attn = masks + self.eps
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)  # (B,K,N)

            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            slots = self.gru(
                updates.reshape(B * self.num_slots, self.dim),
                slots_prev.reshape(B * self.num_slots, self.dim),
            ).reshape(B, self.num_slots, self.dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, masks
