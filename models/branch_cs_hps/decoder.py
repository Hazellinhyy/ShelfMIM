# -*- coding: utf-8 -*-
"""
Conditional masked patch decoder for reconstruction & style swap.

Decoder predicts pixel patches:
  pred_patches: (B,N,patch_dim) where patch_dim = ps*ps*3

Conditioning:
  cond = concat(c, s)  -> produces FiLM (gamma,beta) to modulate token features

Input tokens should already be masked via:
  F_mask = m * mask_token + (1-m) * F
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.position_embed import interpolate_pos_embed


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CSFiLMDecoder(nn.Module):
    """
    Decoder that takes masked patch tokens + (c,s) condition and predicts pixel patches.
    """
    def __init__(
        self,
        token_dim: int,               # encoder token dim (e.g., 768)
        cond_dim: int,                # content_dim + style_dim
        patch_size: int = 16,
        out_channels: int = 3,
        decoder_dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        pos_base_grid_hw: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.patch_dim = self.patch_size * self.patch_size * self.out_channels
        self.pos_base_grid_hw = pos_base_grid_hw

        self.in_proj = nn.Linear(token_dim, decoder_dim)
        self.blocks = nn.ModuleList([DecoderBlock(decoder_dim, heads=heads, mlp_ratio=mlp_ratio, drop=drop)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        # FiLM from cond -> (gamma,beta)
        self.film = nn.Sequential(
            nn.LayerNorm(cond_dim, eps=1e-6),
            nn.Linear(cond_dim, decoder_dim * 2),
        )

        # learned absolute pos embed at base grid (no cls token)
        base_n = pos_base_grid_hw[0] * pos_base_grid_hw[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, base_n, decoder_dim))

        # predict pixel patch vector
        self.pred = nn.Linear(decoder_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.trunc_normal_(self.pred.weight, std=0.02)
        nn.init.zeros_(self.pred.bias)

    def forward(
        self,
        masked_tokens: torch.Tensor,     # (B,N,token_dim)
        cond: torch.Tensor,             # (B,cond_dim)
        grid_hw: Tuple[int, int],        # (Hp,Wp) for pos interpolation
    ) -> torch.Tensor:
        if masked_tokens.dim() != 3:
            raise ValueError(f"masked_tokens must be (B,N,D), got {masked_tokens.shape}")
        B, N, _ = masked_tokens.shape

        x = self.in_proj(masked_tokens)  # (B,N,dec_dim)

        # add interpolated pos embed
        pos = interpolate_pos_embed(
            self.pos_embed,
            old_grid_hw=self.pos_base_grid_hw,
            new_grid_hw=grid_hw,
            num_extra_tokens=0,
        )
        if pos.shape[1] != N:
            raise RuntimeError(f"pos N {pos.shape[1]} != tokens N {N} (grid_hw={grid_hw})")
        x = x + pos

        # FiLM modulation
        gamma_beta = self.film(cond)  # (B, 2*dec_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # (B,dec_dim)
        gamma = gamma.unsqueeze(1)  # (B,1,dec_dim)
        beta = beta.unsqueeze(1)

        # modulate after LN in each block (simple, stable)
        for blk in self.blocks:
            h = blk.norm1(x)
            h = (1.0 + gamma) * h + beta
            attn_out, _ = blk.attn(h, h, h, need_weights=False)
            x = x + attn_out
            x = x + blk.mlp(blk.norm2(x))

        x = self.norm(x)
        pred_patches = self.pred(x)  # (B,N,patch_dim)
        return pred_patches


def apply_mask_token(
    tokens: torch.Tensor,     # (B,N,D)
    mask: torch.Tensor,       # (B,N) float in [0,1] (ST mask ok)
    mask_token: torch.Tensor  # (1,1,D) or (D,)
) -> torch.Tensor:
    """
    F_mask(n) = m(n)*[MASK] + (1-m(n))*F(n)
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be (B,N), got {mask.shape}")
    B, N, D = tokens.shape

    if mask_token.dim() == 1:
        mt = mask_token.view(1, 1, D)
    elif mask_token.dim() == 3:
        mt = mask_token
    else:
        raise ValueError(f"mask_token must be (D,) or (1,1,D), got {mask_token.shape}")

    m = mask.to(dtype=tokens.dtype).unsqueeze(-1)  # (B,N,1)
    return m * mt + (1.0 - m) * tokens
