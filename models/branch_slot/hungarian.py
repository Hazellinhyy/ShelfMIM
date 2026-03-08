# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import itertools
import torch

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _flatten(m: torch.Tensor) -> torch.Tensor:
    if m.dim() == 4:
        B, K, H, W = m.shape
        return m.reshape(B, K, H * W)
    if m.dim() == 3:
        return m
    raise ValueError(f"m must be (B,K,N) or (B,K,H,W), got {m.shape}")


@torch.no_grad()
def dice_overlap_matrix(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    A = torch.nan_to_num(_flatten(masks_a), nan=0.0, posinf=0.0, neginf=0.0)
    Bm = torch.nan_to_num(_flatten(masks_b), nan=0.0, posinf=0.0, neginf=0.0)
    B, K, N = A.shape

    if valid_mask is not None:
        if valid_mask.dim() == 4:
            vm = valid_mask.reshape(B, 1, N).to(A.dtype)
        elif valid_mask.dim() == 3:
            vm = valid_mask.reshape(B, 1, N).to(A.dtype)
        elif valid_mask.dim() == 2:
            vm = valid_mask.reshape(B, 1, N).to(A.dtype)
        else:
            raise ValueError(f"valid_mask dim not supported: {valid_mask.shape}")
        A = A * vm
        Bm = Bm * vm

    inter = torch.einsum("bkn,bjn->bkj", A, Bm)
    sum_a = A.sum(dim=-1, keepdim=True)
    sum_b = Bm.sum(dim=-1, keepdim=True)
    dice = (2.0 * inter) / (sum_a + sum_b.transpose(1, 2) + eps)
    return torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def _bruteforce(cost: torch.Tensor) -> torch.Tensor:
    B, K, _ = cost.shape
    if K > 8:
        raise RuntimeError("Install scipy for K>8 Hungarian.")
    perms = list(itertools.permutations(range(K)))
    perms_t = torch.tensor(perms, dtype=torch.long, device=cost.device)
    pi = torch.zeros((B, K), dtype=torch.long, device=cost.device)
    idx = torch.arange(K, device=cost.device)
    for b in range(B):
        best = None
        best_p = None
        for p in perms_t:
            v = cost[b][idx, p].sum()
            if best is None or v < best:
                best = v
                best_p = p
        pi[b] = best_p
    return pi


@torch.no_grad()
def hungarian_match_by_dice(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    dice = dice_overlap_matrix(masks_a, masks_b, valid_mask=valid_mask, eps=eps)
    cost = (1.0 - dice).clamp(0.0, 1.0)

    B, K, _ = cost.shape
    pi = torch.zeros((B, K), dtype=torch.long, device=cost.device)

    if _HAS_SCIPY:
        for b in range(B):
            c = cost[b].cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(c)
            order = row_ind.argsort()
            pi[b] = torch.from_numpy(col_ind[order]).to(device=cost.device)
        return pi

    return _bruteforce(cost)
