# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.position_embed import patchify, unpatchify
from .decoder import apply_mask_token


def flatten_valid_mask(valid_mask: Optional[torch.Tensor], N: int) -> Optional[torch.Tensor]:
    if valid_mask is None:
        return None
    if valid_mask.dim() == 2:
        if valid_mask.shape[1] != N:
            raise ValueError("valid_mask N mismatch")
        return valid_mask.float()
    if valid_mask.dim() == 4:
        B, C, H, W = valid_mask.shape
        if C != 1:
            raise ValueError("valid_mask expected (B,1,H,W)")
        if H * W != N:
            raise ValueError("valid_mask H*W mismatch")
        return valid_mask.reshape(B, N).float()
    if valid_mask.dim() == 3:
        B, H, W = valid_mask.shape
        if H * W != N:
            raise ValueError("valid_mask H*W mismatch")
        return valid_mask.reshape(B, N).float()
    raise ValueError(f"valid_mask shape not supported: {valid_mask.shape}")


def content_loss(c_a: torch.Tensor, c_b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(c_a, c_b)


def decorr_loss(c: torch.Tensor, s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if c.dim() != 2 or s.dim() != 2:
        raise ValueError("c and s must be (B,dim)")
    B = c.size(0)
    c_norm = (c - c.mean(dim=0, keepdim=True)) / (c.std(dim=0, keepdim=True, unbiased=False) + eps)
    s_norm = (s - s.mean(dim=0, keepdim=True)) / (s.std(dim=0, keepdim=True, unbiased=False) + eps)
    C = (c_norm.t() @ s_norm) / float(B)
    return (C ** 2).mean()


def sobel_grad_mag(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 4:
        raise ValueError("img must be (B,C,H,W)")
    B, C, H, W = img.shape
    if C > 1:
        x = img.mean(dim=1, keepdim=True)
    else:
        x = img

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


def _sample_random_khot_mask(valid_mask: Optional[torch.Tensor], B: int, N: int, k: int, device, dtype) -> torch.Tensor:
    m = torch.zeros((B, N), device=device, dtype=dtype)
    if valid_mask is None:
        valid_mask = torch.ones((B, N), device=device, dtype=dtype)
    valid_mask = (valid_mask > 0.5)
    for b in range(B):
        idx = torch.where(valid_mask[b])[0]
        if idx.numel() == 0:
            idx = torch.arange(N, device=device)
        kk = int(max(1, min(int(k), int(idx.numel()))))
        perm = torch.randperm(idx.numel(), device=device)[:kk]
        sel = idx[perm]
        m[b, sel] = 1.0
    return m


def _patch_weight_from_variance(target_patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Reduce over-emphasis on trivial smooth patches in visible-region MIM.
    v = target_patches.detach().var(dim=-1)
    v = v / (v.mean(dim=1, keepdim=True).clamp_min(eps))
    return v.clamp(min=0.25, max=2.5)


def _valid_patch_to_img_mask(valid_mask_flat: Optional[torch.Tensor], grid_hw: Tuple[int, int], img_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    if valid_mask_flat is None:
        return None
    Hp, Wp = grid_hw
    H, W = img_hw
    B, N = valid_mask_flat.shape
    if N != Hp * Wp:
        raise ValueError(f"valid mask patch count mismatch: {N} vs {Hp*Wp}")
    vm = valid_mask_flat.reshape(B, 1, Hp, Wp)
    return F.interpolate(vm, size=(H, W), mode="nearest")


def mim_loss(
    pred_patches: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    patch_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_patches.shape != target_patches.shape:
        raise ValueError("pred/target patch shape mismatch")
    m = mask.float()
    if valid_mask is not None:
        m = m * valid_mask.float()
    if patch_weight is not None:
        m = m * patch_weight.float()

    # Charbonnier-style robust penalty instead of pure L2.
    per_patch = torch.sqrt(((pred_patches - target_patches) ** 2).mean(dim=-1) + eps)
    denom = m.sum().clamp_min(eps)
    return (per_patch * m).sum() / denom


def geom_loss(recon_img: torch.Tensor, target_img: torch.Tensor, valid_mask_img: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    g1 = sobel_grad_mag(recon_img)
    g2 = sobel_grad_mag(target_img)
    if valid_mask_img is None:
        return F.l1_loss(g1, g2)
    vm = valid_mask_img.to(device=g1.device, dtype=g1.dtype)
    diff = (g1 - g2).abs() * vm
    return diff.sum() / vm.sum().clamp_min(eps)


def _recon_from_masked(pred_patches: torch.Tensor, tgt_patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask.unsqueeze(-1) * pred_patches + (1.0 - mask.unsqueeze(-1)) * tgt_patches


def _masked_input_patches(tgt_patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # For visualization only: masked patches are set to zero.
    return (1.0 - mask.unsqueeze(-1)) * tgt_patches


class CSHPSLoss(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        lambda_cont: float = 1.0,
        lambda_decorr: float = 0.1,
        lambda_geom: float = 0.5,
        lambda_mim: float = 1.0,
        lambda_mim_visible: float = 0.15,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.lambda_cont = float(lambda_cont)
        self.lambda_decorr = float(lambda_decorr)
        self.lambda_geom = float(lambda_geom)
        self.lambda_mim = float(lambda_mim)
        self.lambda_mim_visible = float(lambda_mim_visible)

    def forward(
        self,
        *,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        pooled_a: torch.Tensor,
        pooled_b: torch.Tensor,
        grid_hw: Tuple[int, int],
        valid_mask_a: Optional[torch.Tensor],
        valid_mask_b: Optional[torch.Tensor],
        cs_heads: nn.Module,
        policy: nn.Module,
        sampler,
        decoder: nn.Module,
        mask_token: torch.Tensor,
        mask_ratio: float,
        gumbel_tau: float = 1.0,
        use_learned_policy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, N, D = tokens_a.shape

        vm_a = flatten_valid_mask(valid_mask_a, N)
        vm_b = flatten_valid_mask(valid_mask_b, N)

        c_a, s_a = cs_heads(pooled_a)
        c_b, s_b = cs_heads(pooled_b)

        L_cont = content_loss(c_a, c_b)
        L_decorr = 0.5 * (decorr_loss(c_a, s_a) + decorr_loss(c_b, s_b))

        from .gumbel_topk import k_from_ratio
        k = k_from_ratio(N, mask_ratio)

        if use_learned_policy:
            logits_a = policy(tokens_a, valid_mask=vm_a, detach_input=True)
            logits_b = policy(tokens_b, valid_mask=vm_b, detach_input=True)
            m_a, _, _ = sampler(logits_a, k=k, tau=gumbel_tau, hard=True)
            m_b, _, _ = sampler(logits_b, k=k, tau=gumbel_tau, hard=True)
            policy_mode = torch.tensor(1.0, device=tokens_a.device, dtype=tokens_a.dtype)
        else:
            m_a = _sample_random_khot_mask(vm_a, B, N, k, tokens_a.device, tokens_a.dtype)
            m_b = _sample_random_khot_mask(vm_b, B, N, k, tokens_b.device, tokens_b.dtype)
            policy_mode = torch.tensor(0.0, device=tokens_a.device, dtype=tokens_a.dtype)

        tgt_a = patchify(images_a, patch_size=self.patch_size)
        tgt_b = patchify(images_b, patch_size=self.patch_size)

        tok_a_masked = apply_mask_token(tokens_a, m_a, mask_token)
        tok_b_masked = apply_mask_token(tokens_b, m_b, mask_token)

        cond_aa = torch.cat([c_a, s_a], dim=-1)
        cond_bb = torch.cat([c_b, s_b], dim=-1)

        pred_a = decoder(tok_a_masked, cond_aa, grid_hw=grid_hw)
        pred_b = decoder(tok_b_masked, cond_bb, grid_hw=grid_hw)

        L_mim = 0.5 * (
            mim_loss(pred_a, tgt_a, m_a, valid_mask=vm_a) +
            mim_loss(pred_b, tgt_b, m_b, valid_mask=vm_b)
        )

        vis_w_a = _patch_weight_from_variance(tgt_a)
        vis_w_b = _patch_weight_from_variance(tgt_b)

        # Visible-region reconstruction is weighted by patch variance to avoid over-rewarding easy flat areas.
        L_mim_vis = 0.5 * (
            mim_loss(pred_a, tgt_a, 1.0 - m_a, valid_mask=vm_a, patch_weight=vis_w_a) +
            mim_loss(pred_b, tgt_b, 1.0 - m_b, valid_mask=vm_b, patch_weight=vis_w_b)
        )

        cond_ab = torch.cat([c_a, s_b], dim=-1)
        cond_ba = torch.cat([c_b, s_a], dim=-1)

        pred_ab = decoder(tok_a_masked, cond_ab, grid_hw=grid_hw)
        pred_ba = decoder(tok_b_masked, cond_ba, grid_hw=grid_hw)

        recon_ab_patches = _recon_from_masked(pred_ab, tgt_a, m_a)
        recon_ba_patches = _recon_from_masked(pred_ba, tgt_b, m_b)

        H, W = images_a.shape[-2], images_a.shape[-1]
        recon_ab_img = unpatchify(recon_ab_patches, patch_size=self.patch_size, img_hw=(H, W), channels=3)
        recon_ba_img = unpatchify(recon_ba_patches, patch_size=self.patch_size, img_hw=(H, W), channels=3)

        valid_img_a = _valid_patch_to_img_mask(vm_a, grid_hw=grid_hw, img_hw=(H, W))
        valid_img_b = _valid_patch_to_img_mask(vm_b, grid_hw=grid_hw, img_hw=(H, W))
        L_geom = 0.5 * (
            geom_loss(recon_ab_img, images_a, valid_mask_img=valid_img_a) +
            geom_loss(recon_ba_img, images_b, valid_mask_img=valid_img_b)
        )

        contrib_cont = L_cont
        contrib_decorr = L_decorr
        contrib_geom = L_geom
        contrib_mim = L_mim
        contrib_mim_visible = L_mim_vis
        # raw total (external weighting in losses/total_loss.py)
        loss_total = contrib_cont + contrib_decorr + contrib_geom + contrib_mim + contrib_mim_visible

        # Visualization tensors (self reconstruction + masked inputs)
        pred_a_vis = pred_a.clamp(-2.0, 2.0)
        pred_b_vis = pred_b.clamp(-2.0, 2.0)

        recon_a_patches = _recon_from_masked(pred_a_vis, tgt_a, m_a)
        recon_b_patches = _recon_from_masked(pred_b_vis, tgt_b, m_b)
        masked_a_patches = _masked_input_patches(tgt_a, m_a)
        masked_b_patches = _masked_input_patches(tgt_b, m_b)

        # Two recon views for monitoring:
        # 1) stitched: predicted on masked patches + gt on visible patches
        # 2) full: decoder prediction on all patches (harder, reveals real reconstruction quality)
        recon_a_img = unpatchify(recon_a_patches, patch_size=self.patch_size, img_hw=(H, W), channels=3)
        recon_b_img = unpatchify(recon_b_patches, patch_size=self.patch_size, img_hw=(H, W), channels=3)
        recon_full_a_img = unpatchify(pred_a_vis, patch_size=self.patch_size, img_hw=(H, W), channels=3)
        recon_full_b_img = unpatchify(pred_b_vis, patch_size=self.patch_size, img_hw=(H, W), channels=3)
        masked_a_img = unpatchify(masked_a_patches, patch_size=self.patch_size, img_hw=(H, W), channels=3)
        masked_b_img = unpatchify(masked_b_patches, patch_size=self.patch_size, img_hw=(H, W), channels=3)

        # Numerical safety: avoid propagating NaN/Inf from rare unstable batches.
        L_cont = torch.nan_to_num(L_cont, nan=0.0, posinf=1e3, neginf=1e3)
        L_decorr = torch.nan_to_num(L_decorr, nan=0.0, posinf=1e3, neginf=1e3)
        L_geom = torch.nan_to_num(L_geom, nan=0.0, posinf=1e3, neginf=1e3)
        L_mim = torch.nan_to_num(L_mim, nan=0.0, posinf=1e3, neginf=1e3)
        L_mim_vis = torch.nan_to_num(L_mim_vis, nan=0.0, posinf=1e3, neginf=1e3)
        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e3, neginf=1e3)

        return {
            "loss_cont": L_cont,
            "loss_decorr": L_decorr,
            "loss_geom": L_geom,
            "loss_mim": L_mim,
            "loss_mim_visible": L_mim_vis,
            "policy_mode": policy_mode,
            "lambda_cont_eff": torch.tensor(float(self.lambda_cont), device=loss_total.device, dtype=loss_total.dtype),
            "lambda_decorr_eff": torch.tensor(float(self.lambda_decorr), device=loss_total.device, dtype=loss_total.dtype),
            "lambda_geom_eff": torch.tensor(float(self.lambda_geom), device=loss_total.device, dtype=loss_total.dtype),
            "lambda_mim_eff": torch.tensor(float(self.lambda_mim), device=loss_total.device, dtype=loss_total.dtype),
            "lambda_mim_visible_eff": torch.tensor(float(self.lambda_mim_visible), device=loss_total.device, dtype=loss_total.dtype),
            "contrib_cs_cont": contrib_cont,
            "contrib_cs_decorr": contrib_decorr,
            "contrib_cs_geom": contrib_geom,
            "contrib_cs_mim": contrib_mim,
            "contrib_cs_mim_visible": contrib_mim_visible,
            "contrib_cs_internal_total": loss_total,
            "loss_cs_hps_total": loss_total,
            "mask_a": m_a.detach(),
            "mask_b": m_b.detach(),
            "masked_img_a": masked_a_img.detach(),
            "masked_img_b": masked_b_img.detach(),
            "recon_img_a": recon_a_img.detach(),
            "recon_img_b": recon_b_img.detach(),
            "recon_full_img_a": recon_full_a_img.detach(),
            "recon_full_img_b": recon_full_b_img.detach(),
        }
