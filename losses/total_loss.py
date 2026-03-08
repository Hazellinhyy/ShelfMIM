# -*- coding: utf-8 -*-
"""
losses/total_loss.py

ShelfMIM total loss assembly.
Design goal: single-source branch cores + explicit outer weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from losses.dice import dice_loss, _to_bk_hw
from losses.info_nce import invert_permutation, slot_info_nce
from models.branch_slot.hungarian import hungarian_match_by_dice
from models.branch_slot.warp import warp_masks_a_to_b, warp_masks_b_to_a


def _normalize_over_slots(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if m.dim() not in (3, 4):
        raise ValueError(f"mask dim not supported: {m.shape}")
    denom = m.sum(dim=1, keepdim=True).clamp_min(eps)
    return m / denom


def _gather_slots(m: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    if m.dim() < 3:
        raise ValueError("m must be (B,K,...)")
    bsz, k = pi.shape
    if m.shape[0] != bsz or m.shape[1] != k:
        raise ValueError(f"shape mismatch: m {m.shape[:2]} vs pi {pi.shape}")
    idx = pi.view(bsz, k, *([1] * (m.dim() - 2))).expand_as(m[:, :k, ...])
    return torch.gather(m, dim=1, index=idx)


def _slot_mass_balance_loss(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b, k, _, _ = m.shape
    mass = m.mean(dim=(2, 3))
    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(eps)
    target = torch.full_like(mass, 1.0 / float(max(k, 1)))
    return torch.mean((mass - target) ** 2)


def _slot_diversity_loss(z: torch.Tensor) -> torch.Tensor:
    if z.dim() != 3:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2))
    k = sim.size(1)
    eye = torch.eye(k, device=sim.device, dtype=sim.dtype).unsqueeze(0)
    off = sim * (1.0 - eye)
    return (off ** 2).mean()


def _merge_valid_masks(
    valid_mask: Optional[torch.Tensor],
    warp_valid_mask: Optional[torch.Tensor],
    grid_hw: Tuple[int, int],
    device,
    dtype,
) -> Optional[torch.Tensor]:
    H, W = grid_hw

    def _to_b1_hw(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 4:
            if mask.shape[1] != 1:
                raise ValueError("valid mask expected shape (B,1,H,W)")
            return mask.to(device=device, dtype=dtype)
        if mask.dim() == 3:
            if mask.shape[-2:] != (H, W):
                raise ValueError(f"valid mask HW mismatch: {mask.shape[-2:]} vs {(H, W)}")
            return mask.unsqueeze(1).to(device=device, dtype=dtype)
        if mask.dim() == 2:
            if mask.shape[1] != H * W:
                raise ValueError(f"valid mask N mismatch: {mask.shape[1]} vs {H*W}")
            return mask.reshape(mask.shape[0], 1, H, W).to(device=device, dtype=dtype)
        raise ValueError(f"Unsupported valid mask shape: {mask.shape}")

    vm = _to_b1_hw(valid_mask)
    wv = _to_b1_hw(warp_valid_mask)

    if vm is None and wv is None:
        return None
    if vm is None:
        return wv
    if wv is None:
        return vm
    return vm * wv




def _soft_slot_dice_per_slot(
    m1: torch.Tensor,
    m2: torch.Tensor,
    valid_mask_b1hw: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    if m1.shape != m2.shape:
        raise ValueError(f"shape mismatch for slot dice: {m1.shape} vs {m2.shape}")
    if m1.dim() != 4:
        raise ValueError(f"expected (B,K,H,W), got {m1.shape}")

    if valid_mask_b1hw is None:
        vm = torch.ones((m1.shape[0], 1, m1.shape[2], m1.shape[3]), device=m1.device, dtype=m1.dtype)
    else:
        vm = valid_mask_b1hw.to(device=m1.device, dtype=m1.dtype)

    inter = (m1 * m2 * vm).sum(dim=(2, 3))
    denom = ((m1 + m2) * vm).sum(dim=(2, 3)).clamp_min(eps)
    return (2.0 * inter / denom).clamp(min=0.0, max=1.0)


def compute_branch_i_slot_losses(
    *,
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    slots_a: torch.Tensor,
    slots_b: torch.Tensor,
    H_b2a_pix: torch.Tensor,
    grid_hw: Tuple[int, int],
    valid_mask_a: Optional[torch.Tensor] = None,
    valid_mask_b: Optional[torch.Tensor] = None,
    patch_size: int = 16,
    alpha_mask: float = 1.0,
    alpha_nce: float = 0.1,
    tau_nce: float = 0.2,
    compute_pi: bool = True,
    pi: Optional[torch.Tensor] = None,
    nce_use_valid_slots: bool = True,
    slot_conf_thresh: float = 0.25,
    alpha_slot_balance: float = 0.2,
    alpha_slot_diversity: float = 0.1,
) -> Dict[str, torch.Tensor]:
    ma = _to_bk_hw(masks_a, grid_hw)
    mb = _to_bk_hw(masks_b, grid_hw)
    ma = _normalize_over_slots(ma)
    mb = _normalize_over_slots(mb)

    mb2a, warp_valid_a = warp_masks_b_to_a(mb, H_b2a_pix, grid_hw=grid_hw, patch_size=patch_size, return_valid_mask=True)
    ma2b, warp_valid_b = warp_masks_a_to_b(ma, H_b2a_pix, grid_hw=grid_hw, patch_size=patch_size, return_valid_mask=True)

    valid_a_total = _merge_valid_masks(valid_mask_a, warp_valid_a, grid_hw, device=ma.device, dtype=ma.dtype)
    valid_b_total = _merge_valid_masks(valid_mask_b, warp_valid_b, grid_hw, device=mb.device, dtype=mb.dtype)

    if compute_pi:
        if pi is None:
            pi = hungarian_match_by_dice(ma, mb2a, valid_mask=valid_a_total)
    elif pi is None:
        raise ValueError("compute_pi=False requires pi to be provided")

    mb2a_aligned = _gather_slots(mb2a, pi)
    loss_a = dice_loss(ma, mb2a_aligned, valid_mask=valid_a_total, reduction="mean")

    inv_pi = invert_permutation(pi)
    ma2b_aligned = _gather_slots(ma2b, inv_pi)
    loss_b = dice_loss(mb, ma2b_aligned, valid_mask=valid_b_total, reduction="mean")
    loss_mask = 0.5 * (loss_a + loss_b)

    # Patch-level geometric validity for proto part consistency.
    geom_w_a = _soft_slot_dice_per_slot(ma, mb2a_aligned, valid_a_total)
    geom_w_b = _soft_slot_dice_per_slot(mb, ma2b_aligned, valid_b_total)
    proto_part_geom_weight = (0.5 * (geom_w_a + geom_w_b)).detach()

    valid_slots = None
    if nce_use_valid_slots:
        conf_a = ma.amax(dim=(2, 3))
        conf_b = mb.amax(dim=(2, 3))
        valid_slots = ((conf_a > float(slot_conf_thresh)) | (conf_b > float(slot_conf_thresh))).float()
        if (valid_slots.sum(dim=1) <= 0).any():
            fallback = torch.ones_like(valid_slots)
            bad = (valid_slots.sum(dim=1) <= 0).unsqueeze(1)
            valid_slots = torch.where(bad, fallback, valid_slots)

    loss_nce = slot_info_nce(
        slots_a,
        slots_b,
        pi,
        temperature=tau_nce,
        symmetric=True,
        valid_slots=valid_slots,
    )

    loss_balance = 0.5 * (_slot_mass_balance_loss(ma) + _slot_mass_balance_loss(mb))
    loss_div = 0.5 * (_slot_diversity_loss(slots_a) + _slot_diversity_loss(slots_b))

    contrib_slot_mask = float(alpha_mask) * loss_mask
    contrib_slot_nce = float(alpha_nce) * loss_nce
    contrib_slot_balance = float(alpha_slot_balance) * loss_balance
    contrib_slot_div = float(alpha_slot_diversity) * loss_div

    slot_branch_core = contrib_slot_mask + contrib_slot_nce + contrib_slot_balance + contrib_slot_div

    return {
        "pi": pi,
        "loss_mask": loss_mask,
        "loss_slot_nce": loss_nce,
        "loss_slot_balance": loss_balance,
        "loss_slot_diversity": loss_div,
        "slot_branch_core": slot_branch_core,
        # legacy key
        "loss_slot_total": slot_branch_core,
        "contrib_slot_mask_real": contrib_slot_mask,
        "contrib_slot_nce_real": contrib_slot_nce,
        "contrib_slot_balance_real": contrib_slot_balance,
        "contrib_slot_diversity_real": contrib_slot_div,
        "slot_geom_weight": proto_part_geom_weight,
        "proto_part_geom_weight": proto_part_geom_weight,
    }


@dataclass
class TotalLossWeights:
    # Outer branch weights
    lambda_proto: float = 0.10
    lambda_cs_hps: float = 0.30

    # Proto internal weights
    lambda_pp: float = 1.0
    lambda_bal: float = 0.0
    lambda_sharp: float = 0.03
    lambda_consistency: float = 0.0

    # CS/HPS internal weights
    lambda_cont: float = 1.0
    lambda_decorr: float = 0.1
    lambda_geom: float = 0.25
    lambda_mim: float = 1.0
    lambda_mim_visible: float = 0.10

    # Slot weights (for completeness / logging parity)
    alpha_mask: float = 1.0
    alpha_nce: float = 0.1
    alpha_slot_balance: float = 0.2
    alpha_slot_diversity: float = 0.1
    tau_nce: float = 0.2

    # Deprecated compatibility flags
    auto_scale_proto: bool = False
    auto_scale_cs: bool = False
    scale_eps: float = 1e-6
    scale_clip: float = 10.0
    scale_min: float = 0.5
    scale_power: float = 0.5


def _safe(x: torch.Tensor, *, pos: float = 1e3, neg: float = -1e3) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)


def assemble_total_loss(
    *,
    slot_loss_dict: Dict[str, torch.Tensor],
    proto_loss_dict: Optional[Dict[str, torch.Tensor]] = None,
    cs_hps_loss_dict: Optional[Dict[str, torch.Tensor]] = None,
    # Backward-compatible fallback args.
    proto_loss_total: Optional[torch.Tensor] = None,
    cs_hps_loss_total: Optional[torch.Tensor] = None,
    weights: TotalLossWeights = TotalLossWeights(),
) -> Dict[str, torch.Tensor]:
    # Single source of truth for slot contribution.
    slot_branch_core = _safe(slot_loss_dict.get("slot_branch_core", slot_loss_dict["loss_slot_total"]))
    slot_train_contrib = slot_branch_core

    proto_pair = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    proto_bal = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    proto_ent = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    proto_cons = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)

    if proto_loss_dict is not None:
        proto_pair = _safe(proto_loss_dict.get("loss_proto_pair", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        proto_bal = _safe(proto_loss_dict.get("loss_proto_balance", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        proto_ent = _safe(proto_loss_dict.get("loss_proto_sharpness", proto_loss_dict.get("loss_proto_entropy", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))))
        proto_cons = _safe(proto_loss_dict.get("loss_proto_consistency", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
    elif proto_loss_total is not None:
        proto_pair = _safe(proto_loss_total)

    proto_branch_core = (
        float(weights.lambda_pp) * proto_pair
        + float(weights.lambda_bal) * proto_bal
        + float(weights.lambda_sharp) * proto_ent
        + float(weights.lambda_consistency) * proto_cons
    )

    contrib_proto_pp_real = float(weights.lambda_proto) * float(weights.lambda_pp) * proto_pair
    contrib_proto_bal_real = float(weights.lambda_proto) * float(weights.lambda_bal) * proto_bal
    contrib_proto_sharp_real = float(weights.lambda_proto) * float(weights.lambda_sharp) * proto_ent
    contrib_proto_consistency_real = float(weights.lambda_proto) * float(weights.lambda_consistency) * proto_cons
    proto_train_contrib_atoms_sum = contrib_proto_pp_real + contrib_proto_bal_real + contrib_proto_sharp_real + contrib_proto_consistency_real
    proto_train_contrib = proto_train_contrib_atoms_sum

    cs_cont = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    cs_decorr = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    cs_geom = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    cs_mim = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)
    cs_mim_vis = torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)

    if cs_hps_loss_dict is not None:
        cs_cont = _safe(cs_hps_loss_dict.get("loss_cont", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        cs_decorr = _safe(cs_hps_loss_dict.get("loss_decorr", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        cs_geom = _safe(cs_hps_loss_dict.get("loss_geom", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        cs_mim = _safe(cs_hps_loss_dict.get("loss_mim", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        cs_mim_vis = _safe(cs_hps_loss_dict.get("loss_mim_visible", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
    elif cs_hps_loss_total is not None:
        cs_mim = _safe(cs_hps_loss_total)

    cs_branch_core = (
        float(weights.lambda_cont) * cs_cont
        + float(weights.lambda_decorr) * cs_decorr
        + float(weights.lambda_geom) * cs_geom
        + float(weights.lambda_mim) * cs_mim
        + float(weights.lambda_mim_visible) * cs_mim_vis
    )

    contrib_cs_cont_real = float(weights.lambda_cs_hps) * float(weights.lambda_cont) * cs_cont
    contrib_cs_decorr_real = float(weights.lambda_cs_hps) * float(weights.lambda_decorr) * cs_decorr
    contrib_cs_geom_real = float(weights.lambda_cs_hps) * float(weights.lambda_geom) * cs_geom
    contrib_cs_mim_real = float(weights.lambda_cs_hps) * float(weights.lambda_mim) * cs_mim
    contrib_cs_mim_visible_real = float(weights.lambda_cs_hps) * float(weights.lambda_mim_visible) * cs_mim_vis
    cs_train_contrib_atoms_sum = contrib_cs_cont_real + contrib_cs_decorr_real + contrib_cs_geom_real + contrib_cs_mim_real + contrib_cs_mim_visible_real
    cs_train_contrib = cs_train_contrib_atoms_sum

    loss_total_train = _safe(slot_train_contrib + proto_train_contrib + cs_train_contrib)

    # Consistency checks for accounting integrity.
    slot_atoms_sum = _safe(
        _safe(slot_loss_dict.get("contrib_slot_mask_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        + _safe(slot_loss_dict.get("contrib_slot_nce_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        + _safe(slot_loss_dict.get("contrib_slot_balance_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
        + _safe(slot_loss_dict.get("contrib_slot_diversity_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype)))
    )

    if not torch.allclose(_safe(slot_train_contrib), slot_atoms_sum, atol=1e-6, rtol=1e-5):
        raise RuntimeError("slot contribution mismatch")

    if not torch.allclose(_safe(proto_train_contrib), _safe(proto_train_contrib_atoms_sum), atol=1e-6, rtol=1e-5):
        raise RuntimeError("proto contribution mismatch")

    if not torch.allclose(_safe(cs_train_contrib), _safe(cs_train_contrib_atoms_sum), atol=1e-6, rtol=1e-5):
        raise RuntimeError("cs contribution mismatch")

    if not torch.allclose(
        _safe(loss_total_train),
        _safe(slot_train_contrib + proto_train_contrib + cs_train_contrib),
        atol=1e-6,
        rtol=1e-5,
    ):
        raise RuntimeError("total contribution mismatch")

    out = {
        "loss_total": loss_total_train,                 # legacy
        "loss_total_train": loss_total_train,

        "slot_branch_core": slot_branch_core,
        "proto_branch_core": _safe(proto_branch_core),
        "cs_branch_core": _safe(cs_branch_core),

        "slot_train_contrib": slot_train_contrib,
        "proto_train_contrib": _safe(proto_train_contrib),
        "cs_train_contrib": _safe(cs_train_contrib),

        # legacy names
        "contrib_slot_real": slot_train_contrib,
        "contrib_proto_real": _safe(proto_train_contrib),
        "contrib_cs_real": _safe(cs_train_contrib),
        "contrib_total_check": _safe(slot_train_contrib + proto_train_contrib + cs_train_contrib),
        "contrib_total_delta": _safe(loss_total_train - (slot_train_contrib + proto_train_contrib + cs_train_contrib)),

        # clearer branch-weighted names (keep legacy keys above)
        "contrib_slot_branch_weighted": slot_train_contrib,
        "contrib_proto_branch_weighted": _safe(proto_train_contrib),
        "contrib_cs_branch_weighted": _safe(cs_train_contrib),
        "contrib_slot_global_weighted": slot_train_contrib,
        "contrib_proto_global_weighted": _safe(proto_train_contrib),
        "contrib_cs_global_weighted": _safe(cs_train_contrib),

        # slot atoms (already weighted in compute_branch_i_slot_losses)
        "contrib_slot_mask_real": _safe(slot_loss_dict.get("contrib_slot_mask_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),
        "contrib_slot_nce_real": _safe(slot_loss_dict.get("contrib_slot_nce_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),
        "contrib_slot_balance_real": _safe(slot_loss_dict.get("contrib_slot_balance_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),
        "contrib_slot_diversity_real": _safe(slot_loss_dict.get("contrib_slot_diversity_real", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),

        # proto atoms (real)
        "contrib_proto_pp_real": _safe(contrib_proto_pp_real),
        "contrib_proto_bal_real": _safe(contrib_proto_bal_real),
        "contrib_proto_sharp_real": _safe(contrib_proto_sharp_real),
        "contrib_proto_consistency_real": _safe(contrib_proto_consistency_real),
        "contrib_proto_atoms_sum_real": _safe(proto_train_contrib_atoms_sum),
        "contrib_proto_atoms_delta_real": _safe(proto_train_contrib - proto_train_contrib_atoms_sum),

        # cs atoms (real)
        "contrib_cs_cont_real": _safe(contrib_cs_cont_real),
        "contrib_cs_decorr_real": _safe(contrib_cs_decorr_real),
        "contrib_cs_geom_real": _safe(contrib_cs_geom_real),
        "contrib_cs_mim_real": _safe(contrib_cs_mim_real),
        "contrib_cs_mim_visible_real": _safe(contrib_cs_mim_visible_real),
        "contrib_cs_atoms_sum_real": _safe(cs_train_contrib_atoms_sum),
        "contrib_cs_atoms_delta_real": _safe(cs_train_contrib - cs_train_contrib_atoms_sum),

        # raw atoms for diagnostics
        "loss_mask": _safe(slot_loss_dict.get("loss_mask", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),
        "loss_slot_nce": _safe(slot_loss_dict.get("loss_slot_nce", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),
        "loss_slot_balance": _safe(slot_loss_dict.get("loss_slot_balance", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),
        "loss_slot_diversity": _safe(slot_loss_dict.get("loss_slot_diversity", torch.tensor(0.0, device=slot_branch_core.device, dtype=slot_branch_core.dtype))),

        "loss_proto_pair": _safe(proto_pair),
        "loss_proto_balance": _safe(proto_bal),
        "loss_proto_entropy": _safe(proto_ent),
        "loss_proto_sharpness": _safe(proto_ent),
        "loss_proto_consistency": _safe(proto_cons),

        "loss_cs_cont": _safe(cs_cont),
        "loss_cs_decorr": _safe(cs_decorr),
        "loss_cs_geom": _safe(cs_geom),
        "loss_cs_mim": _safe(cs_mim),
        "loss_cs_mim_visible": _safe(cs_mim_vis),

        # clearer monitor aliases
        "proto_monitor_total_before_outer_weight": _safe(proto_branch_core),
        "cs_monitor_total_before_outer_weight": _safe(cs_branch_core),
        # legacy compatibility
        "loss_slot_total": slot_branch_core,
        "loss_proto_total": _safe(proto_branch_core),
        "loss_cs_hps_total": _safe(cs_branch_core),
        "contrib_proto_scaled": _safe(proto_branch_core),
        "contrib_cs_scaled": _safe(cs_branch_core),
        "lambda_proto_eff": torch.tensor(float(weights.lambda_proto), device=slot_branch_core.device, dtype=slot_branch_core.dtype),
        "lambda_cs_hps_eff": torch.tensor(float(weights.lambda_cs_hps), device=slot_branch_core.device, dtype=slot_branch_core.dtype),
    }
    return out
