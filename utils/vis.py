# -*- coding: utf-8 -*-
"""
Visualization utilities for ShelfMIM.

A) Branch I slot grid
   Image | soft segmentation | Slot1..SlotK

B) Branch III reconstruction triplets
   masked | recon | gt
"""

from __future__ import annotations

from typing import Tuple, Optional
import os

import cv2
import numpy as np
import torch


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_uint8_img(t: torch.Tensor, mean=MEAN, std=STD) -> np.ndarray:
    x = torch.nan_to_num(t.detach().cpu().float().clone(), nan=0.0, posinf=1.0, neginf=0.0)
    for c in range(3):
        x[c] = x[c] * std[c] + mean[c]
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / max(h, 1)
    return cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)


def _pad_to_width(img: np.ndarray, target_w: int, value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = np.full((h, target_w - w, 3), value, dtype=np.uint8)
    return np.concatenate([img, pad], axis=1)


def _title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (250, 250, 250), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    return out


def _border(img: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int = 4) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color_bgr, thickness, cv2.LINE_AA)
    return out


def _colors_k(k: int) -> np.ndarray:
    palette = np.array([
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
        [153, 153, 153], [102, 194, 165], [252, 141, 98], [141, 160, 203],
    ], dtype=np.uint8)
    if k <= len(palette):
        return palette[:k]
    reps = int(np.ceil(k / len(palette)))
    return np.tile(palette, (reps, 1))[:k]


def masks_to_khw(masks: torch.Tensor, grid_hw: Tuple[int, int], b: int = 0) -> np.ndarray:
    hp, wp = grid_hw
    if masks.dim() == 3:
        k = masks.shape[1]
        return masks[b].detach().cpu().float().reshape(k, hp, wp).numpy()
    if masks.dim() == 4:
        return masks[b].detach().cpu().float().numpy()
    raise ValueError(f"masks must be (B,K,N) or (B,K,Hp,Wp), got {masks.shape}")


def valid_to_hw(valid_patch: Optional[torch.Tensor], grid_hw: Tuple[int, int], b: int = 0) -> Optional[np.ndarray]:
    if valid_patch is None:
        return None
    hp, wp = grid_hw
    vp = valid_patch.detach().cpu()
    if vp.dim() == 4:
        return vp[b, 0].float().numpy() > 0.5
    if vp.dim() == 3:
        return vp[b].float().numpy() > 0.5
    if vp.dim() == 2:
        return vp[b].float().reshape(hp, wp).numpy() > 0.5
    return None


def to_per_patch_dist(m_khw: np.ndarray, valid_hw: Optional[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    m = np.nan_to_num(m_khw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if valid_hw is not None:
        m[:, ~valid_hw] = 0.0
    denom = m.sum(axis=0, keepdims=True) + eps
    return m / denom


def upsample_valid(valid_hw: Optional[np.ndarray], out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if valid_hw is None:
        return None
    h, w = out_hw
    return cv2.resize(valid_hw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _upsample_prob_map(p_hw: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(p_hw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def render_combined_mask_soft(p_khw: np.ndarray, valid_up: Optional[np.ndarray], colors_rgb: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    # Hard winner-takes-all mask, blended with image-like background style.
    k, hp, wp = p_khw.shape
    h, w = out_hw

    winner = np.argmax(p_khw, axis=0).astype(np.int32)  # (Hp,Wp)
    seg_small = colors_rgb[winner]  # RGB
    seg_up = cv2.resize(seg_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    seg_bgr = seg_up[..., ::-1].copy()

    if valid_up is not None:
        seg_bgr[~valid_up] = 255
    return seg_bgr


def render_slot_foreground(img_bgr: np.ndarray, win_up: np.ndarray, slot_id: int, valid_up: Optional[np.ndarray]) -> np.ndarray:
    # Object-centric rendering: keep this slot in color, non-slot in light grayscale.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    gray = 0.82 * gray + 42.0

    out = gray.copy()
    sel = (win_up == int(slot_id))
    out[sel] = img_bgr[sel].astype(np.float32)

    if valid_up is not None:
        out[~valid_up] = 255.0
    return out.clip(0, 255).astype(np.uint8)


def make_slot_grid_row_hard(img_bgr: np.ndarray, m_khw: np.ndarray, valid_hw: Optional[np.ndarray], prefix: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    k = m_khw.shape[0]
    colors_rgb = _colors_k(k)

    p_khw = to_per_patch_dist(m_khw, valid_hw)
    valid_up = upsample_valid(valid_hw, (h, w))

    winner_small = np.argmax(p_khw, axis=0).astype(np.int32)
    winner_up = cv2.resize(winner_small.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    cells = []
    cells.append(_border(_title(img_bgr.copy(), f"{prefix}Image"), (255, 255, 255), 4))

    mask_cell = render_combined_mask_soft(p_khw, valid_up, colors_rgb, out_hw=(h, w))
    mask_overlay = cv2.addWeighted(img_bgr, 0.30, mask_cell, 0.70, 0.0)
    if valid_up is not None:
        mask_overlay[~valid_up] = 255
    cells.append(_border(_title(mask_overlay, f"{prefix}Mask"), (255, 255, 255), 4))

    for i in range(k):
        c_bgr = tuple(int(x) for x in colors_rgb[i][::-1])
        slot_cell = render_slot_foreground(img_bgr, winner_up, i, valid_up)
        slot_cell = _border(_title(slot_cell, f"{prefix}Slot {i + 1}"), c_bgr, 5)
        cells.append(slot_cell)

    target_h = max(x.shape[0] for x in cells)
    cells = [_resize_to_height(x, target_h) for x in cells]
    target_w = max(x.shape[1] for x in cells)
    cells = [_pad_to_width(x, target_w, value=255) for x in cells]
    return np.concatenate(cells, axis=1)


def save_slot_grid_ab(
    out_dir: str,
    step: int,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    M_a: torch.Tensor,
    M_b: torch.Tensor,
    grid_hw: Tuple[int, int],
    valid_a_patch: Optional[torch.Tensor] = None,
    valid_b_patch: Optional[torch.Tensor] = None,
    b: int = 0,
) -> str:
    vis_dir = os.path.join(out_dir, "vis")
    _ensure_dir(vis_dir)

    A_bgr = _to_uint8_img(img_a[b])
    B_bgr = _to_uint8_img(img_b[b])
    Ma_khw = masks_to_khw(M_a, grid_hw, b=b)
    Mb_khw = masks_to_khw(M_b, grid_hw, b=b)
    va = valid_to_hw(valid_a_patch, grid_hw, b=b)
    vb = valid_to_hw(valid_b_patch, grid_hw, b=b)

    rowA = make_slot_grid_row_hard(A_bgr, Ma_khw, va, prefix="A ")
    rowB = make_slot_grid_row_hard(B_bgr, Mb_khw, vb, prefix="B ")

    target_w = max(rowA.shape[1], rowB.shape[1])
    rowA = _pad_to_width(rowA, target_w, value=255)
    rowB = _pad_to_width(rowB, target_w, value=255)
    grid = np.concatenate([rowA, rowB], axis=0)

    path = os.path.join(vis_dir, f"slots_grid_AB_{step:07d}.png")
    cv2.imwrite(path, grid)
    return path


def _concat_h3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    target_h = max(a.shape[0], b.shape[0], c.shape[0])
    a = _resize_to_height(a, target_h)
    b = _resize_to_height(b, target_h)
    c = _resize_to_height(c, target_h)
    return np.concatenate([a, b, c], axis=1)


def save_recon_triplets(
    out_dir: str,
    step: int,
    *,
    masked: torch.Tensor,
    recon: torch.Tensor,
    gt: torch.Tensor,
    prefix: str = "A",
    max_rows: int = 4,
) -> str:
    vis_dir = os.path.join(out_dir, "vis")
    _ensure_dir(vis_dir)

    bsz = int(masked.shape[0])
    n = min(bsz, int(max_rows))
    rows = []
    for i in range(n):
        m = _title(_to_uint8_img(masked[i]), f"{prefix} masked")
        r = _title(_to_uint8_img(recon[i]), f"{prefix} recon")
        g = _title(_to_uint8_img(gt[i]), f"{prefix} gt")
        rows.append(_concat_h3(m, r, g))

    grid = np.concatenate(rows, axis=0) if rows else np.zeros((16, 16, 3), dtype=np.uint8)
    path = os.path.join(vis_dir, f"recon_{prefix}_step_{step:07d}.png")
    cv2.imwrite(path, grid)
    return path
