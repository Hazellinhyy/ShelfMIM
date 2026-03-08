# -*- coding: utf-8 -*-
"""
Metrics for instance masks:
- Dice / IoU (binary)
- ASD (Average Surface Distance)
- HD95 (95th percentile Hausdorff distance)
Also provides greedy instance matching within class by IoU.

Fix:
- When boundary is empty (empty mask), return a configurable penalty instead of NaN
  to avoid silently dropping samples in aggregate.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2


def dice_coef(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def iou_coef(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def _boundary(mask: np.ndarray) -> np.ndarray:
    """1-pixel boundary of a binary mask."""
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(mask_u8, kernel, iterations=1)
    b = (mask_u8 ^ er).astype(np.uint8)
    return b


def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Distances from boundary(a) points to nearest boundary(b).
    Uses cv2.distanceTransform: distance to nearest zero.
    """
    ba = _boundary(a)
    bb = _boundary(b)

    if ba.sum() == 0 or bb.sum() == 0:
        return np.array([], dtype=np.float32)

    inv = (1 - bb).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    return dist_map[ba.astype(bool)]


def asd_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    empty_penalty: float = 0.0,
) -> Tuple[float, float]:
    """
    ASD: mean symmetric surface distance
    HD95: 95th percentile symmetric surface distance

    Policy:
    - If both masks empty -> (0,0)
    - If only one empty or boundary empty -> (empty_penalty, empty_penalty)
      Suggest: empty_penalty = image_diag to strongly penalize.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(empty_penalty), float(empty_penalty)

    d1 = _surface_distances(pred, gt)
    d2 = _surface_distances(gt, pred)
    if d1.size == 0 or d2.size == 0:
        return float(empty_penalty), float(empty_penalty)

    d = np.concatenate([d1, d2], axis=0)
    return float(d.mean()), float(np.percentile(d, 95))


def iou_matrix(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> np.ndarray:
    P, G = len(pred_masks), len(gt_masks)
    M = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            M[i, j] = iou_coef(pred_masks[i], gt_masks[j])
    return M


@dataclass
class MatchResult:
    matched_pairs: List[Tuple[int, int, float]]  # (pred_idx, gt_idx, iou)
    unmatched_pred: List[int]
    unmatched_gt: List[int]


def greedy_match_by_iou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_thr: float = 0.5,
) -> MatchResult:
    P, G = len(pred_masks), len(gt_masks)
    if P == 0 and G == 0:
        return MatchResult([], [], [])
    if P == 0:
        return MatchResult([], [], list(range(G)))
    if G == 0:
        return MatchResult([], list(range(P)), [])

    M = iou_matrix(pred_masks, gt_masks)
    pairs = [(i, j, float(M[i, j])) for i in range(P) for j in range(G)]
    pairs.sort(key=lambda x: x[2], reverse=True)

    used_p, used_g = set(), set()
    matched = []
    for i, j, v in pairs:
        if v < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, v))

    unmatched_pred = [i for i in range(P) if i not in used_p]
    unmatched_gt = [j for j in range(G) if j not in used_g]
    return MatchResult(matched, unmatched_pred, unmatched_gt)


def compute_pair_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, empty_penalty: float = 0.0) -> Dict[str, float]:
    d = dice_coef(pred_mask, gt_mask)
    i = iou_coef(pred_mask, gt_mask)
    asd, hd95 = asd_hd95(pred_mask, gt_mask, empty_penalty=empty_penalty)
    return {"dice": d, "iou": i, "asd": asd, "hd95": hd95}


def aggregate_metrics(values: List[float]) -> float:
    vals = np.array([v for v in values if not (np.isnan(v) or np.isinf(v))], dtype=np.float32)
    return float(vals.mean()) if vals.size > 0 else float("nan")
