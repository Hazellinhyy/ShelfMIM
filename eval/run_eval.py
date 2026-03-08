# # -*- coding: utf-8 -*-
# """
# Run evaluation on a COCO-format test set for instance masks.
#
# Default checkpoint behavior (paper-style):
# - If user passes a directory as checkpoint, try:
#     <dir>/best.pth -> <dir>/last.pth
# - If user passes a file path, use it directly.
# """
#
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, Any, List, Optional, Tuple
#
# import os
# import json
# import numpy as np
# import cv2
# import torch
#
# from pycocotools.coco import COCO
# from pycocotools import mask as maskUtils
#
# from mmengine.config import Config
# from mmdet.apis import init_detector, inference_detector
#
# from .metrics import greedy_match_by_iou, compute_pair_metrics, aggregate_metrics
#
#
# def _resolve_ckpt_path(ckpt: str) -> str:
#     """
#     Accept:
#       - a .pth file path
#       - a directory path containing best.pth / last.pth
#     Prefer best.pth (paper-style).
#     """
#     ckpt = os.path.expanduser(ckpt)
#     if os.path.isdir(ckpt):
#         best = os.path.join(ckpt, "best.pth")
#         last = os.path.join(ckpt, "last.pth")
#         if os.path.exists(best):
#             return best
#         if os.path.exists(last):
#             return last
#         raise FileNotFoundError(f"No best.pth/last.pth found under directory: {ckpt}")
#     if os.path.isfile(ckpt):
#         return ckpt
#     raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")
#
#
# def _catid_to_label_map(coco: COCO) -> Dict[int, int]:
#     cat_ids = sorted(coco.getCatIds())
#     return {cid: i for i, cid in enumerate(cat_ids)}
#
#
# def _decode_gt_masks(
#     coco: COCO,
#     img_id: int,
#     catid2label: Dict[int, int],
#     skip_crowd: bool = True,
# ) -> Tuple[List[np.ndarray], List[int]]:
#     ann_ids = coco.getAnnIds(imgIds=[img_id])
#     anns = coco.loadAnns(ann_ids)
#     info = coco.loadImgs([img_id])[0]
#     H, W = info["height"], info["width"]
#
#     masks, labels = [], []
#     for a in anns:
#         if skip_crowd and int(a.get("iscrowd", 0)) == 1:
#             continue
#         seg = a.get("segmentation", None)
#         if seg is None:
#             continue
#         rle = maskUtils.frPyObjects(seg, H, W) if isinstance(seg, list) else seg
#         if isinstance(rle, list):
#             rle = maskUtils.merge(rle)
#         m = maskUtils.decode(rle).astype(bool)
#         lab = catid2label.get(int(a["category_id"]), -1)
#         if lab >= 0:
#             masks.append(m)
#             labels.append(lab)
#     return masks, labels
#
#
# def _extract_pred_instances(
#     pred,
#     score_thr: float = 0.05,
#     max_per_img: int = 100,
# ) -> Tuple[List[np.ndarray], List[int], List[float]]:
#     if isinstance(pred, list):
#         pred = pred[0]
#     if not hasattr(pred, "pred_instances"):
#         return [], [], []
#
#     inst = pred.pred_instances
#     if not hasattr(inst, "scores") or not hasattr(inst, "labels"):
#         return [], [], []
#
#     scores = inst.scores.detach().cpu().numpy().astype(np.float32)
#     labels = inst.labels.detach().cpu().numpy().astype(np.int32)
#
#     masks = None
#     if hasattr(inst, "masks"):
#         m = inst.masks
#         if torch.is_tensor(m):
#             masks = m.detach().cpu().numpy().astype(bool)
#         else:
#             try:
#                 masks = m.to_ndarray().astype(bool)
#             except Exception:
#                 masks = None
#     if masks is None:
#         return [], [], []
#
#     keep = scores >= float(score_thr)
#     scores, labels, masks = scores[keep], labels[keep], masks[keep]
#
#     if masks.shape[0] > int(max_per_img):
#         idx = np.argsort(-scores)[: int(max_per_img)]
#         scores, labels, masks = scores[idx], labels[idx], masks[idx]
#
#     return [masks[i] for i in range(masks.shape[0])], [int(x) for x in labels.tolist()], [float(x) for x in scores.tolist()]
#
#
# @dataclass
# class EvalConfig:
#     base_cfg_py: str
#     checkpoint: str  # can be .pth file or directory containing best.pth/last.pth
#     ann_file: str
#     img_dir: str
#     device: str = "cpu"
#     score_thr: float = 0.05
#     max_per_img: int = 100
#     iou_match_thr: float = 0.5
#     skip_crowd: bool = True
#     empty_penalty: float = 0.0
#     save_json: Optional[str] = None
#
#
# def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
#     coco = COCO(cfg.ann_file)
#     catid2label = _catid_to_label_map(coco)
#     num_classes_gt = len(catid2label)
#
#     mmcfg = Config.fromfile(cfg.base_cfg_py)
#
#     ckpt_path = _resolve_ckpt_path(cfg.checkpoint)
#     print(f"[eval] using checkpoint: {ckpt_path}")
#
#     model = init_detector(mmcfg, ckpt_path, device=cfg.device)
#
#     # sanity check (best effort)
#     try:
#         head = getattr(model, "panoptic_head", None)
#         if head is not None and hasattr(head, "num_things_classes"):
#             nc = int(head.num_things_classes)
#             if nc != num_classes_gt:
#                 print(f"[WARN] num_classes mismatch: model={nc} vs GT={num_classes_gt}. "
#                       f"Make sure your finetune cfg set dataset.metainfo.classes correctly.")
#     except Exception:
#         pass
#
#     img_ids = coco.getImgIds()
#     dice_vals, iou_vals, asd_vals, hd95_vals = [], [], [], []
#     counts = {"matched": 0, "pred": 0, "gt": 0}
#
#     for img_id in img_ids:
#         info = coco.loadImgs([img_id])[0]
#         file_name = info["file_name"]
#         img_path = os.path.join(cfg.img_dir, file_name)
#         if not os.path.exists(img_path):
#             img_path = os.path.join(cfg.img_dir, os.path.basename(file_name))
#             if not os.path.exists(img_path):
#                 continue
#
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         if img is None:
#             continue
#
#         gt_masks, gt_labels = _decode_gt_masks(coco, img_id, catid2label, skip_crowd=cfg.skip_crowd)
#         pred = inference_detector(model, img)
#         pred_masks, pred_labels, _ = _extract_pred_instances(pred, cfg.score_thr, cfg.max_per_img)
#
#         counts["pred"] += len(pred_masks)
#         counts["gt"] += len(gt_masks)
#
#         all_labels = sorted(set(gt_labels + pred_labels))
#         for lab in all_labels:
#             gt_idx = [i for i, l in enumerate(gt_labels) if l == lab]
#             pr_idx = [i for i, l in enumerate(pred_labels) if l == lab]
#             gt_m = [gt_masks[i] for i in gt_idx]
#             pr_m = [pred_masks[i] for i in pr_idx]
#
#             match = greedy_match_by_iou(pr_m, gt_m, iou_thr=cfg.iou_match_thr)
#             counts["matched"] += len(match.matched_pairs)
#
#             for pi, gi, _ in match.matched_pairs:
#                 m = compute_pair_metrics(pr_m[pi], gt_m[gi], empty_penalty=cfg.empty_penalty)
#                 dice_vals.append(m["dice"])
#                 iou_vals.append(m["iou"])
#                 asd_vals.append(m["asd"])
#                 hd95_vals.append(m["hd95"])
#
#     results = {
#         "checkpoint_used": ckpt_path,
#         "dice": aggregate_metrics(dice_vals),
#         "iou": aggregate_metrics(iou_vals),
#         "asd": aggregate_metrics(asd_vals),
#         "hd95": aggregate_metrics(hd95_vals),
#         "counts": counts,
#         "num_images": len(img_ids),
#         "score_thr": cfg.score_thr,
#         "max_per_img": cfg.max_per_img,
#         "iou_match_thr": cfg.iou_match_thr,
#         "skip_crowd": cfg.skip_crowd,
#         "empty_penalty": cfg.empty_penalty,
#     }
#
#     if cfg.save_json:
#         os.makedirs(os.path.dirname(cfg.save_json), exist_ok=True)
#         with open(cfg.save_json, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
#
#     return results
# -*- coding: utf-8 -*-
"""
eval/run_eval.py

- Map GT category_id -> label by name using checkpoint classes.
- Remap prediction labels into GT label space using checkpoint cat_ids.
- Auto-detect label offset (0 / +1 / -1) per-image by maximizing mapped-valid labels.
- Add diagnostics to understand why matched=0 (too few preds? low scores? label mismatch?).

Note:
- If ignore_labels=True: match all instances globally (ignoring label).
- If ignore_labels=False: match within each label after remap.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import os
import json
import numpy as np
import cv2
import torch

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

from .metrics import greedy_match_by_iou, compute_pair_metrics, aggregate_metrics


def _resolve_ckpt_path(ckpt: str) -> str:
    ckpt = os.path.expanduser(ckpt)
    if os.path.isdir(ckpt):
        best = os.path.join(ckpt, "best.pth")
        last = os.path.join(ckpt, "last.pth")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"No best.pth/last.pth found under directory: {ckpt}")
    if os.path.isfile(ckpt):
        return ckpt
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")


def _catid_to_label_map(coco: COCO, classes: Optional[List[str]] = None) -> Dict[int, int]:
    """COCO category_id -> label index (aligned to model classes by name if provided)."""
    cats = coco.loadCats(coco.getCatIds())

    if not classes:
        cat_ids = sorted([int(c["id"]) for c in cats])
        return {cid: i for i, cid in enumerate(cat_ids)}

    def norm(s: str) -> str:
        return str(s).strip().lower()

    name2label = {norm(n): i for i, n in enumerate(classes)}

    catid2label: Dict[int, int] = {}
    missing = []
    for c in cats:
        cid = int(c["id"])
        cname = norm(c.get("name", ""))
        if cname in name2label:
            catid2label[cid] = int(name2label[cname])
        else:
            missing.append((cid, c.get("name", "")))

    if missing:
        print(f"[WARN] {len(missing)} GT categories not found in model classes (show first 20): {missing[:20]}")

    return catid2label


def _decode_gt_masks(
    coco: COCO,
    img_id: int,
    catid2label: Dict[int, int],
    skip_crowd: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    info = coco.loadImgs([img_id])[0]
    H, W = info["height"], info["width"]

    masks: List[np.ndarray] = []
    labels: List[int] = []
    for a in anns:
        if skip_crowd and int(a.get("iscrowd", 0)) == 1:
            continue
        seg = a.get("segmentation", None)
        if seg is None:
            continue

        rle = maskUtils.frPyObjects(seg, H, W) if isinstance(seg, list) else seg
        if isinstance(rle, list):
            rle = maskUtils.merge(rle)

        m = maskUtils.decode(rle).astype(bool)
        lab = catid2label.get(int(a["category_id"]), -1)
        if lab >= 0:
            masks.append(m)
            labels.append(lab)

    return masks, labels


def _extract_pred_instances(
    pred,
    score_thr: float = 0.05,
    max_per_img: int = 100,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Extract (masks, labels, scores) from MMDet prediction."""
    if isinstance(pred, list):
        pred = pred[0]
    if not hasattr(pred, "pred_instances"):
        return [], [], []

    inst = pred.pred_instances
    if not hasattr(inst, "scores") or not hasattr(inst, "labels"):
        return [], [], []

    scores = inst.scores.detach().cpu().numpy().astype(np.float32)
    labels = inst.labels.detach().cpu().numpy().astype(np.int32)

    masks = None
    if hasattr(inst, "masks"):
        m = inst.masks
        if torch.is_tensor(m):
            masks = m.detach().cpu().numpy().astype(bool)
        else:
            try:
                masks = m.to_ndarray().astype(bool)
            except Exception:
                masks = None

    if masks is None:
        return [], [], []

    keep = scores >= float(score_thr)
    scores, labels, masks = scores[keep], labels[keep], masks[keep]

    if masks.shape[0] > int(max_per_img):
        idx = np.argsort(-scores)[: int(max_per_img)]
        scores, labels, masks = scores[idx], labels[idx], masks[idx]

    return (
        [masks[i] for i in range(masks.shape[0])],
        [int(x) for x in labels.tolist()],
        [float(x) for x in scores.tolist()],
    )


@dataclass
class EvalConfig:
    base_cfg_py: str
    checkpoint: str
    ann_file: str
    img_dir: str
    device: str = "cpu"
    score_thr: float = 0.05
    max_per_img: int = 100
    iou_match_thr: float = 0.5
    skip_crowd: bool = True
    empty_penalty: float = 0.0
    save_json: Optional[str] = None
    ignore_labels: bool = False


def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
    coco = COCO(cfg.ann_file)
    mmcfg = Config.fromfile(cfg.base_cfg_py)

    ckpt_path = _resolve_ckpt_path(cfg.checkpoint)
    print(f"[eval] using checkpoint: {ckpt_path}")

    # ---- Read dataset_meta from checkpoint ----
    classes = None
    cat_ids = None
    try:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        meta = ckpt_obj.get("meta", {}) if isinstance(ckpt_obj, dict) else {}
        dmeta = meta.get("dataset_meta", {}) if isinstance(meta, dict) else {}
        if isinstance(dmeta, dict):
            classes = dmeta.get("classes", None)
            cat_ids = dmeta.get("cat_ids", None)
    except Exception as e:
        print(f"[WARN] cannot read dataset_meta from checkpoint meta: {e}")

    catid2label = _catid_to_label_map(coco, classes=classes)

    # pred label index -> gt label index (base, before offset)
    predlabel2gtlabel: Dict[int, int] = {}
    if cat_ids is not None:
        for i, cid in enumerate(list(cat_ids)):
            cid_int = int(cid)
            if cid_int in catid2label:
                predlabel2gtlabel[int(i)] = int(catid2label[cid_int])

    model = init_detector(mmcfg, ckpt_path, device=cfg.device)

    img_ids = coco.getImgIds()

    dice_vals: List[float] = []
    iou_vals: List[float] = []
    asd_vals: List[float] = []
    hd95_vals: List[float] = []

    counts = {"matched": 0, "pred": 0, "gt": 0}

    # diagnostics
    diag = {
        "images_with_pred": 0,
        "images_with_gt": 0,
        "images_with_both": 0,
        "remap_valid_pred_labels_total": 0,
        "remap_total_pred_labels_total": 0,
        "offset_chosen_hist": { "-1": 0, "0": 0, "1": 0 },
        "label_agree_on_iou_matches": 0,
        "label_disagree_on_iou_matches": 0,
        "iou_matches_total_ignore_labels": 0,
    }

    def remap_with_best_offset(pred_labels: List[int]) -> Tuple[List[int], int, int]:
        """
        Try offsets in [0, +1, -1], choose the one maximizing #mapped labels (>=0).
        Return: (remapped_labels, best_offset, valid_count)
        """
        best_offset = 0
        best_valid = -1
        best_out: List[int] = []

        for offset in (0, 1, -1):
            out = []
            valid = 0
            for pl in pred_labels:
                key = pl + offset
                if key in predlabel2gtlabel:
                    out.append(predlabel2gtlabel[key])
                    valid += 1
                else:
                    out.append(-1)
            if valid > best_valid:
                best_valid = valid
                best_offset = offset
                best_out = out

        return best_out, best_offset, best_valid

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        file_name = info["file_name"]

        img_path = os.path.join(cfg.img_dir, file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(cfg.img_dir, os.path.basename(file_name))
            if not os.path.exists(img_path):
                continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        gt_masks, gt_labels = _decode_gt_masks(coco, img_id, catid2label, skip_crowd=cfg.skip_crowd)
        pred = inference_detector(model, img)
        pred_masks, pred_labels_raw, _ = _extract_pred_instances(pred, cfg.score_thr, cfg.max_per_img)

        counts["pred"] += len(pred_masks)
        counts["gt"] += len(gt_masks)

        if len(pred_masks) > 0:
            diag["images_with_pred"] += 1
        if len(gt_masks) > 0:
            diag["images_with_gt"] += 1
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            diag["images_with_both"] += 1

        # Always compute ignore-label matches for diagnostics
        ig = greedy_match_by_iou(pred_masks, gt_masks, iou_thr=cfg.iou_match_thr)
        diag["iou_matches_total_ignore_labels"] += len(ig.matched_pairs)

        # If we are doing label-aware eval, remap prediction labels first
        if cfg.ignore_labels:
            pred_labels = pred_labels_raw
        else:
            pred_labels, best_offset, valid = remap_with_best_offset(pred_labels_raw)
            diag["offset_chosen_hist"][str(best_offset)] += 1
            diag["remap_valid_pred_labels_total"] += int(valid)
            diag["remap_total_pred_labels_total"] += int(len(pred_labels_raw))

            # also compute how many iou-matched pairs have same label (diagnostic)
            for pi, gi, _ in ig.matched_pairs:
                pl = pred_labels[pi] if 0 <= pi < len(pred_labels) else -1
                gl = gt_labels[gi] if 0 <= gi < len(gt_labels) else -2
                if pl == gl and pl >= 0:
                    diag["label_agree_on_iou_matches"] += 1
                else:
                    diag["label_disagree_on_iou_matches"] += 1

        if cfg.ignore_labels:
            match = ig
            counts["matched"] += len(match.matched_pairs)
            for pi, gi, _ in match.matched_pairs:
                m = compute_pair_metrics(pred_masks[pi], gt_masks[gi], empty_penalty=cfg.empty_penalty)
                dice_vals.append(m["dice"])
                iou_vals.append(m["iou"])
                asd_vals.append(m["asd"])
                hd95_vals.append(m["hd95"])
        else:
            # label-aware: match within each label
            all_labels = sorted(set([l for l in gt_labels if l >= 0] + [l for l in pred_labels if l >= 0]))
            for lab in all_labels:
                gt_idx = [i for i, l in enumerate(gt_labels) if l == lab]
                pr_idx = [i for i, l in enumerate(pred_labels) if l == lab]
                gt_m = [gt_masks[i] for i in gt_idx]
                pr_m = [pred_masks[i] for i in pr_idx]

                match = greedy_match_by_iou(pr_m, gt_m, iou_thr=cfg.iou_match_thr)
                counts["matched"] += len(match.matched_pairs)

                for pi, gi, _ in match.matched_pairs:
                    m = compute_pair_metrics(pr_m[pi], gt_m[gi], empty_penalty=cfg.empty_penalty)
                    dice_vals.append(m["dice"])
                    iou_vals.append(m["iou"])
                    asd_vals.append(m["asd"])
                    hd95_vals.append(m["hd95"])

    results: Dict[str, Any] = {
        "checkpoint_used": ckpt_path,
        "dice": aggregate_metrics(dice_vals),
        "iou": aggregate_metrics(iou_vals),
        "asd": aggregate_metrics(asd_vals),
        "hd95": aggregate_metrics(hd95_vals),
        "counts": counts,
        "num_images": len(img_ids),
        "score_thr": cfg.score_thr,
        "max_per_img": cfg.max_per_img,
        "iou_match_thr": cfg.iou_match_thr,
        "skip_crowd": cfg.skip_crowd,
        "empty_penalty": cfg.empty_penalty,
        "ignore_labels": bool(cfg.ignore_labels),
        "diagnostics": diag,
        "note": (
            "If counts.matched==0 then metrics will be empty/NaN. "
            "Check diagnostics.iou_matches_total_ignore_labels, remap_valid_pred_labels_total, "
            "and label_agree_on_iou_matches to know whether it's geometry or label issue."
        )
    }

    if cfg.save_json:
        os.makedirs(os.path.dirname(cfg.save_json), exist_ok=True)
        with open(cfg.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results