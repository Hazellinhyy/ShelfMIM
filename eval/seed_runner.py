# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, List
import os
import json
import random

import numpy as np
import torch

from .run_eval import EvalConfig, run_eval


def mean_std(x: List[float]) -> Dict[str, float]:
    arr = np.array(x, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def run_seeds(
    *,
    seeds: List[int],
    base_cfg_py: str,
    ann_file: str,
    img_dir: str,
    checkpoint: str | None = None,
    ckpt_template: str | None = None,
    device: str = "cpu",
    out_dir: str = "eval_results",
    score_thr: float = 0.05,
    max_per_img: int = 100,
    iou_match_thr: float = 0.5,
    skip_crowd: bool = True,
    empty_penalty: float = 0.0,
    ignore_labels: bool = False,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    if checkpoint is None and ckpt_template is None:
        raise ValueError("Either `checkpoint` or `ckpt_template` must be provided.")

    per_seed = {}
    dice_list, iou_list, asd_list, hd95_list = [], [], [], []

    for s in seeds:
        s = int(s)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        ckpt = ckpt_template.format(seed=s) if ckpt_template is not None else str(checkpoint)
        save_json = os.path.join(out_dir, f"seed{s}_metrics.json")

        ecfg = EvalConfig(
            base_cfg_py=base_cfg_py,
            checkpoint=ckpt,
            ann_file=ann_file,
            img_dir=img_dir,
            device=device,
            score_thr=score_thr,
            max_per_img=max_per_img,
            iou_match_thr=iou_match_thr,
            skip_crowd=skip_crowd,
            empty_penalty=empty_penalty,
            save_json=save_json,
            ignore_labels=ignore_labels,
        )
        res = run_eval(ecfg)
        per_seed[str(s)] = res

        dice_list.append(res["dice"])
        iou_list.append(res["iou"])
        asd_list.append(res["asd"])
        hd95_list.append(res["hd95"])

    summary = {
        "seeds": [int(s) for s in seeds],
        "dice": mean_std(dice_list),
        "iou": mean_std(iou_list),
        "asd": mean_std(asd_list),
        "hd95": mean_std(hd95_list),
        "paper_postprocess": {"score_thr": score_thr, "max_per_img": max_per_img},
        "iou_match_thr": iou_match_thr,
        "skip_crowd": skip_crowd,
        "empty_penalty": empty_penalty,
        "ignore_labels": bool(ignore_labels),
        "per_seed": per_seed,
    }

    with open(os.path.join(out_dir, "summary_mean_std.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
