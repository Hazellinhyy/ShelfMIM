# -*- coding: utf-8 -*-
"""
inference.py

Paper-style checkpoint selection:
- If checkpoint is a directory -> prefer best.pth then last.pth
- If checkpoint is a file -> use it directly

This is a minimal MMDet inference utility:
- loads config + checkpoint
- runs inference on one image or a directory of images
- optionally saves visualizations (if you implement visualization yourself)

Adjust / extend as needed for your project.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import os
import glob

import cv2
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector


def resolve_ckpt_path(ckpt: str) -> str:
    ckpt = os.path.expanduser(ckpt)
    if os.path.isdir(ckpt):
        best = os.path.join(ckpt, "best.pth")
        last = os.path.join(ckpt, "last.pth")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"No best.pth/last.pth found in directory: {ckpt}")
    if os.path.isfile(ckpt):
        return ckpt
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")


@dataclass
class InferArgs:
    base_cfg_py: str
    checkpoint: str            # can be file or directory
    input_path: str            # image path OR directory
    device: str = "cuda:0"
    score_thr: float = 0.05
    max_per_img: int = 100


def _list_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"]
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, e)))
        files.sort()
        return files
    return [input_path]


def run_inference(args: InferArgs):
    cfg = Config.fromfile(args.base_cfg_py)
    ckpt = resolve_ckpt_path(args.checkpoint)
    print(f"[infer] using checkpoint: {ckpt}")

    model = init_detector(cfg, ckpt, device=args.device)

    img_list = _list_images(args.input_path)
    for p in img_list:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[infer] skip unreadable image: {p}")
            continue
        pred = inference_detector(model, img)
        # 这里 pred 就是 MMDet 的结果对象
        # 你可以按自己的项目需要解析 masks/boxes/labels/scores
        print(f"[infer] done: {p}")

    return True
