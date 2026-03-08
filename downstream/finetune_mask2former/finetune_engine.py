# -*- coding: utf-8 -*-
"""
downstream/finetune_mask2former/finetune_engine.py

- Build MMDet config via wrapper.
- Load SSL backbone weights into MMDet backbone.vit.
- Train with MMEngine runner (train has backprop, val has no backprop).
- Export canonical best.pth / last.pth and keep only one best checkpoint file.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import glob
import shutil

import torch
from mmengine.runner import Runner

from .mask2former_wrapper import build_mmdet_cfg_from_finetune_yaml, cfg_get


def _strip_prefix(k: str, prefixes: List[str]) -> str:
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k


def load_ssl_vit_weights_into_mmdet_model(model, ssl_ckpt_path: str, strict: bool = False) -> Dict[str, int]:
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    vit_sd = {}
    for k, v in state.items():
        kk = _strip_prefix(k, ["module.", "model."])
        if kk.startswith("backbone."):
            vit_sd[kk[len("backbone."):]] = v
        elif kk.startswith("vit."):
            vit_sd[kk[len("vit."):]] = v

    if not vit_sd:
        raise KeyError("No ViT weights found in ssl checkpoint (expected 'backbone.' or 'vit.' prefixes).")

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "vit"):
        raise AttributeError("MMDet model backbone does not have .vit. Ensure config uses MMShelfMIMViTWithSimpleFPN.")

    missing, unexpected = model.backbone.vit.load_state_dict(vit_sd, strict=strict)
    return {"missing": len(missing), "unexpected": len(unexpected)}


def _export_best_last(work_dir: str) -> Dict[str, str]:
    os.makedirs(work_dir, exist_ok=True)
    out = {"best": "", "last": ""}

    best_glob = sorted(glob.glob(os.path.join(work_dir, "best_*.pth")))
    if best_glob:
        src_best = best_glob[-1]
        dst_best = os.path.join(work_dir, "best.pth")
        shutil.copy2(src_best, dst_best)
        out["best"] = dst_best

        for old in best_glob[:-1]:
            try:
                os.remove(old)
            except OSError:
                pass
    elif os.path.isfile(os.path.join(work_dir, "best.pth")):
        out["best"] = os.path.join(work_dir, "best.pth")

    latest = os.path.join(work_dir, "latest.pth")
    if os.path.isfile(latest):
        dst_last = os.path.join(work_dir, "last.pth")
        shutil.copy2(latest, dst_last)
        out["last"] = dst_last
    else:
        epoch_glob = sorted(glob.glob(os.path.join(work_dir, "epoch_*.pth")))
        if epoch_glob:
            src_last = epoch_glob[-1]
            dst_last = os.path.join(work_dir, "last.pth")
            shutil.copy2(src_last, dst_last)
            out["last"] = dst_last

    return out


class FinetuneEngineMMDet:
    def __init__(self, finetune_yaml_cfg: Any):
        import models.backbone.mmshelfmim_vit  # noqa: F401

        self.fcfg = finetune_yaml_cfg
        self.cfg = build_mmdet_cfg_from_finetune_yaml(self.fcfg)

    def train(self) -> Dict[str, Any]:
        runner = Runner.from_cfg(self.cfg)

        ssl_ckpt = cfg_get(self.fcfg, "paths.pretrained_backbone_ckpt", None)
        if ssl_ckpt:
            rep = load_ssl_vit_weights_into_mmdet_model(runner.model, ssl_ckpt, strict=False)
            print(f"[SSL load] missing={rep['missing']} unexpected={rep['unexpected']}")

        runner.train()
        return {"work_dir": self.cfg.work_dir}


class FinetuneEngine:
    def __init__(self, cfg: Any, output_dir: str, device: str = "cuda:0"):
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = device

    def run(self) -> Dict[str, Any]:
        self.cfg.setdefault("experiment", {})
        self.cfg["experiment"]["output_dir"] = self.output_dir

        engine = FinetuneEngineMMDet(self.cfg)
        engine.cfg.work_dir = self.output_dir

        ret = engine.train()
        work_dir = ret.get("work_dir", self.output_dir)

        exported = _export_best_last(work_dir)
        print("[finetune] exported:", exported)
        return {"work_dir": work_dir, "exported": exported}
