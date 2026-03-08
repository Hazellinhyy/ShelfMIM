# -*- coding: utf-8 -*-
"""
Mask2Former config builder for ShelfMIM finetuning.

Implements thesis protocol:
- Train/val/test official split usage.
- SSL-pretrained ViT initialization (backbone only loaded by finetune_engine).
- End-to-end finetuning with LayerNorm not frozen.
- Backbone LR multiplier = 0.1 (configurable).
- Validation-based model selection and canonical best/last export.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Tuple

from mmengine.config import Config
from mmengine.registry import init_default_scope


def cfg_get(cfg: Any, key: str, default=None):
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if cur is None:
            return default
        if hasattr(cur, p):
            cur = getattr(cur, p)
        elif isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _resolve_placeholders(value: Any, cfg: Any) -> Any:
    if not isinstance(value, str):
        return value

    out = value
    pat = re.compile(r"\$\{([^}]+)\}")
    for _ in range(8):
        ms = list(pat.finditer(out))
        if not ms:
            break
        changed = False
        for m in ms:
            key = m.group(1).strip()
            rep = cfg_get(cfg, key, m.group(0))
            rep = str(rep)
            if rep != m.group(0):
                changed = True
            out = out.replace(m.group(0), rep)
        if not changed:
            break
    return out


def _abspath(base: str, p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(base, p))


def _set_dataset(cfg: Config, key: str, ann: str, img: str, classes: Tuple[str, ...], cat_ids: List[int]):
    dl = cfg.get(key)
    if not dl:
        return

    ds = dl.get("dataset", dl)
    ds["type"] = "CocoDataset"
    ds["ann_file"] = ann
    ds["data_prefix"] = dict(img=img)
    ds["data_root"] = ""
    ds["metainfo"] = dict(classes=classes, cat_ids=cat_ids)


def _disable_filter_empty_gt(cfg: Config):
    td = cfg.get("train_dataloader")
    if not td:
        return
    ds = td.get("dataset", td)
    ds.setdefault("filter_cfg", {})
    ds["filter_cfg"]["filter_empty_gt"] = False


def _replace_backbone(cfg: Config, fcfg: Any):
    cfg.model["backbone"] = dict(
        type="MMShelfMIMViTWithSimpleFPN",
        out_channels=256,
        patch_size=int(cfg_get(fcfg, "model.backbone.patch_size", 16)),
        vit_drop_path_rate=0.1,
        vit_use_cls_token=True,
        vit_pos_base_grid_hw=(14, 14),
    )

    cfg.model["init_cfg"] = None
    cfg.model["neck"] = None

    ph = cfg.model["panoptic_head"]
    ph["in_channels"] = [256, 256, 256, 256]
    ph["pixel_decoder"]["in_channels"] = [256, 256, 256, 256]


def _override_resize(cfg: Config, scale: Tuple[int, int]):
    def patch_pipeline(pipeline):
        for t in pipeline:
            if isinstance(t, dict) and t.get("type") in ["Resize", "RandomResize"]:
                t["scale"] = scale
                t["keep_ratio"] = True

    for k in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        dl = cfg.get(k)
        if not dl:
            continue
        ds = dl.get("dataset", dl)
        if isinstance(ds, dict) and "pipeline" in ds:
            patch_pipeline(ds["pipeline"])


def _fix_num_classes_for_mask2former(cfg: Config, num_classes: int):
    ph = cfg.model.get("panoptic_head", None)
    if not isinstance(ph, dict):
        return

    ph["num_classes"] = num_classes
    ph["num_things_classes"] = num_classes
    ph["num_stuff_classes"] = 0

    loss_cls = ph.get("loss_cls", None)
    if isinstance(loss_cls, dict):
        loss_cls.pop("num_classes", None)
        loss_cls["class_weight"] = [1.0] * num_classes + [0.1]


def _fix_fusion_head(cfg: Config, num_classes: int):
    fh = cfg.model.get("panoptic_fusion_head", None)
    if not isinstance(fh, dict):
        return

    fh["num_classes"] = num_classes
    fh["num_things_classes"] = num_classes
    fh["num_stuff_classes"] = 0

    tc = fh.get("test_cfg", None)
    if isinstance(tc, dict):
        tc["num_classes"] = num_classes
        tc["num_things_classes"] = num_classes
        tc["num_stuff_classes"] = 0

        if "max_per_image" in tc and isinstance(tc["max_per_image"], int):
            ph = cfg.model.get("panoptic_head", {})
            nq = int(ph.get("num_queries", 100))
            tc["max_per_image"] = min(tc["max_per_image"], nq)

    for k in ["val_dataloader", "test_dataloader"]:
        dl = cfg.get(k, None)
        if isinstance(dl, dict):
            dl["persistent_workers"] = False


def _set_dataloader_runtime(cfg: Config, fcfg: Any):
    bs = int(cfg_get(fcfg, "train.batch_size_total", 1))
    nw = int(cfg_get(fcfg, "train.num_workers", 2))

    train_dl = cfg.get("train_dataloader")
    if isinstance(train_dl, dict):
        train_dl["batch_size"] = bs
        train_dl["num_workers"] = nw
        train_dl["persistent_workers"] = bool(nw > 0)

    for k in ["val_dataloader", "test_dataloader"]:
        dl = cfg.get(k)
        if isinstance(dl, dict):
            dl.setdefault("batch_size", 1)
            dl["num_workers"] = nw
            dl["persistent_workers"] = False


def _set_checkpoint_hook(cfg: Config, fcfg: Any):
    cfg.setdefault("default_hooks", {})
    ckpt_hook = cfg["default_hooks"].get("checkpoint", {})
    ckpt_hook["type"] = "CheckpointHook"
    ckpt_hook["interval"] = int(cfg_get(fcfg, "train.eval_interval_epochs", 1))
    ckpt_hook["save_best"] = str(cfg_get(fcfg, "train.checkpoint.save_best_metric", "coco/segm_mAP"))
    ckpt_hook["rule"] = str(cfg_get(fcfg, "train.checkpoint.save_best_rule", "greater"))
    ckpt_hook["max_keep_ckpts"] = int(cfg_get(fcfg, "train.checkpoint.max_keep_ckpts", 1))
    ckpt_hook["save_last"] = bool(cfg_get(fcfg, "train.checkpoint.save_last", True))
    cfg["default_hooks"]["checkpoint"] = ckpt_hook

    log_interval = int(cfg_get(fcfg, "train.log_interval", 50))
    logger_hook = cfg["default_hooks"].get("logger", {})
    logger_hook["type"] = "LoggerHook"
    logger_hook["interval"] = log_interval
    cfg["default_hooks"]["logger"] = logger_hook


def _set_seed(cfg: Config, fcfg: Any):
    seed = cfg_get(fcfg, "experiment.seed", None)
    if seed is None:
        seeds = cfg_get(fcfg, "experiment.seeds", None)
        if isinstance(seeds, list) and seeds:
            seed = int(seeds[0])
    if seed is not None:
        cfg["randomness"] = dict(seed=int(seed), deterministic=False)


def _set_optim_and_schedule(cfg: Config, fcfg: Any):
    base_lr = float(cfg_get(fcfg, "optimizer.base_lr", 1e-4))
    weight_decay = float(cfg_get(fcfg, "optimizer.weight_decay", 0.05))
    betas = tuple(cfg_get(fcfg, "optimizer.betas", [0.9, 0.999]))
    grad_clip_norm = float(cfg_get(fcfg, "train.grad_clip_norm", 0.0))

    lr_mult = float(cfg_get(fcfg, "model.backbone.lr_multiplier", 0.1))
    amp_cfg = cfg_get(fcfg, "train.amp", False)
    amp_enabled = bool(amp_cfg.get("enabled", False)) if isinstance(amp_cfg, dict) else bool(amp_cfg)

    optim_wrapper = dict(
        type="AmpOptimWrapper" if amp_enabled else "OptimWrapper",
        optimizer=dict(type="AdamW", lr=base_lr, betas=betas, weight_decay=weight_decay),
        paramwise_cfg=dict(custom_keys={"backbone.vit": dict(lr_mult=lr_mult)}),
    )
    if amp_enabled:
        optim_wrapper["loss_scale"] = "dynamic"
    if grad_clip_norm > 0:
        optim_wrapper["clip_grad"] = dict(max_norm=grad_clip_norm, norm_type=2)

    cfg.optim_wrapper = optim_wrapper

    max_epochs = int(cfg_get(fcfg, "train.epochs", 30))
    warmup_epochs = int(cfg_get(fcfg, "scheduler.warmup_epochs", 0))
    min_lr = float(cfg_get(fcfg, "scheduler.min_lr", 0.0))

    sched = []
    if warmup_epochs > 0:
        sched.append(
            dict(
                type="LinearLR",
                start_factor=1e-3,
                by_epoch=True,
                begin=0,
                end=warmup_epochs,
                convert_to_iter_based=True,
            )
        )
    sched.append(
        dict(
            type="CosineAnnealingLR",
            by_epoch=True,
            begin=warmup_epochs,
            end=max_epochs,
            T_max=max(1, max_epochs - warmup_epochs),
            eta_min=min_lr,
        )
    )
    cfg.param_scheduler = sched


def _set_inference_postprocess(cfg: Config, fcfg: Any):
    score_thr = float(cfg_get(fcfg, "inference.score_threshold", 0.05))
    max_per_img = int(cfg_get(fcfg, "inference.max_instances_per_image", 100))

    ph = cfg.model.get("panoptic_head", {})
    if isinstance(ph, dict):
        tcfg = ph.get("test_cfg", {})
        if isinstance(tcfg, dict):
            tcfg["max_per_image"] = max_per_img
            ph["test_cfg"] = tcfg

    fh = cfg.model.get("panoptic_fusion_head", {})
    if isinstance(fh, dict):
        tcfg = fh.get("test_cfg", {})
        if isinstance(tcfg, dict):
            tcfg["max_per_image"] = max_per_img
            fh["test_cfg"] = tcfg

    cfg.setdefault("paper_infer", {})
    cfg.paper_infer["score_thr"] = score_thr
    cfg.paper_infer["max_per_img"] = max_per_img


def build_mmdet_cfg_from_finetune_yaml(fcfg: Any) -> Config:
    init_default_scope("mmdet")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_root_raw = _resolve_placeholders(cfg_get(fcfg, "data.root", ""), fcfg)
    dataset_root = _abspath(project_root, data_root_raw)

    base_cfg_raw = _resolve_placeholders(cfg_get(fcfg, "paths.mask2former_base_config", ""), fcfg)
    base_cfg = _abspath(project_root, base_cfg_raw)

    cfg = Config.fromfile(base_cfg)
    cfg.data_root = ""

    train_ann = _abspath(dataset_root, _resolve_placeholders(cfg_get(fcfg, "data.splits.train.ann_file"), fcfg))
    val_ann = _abspath(dataset_root, _resolve_placeholders(cfg_get(fcfg, "data.splits.val.ann_file"), fcfg))
    test_ann = _abspath(dataset_root, _resolve_placeholders(cfg_get(fcfg, "data.splits.test.ann_file"), fcfg))

    train_img = _abspath(dataset_root, _resolve_placeholders(cfg_get(fcfg, "data.splits.train.image_dir"), fcfg))
    val_img = _abspath(dataset_root, _resolve_placeholders(cfg_get(fcfg, "data.splits.val.image_dir"), fcfg))
    test_img = _abspath(dataset_root, _resolve_placeholders(cfg_get(fcfg, "data.splits.test.image_dir"), fcfg))

    with open(train_ann, "r", encoding="utf-8") as f:
        ann_json = json.load(f)

    categories = sorted(ann_json["categories"], key=lambda x: int(x["id"]))
    classes = tuple([c["name"] for c in categories])
    cat_ids = [int(c["id"]) for c in categories]
    num_classes = len(classes)

    print(f"[finetune] detected num_classes = {num_classes}")

    _set_dataset(cfg, "train_dataloader", train_ann, train_img, classes, cat_ids)
    _set_dataset(cfg, "val_dataloader", val_ann, val_img, classes, cat_ids)
    _set_dataset(cfg, "test_dataloader", test_ann, test_img, classes, cat_ids)

    _fix_num_classes_for_mask2former(cfg, num_classes)
    _fix_fusion_head(cfg, num_classes)

    cfg.val_evaluator = dict(type="CocoMetric", ann_file=val_ann, metric=["segm"])
    cfg.test_evaluator = dict(type="CocoMetric", ann_file=test_ann, metric=["segm"])

    _disable_filter_empty_gt(cfg)
    _replace_backbone(cfg, fcfg)

    short_side = int(cfg_get(fcfg, "input.resize.short_side", 768))
    long_side_max = int(cfg_get(fcfg, "input.resize.long_side_max", 1536))
    _override_resize(cfg, (long_side_max, short_side))

    _set_dataloader_runtime(cfg, fcfg)

    max_epochs = int(cfg_get(fcfg, "train.epochs", 30))
    val_interval = int(cfg_get(fcfg, "train.eval_interval_epochs", 1))
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=val_interval)

    _set_optim_and_schedule(cfg, fcfg)
    _set_checkpoint_hook(cfg, fcfg)
    _set_inference_postprocess(cfg, fcfg)
    _set_seed(cfg, fcfg)

    return cfg
