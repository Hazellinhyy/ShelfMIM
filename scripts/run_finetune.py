# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, List, Optional

import yaml


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _project_root_from_this_file() -> str:
    return _abspath(os.path.join(os.path.dirname(__file__), ".."))


_PROJ_ROOT = _project_root_from_this_file()
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _touch_init_py(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    init_py = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_py):
        with open(init_py, "w", encoding="utf-8") as f:
            f.write("# auto-created\n")


def _ensure_downstream_packages() -> None:
    _touch_init_py(os.path.join(_PROJ_ROOT, "downstream"))
    _touch_init_py(os.path.join(_PROJ_ROOT, "downstream", "finetune_mask2former"))


def _load_cfg(cfg_path: str) -> Any:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cfg_get(cfg: Any, key: str, default=None):
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if cur is None:
            return default
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _try_candidates(cands: List[str]) -> Optional[str]:
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _resolve_path_relative(cfg_path: str, p: str) -> str:
    if p is None:
        return ""
    p = str(p).strip()
    if p == "":
        return ""
    if os.path.isabs(p):
        return _abspath(p)
    cfg_dir = _abspath(os.path.dirname(cfg_path))
    cand_cfg = _abspath(os.path.join(cfg_dir, p))
    if os.path.exists(cand_cfg):
        return cand_cfg
    return _abspath(os.path.join(_PROJ_ROOT, p))


def _resolve_ckpt_path(cfg_path: str, ckpt_raw: str) -> str:
    if ckpt_raw is None or str(ckpt_raw).strip() == "":
        return ""
    ckpt = _resolve_path_relative(cfg_path, ckpt_raw)
    if os.path.isdir(ckpt):
        return _try_candidates([os.path.join(ckpt, "best.pth"), os.path.join(ckpt, "last.pth")]) or ""
    if os.path.isfile(ckpt):
        return ckpt
    base_dir = _abspath(os.path.dirname(ckpt))
    return _try_candidates([os.path.join(base_dir, "best.pth"), os.path.join(base_dir, "last.pth")]) or ""


def _is_subpath(path: str, root: str) -> bool:
    path_abs = _abspath(path)
    root_abs = _abspath(root)
    try:
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except Exception:
        return False


def _ensure_external_output_dir(cfg: Any, cfg_path: str, out_dir: str) -> str:
    out_dir = _abspath(out_dir)
    if not _is_subpath(out_dir, _PROJ_ROOT):
        return out_dir

    ext_root = (
        _cfg_get(cfg, "train.external_output_root")
        or _cfg_get(cfg, "experiment.external_output_root")
        or os.environ.get("SHELFMIM_OUTPUT_ROOT")
        or _abspath(os.path.join(_PROJ_ROOT, "..", "shelfmim_runs"))
    )
    run_name = _cfg_get(cfg, "experiment.name") or os.path.splitext(os.path.basename(cfg_path))[0]
    new_out = _abspath(os.path.join(str(ext_root), str(run_name), "finetune"))
    print(f"[output] redirect output_dir: {out_dir} -> {new_out}")
    return new_out


def main(cfg_path: str) -> None:
    logging.getLogger("mmengine").setLevel(logging.WARNING)
    logging.getLogger("mmdet").setLevel(logging.WARNING)

    print(f"[run_finetune] project_root: {_PROJ_ROOT}")
    print(f"[run_finetune] using cfg: {cfg_path}")

    cfg = _load_cfg(cfg_path)

    cfg.setdefault("paths", {})
    cfg["paths"]["project_root"] = _PROJ_ROOT
    cfg["paths"].pop("data_root", None)

    out_dir = _cfg_get(cfg, "experiment.output_dir")
    if not out_dir:
        out_dir = os.path.join(_PROJ_ROOT, "scripts", "outputs", "finetune", "rpc")
    out_dir = _ensure_external_output_dir(cfg, cfg_path, out_dir)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    base_cfg = _cfg_get(cfg, "paths.mask2former_base_config") or _cfg_get(cfg, "model.mmdet_base_config")
    if not base_cfg:
        raise ValueError("Base config is required (paths.mask2former_base_config).")
    cfg["paths"]["mask2former_base_config"] = _resolve_path_relative(cfg_path, base_cfg)

    ckpt_raw = _cfg_get(cfg, "paths.pretrained_backbone_ckpt", "") or _cfg_get(cfg, "model.pretrained_backbone_ckpt", "")
    ckpt = _resolve_ckpt_path(cfg_path, ckpt_raw)
    if not ckpt or (not os.path.isfile(ckpt)):
        raise FileNotFoundError(f"pretrained_backbone_ckpt not found: {ckpt_raw}")
    cfg["paths"]["pretrained_backbone_ckpt"] = ckpt

    _ensure_downstream_packages()
    from downstream.finetune_mask2former.finetune_engine import FinetuneEngine

    device = _cfg_get(cfg, "device", "cpu")
    engine = FinetuneEngine(cfg=cfg, output_dir=out_dir, device=device)
    engine.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/finetune_rpc_mask2former.yaml")
    args = parser.parse_args()

    cfg_path = args.cfg.strip()
    if not os.path.isabs(cfg_path):
        cfg_path = _abspath(os.path.join(_PROJ_ROOT, cfg_path))
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config yaml not found: {cfg_path}")

    main(cfg_path)
