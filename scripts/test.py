# -*- coding: utf-8 -*-
"""
scripts/test.py

Default behavior:
- Resolve inputs from finetune yaml.
- Resolve checkpoint (prefer best.pth then last.pth when a work dir is passed).
- Run three-seed evaluation on the same best checkpoint and report mean ± std.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Optional, List

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import models.backbone.mmshelfmim_vit  # noqa: F401,E402

from eval.run_eval import EvalConfig, run_eval  # noqa: E402
from eval.seed_runner import run_seeds  # noqa: E402


def _abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
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


def _list_pth_recursively(root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".pth"):
                out.append(os.path.join(dp, fn))
    return out


def _pick_latest_by_mtime(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def resolve_ckpt_path(ckpt_or_dir: str) -> str:
    p = _abspath(ckpt_or_dir)

    if os.path.isdir(p):
        best = os.path.join(p, "best.pth")
        last = os.path.join(p, "last.pth")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last

        common = [
            os.path.join(p, "latest.pth"),
            os.path.join(p, "latest_checkpoint.pth"),
            os.path.join(p, "checkpoint.pth"),
            os.path.join(p, "model.pth"),
            os.path.join(p, "final.pth"),
        ]
        for c in common:
            if os.path.isfile(c):
                return c

        all_pth = _list_pth_recursively(p)
        picked = _pick_latest_by_mtime(all_pth)
        if picked:
            print(f"[test] WARN: best/last not found. Using latest modified checkpoint: {picked}")
            return picked

        raise FileNotFoundError(
            f"No checkpoint found under: {p}. "
            f"Tried best.pth/last.pth/common names and recursive *.pth search."
        )

    if os.path.isfile(p):
        return p

    raise FileNotFoundError(f"Checkpoint path not found: {p}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ShelfMIM finetune test/eval runner.")

    ap.add_argument(
        "--cfg",
        default="configs/finetune_rpc_mask2former.yaml",
        help="Finetune yaml to take defaults from",
    )

    ap.add_argument("--base-cfg", dest="base_cfg_py", default="", help="MMDet config .py used for inference.")
    ap.add_argument("--ann", dest="ann_file", default="", help="COCO-format annotation json.")
    ap.add_argument("--img-dir", dest="img_dir", default="", help="Image directory for evaluation.")

    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--ckpt", dest="checkpoint", default="", help="Checkpoint .pth file OR a work_dir")
    grp.add_argument("--work-dir", dest="work_dir", default="", help="Work directory containing checkpoints")

    ap.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    ap.add_argument("--score-thr", type=float, default=0.05, help="Score threshold")
    ap.add_argument("--max-per-img", type=int, default=100, help="Max predictions per image")
    ap.add_argument("--iou-match-thr", type=float, default=0.5, help="IoU threshold for matching")

    ap.add_argument("--skip-crowd", action="store_true", help="Skip iscrowd==1 GT annotations")
    ap.add_argument("--no-skip-crowd", dest="skip_crowd", action="store_false")
    ap.set_defaults(skip_crowd=True)

    ap.add_argument("--empty-penalty", type=float, default=0.0)
    ap.add_argument("--out-json", default=None, help="Optional path to save json")

    ap.add_argument("--ignore-labels", action="store_true", help="Ignore class labels and match globally")
    ap.add_argument("--single-run", action="store_true", help="Run one eval only (no seed aggregation)")
    ap.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds for multi-seed test, e.g. 0,1,2. Empty -> read from cfg or default 0,1,2",
    )

    return ap.parse_args()


def _resolve_default_paths_from_yaml(cfg_path: str) -> dict:
    cfg_abs = _abspath(os.path.join(_PROJ_ROOT, cfg_path)) if not os.path.isabs(cfg_path) else _abspath(cfg_path)
    cfg = _load_yaml(cfg_abs)

    base_cfg = _cfg_get(cfg, "paths.mask2former_base_config", "") or _cfg_get(cfg, "model.mmdet_base_config", "")
    ann = _cfg_get(cfg, "paths.eval_ann_file", "") or _cfg_get(cfg, "data.eval_ann_file", "")
    img_dir = _cfg_get(cfg, "paths.eval_img_dir", "") or _cfg_get(cfg, "data.eval_img_dir", "")

    data_root = _cfg_get(cfg, "data.root", "")
    if not ann:
        ann = _cfg_get(cfg, "data.splits.test.ann_file", "")
        if ann and data_root:
            ann = os.path.join(data_root, ann)
    if not img_dir:
        img_dir = _cfg_get(cfg, "data.splits.test.image_dir", "")
        if img_dir and data_root:
            img_dir = os.path.join(data_root, img_dir)

    out_dir = _cfg_get(cfg, "experiment.output_dir", "")

    cfg_seeds = _cfg_get(cfg, "test.seeds", None)
    if not cfg_seeds:
        cfg_seeds = _cfg_get(cfg, "experiment.seeds", None)

    def _make_abs(p: str) -> str:
        if not p:
            return ""
        return _abspath(os.path.join(_PROJ_ROOT, p)) if not os.path.isabs(p) else _abspath(p)

    return {
        "cfg_path": cfg_abs,
        "base_cfg_py": _make_abs(base_cfg),
        "ann_file": _make_abs(ann),
        "img_dir": _make_abs(img_dir),
        "default_work_dir": _make_abs(out_dir),
        "cfg_seeds": cfg_seeds,
    }


def _parse_seeds(seed_text: str, cfg_seeds: Any) -> List[int]:
    if seed_text.strip():
        return [int(x.strip()) for x in seed_text.split(",") if x.strip()]
    if isinstance(cfg_seeds, list) and len(cfg_seeds) > 0:
        return [int(x) for x in cfg_seeds]
    return [0, 1, 2]


def _safe_write_json(path: str, obj: Any):
    out_path = _abspath(path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    defaults = _resolve_default_paths_from_yaml(args.cfg)

    base_cfg_py = args.base_cfg_py.strip() or defaults["base_cfg_py"]
    ann_file = args.ann_file.strip() or defaults["ann_file"]
    img_dir = args.img_dir.strip() or defaults["img_dir"]

    ckpt_in = ""
    if args.checkpoint:
        ckpt_in = args.checkpoint
    elif args.work_dir:
        ckpt_in = args.work_dir
    elif defaults["default_work_dir"]:
        ckpt_in = defaults["default_work_dir"]

    missing = []
    if not base_cfg_py:
        missing.append("--base-cfg (or set paths.mask2former_base_config in yaml)")
    if not ann_file:
        missing.append("--ann (or set data.splits.test.ann_file in yaml)")
    if not img_dir:
        missing.append("--img-dir (or set data.splits.test.image_dir in yaml)")
    if not ckpt_in:
        missing.append("--ckpt/--work-dir (or set experiment.output_dir in yaml)")
    if missing:
        raise SystemExit("Missing required inputs:\n  - " + "\n  - ".join(missing))

    ckpt_path = resolve_ckpt_path(ckpt_in)

    if args.single_run:
        ecfg = EvalConfig(
            base_cfg_py=base_cfg_py,
            checkpoint=ckpt_path,
            ann_file=ann_file,
            img_dir=img_dir,
            device=args.device,
            score_thr=args.score_thr,
            max_per_img=args.max_per_img,
            iou_match_thr=args.iou_match_thr,
            skip_crowd=args.skip_crowd,
            empty_penalty=args.empty_penalty,
            save_json=args.out_json,
            ignore_labels=bool(args.ignore_labels),
        )
        print("[test] finetune_yaml:", defaults["cfg_path"])
        print("[test] EvalConfig:", json.dumps(asdict(ecfg), ensure_ascii=False, indent=2))

        res = run_eval(ecfg)
        print("\n[test] Results:")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    seeds = _parse_seeds(args.seeds, defaults["cfg_seeds"])
    out_dir = args.work_dir if args.work_dir else os.path.dirname(ckpt_path)
    out_dir = _abspath(out_dir)
    seed_out_dir = os.path.join(out_dir, "test_seed_runs")

    print("[test] finetune_yaml:", defaults["cfg_path"])
    print(f"[test] checkpoint={ckpt_path}")
    print(f"[test] seeds={seeds}")

    summary = run_seeds(
        seeds=seeds,
        base_cfg_py=base_cfg_py,
        ann_file=ann_file,
        img_dir=img_dir,
        checkpoint=ckpt_path,
        device=args.device,
        out_dir=seed_out_dir,
        score_thr=args.score_thr,
        max_per_img=args.max_per_img,
        iou_match_thr=args.iou_match_thr,
        skip_crowd=args.skip_crowd,
        empty_penalty=args.empty_penalty,
        ignore_labels=bool(args.ignore_labels),
    )

    compact = {
        "dice": summary["dice"],
        "iou": summary["iou"],
        "asd": summary["asd"],
        "hd95": summary["hd95"],
        "seeds": summary["seeds"],
        "summary_file": os.path.join(seed_out_dir, "summary_mean_std.json"),
    }

    if args.out_json:
        _safe_write_json(args.out_json, summary)
        compact["out_json"] = _abspath(args.out_json)

    print("\n[test] Results (mean ± std over seeds):")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
