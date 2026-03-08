#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click pipeline runner:
pretrain -> finetune -> 3-seed test(mean/std)

Designed for server setup:
- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1
- Single GPU
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, cfg: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("[pipeline] run:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _maybe_override(cfg: dict, key_path: list[str], value) -> None:
    cur = cfg
    for k in key_path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[key_path[-1]] = value


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def _extract_epoch_num(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pth$", path.name)
    if not m:
        return -1
    return int(m.group(1))


def _cleanup_epoch_ckpts(work_dir: Path, keep_last_n_epoch_ckpt: int) -> None:
    keep_n = max(0, int(keep_last_n_epoch_ckpt))
    ckpts = sorted(
        [p for p in work_dir.glob("epoch_*.pth") if p.is_file()],
        key=lambda p: (_extract_epoch_num(p), p.stat().st_mtime),
    )
    if len(ckpts) <= keep_n:
        return

    to_delete = ckpts[: len(ckpts) - keep_n]
    for p in to_delete:
        try:
            p.unlink()
            print(f"[cleanup] removed ckpt: {p}")
        except OSError as e:
            print(f"[cleanup][warn] cannot remove {p}: {e}")


def _cleanup_vis(work_dir: Path, keep_last_n_vis: int) -> None:
    keep_n = max(0, int(keep_last_n_vis))
    vis_dir = work_dir / "vis"
    proto_dir = work_dir / "proto_vis"

    for d in [vis_dir, proto_dir]:
        if not d.is_dir():
            continue
        imgs = sorted([p for p in d.glob("*.png") if p.is_file()], key=lambda p: p.stat().st_mtime)
        if len(imgs) <= keep_n:
            continue
        for p in imgs[: len(imgs) - keep_n]:
            try:
                p.unlink()
                print(f"[cleanup] removed vis: {p}")
            except OSError as e:
                print(f"[cleanup][warn] cannot remove {p}: {e}")


def _cleanup_run_outputs(
    *,
    pretrain_out: Path,
    finetune_out: Path,
    enabled: bool,
    keep_last_n_epoch_ckpt: int,
    keep_last_n_vis: int,
) -> None:
    if not enabled:
        print("[cleanup] disabled")
        return

    print("[cleanup] start")
    _cleanup_epoch_ckpts(pretrain_out, keep_last_n_epoch_ckpt)
    _cleanup_vis(pretrain_out, keep_last_n_vis)
    _cleanup_epoch_ckpts(finetune_out, keep_last_n_epoch_ckpt)
    _cleanup_vis(finetune_out, keep_last_n_vis)
    print("[cleanup] done")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-click ShelfMIM pipeline")
    ap.add_argument("--project-root", default=".", help="Repository root")
    ap.add_argument("--python", default=sys.executable, help="Python executable")

    ap.add_argument("--pretrain-cfg", default="configs/pretrain_rpc.yaml")
    ap.add_argument("--finetune-cfg", default="configs/finetune_rpc_mask2former.yaml")

    ap.add_argument("--data-root", required=True, help="Dataset root containing train/val/test splits")
    ap.add_argument("--output-root", default="outputs/oneclick_rpc", help="Output root for pipeline")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES id")

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--test-seeds", default="0,1,2")

    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--skip-test", action="store_true")

    ap.add_argument("--min-free-gb", type=float, default=8.0, help="Minimum required free disk before running")

    # cleanup switch
    ap.add_argument("--cleanup", dest="cleanup_enabled", action="store_true", default=True)
    ap.add_argument("--no-cleanup", dest="cleanup_enabled", action="store_false")
    ap.add_argument("--keep-last-n-epoch-ckpt", type=int, default=2)
    ap.add_argument("--keep-last-n-vis", type=int, default=2)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)

    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    free_gb = _disk_free_gb(output_root)
    print(f"[pipeline] disk free at output_root: {free_gb:.2f} GB")
    if free_gb < args.min_free_gb:
        raise SystemExit(f"Insufficient free disk ({free_gb:.2f} GB < {args.min_free_gb:.2f} GB)")

    pretrain_base = (project_root / args.pretrain_cfg).resolve()
    finetune_base = (project_root / args.finetune_cfg).resolve()

    if not pretrain_base.is_file():
        raise SystemExit(f"pretrain cfg not found: {pretrain_base}")
    if not finetune_base.is_file():
        raise SystemExit(f"finetune cfg not found: {finetune_base}")

    pretrain_cfg = _load_yaml(pretrain_base)
    finetune_cfg = _load_yaml(finetune_base)

    pretrain_out = output_root / "pretrain"
    finetune_out = output_root / "finetune"
    run_cfg_dir = output_root / "run_cfg"

    _maybe_override(pretrain_cfg, ["data", "root"], str(data_root))
    _maybe_override(pretrain_cfg, ["experiment", "output_dir"], str(pretrain_out))
    _maybe_override(pretrain_cfg, ["runtime", "device"], "cuda")
    _maybe_override(pretrain_cfg, ["train", "num_workers"], int(args.num_workers))
    _maybe_override(pretrain_cfg, ["train", "save_interval_epochs"], 10)
    _maybe_override(pretrain_cfg, ["train", "vis_epoch_interval"], 10)
    _maybe_override(pretrain_cfg, ["train", "proto_vis_epoch_interval"], 10)
    _maybe_override(
        pretrain_cfg,
        ["cleanup"],
        {
            "enabled": bool(args.cleanup_enabled),
            "keep_last_n_epoch_ckpt": int(args.keep_last_n_epoch_ckpt),
            "keep_last_n_vis": int(args.keep_last_n_vis),
            "keep_best_last": True,
        },
    )
    if args.pretrain_epochs is not None:
        _maybe_override(pretrain_cfg, ["train", "epochs"], int(args.pretrain_epochs))

    _maybe_override(finetune_cfg, ["data", "root"], str(data_root))
    _maybe_override(finetune_cfg, ["experiment", "output_dir"], str(finetune_out))
    _maybe_override(finetune_cfg, ["device"], str(args.device))
    _maybe_override(finetune_cfg, ["train", "num_workers"], int(args.num_workers))
    _maybe_override(finetune_cfg, ["paths", "pretrained_backbone_ckpt"], str(pretrain_out / "last.pth"))
    _maybe_override(finetune_cfg, ["test", "seeds"], [int(x.strip()) for x in args.test_seeds.split(",") if x.strip()])
    _maybe_override(
        finetune_cfg,
        ["cleanup"],
        {
            "enabled": bool(args.cleanup_enabled),
            "keep_last_n_epoch_ckpt": int(args.keep_last_n_epoch_ckpt),
            "keep_last_n_vis": int(args.keep_last_n_vis),
            "keep_best_last": True,
        },
    )
    if args.finetune_epochs is not None:
        _maybe_override(finetune_cfg, ["train", "epochs"], int(args.finetune_epochs))

    pretrain_cfg_path = run_cfg_dir / "pretrain.auto.yaml"
    finetune_cfg_path = run_cfg_dir / "finetune.auto.yaml"
    _save_yaml(pretrain_cfg_path, pretrain_cfg)
    _save_yaml(finetune_cfg_path, finetune_cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if not args.skip_pretrain:
        _run([args.python, "scripts/run_pretrain.py", "--cfg", str(pretrain_cfg_path)], cwd=project_root, env=env)

    if not args.skip_finetune:
        if not (pretrain_out / "last.pth").is_file() and not args.skip_pretrain:
            raise SystemExit(f"Expected pretrain checkpoint missing: {pretrain_out / 'last.pth'}")
        _run([args.python, "scripts/run_finetune.py", "--cfg", str(finetune_cfg_path)], cwd=project_root, env=env)

    _cleanup_run_outputs(
        pretrain_out=pretrain_out,
        finetune_out=finetune_out,
        enabled=bool(args.cleanup_enabled),
        keep_last_n_epoch_ckpt=int(args.keep_last_n_epoch_ckpt),
        keep_last_n_vis=int(args.keep_last_n_vis),
    )

    if not args.skip_test:
        summary_json = output_root / "test_mean_std.json"
        _run(
            [
                args.python,
                "scripts/test.py",
                "--cfg",
                str(finetune_cfg_path),
                "--work-dir",
                str(finetune_out),
                "--device",
                str(args.device),
                "--seeds",
                str(args.test_seeds),
                "--out-json",
                str(summary_json),
            ],
            cwd=project_root,
            env=env,
        )

    print("[pipeline] done")
    print(f"[pipeline] pretrain dir: {pretrain_out}")
    print(f"[pipeline] finetune dir: {finetune_out}")
    print(f"[pipeline] test summary: {output_root / 'test_mean_std.json'}")


if __name__ == "__main__":
    main()
