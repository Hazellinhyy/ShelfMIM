from __future__ import annotations

import csv
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.collate import collate_ssl_two_view
from data.datasets import D2SSSL, RPCSSL
from data.transforms import (
    AppearanceAugment,
    GeometryAugmentWithHomography,
    PadRightBottomToDivisor,
    ResizeKeepARShortLong,
    TwoViewSSLTransform,
)
from train.checkpoint import CheckpointManager
from train.optim import WarmupCosineLR, build_adamw, build_amp_scaler, compute_total_steps
from train.pretrain_engine import PretrainEngine, ShelfMIMPretrainModel
from utils.misc import set_seed


def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _is_subpath(path: str, root: str) -> bool:
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    try:
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except Exception:
        return False


def _ensure_external_output_dir(cfg: dict, cfg_path: str) -> None:
    exp_cfg = cfg.setdefault("experiment", {})
    out_dir = exp_cfg.get("output_dir", "outputs/pretrain")
    if not isinstance(out_dir, str) or not out_dir.strip():
        out_dir = "outputs/pretrain"
    out_dir_abs = os.path.abspath(out_dir)

    if not _is_subpath(out_dir_abs, _PROJECT_ROOT):
        exp_cfg["output_dir"] = out_dir_abs
        return

    train_cfg = cfg.setdefault("train", {})
    base_root = (
        train_cfg.get("external_output_root")
        or exp_cfg.get("external_output_root")
        or os.environ.get("SHELFMIM_OUTPUT_ROOT")
        or os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "shelfmim_runs"))
    )
    run_name = (
        exp_cfg.get("name")
        or os.path.splitext(os.path.basename(cfg_path))[0]
        or "pretrain"
    )
    new_out = os.path.abspath(os.path.join(str(base_root), run_name))
    exp_cfg["output_dir"] = new_out
    print(f"[output] redirect output_dir: {out_dir_abs} -> {new_out}")


def _apply_small_data_autotune(cfg: dict, dataset_len: int):
    if int(dataset_len) <= 0 or int(dataset_len) >= 1000:
        return

    model_cfg = cfg.setdefault("model", {})
    slot_cfg = model_cfg.setdefault("branch_i_slot", {})
    slot_cfg.setdefault("slot_attention", {})

    proto_cfg = model_cfg.setdefault("branch_ii_proto", {})
    p_cfg = proto_cfg.setdefault("prototypes", {})
    a_cfg = proto_cfg.setdefault("assignment", {})

    loss_cfg = cfg.setdefault("loss", {})
    b3_cfg = model_cfg.setdefault("branch_iii_cs_hps", {})
    mask_cfg = b3_cfg.setdefault("masking", {})
    train_cfg = cfg.setdefault("train", {})

    old_slots_k = int(slot_cfg.get("slots_k", 7))
    old_kg = int(p_cfg.get("kg", 4096))
    old_kp = int(p_cfg.get("kp", 1024))
    old_tau = float(a_cfg.get("temperature_tau_p", 0.1))
    old_bal = float(loss_cfg.get("lambda_bal", 0.0))
    old_lambda_proto = float(loss_cfg.get("lambda_proto", 0.10))
    old_lambda_cs = float(loss_cfg.get("lambda_cs_hps", 0.30))
    old_cons = float(loss_cfg.get("lambda_consistency", 0.0))
    old_sharp = float(loss_cfg.get("lambda_sharp", 0.03))
    old_geom = float(loss_cfg.get("lambda_geom", 0.25))
    old_mim_vis = float(loss_cfg.get("lambda_mim_visible", 0.10))
    old_mask_ratio = float(mask_cfg.get("ratio_r", 0.25))
    old_drop_last = bool(train_cfg.get("drop_last", False))

    slot_cfg["slots_k"] = int(min(old_slots_k, 3))
    p_cfg["kg"] = int(min(old_kg, 64))
    p_cfg["kp"] = int(min(old_kp, 32))
    a_cfg["temperature_tau_p"] = float(min(old_tau, 0.07))

    # Keep proto/cs as auxiliary branches under small-data regime.
    loss_cfg["auto_scale_proto"] = False
    loss_cfg["auto_scale_cs"] = False
    loss_cfg["lambda_proto"] = float(min(old_lambda_proto, 0.10))
    loss_cfg["lambda_cs_hps"] = float(min(old_lambda_cs, 0.30))
    loss_cfg["lambda_bal"] = 0.0
    loss_cfg["lambda_consistency"] = 0.0
    loss_cfg["lambda_sharp"] = float(min(old_sharp, 0.03))
    loss_cfg["lambda_geom"] = float(min(old_geom, 0.25))
    loss_cfg["lambda_mim_visible"] = float(min(old_mim_vis, 0.10))

    # Keep masking ratio moderate to reduce noisy reconstruction pressure.
    mask_cfg["ratio_r"] = float(max(min(old_mask_ratio, 0.50), 0.30))
    mask_cfg["ratio_r_min"] = float(min(mask_cfg["ratio_r"], 0.10))
    train_cfg["drop_last"] = True

    print(
        "[autotune][small-data] "
        f"N={dataset_len}, slots {old_slots_k}->{slot_cfg['slots_k']}, "
        f"kg {old_kg}->{p_cfg['kg']}, kp {old_kp}->{p_cfg['kp']}, "
        f"tau_p {old_tau:.4f}->{a_cfg['temperature_tau_p']:.4f}, "
        f"lambda_proto {old_lambda_proto:.2f}->{loss_cfg['lambda_proto']:.2f}, "
        f"lambda_cs_hps {old_lambda_cs:.2f}->{loss_cfg['lambda_cs_hps']:.2f}, "
        f"lambda_consistency {old_cons:.3f}->{loss_cfg['lambda_consistency']:.3f}, "
        f"lambda_sharp {old_sharp:.3f}->{loss_cfg['lambda_sharp']:.3f}, "
        f"lambda_geom {old_geom:.2f}->{loss_cfg['lambda_geom']:.2f}, "
        f"lambda_mim_visible {old_mim_vis:.2f}->{loss_cfg['lambda_mim_visible']:.2f}, "
        f"mask_ratio {old_mask_ratio:.2f}->{mask_cfg['ratio_r']:.2f}, "
        f"drop_last {old_drop_last}->{train_cfg['drop_last']}"
    )


def _append_csv_row(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_ssl_transform(cfg: dict):
    return TwoViewSSLTransform(
        resize=ResizeKeepARShortLong(
            short_side=cfg["input"]["resize"]["short_side"],
            long_side_max=cfg["input"]["resize"]["long_side_max"],
        ),
        pad=PadRightBottomToDivisor(
            divisor=cfg["input"]["pad"]["size_divisor"],
            pad_value=cfg["input"]["pad"]["value"],
        ),
        view_a=AppearanceAugment(
            brightness=cfg["augmentation"]["view_a"]["color_jitter"]["brightness"],
            contrast=cfg["augmentation"]["view_a"]["color_jitter"]["contrast"],
            saturation=cfg["augmentation"]["view_a"]["color_jitter"]["saturation"],
            hue=cfg["augmentation"]["view_a"]["color_jitter"]["hue"],
            grayscale_prob=cfg["augmentation"]["view_a"]["grayscale_prob"],
            blur_prob=cfg["augmentation"]["view_a"]["gaussian_blur_prob"],
            hflip_prob=cfg["augmentation"]["view_a"]["horizontal_flip_prob"],
        ),
        view_b=GeometryAugmentWithHomography(
            use_rrc=cfg["augmentation"]["view_b"]["random_resized_crop"]["enabled"],
            rrc_scale=tuple(cfg["augmentation"]["view_b"]["random_resized_crop"]["scale"]),
            rrc_ratio=tuple(cfg["augmentation"]["view_b"]["random_resized_crop"]["ratio"]),
            use_affine=bool(cfg["augmentation"]["view_b"].get("random_affine", {}).get("enabled", False)),
            affine_degrees=float(cfg["augmentation"]["view_b"].get("random_affine", {}).get("degrees", 8.0)),
            affine_translate=tuple(cfg["augmentation"]["view_b"].get("random_affine", {}).get("translate", [0.06, 0.06])),
            affine_scale=tuple(cfg["augmentation"]["view_b"].get("random_affine", {}).get("scale", [0.95, 1.05])),
            affine_prob=float(cfg["augmentation"]["view_b"].get("random_affine", {}).get("prob", 0.3)),
            use_perspective=cfg["augmentation"]["view_b"]["random_perspective"]["enabled"],
            perspective_distortion=cfg["augmentation"]["view_b"]["random_perspective"]["distortion_scale"],
            perspective_prob=cfg["augmentation"]["view_b"]["random_perspective"]["prob"],
            hflip_prob=cfg["augmentation"]["view_b"]["horizontal_flip_prob"],
        ),
        mean=tuple(cfg["input"]["normalize"]["mean"]),
        std=tuple(cfg["input"]["normalize"]["std"]),
    )


def build_dataset(cfg: dict, transform):
    ds_name = cfg["data"]["dataset"].lower()
    root = cfg["data"]["root"]
    split = cfg["data"]["split"]["pretrain"]

    if ds_name == "rpc":
        return RPCSSL(root=root, split=split, transform=transform)
    if ds_name in ["mvtec_d2s", "d2s"]:
        return D2SSSL(root=root, transform=transform)
    raise ValueError(f"Unknown dataset: {ds_name}")


def _resolve_default_cfg(user_cfg_path: str | None) -> str:
    if user_cfg_path is not None:
        return user_cfg_path

    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "configs", "pretrain_rpc.yaml"),
        os.path.join(os.path.dirname(here), "configs", "pretrain_rpc.yaml"),
        os.path.join(os.getcwd(), "configs", "pretrain_rpc.yaml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Cannot locate default config 'configs/pretrain_rpc.yaml'. "
        "Please pass --cfg explicitly."
    )


def main(
    cfg_path: str,
    *,
    disable_autotune: bool = False,
    proto_ablation_mode: str | None = None,
):
    cfg = load_yaml(cfg_path)
    _ensure_external_output_dir(cfg, cfg_path)
    set_seed(cfg["experiment"]["seed"])

    cfg.setdefault("train", {})
    cfg["train"]["drop_last"] = bool(cfg["train"].get("drop_last", True))
    if proto_ablation_mode:
        cfg.setdefault("loss", {})["proto_ablation_mode"] = str(proto_ablation_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.get("train", {}).get("amp", {}).get("enabled", False)) and device.type == "cuda"

    transform = build_ssl_transform(cfg)
    dataset = build_dataset(cfg, transform)

    autotune_cfg = bool(cfg.get("train", {}).get("autotune", {}).get("enabled", False))
    use_autotune = autotune_cfg and (not disable_autotune)
    if use_autotune:
        _apply_small_data_autotune(cfg, len(dataset))
    else:
        print("[autotune] disabled")

    num_workers = int(cfg["train"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size_total"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_ssl_two_view,
        drop_last=bool(cfg["train"].get("drop_last", True)),
    )
    print("len(dataset)=", len(dataset), "len(loader)=", len(loader))
    steps_per_epoch = len(loader)

    model = ShelfMIMPretrainModel(cfg)
    model.to(device)

    optimizer = build_adamw(
        model,
        base_lr=cfg["optimizer"]["base_lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )

    total_steps = compute_total_steps(cfg["train"]["epochs"], steps_per_epoch)
    warmup_steps = int(cfg["scheduler"]["warmup_epochs"] * steps_per_epoch)
    scheduler = WarmupCosineLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=cfg["optimizer"]["base_lr"],
        min_lr=cfg["scheduler"]["min_lr"],
    )

    scaler = build_amp_scaler(amp_enabled)
    engine = PretrainEngine(cfg, model, optimizer, scheduler, scaler, device)

    out_dir = cfg["experiment"]["output_dir"]
    ckpt = CheckpointManager(out_dir)
    csv_path = os.path.join(out_dir, "train_log.csv")

    resume_path = cfg.get("train", {}).get("resume", None)
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    best_epoch = 0
    if resume_path:
        info = ckpt.load(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
            strict=True,
            load_rng=True,
        )
        start_epoch = int(info.get("epoch", 0))
        global_step = int(info.get("step", 0))
        extra = info.get("extra", {}) or {}
        best_metric = _safe_float(extra.get("best_metric", float("inf")), float("inf"))
        best_epoch = int(extra.get("best_epoch", 0))
        print(f"[resume] loaded from {resume_path}, epoch={start_epoch}, step={global_step}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        stats = engine.train_one_epoch(loader, epoch, start_step=global_step)
        global_step += steps_per_epoch
        print(f"[epoch {epoch + 1}] {stats}")

        row = {"epoch": epoch + 1, "step": global_step}
        row.update({k: float(v) for k, v in stats.items()})
        _append_csv_row(csv_path, row)

        metric = _safe_float(stats.get("loss_total_opt", stats.get("loss_total", float("inf"))), float("inf"))
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            ckpt.save(
                name="best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                step=global_step,
                extra={"stats": stats, "best_metric": best_metric, "best_epoch": best_epoch},
            )
            print(f"[best] epoch={best_epoch}, loss_total_opt={best_metric:.6f}")

        if (epoch + 1) % int(cfg["train"].get("save_interval_epochs", 1)) == 0:
            ckpt.save(
                name=f"epoch_{epoch + 1}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                step=global_step,
                extra={"stats": stats, "best_metric": best_metric, "best_epoch": best_epoch},
            )

    ckpt.save(
        name="last.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=cfg["train"]["epochs"],
        step=global_step,
        extra={"best_metric": best_metric, "best_epoch": best_epoch},
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None, help="Path to pretrain yaml (default: auto detect configs/pretrain_rpc.yaml)")
    ap.add_argument("--no-autotune", action="store_true", help="Disable small-data autotune.")
    ap.add_argument(
        "--proto-ablation",
        default=None,
        choices=["both", "global-only", "part-only"],
        help="Prototype branch ablation mode override.",
    )
    args = ap.parse_args()
    args.cfg = _resolve_default_cfg(args.cfg)
    print(f"[run_pretrain] using cfg: {args.cfg}")
    main(
        args.cfg,
        disable_autotune=bool(args.no_autotune),
        proto_ablation_mode=args.proto_ablation,
    )
