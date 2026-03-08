# -*- coding: utf-8 -*-
"""
train/pretrain_engine.py

- Always push proto points each step
- Save proto visualization every N epochs (default 5)
- Compatible with NEW utils/proto_vis.py (cluster-faithful view)
  NOTE: plot_proto_pca_2x2() has NO 'color_mod' argument in that version.

If you still want the old "mod-10 color" style, use the older proto_vis.py.
"""

from __future__ import annotations
from typing import Dict, Any
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.vit import vit_base_patch16
from models.branch_slot.slot_attention import SlotAttention
from models.branch_slot.warp import warp_quality_metrics
from models.branch_proto.prototype_bank import MultiPrototypeBank
from models.branch_proto.proto_loss import ProtoLoss
from models.branch_cs_hps.heads import ContentStyleHeads
from models.branch_cs_hps.policy import PatchImportancePolicy
from models.branch_cs_hps.gumbel_topk import gumbel_topk_st
from models.branch_cs_hps.decoder import CSFiLMDecoder, apply_mask_token
from models.branch_cs_hps.cs_hps_loss import CSHPSLoss

from models.backbone.position_embed import patchify, unpatchify
from losses.total_loss import compute_branch_i_slot_losses, assemble_total_loss, TotalLossWeights

# NEW proto_vis signature (no color_mod)
from utils.proto_vis import plot_proto_pca_2x2, ProtoVisBuffer
from utils.vis import save_slot_grid_ab, save_recon_triplets


def cfg_get(cfg: Any, key: str, default=None):
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if cur is None:
            return default
        if hasattr(cur, p):
            cur = getattr(cur, p)
            continue
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
            continue
        return default
    return cur


def _pixel_valid_to_patch(valid_pix: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    inv = 1.0 - valid_pix.float()
    inv_p = F.max_pool2d(inv, kernel_size=patch_size, stride=patch_size)
    return 1.0 - inv_p


def _patch_valid_flat(valid_patch: torch.Tensor) -> torch.Tensor:
    B, _, Hp, Wp = valid_patch.shape
    return valid_patch.reshape(B, Hp * Wp)


def _align_by_pi(z_b: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    B, K, D = z_b.shape
    idx = pi.unsqueeze(-1).expand(B, K, D)
    return torch.gather(z_b, dim=1, index=idx)


def _try_get(d: Dict[str, Any], *keys: str):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class ShelfMIMPretrainModel(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.patch_size = int(cfg_get(cfg, "model.backbone.patch_size", 16))

        drop_path = float(cfg_get(cfg, "model.backbone.drop_path_rate", 0.1))
        self.backbone = vit_base_patch16(use_cls_token=True, drop_path_rate=drop_path)

        dim = 768

        # Branch I
        K = int(cfg_get(cfg, "model.branch_i_slot.slots_k", 8))
        iters = int(cfg_get(cfg, "model.branch_i_slot.slot_attention.iters", 3))
        eps = float(cfg_get(cfg, "model.branch_i_slot.slot_attention.epsilon", 1e-6))
        slot_temp = float(cfg_get(cfg, "model.branch_i_slot.slot_attention.temperature", 0.6))
        self.slot_attn = SlotAttention(num_slots=K, dim=dim, iters=iters, epsilon=eps, slot_temperature=slot_temp)

        # Branch II (proto)
        kg = int(cfg_get(cfg, "model.branch_ii_proto.prototypes.kg", 8192))
        kp = int(cfg_get(cfg, "model.branch_ii_proto.prototypes.kp", 2048))
        m = float(cfg_get(cfg, "model.branch_ii_proto.ema.prototype_momentum_m", 0.95))  # suggested for bs=1
        logit_scale = float(cfg_get(cfg, "model.branch_ii_proto.assignment.logit_scale", 1.0))
        self.proto_banks = MultiPrototypeBank(
            dim=dim,
            kg=kg,
            kp=kp,
            momentum=m,
            logit_scale=logit_scale,
            enable_revive=bool(cfg_get(cfg, "model.branch_ii_proto.ema.enable_revive", True)),
            revive_after_steps=int(cfg_get(cfg, "model.branch_ii_proto.ema.revive_after_steps", 100)),
            revive_noise_std=float(cfg_get(cfg, "model.branch_ii_proto.ema.revive_noise_std", 0.01)),
        )

        tau_p = float(cfg_get(cfg, "model.branch_ii_proto.assignment.temperature_tau_p", 0.05))  # suggested
        lambda_pp = float(cfg_get(cfg, "loss.lambda_pp", 1.0))
        lambda_bal = float(cfg_get(cfg, "loss.lambda_bal", 0.0))  # suggested for bs=1

        self.proto_loss = ProtoLoss(
            global_bank=self.proto_banks.global_bank,
            part_bank=self.proto_banks.part_bank,
            tau_p=tau_p,
            lambda_pp=lambda_pp,
            lambda_bal=lambda_bal,
            lambda_sharp=float(cfg_get(cfg, "loss.lambda_sharp", 0.03)),
            sinkhorn_iters=int(cfg_get(cfg, "loss.sinkhorn_iters", 3)),
            lambda_consistency=float(cfg_get(cfg, "loss.lambda_consistency", 0.0)),
            consistency_temperature=float(cfg_get(cfg, "loss.proto_consistency_temperature", 1.6)),
            consistency_conf_thresh=float(cfg_get(cfg, "loss.proto_consistency_conf_thresh", 0.40)),
            consistency_part_weight=float(cfg_get(cfg, "loss.proto_consistency_part_weight", 0.25)),
            consistency_use_part=bool(cfg_get(cfg, "loss.proto_consistency_use_part", False)),
            return_assignments=True,
            proto_mode=str(cfg_get(cfg, "loss.proto_ablation_mode", "both")),
        )

        # Branch III
        heads_cfg = cfg_get(cfg, "model.branch_iii_cs_hps.content_style.heads", {}) or {}
        policy_cfg = cfg_get(cfg, "model.branch_iii_cs_hps.policy", {}) or {}
        dec_cfg = cfg_get(cfg, "model.branch_iii_cs_hps.decoder", {}) or {}

        content_dim = int(heads_cfg.get("content_dim", 256))
        style_dim = int(heads_cfg.get("style_dim", 256))
        hidden_dim = int(heads_cfg.get("hidden_dim", 1536))
        head_drop = float(heads_cfg.get("dropout", 0.0))
        l2n = bool(heads_cfg.get("l2_normalize", False))

        self.cs_heads = ContentStyleHeads(
            in_dim=dim,
            content_dim=content_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
            dropout=head_drop,
            l2_normalize=l2n,
        )
        self.policy = PatchImportancePolicy(
            dim=dim,
            hidden_dim=int(policy_cfg.get("hidden_dim", 256)),
            dropout=float(policy_cfg.get("dropout", 0.0)),
        )

        self.decoder = CSFiLMDecoder(
            token_dim=dim,
            cond_dim=content_dim + style_dim,
            patch_size=self.patch_size,
            decoder_dim=int(dec_cfg.get("decoder_dim", 512)),
            depth=int(dec_cfg.get("depth", 4)),
            heads=int(dec_cfg.get("heads", 8)),
            mlp_ratio=float(dec_cfg.get("mlp_ratio", 4.0)),
            drop=float(dec_cfg.get("drop", 0.0)),
            pos_base_grid_hw=tuple(dec_cfg.get("pos_base_grid_hw", [14, 14])),
        )
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.cs_hps_loss = CSHPSLoss(
            patch_size=self.patch_size,
            lambda_cont=float(cfg_get(cfg, "loss.lambda_cont", 1.0)),
            lambda_decorr=float(cfg_get(cfg, "loss.lambda_decorr", 0.1)),
            lambda_geom=float(cfg_get(cfg, "loss.lambda_geom", 0.25)),
            lambda_mim=float(cfg_get(cfg, "loss.lambda_mim", 1.0)),
            lambda_mim_visible=float(cfg_get(cfg, "loss.lambda_mim_visible", 0.10)),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        img_a = batch["img_a"]
        img_b = batch["img_b"]
        H_b2a = batch["H_b2a"]

        valid_a_pix = batch["valid_mask_a"]
        valid_b_pix = batch["valid_mask_b"]

        out_a = self.backbone(img_a)
        out_b = self.backbone(img_b)

        F_a = out_a["patch_tokens"]
        F_b = out_b["patch_tokens"]
        u_a = out_a["global_u"]
        u_b = out_b["global_u"]  # correct
        grid_hw = out_a["grid_hw"]

        valid_a_patch = _pixel_valid_to_patch(valid_a_pix, self.patch_size)
        valid_b_patch = _pixel_valid_to_patch(valid_b_pix, self.patch_size)
        valid_a_flat = _patch_valid_flat(valid_a_patch)
        valid_b_flat = _patch_valid_flat(valid_b_patch)

        Z_a, M_a = self.slot_attn(F_a, valid_mask=valid_a_flat)
        Z_b, M_b = self.slot_attn(F_b, valid_mask=valid_b_flat)

        return {
            "img_a": img_a, "img_b": img_b,
            "H_b2a": H_b2a,
            "grid_hw": grid_hw,
            "F_a": F_a, "F_b": F_b,
            "u_a": u_a, "u_b": u_b,
            "Z_a": Z_a, "Z_b": Z_b,
            "M_a": M_a, "M_b": M_b,
            "valid_a_patch": valid_a_patch,
            "valid_b_patch": valid_b_patch,
        }


class PretrainEngine:
    def __init__(self, cfg, model, optimizer, scheduler, scaler, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.output_dir = cfg_get(cfg, "experiment.output_dir", "work_dirs/pretrain")
        os.makedirs(self.output_dir, exist_ok=True)

        self.weights = TotalLossWeights(
            lambda_proto=float(cfg_get(cfg, "loss.lambda_proto", 0.10)),
            lambda_cs_hps=float(cfg_get(cfg, "loss.lambda_cs_hps", 0.30)),
            lambda_pp=float(cfg_get(cfg, "loss.lambda_pp", 1.0)),
            lambda_bal=float(cfg_get(cfg, "loss.lambda_bal", 0.0)),
            lambda_sharp=float(cfg_get(cfg, "loss.lambda_sharp", 0.03)),
            lambda_consistency=float(cfg_get(cfg, "loss.lambda_consistency", 0.0)),
            lambda_cont=float(cfg_get(cfg, "loss.lambda_cont", 1.0)),
            lambda_decorr=float(cfg_get(cfg, "loss.lambda_decorr", 0.1)),
            lambda_geom=float(cfg_get(cfg, "loss.lambda_geom", 0.25)),
            lambda_mim=float(cfg_get(cfg, "loss.lambda_mim", 1.0)),
            lambda_mim_visible=float(cfg_get(cfg, "loss.lambda_mim_visible", 0.10)),
            alpha_mask=float(cfg_get(cfg, "loss.alpha_mask", 1.0)),
            alpha_nce=float(cfg_get(cfg, "loss.alpha_nce", 0.1)),
            alpha_slot_balance=float(cfg_get(cfg, "loss.alpha_slot_balance", 0.2)),
            alpha_slot_diversity=float(cfg_get(cfg, "loss.alpha_slot_diversity", 0.1)),
            tau_nce=float(cfg_get(cfg, "model.branch_i_slot.contrastive.temperature_tau", 0.2)),
            auto_scale_proto=False,
            auto_scale_cs=False,
            scale_eps=float(cfg_get(cfg, "loss.scale_eps", 1e-6)),
            scale_clip=float(cfg_get(cfg, "loss.scale_clip", 10.0)),
            scale_min=float(cfg_get(cfg, "loss.scale_min", 0.5)),
            scale_power=float(cfg_get(cfg, "loss.scale_power", 0.5)),
        )
        self.branch_warmup_epochs = int(cfg_get(cfg, "loss.branch_warmup_epochs", 10))
        self.nce_warmup_epochs = int(cfg_get(cfg, "loss.alpha_nce_warmup_epochs", 10))

        self.proto_vis_buf = ProtoVisBuffer(
            max_u=int(cfg_get(cfg, "train.proto_vis.max_u", 2000)),
            max_z=int(cfg_get(cfg, "train.proto_vis.max_z", 5000)),
            device="cpu",
        )

    def _autocast_context(self):
        enabled = bool(getattr(self.scaler, "is_enabled", lambda: False)()) and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=self.device.type, enabled=enabled)
        return torch.cuda.amp.autocast(enabled=enabled)

    @torch.no_grad()
    def _prototype_ema_update(self, u_a, u_b, Z_a, Z_b, pi: torch.Tensor, proto_out: Optional[Dict[str, Any]] = None):
        tau_p = float(cfg_get(self.cfg, "model.branch_ii_proto.assignment.temperature_tau_p", 0.05))
        proto_mode = str(getattr(self.model.proto_loss, "proto_mode", "both")).lower()
        Zb_aligned = _align_by_pi(Z_b, pi)

        u_a = F.normalize(u_a, p=2, dim=-1)
        u_b = F.normalize(u_b, p=2, dim=-1)
        Z_a = F.normalize(Z_a, p=2, dim=-1)
        Zb_aligned = F.normalize(Zb_aligned, p=2, dim=-1)

        B, K, D = Z_a.shape

        qg_a = None
        qg_b = None
        qp_a = None
        qp_b = None
        if proto_out is not None:
            qg_a = proto_out.get("qg_a_t", None)
            qg_b = proto_out.get("qg_b_t", None)
            qp_a = proto_out.get("qp_a_t", None)
            qp_b = proto_out.get("qp_b_t", None)

        if qg_a is None or qg_b is None:
            qg_a = self.model.proto_banks.global_bank.assign(u_a, temperature=tau_p, detach_emb=True)
            qg_b = self.model.proto_banks.global_bank.assign(u_b, temperature=tau_p, detach_emb=True)
        if qp_a is None or qp_b is None:
            qp_a = self.model.proto_banks.part_bank.assign(Z_a.reshape(B * K, D), temperature=tau_p, detach_emb=True)
            qp_b = self.model.proto_banks.part_bank.assign(Zb_aligned.reshape(B * K, D), temperature=tau_p, detach_emb=True)

        if proto_mode in {"both", "global-only"}:
            self.model.proto_banks.global_bank.ema_update(torch.cat([u_a, u_b], dim=0), torch.cat([qg_a, qg_b], dim=0))
            self.model.proto_banks.global_bank.prototypes.data = F.normalize(
                self.model.proto_banks.global_bank.prototypes.data, p=2, dim=-1
            )
        if proto_mode in {"both", "part-only"}:
            self.model.proto_banks.part_bank.ema_update(
                torch.cat([Z_a.reshape(B * K, D), Zb_aligned.reshape(B * K, D)], dim=0),
                torch.cat([qp_a, qp_b], dim=0),
            )
            self.model.proto_banks.part_bank.prototypes.data = F.normalize(
                self.model.proto_banks.part_bank.prototypes.data, p=2, dim=-1
            )

    @torch.no_grad()
    def _proto_vis_push(self, fw: Dict[str, Any], proto_out: Dict[str, Any], pi: torch.Tensor):
        try:
            Zb_aligned = _align_by_pi(fw["Z_b"].detach(), pi)
            Bsz, K, D = fw["Z_a"].shape
            self.proto_vis_buf.push_global(
                u_a=fw["u_a"].detach(),
                u_b=fw["u_b"].detach(),
                qg_a=proto_out.get("qg_a_t", proto_out.get("qg_a", None)),
                qg_b=proto_out.get("qg_b_t", proto_out.get("qg_b", None)),
            )
            self.proto_vis_buf.push_part(
                z_a=fw["Z_a"].detach().reshape(Bsz * K, D),
                z_b=Zb_aligned.reshape(Bsz * K, D),
                qp_a=proto_out.get("qp_a_t", proto_out.get("qp_a", None)),
                qp_b=proto_out.get("qp_b_t", proto_out.get("qp_b", None)),
            )
        except Exception as e:
            print(f"[proto_vis_push] skipped: {e}")

    @torch.no_grad()
    def _proto_vis_save_epoch(self, epoch: int):
        proto_ep_int = int(cfg_get(self.cfg, "train.proto_vis_epoch_interval", 1))
        proto_force = bool(cfg_get(self.cfg, "train.proto_vis_force", True))
        if proto_ep_int <= 0 or ((epoch + 1) % proto_ep_int) != 0:
            return
        if (not self.proto_vis_buf.ready()) and (not proto_force):
            print("[proto_vis] skip: buffer not ready.")
            return

        buf = self.proto_vis_buf.get()
        vis_path = os.path.join(self.output_dir, "proto_vis", f"proto_pca_ep{epoch + 1:03d}.png")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        plot_proto_pca_2x2(
            save_path=vis_path,
            u_a=buf["u_a"],
            u_b=buf["u_b"],
            z_a=buf["z_a"],
            z_b=buf["z_b"],
            proto_g=self.model.proto_banks.global_bank.prototypes.detach(),
            proto_p=self.model.proto_banks.part_bank.prototypes.detach(),
            qg_a=buf.get("qg_a", None),
            qg_b=buf.get("qg_b", None),
            qp_a=buf.get("qp_a", None),
            qp_b=buf.get("qp_b", None),
            topM_global=int(cfg_get(self.cfg, "train.proto_vis.topM_global", 100)),
            topM_part=int(cfg_get(self.cfg, "train.proto_vis.topM_part", 60)),
            topK_per_proto_global=int(cfg_get(self.cfg, "train.proto_vis.topK_per_proto_global", 80)),
            topK_per_proto_part=int(cfg_get(self.cfg, "train.proto_vis.topK_per_proto_part", 120)),
            title=f"ShelfMIM Prototype-Alignment PCA (epoch {epoch + 1})",
        )
        print(f"[proto_vis] saved: {vis_path}")

    @torch.no_grad()
    def _save_slot_vis(self, fw: Dict[str, Any], step: int):
        save_slot_grid_ab(
            out_dir=self.output_dir,
            step=step,
            img_a=fw["img_a"],
            img_b=fw["img_b"],
            M_a=fw["M_a"],
            M_b=fw["M_b"],
            grid_hw=fw["grid_hw"],
            valid_a_patch=fw.get("valid_a_patch", None),
            valid_b_patch=fw.get("valid_b_patch", None),
            b=0,
        )

    @torch.no_grad()
    def _save_recon_vis(self, fw: Dict[str, Any], cs_out: Dict[str, Any], step: int):
        masked_a = _try_get(cs_out, "masked_img_a", "img_masked_a", "masked_a")
        masked_b = _try_get(cs_out, "masked_img_b", "img_masked_b", "masked_b")
        recon_a = _try_get(cs_out, "recon_img_a", "recon_full_img_a", "recon_a", "img_recon_a")
        recon_b = _try_get(cs_out, "recon_img_b", "recon_full_img_b", "recon_b", "img_recon_b")

        if masked_a is not None and recon_a is not None:
            save_recon_triplets(
                out_dir=self.output_dir,
                step=step,
                masked=masked_a,
                recon=recon_a,
                gt=fw["img_a"],
                prefix="A",
                max_rows=int(cfg_get(self.cfg, "train.vis.max_rows", 4)),
            )
        if masked_b is not None and recon_b is not None:
            save_recon_triplets(
                out_dir=self.output_dir,
                step=step,
                masked=masked_b,
                recon=recon_b,
                gt=fw["img_b"],
                prefix="B",
                max_rows=int(cfg_get(self.cfg, "train.vis.max_rows", 4)),
            )

    @torch.no_grad()
    def _save_epoch_visuals(self, fw: Dict[str, Any], cs_out: Dict[str, Any], epoch: int, last_global_step: int):
        vis_epoch_int = int(cfg_get(self.cfg, "train.vis_epoch_interval", 1))
        if vis_epoch_int <= 0 or ((epoch + 1) % vis_epoch_int) != 0:
            return
        try:
            self._save_slot_vis(fw, last_global_step)
        except Exception as e:
            print(f"[slot_vis] skipped: {e}")
        try:
            self._save_recon_vis(fw, cs_out, last_global_step)
        except Exception as e:
            print(f"[recon_vis] skipped: {e}")

    def train_one_epoch(self, dataloader, epoch: int, start_step: int = 0) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        last_fw: Optional[Dict[str, Any]] = None
        last_cs_out: Optional[Dict[str, Any]] = None
        last_global_step = start_step
        num_optim_steps = 0
        skipped_batches = 0

        if self.branch_warmup_epochs > 0:
            branch_scale = min(1.0, float(epoch + 1) / float(self.branch_warmup_epochs))
        else:
            branch_scale = 1.0
        if self.nce_warmup_epochs > 0:
            nce_scale = min(1.0, float(epoch + 1) / float(self.nce_warmup_epochs))
        else:
            nce_scale = 1.0
        alpha_nce_eff = float(self.weights.alpha_nce) * nce_scale

        proto_cons_warmup_epochs = int(cfg_get(self.cfg, "loss.proto_consistency_warmup_epochs", 15))
        if proto_cons_warmup_epochs > 0:
            proto_cons_scale = min(1.0, float(epoch + 1) / float(proto_cons_warmup_epochs))
        else:
            proto_cons_scale = 1.0
        proto_cons_eff = float(self.model.proto_loss.lambda_consistency) * proto_cons_scale
        epoch_weights = TotalLossWeights(
            lambda_proto=float(self.weights.lambda_proto) * branch_scale,
            lambda_cs_hps=float(self.weights.lambda_cs_hps) * branch_scale,
            lambda_pp=float(self.weights.lambda_pp),
            lambda_bal=float(self.weights.lambda_bal),
            lambda_sharp=float(self.weights.lambda_sharp),
            lambda_consistency=float(self.weights.lambda_consistency) * proto_cons_scale,
            lambda_cont=float(self.weights.lambda_cont),
            lambda_decorr=float(self.weights.lambda_decorr),
            lambda_geom=float(self.weights.lambda_geom),
            lambda_mim=float(self.weights.lambda_mim),
            lambda_mim_visible=float(self.weights.lambda_mim_visible),
            alpha_mask=float(self.weights.alpha_mask),
            alpha_nce=float(alpha_nce_eff),
            alpha_slot_balance=float(self.weights.alpha_slot_balance),
            alpha_slot_diversity=float(self.weights.alpha_slot_diversity),
            tau_nce=float(self.weights.tau_nce),
            auto_scale_proto=False,
            auto_scale_cs=False,
            scale_eps=float(self.weights.scale_eps),
            scale_clip=float(self.weights.scale_clip),
            scale_min=float(self.weights.scale_min),
            scale_power=float(self.weights.scale_power),
        )

        for it, batch in enumerate(dataloader):
            global_step = start_step + it
            last_global_step = global_step
            batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

            with self._autocast_context():
                fw = self.model(batch)
                slot_dict = compute_branch_i_slot_losses(
                    masks_a=fw["M_a"],
                    masks_b=fw["M_b"],
                    slots_a=fw["Z_a"],
                    slots_b=fw["Z_b"],
                    H_b2a_pix=fw["H_b2a"],
                    grid_hw=fw["grid_hw"],
                    valid_mask_a=fw["valid_a_patch"],
                    valid_mask_b=fw["valid_b_patch"],
                    patch_size=self.model.patch_size,
                    alpha_mask=float(cfg_get(self.cfg, "loss.alpha_mask", 1.0)),
                    alpha_nce=float(alpha_nce_eff),
                    tau_nce=float(cfg_get(self.cfg, "model.branch_i_slot.contrastive.temperature_tau", 0.2)),
                    nce_use_valid_slots=bool(cfg_get(self.cfg, "loss.nce_use_valid_slots", True)),
                    slot_conf_thresh=float(cfg_get(self.cfg, "loss.slot_conf_thresh", 0.25)),
                    alpha_slot_balance=float(cfg_get(self.cfg, "loss.alpha_slot_balance", 0.35)),
                    alpha_slot_diversity=float(cfg_get(self.cfg, "loss.alpha_slot_diversity", 0.30)),
                )
                pi = slot_dict["pi"]

                proto_out = self.model.proto_loss(
                    u_a=fw["u_a"],
                    u_b=fw["u_b"],
                    z_a=fw["Z_a"],
                    z_b=fw["Z_b"],
                    pi=pi,
                    part_geom_weight=slot_dict.get("proto_part_geom_weight", None),
                )
                proto_total = proto_out["loss_proto_total"]
                self._proto_vis_push(fw, proto_out, pi)

                mask_ratio_tgt = float(cfg_get(self.cfg, "model.branch_iii_cs_hps.masking.ratio_r", 0.35))
                mask_ratio_min = float(cfg_get(self.cfg, "model.branch_iii_cs_hps.masking.ratio_r_min", min(0.15, mask_ratio_tgt)))
                mask_warm_epochs = int(cfg_get(self.cfg, "model.branch_iii_cs_hps.masking.ratio_warmup_epochs", 20))
                if mask_warm_epochs > 0:
                    mprog = min(1.0, float(epoch + 1) / float(mask_warm_epochs))
                else:
                    mprog = 1.0
                mask_ratio = mask_ratio_min + (mask_ratio_tgt - mask_ratio_min) * mprog

                gumbel_tau = float(cfg_get(self.cfg, "model.branch_iii_cs_hps.masking.gumbel_tau", 1.0))
                policy_warmup_epochs = int(cfg_get(self.cfg, "loss.policy_warmup_epochs", 5))
                use_learned_policy = (epoch + 1) > policy_warmup_epochs
                cs_out = self.model.cs_hps_loss(
                    images_a=fw["img_a"],
                    images_b=fw["img_b"],
                    tokens_a=fw["F_a"],
                    tokens_b=fw["F_b"],
                    pooled_a=fw["u_a"],
                    pooled_b=fw["u_b"],
                    grid_hw=fw["grid_hw"],
                    valid_mask_a=fw["valid_a_patch"],
                    valid_mask_b=fw["valid_b_patch"],
                    cs_heads=self.model.cs_heads,
                    policy=self.model.policy,
                    sampler=gumbel_topk_st,
                    decoder=self.model.decoder,
                    mask_token=self.model.mask_token,
                    mask_ratio=mask_ratio,
                    gumbel_tau=gumbel_tau,
                    use_learned_policy=use_learned_policy,
                )

                opt_losses = assemble_total_loss(
                    slot_loss_dict=slot_dict,
                    proto_loss_dict=proto_out,
                    cs_hps_loss_dict=cs_out,
                    weights=epoch_weights,
                )
                monitor_losses = assemble_total_loss(
                    slot_loss_dict=slot_dict,
                    proto_loss_dict=proto_out,
                    cs_hps_loss_dict=cs_out,
                    weights=self.weights,
                )

                def _assert_opt_contrib_sum(total_key: str, atom_keys: list[str], name: str):
                    total = torch.nan_to_num(opt_losses.get(total_key, torch.tensor(0.0, device=self.device)), nan=0.0, posinf=1e3, neginf=-1e3)
                    atoms = torch.tensor(0.0, device=self.device, dtype=total.dtype)
                    for ak in atom_keys:
                        atoms = atoms + torch.nan_to_num(opt_losses.get(ak, torch.tensor(0.0, device=self.device, dtype=total.dtype)), nan=0.0, posinf=1e3, neginf=-1e3)
                    if not torch.allclose(total, atoms, atol=1e-6, rtol=1e-5):
                        raise RuntimeError(f"opt {name} contribution mismatch: total={float(total.item()):.6f}, atoms={float(atoms.item()):.6f}")

                _assert_opt_contrib_sum(
                    "contrib_slot_real",
                    ["contrib_slot_mask_real", "contrib_slot_nce_real", "contrib_slot_balance_real", "contrib_slot_diversity_real"],
                    "slot",
                )
                _assert_opt_contrib_sum(
                    "contrib_proto_real",
                    ["contrib_proto_pp_real", "contrib_proto_bal_real", "contrib_proto_sharp_real", "contrib_proto_consistency_real"],
                    "proto",
                )
                _assert_opt_contrib_sum(
                    "contrib_cs_real",
                    ["contrib_cs_cont_real", "contrib_cs_decorr_real", "contrib_cs_geom_real", "contrib_cs_mim_real", "contrib_cs_mim_visible_real"],
                    "cs",
                )

                cs_total_raw = monitor_losses["cs_branch_core"]
                loss = opt_losses["loss_total"]
                if not torch.isfinite(loss):
                    print(f"[warn] non-finite total loss at epoch={epoch+1}, iter={it+1}; skip this batch")
                    skipped_batches += 1
                    continue

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_clip = float(cfg_get(self.cfg, "train.grad_clip_norm", 1.0))
            if grad_clip > 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            num_optim_steps += 1

            lr = self.scheduler.step() if self.scheduler is not None else self.optimizer.param_groups[0]["lr"]
            with torch.no_grad():
                self._prototype_ema_update(fw["u_a"], fw["u_b"], fw["Z_a"], fw["Z_b"], pi, proto_out=proto_out)

            def _add(k, v):
                running[k] = running.get(k, 0.0) + float(v)

            _add("loss_total", float(torch.nan_to_num(monitor_losses["loss_total"].detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_total_opt", float(torch.nan_to_num(loss.detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_slot_total", float(torch.nan_to_num(monitor_losses["loss_slot_total"].detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_mask", float(torch.nan_to_num(monitor_losses["loss_mask"].detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_slot_nce", float(torch.nan_to_num(monitor_losses["loss_slot_nce"].detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_slot_balance", float(torch.nan_to_num(monitor_losses.get("loss_slot_balance", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_slot_diversity", float(torch.nan_to_num(monitor_losses.get("loss_slot_diversity", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_proto_total", float(torch.nan_to_num(monitor_losses.get("loss_proto_total", proto_total).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_proto_raw", float(torch.nan_to_num(monitor_losses.get("loss_proto_raw", proto_total).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_cs_hps_total", float(torch.nan_to_num(monitor_losses.get("loss_cs_hps_total", cs_total_raw).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("loss_cs_hps_raw", float(torch.nan_to_num(monitor_losses.get("loss_cs_hps_raw", cs_total_raw).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_slot_real", float(torch.nan_to_num(monitor_losses.get("contrib_slot_real", monitor_losses["loss_slot_total"]).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_proto_scaled", float(torch.nan_to_num(monitor_losses.get("contrib_proto_scaled", monitor_losses.get("loss_proto_total", torch.tensor(0.0, device=self.device))).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_proto_real", float(torch.nan_to_num(monitor_losses.get("contrib_proto_real", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_cs_scaled", float(torch.nan_to_num(monitor_losses.get("contrib_cs_scaled", monitor_losses.get("loss_cs_hps_total", torch.tensor(0.0, device=self.device))).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_cs_real", float(torch.nan_to_num(monitor_losses.get("contrib_cs_real", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_total_check", float(torch.nan_to_num(monitor_losses.get("contrib_total_check", monitor_losses["loss_total"]).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))

            _add("contrib_slot_real_opt", float(torch.nan_to_num(opt_losses.get("contrib_slot_real", opt_losses["loss_slot_total"]).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_proto_scaled_opt", float(torch.nan_to_num(opt_losses.get("contrib_proto_scaled", opt_losses.get("loss_proto_total", torch.tensor(0.0, device=self.device))).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_proto_real_opt", float(torch.nan_to_num(opt_losses.get("contrib_proto_real", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_cs_scaled_opt", float(torch.nan_to_num(opt_losses.get("contrib_cs_scaled", opt_losses.get("loss_cs_hps_total", torch.tensor(0.0, device=self.device))).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_cs_real_opt", float(torch.nan_to_num(opt_losses.get("contrib_cs_real", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            _add("contrib_total_check_opt", float(torch.nan_to_num(opt_losses.get("contrib_total_check", opt_losses["loss_total"]).detach(), nan=0.0, posinf=1e3, neginf=1e3).item()))
            for ck in ["contrib_slot_mask_real", "contrib_slot_nce_real", "contrib_slot_balance_real", "contrib_slot_diversity_real", "contrib_proto_pp_real", "contrib_proto_bal_real", "contrib_proto_sharp_real", "contrib_proto_consistency_real", "contrib_cs_cont_real", "contrib_cs_decorr_real", "contrib_cs_geom_real", "contrib_cs_mim_real", "contrib_cs_mim_visible_real"]:
                if ck in monitor_losses:
                    _add(ck, float(torch.nan_to_num(monitor_losses[ck].detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))
                if (ck + "_opt") not in running and ck in opt_losses:
                    _add(ck + "_opt", float(torch.nan_to_num(opt_losses[ck].detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))

            # Slot entropy diagnostics: supervision masks vs update attention.
            with torch.no_grad():
                m_a = fw["M_a"].detach().clamp_min(1e-8)
                m_b = fw["M_b"].detach().clamp_min(1e-8)
                if fw.get("valid_a_patch", None) is not None:
                    vm_a = fw["valid_a_patch"].reshape(m_a.shape[0], -1)
                else:
                    vm_a = torch.ones((m_a.shape[0], m_a.shape[-1]), device=m_a.device, dtype=m_a.dtype)
                if fw.get("valid_b_patch", None) is not None:
                    vm_b = fw["valid_b_patch"].reshape(m_b.shape[0], -1)
                else:
                    vm_b = torch.ones((m_b.shape[0], m_b.shape[-1]), device=m_b.device, dtype=m_b.dtype)

                # masks are softmax over slots per patch.
                ent_a = (-(m_a * torch.log(m_a)).sum(dim=1) * vm_a).sum() / vm_a.sum().clamp_min(1.0)
                ent_b = (-(m_b * torch.log(m_b)).sum(dim=1) * vm_b).sum() / vm_b.sum().clamp_min(1.0)
                _add("slot_mask_entropy", float(torch.nan_to_num(0.5 * (ent_a + ent_b), nan=0.0, posinf=1e3, neginf=0.0).item()))

                # update attention is normalized over patches per slot.
                attn_a = (m_a + 1e-6)
                attn_a = attn_a / attn_a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                attn_b = (m_b + 1e-6)
                attn_b = attn_b / attn_b.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                ent_upd_a = -(attn_a * torch.log(attn_a)).sum(dim=-1).mean()
                ent_upd_b = -(attn_b * torch.log(attn_b)).sum(dim=-1).mean()
                _add("slot_update_attn_entropy", float(torch.nan_to_num(0.5 * (ent_upd_a + ent_upd_b), nan=0.0, posinf=1e3, neginf=0.0).item()))

                wm = warp_quality_metrics(fw["H_b2a"].detach(), grid_hw=fw["grid_hw"], patch_size=self.model.patch_size)
                for wk, wv in wm.items():
                    _add(wk, float(torch.nan_to_num(wv.detach(), nan=0.0, posinf=1e3, neginf=0.0).item()))
            _add("lambda_proto_eff_monitor", float(torch.nan_to_num(monitor_losses.get("lambda_proto_eff", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=0.0).item()))
            _add("lambda_cs_hps_eff_monitor", float(torch.nan_to_num(monitor_losses.get("lambda_cs_hps_eff", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=0.0).item()))
            _add("lambda_proto_eff_opt", float(torch.nan_to_num(opt_losses.get("lambda_proto_eff", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=0.0).item()))
            _add("lambda_cs_hps_eff_opt", float(torch.nan_to_num(opt_losses.get("lambda_cs_hps_eff", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=0.0).item()))
            _add("contrib_total_delta_monitor", float(torch.nan_to_num(monitor_losses.get("contrib_total_delta", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))
            _add("contrib_total_delta_opt", float(torch.nan_to_num(opt_losses.get("contrib_total_delta", torch.tensor(0.0, device=self.device)).detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))
            _add("delta_proto_monitor_vs_opt", float(torch.nan_to_num((monitor_losses.get("contrib_proto_real", torch.tensor(0.0, device=self.device)) - opt_losses.get("contrib_proto_real", torch.tensor(0.0, device=self.device))).detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))
            _add("delta_cs_monitor_vs_opt", float(torch.nan_to_num((monitor_losses.get("contrib_cs_real", torch.tensor(0.0, device=self.device)) - opt_losses.get("contrib_cs_real", torch.tensor(0.0, device=self.device))).detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))

            for dk in ["contrib_proto_pp", "contrib_proto_bal", "contrib_proto_sharp", "contrib_proto_consistency", "contrib_proto_internal_total", "lambda_pp_eff", "lambda_bal_eff", "lambda_sharp_eff", "lambda_consistency_eff"]:
                if dk in proto_out:
                    _add(dk, float(torch.nan_to_num(proto_out[dk].detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))

            for dk in ["contrib_cs_cont", "contrib_cs_decorr", "contrib_cs_geom", "contrib_cs_mim", "contrib_cs_mim_visible", "contrib_cs_internal_total", "lambda_cont_eff", "lambda_decorr_eff", "lambda_geom_eff", "lambda_mim_eff", "lambda_mim_visible_eff"]:
                if dk in cs_out:
                    _add(dk, float(torch.nan_to_num(cs_out[dk].detach(), nan=0.0, posinf=1e3, neginf=-1e3).item()))

            if "proto_scale_factor" in monitor_losses:
                _add("proto_scale_factor", float(torch.nan_to_num(monitor_losses["proto_scale_factor"].detach(), nan=1.0, posinf=10.0, neginf=0.0).item()))
            if "cs_scale_factor" in monitor_losses:
                _add("cs_scale_factor", float(torch.nan_to_num(monitor_losses["cs_scale_factor"].detach(), nan=1.0, posinf=10.0, neginf=0.0).item()))
            _add("branch_warmup_scale", branch_scale)
            _add("nce_warmup_scale", nce_scale)
            _add("mask_ratio_eff", float(mask_ratio))
            _add("proto_consistency_scale", float(proto_cons_scale))
            with torch.no_grad():
                za = torch.nn.functional.normalize(fw["Z_a"].detach(), dim=-1)
                sim = torch.matmul(za, za.transpose(1, 2))
                K = sim.size(1)
                eye = torch.eye(K, device=sim.device, dtype=sim.dtype).unsqueeze(0)
                off = (sim * (1.0 - eye)).sum(dim=(1, 2)) / max(K * (K - 1), 1)
                _add("slot_cos_mean", float(torch.nan_to_num(off.mean(), nan=0.0, posinf=1.0, neginf=0.0).item()))
            for dk in ["proto_g_active_ratio_a", "proto_g_active_ratio_b", "proto_p_active_ratio_a", "proto_p_active_ratio_b", "proto_g_maxprob_a", "proto_g_maxprob_b", "proto_p_maxprob_a", "proto_p_maxprob_b", "loss_proto_entropy", "loss_proto_sharpness", "loss_proto_consistency", "loss_proto_consistency_global", "loss_proto_consistency_part", "proto_part_geom_weight_mean", "proto_g_usage_entropy", "proto_p_usage_entropy", "proto_g_dead_ratio", "proto_p_dead_ratio"]:
                if dk in proto_out:
                    _add(dk, float(torch.nan_to_num(proto_out[dk].detach(), nan=0.0, posinf=1e3, neginf=0.0).item()))
            _add("lr", float(lr))

            last_fw = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in fw.items()}
            last_cs_out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in cs_out.items()}

        if last_fw is not None and last_cs_out is not None:
            with torch.no_grad():
                self._save_epoch_visuals(last_fw, last_cs_out, epoch, last_global_step)
                self._proto_vis_save_epoch(epoch)

        if skipped_batches > 0:
            print(f"[warn] epoch {epoch+1}: skipped {skipped_batches} non-finite batches")

        n = max(1, num_optim_steps)
        return {k: v / n for k, v in running.items()}






