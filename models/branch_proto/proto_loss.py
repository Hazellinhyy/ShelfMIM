from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.branch_proto.prototype_bank import PrototypeBank


def _cross_entropy_probs(p_target: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    logq = torch.log(p_pred.clamp_min(eps))
    return -(p_target * logq).sum(dim=-1).mean()


def _balance_kl_to_uniform(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = q.size(-1)
    q_bar = q.mean(dim=0).clamp_min(eps)
    q_bar = q_bar / q_bar.sum().clamp_min(eps)
    kl = (q_bar * torch.log(q_bar * float(p))).sum()
    return kl / float(p)


def _entropy_mean(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(q * torch.log(q)).sum(dim=-1).mean()


@torch.no_grad()
def _sinkhorn_balanced(logits: torch.Tensor, iters: int = 3, eps: float = 1e-8) -> torch.Tensor:
    if logits.dim() != 2:
        logits = logits.reshape(-1, logits.size(-1))
    q = torch.exp(logits - logits.max(dim=1, keepdim=True).values).t()  # (P,N)
    q = q / q.sum().clamp_min(eps)

    p, n = q.shape
    for _ in range(int(iters)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        q = q / float(p)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(eps)
        q = q / float(n)

    q = q * float(n)
    return q.t().clamp_min(eps)


def _align_by_pi(z_b: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    b, k, d = z_b.shape
    idx = pi.unsqueeze(-1).expand(b, k, d)
    return torch.gather(z_b, dim=1, index=idx)


def _softmax_with_temperature_from_logits(
    logits: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    t = float(max(temperature, eps))
    x = logits / t
    x = x - x.max(dim=-1, keepdim=True).values
    return torch.softmax(x, dim=-1).clamp_min(eps)


def _consistency_soft_ce(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    temperature: float,
    conf_thresh: float,
    sample_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    qa_t = _softmax_with_temperature_from_logits(logits_a, temperature=temperature, eps=eps)
    qb_t = _softmax_with_temperature_from_logits(logits_b, temperature=temperature, eps=eps)

    conf = 0.5 * (qa_t.max(dim=-1).values + qb_t.max(dim=-1).values)
    conf_mask = (conf >= float(conf_thresh)).float()
    if sample_weight is not None:
        sw = sample_weight.to(device=conf_mask.device, dtype=conf_mask.dtype).reshape(-1)
        if sw.numel() != conf_mask.numel():
            raise ValueError(f"sample_weight size mismatch: {sw.numel()} vs {conf_mask.numel()}")
        conf_mask = conf_mask * sw.clamp(min=0.0, max=1.0)
    if conf_mask.sum() <= 0:
        conf_mask = torch.ones_like(conf_mask)

    # Stop-grad teacher on both directions to avoid mutually unstable chasing.
    ce_ab = -(qa_t.detach() * torch.log(qb_t.clamp_min(eps))).sum(dim=-1)
    ce_ba = -(qb_t.detach() * torch.log(qa_t.clamp_min(eps))).sum(dim=-1)
    ce = 0.5 * (ce_ab + ce_ba)
    return (ce * conf_mask).sum() / conf_mask.sum().clamp_min(1.0)


@torch.no_grad()
def _proto_stats(q: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    if q.dim() != 2:
        q = q.reshape(-1, q.size(-1))
    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    q_bar = q.mean(dim=0)
    ent = -(q * torch.log(q)).sum(dim=-1).mean()
    maxp = q.max(dim=-1).values.mean()
    active_ratio = (q_bar > (0.01 / q_bar.numel())).float().mean()
    return {
        "usage": q_bar,
        "entropy_mean": ent,
        "maxprob_mean": maxp,
        "active_ratio": active_ratio,
    }


class ProtoLoss(nn.Module):
    def __init__(
        self,
        global_bank: PrototypeBank,
        part_bank: PrototypeBank,
        tau_p: float = 0.1,
        lambda_pp: float = 1.0,
        lambda_bal: float = 0.05,
        eps: float = 1e-8,
        norm_by_logP: bool = True,
        return_assignments: bool = False,
        lambda_sharp: float = 0.05,
        sinkhorn_iters: int = 3,
        lambda_consistency: float = 0.2,
        proto_mode: str = "both",
        consistency_temperature: float = 1.6,
        consistency_conf_thresh: float = 0.40,
        consistency_part_weight: float = 0.25,
        consistency_use_part: bool = False,
    ):
        super().__init__()
        self.global_bank = global_bank
        self.part_bank = part_bank
        self.tau_p = float(tau_p)
        self.lambda_pp = float(lambda_pp)
        self.lambda_bal = float(lambda_bal)
        self.eps = float(eps)
        self.norm_by_logP = bool(norm_by_logP)
        self.return_assignments = bool(return_assignments)
        self.lambda_sharp = float(lambda_sharp)
        self.sinkhorn_iters = int(max(1, sinkhorn_iters))
        self.lambda_consistency = float(lambda_consistency)
        self.proto_mode = str(proto_mode).lower()
        self.consistency_temperature = float(max(consistency_temperature, 1.0))
        self.consistency_conf_thresh = float(max(0.0, min(1.0, consistency_conf_thresh)))
        self.consistency_part_weight = float(max(0.0, consistency_part_weight))
        self.consistency_use_part = bool(consistency_use_part)
        if self.proto_mode not in {"both", "global-only", "part-only"}:
            raise ValueError(f"Unknown proto_mode: {proto_mode}")

    def forward(
        self,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
        part_geom_weight: Optional[torch.Tensor] = None,
        lambda_consistency_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        b, d = u_a.shape
        _, k, _ = z_a.shape
        z_b_aligned = _align_by_pi(z_b, pi) if pi is not None else z_b

        qg_a, lg_a = self.global_bank.assign(u_a, temperature=self.tau_p, return_logits=True)
        qg_b, lg_b = self.global_bank.assign(u_b, temperature=self.tau_p, return_logits=True)
        za_flat = z_a.reshape(b * k, d)
        zb_flat = z_b_aligned.reshape(b * k, d)
        qp_a, lp_a = self.part_bank.assign(za_flat, temperature=self.tau_p, return_logits=True)
        qp_b, lp_b = self.part_bank.assign(zb_flat, temperature=self.tau_p, return_logits=True)

        qg_a_t = _sinkhorn_balanced(lg_a.detach(), iters=self.sinkhorn_iters, eps=self.eps)
        qg_b_t = _sinkhorn_balanced(lg_b.detach(), iters=self.sinkhorn_iters, eps=self.eps)
        qp_a_t = _sinkhorn_balanced(lp_a.detach(), iters=self.sinkhorn_iters, eps=self.eps)
        qp_b_t = _sinkhorn_balanced(lp_b.detach(), iters=self.sinkhorn_iters, eps=self.eps)

        loss_g = 0.5 * (
            _cross_entropy_probs(qg_a_t.detach(), qg_b, eps=self.eps)
            + _cross_entropy_probs(qg_b_t.detach(), qg_a, eps=self.eps)
        )
        loss_p = 0.5 * (
            _cross_entropy_probs(qp_a_t.detach(), qp_b, eps=self.eps)
            + _cross_entropy_probs(qp_b_t.detach(), qp_a, eps=self.eps)
        )

        if self.norm_by_logP:
            kg = int(self.global_bank.num_prototypes)
            kp = int(self.part_bank.num_prototypes)
            loss_g = loss_g / max(math.log(max(kg, 2)), 1e-6)
            loss_p = loss_p / max(math.log(max(kp, 2)), 1e-6)

        bal_g = _balance_kl_to_uniform(qg_a, eps=self.eps) + _balance_kl_to_uniform(qg_b, eps=self.eps)
        bal_p = _balance_kl_to_uniform(qp_a, eps=self.eps) + _balance_kl_to_uniform(qp_b, eps=self.eps)
        ent_g = _entropy_mean(qg_a, eps=self.eps) + _entropy_mean(qg_b, eps=self.eps)
        ent_p = _entropy_mean(qp_a, eps=self.eps) + _entropy_mean(qp_b, eps=self.eps)

        # Minimized as +lambda_sharp * sharpness_penalty in total loss.
        sharpness_penalty = 0.5 * (ent_g + ent_p)

        cons_g = _consistency_soft_ce(
            lg_a,
            lg_b,
            temperature=self.consistency_temperature,
            conf_thresh=self.consistency_conf_thresh,
            eps=self.eps,
        )
        part_geom_weight_flat = None
        if part_geom_weight is not None:
            part_geom_weight_flat = part_geom_weight.to(device=lp_a.device, dtype=lp_a.dtype).reshape(-1)

        cons_p = _consistency_soft_ce(
            lp_a,
            lp_b,
            temperature=self.consistency_temperature,
            conf_thresh=self.consistency_conf_thresh,
            sample_weight=part_geom_weight_flat,
            eps=self.eps,
        )

        if self.norm_by_logP:
            kg = int(self.global_bank.num_prototypes)
            kp = int(self.part_bank.num_prototypes)
            cons_g = cons_g / max(math.log(max(kg, 2)), 1e-6)
            cons_p = cons_p / max(math.log(max(kp, 2)), 1e-6)

        use_g = 1.0 if self.proto_mode in {"both", "global-only"} else 0.0
        use_p = 1.0 if self.proto_mode in {"both", "part-only"} else 0.0

        proto_pair = use_g * loss_g + use_p * loss_p
        bal = use_g * bal_g + use_p * bal_p
        # Raw entropy is tracked for diagnostics; optimization uses sharpness=-entropy.
        entropy_raw = 0.5 * (use_g * ent_g + use_p * ent_p)
        sharpness_penalty = -entropy_raw
        use_p_cons = use_p * (1.0 if self.consistency_use_part else 0.0)
        cons = use_g * cons_g + use_p_cons * self.consistency_part_weight * cons_p

        lambda_cons_eff = self.lambda_consistency if lambda_consistency_override is None else float(lambda_consistency_override)
        # Keep raw components here; compose weighted total in losses/total_loss.py.
        contrib_pp = proto_pair
        contrib_bal = bal
        contrib_sharp = sharpness_penalty
        contrib_consistency = cons
        loss_proto = proto_pair

        with torch.no_grad():
            st_g_a = _proto_stats(qg_a, eps=self.eps)
            st_g_b = _proto_stats(qg_b, eps=self.eps)
            st_p_a = _proto_stats(qp_a, eps=self.eps)
            st_p_b = _proto_stats(qp_b, eps=self.eps)
            usage_g = 0.5 * (st_g_a["usage"] + st_g_b["usage"])
            usage_p = 0.5 * (st_p_a["usage"] + st_p_b["usage"])
            usage_ent_g = -(usage_g.clamp_min(self.eps) * torch.log(usage_g.clamp_min(self.eps))).sum()
            usage_ent_p = -(usage_p.clamp_min(self.eps) * torch.log(usage_p.clamp_min(self.eps))).sum()
            dead_g = (usage_g < (0.1 / usage_g.numel())).float().mean()
            dead_p = (usage_p < (0.1 / usage_p.numel())).float().mean()

        out: Dict[str, torch.Tensor] = {
            "loss_proto_global": loss_g,
            "loss_proto_part": loss_p,
            "loss_proto_pair": proto_pair,
            "loss_proto_balance": bal,
            "loss_proto_entropy": entropy_raw,
            "loss_proto_entropy_raw": entropy_raw,
            "loss_proto_sharpness": sharpness_penalty,
            "loss_proto_consistency": cons,
            "loss_proto_consistency_global": cons_g,
            "loss_proto_consistency_part": cons_p,
            # legacy key retained for compatibility; now aliases pair term.
            "loss_proto_total": loss_proto,
            "lambda_pp_eff": torch.tensor(float(self.lambda_pp), device=cons.device, dtype=cons.dtype),
            "lambda_bal_eff": torch.tensor(float(self.lambda_bal), device=cons.device, dtype=cons.dtype),
            "lambda_sharp_eff": torch.tensor(float(self.lambda_sharp), device=cons.device, dtype=cons.dtype),
            "lambda_consistency_eff": torch.tensor(lambda_cons_eff, device=cons.device, dtype=cons.dtype),
            # raw components (weighted externally)
            "contrib_proto_pp": contrib_pp,
            "contrib_proto_bal": contrib_bal,
            "contrib_proto_sharp": contrib_sharp,
            "contrib_proto_consistency": contrib_consistency,
            "contrib_proto_internal_total": proto_pair,
            "proto_mode": torch.tensor(0.0, device=cons.device),
            "proto_g_usage_entropy": usage_ent_g,
            "proto_p_usage_entropy": usage_ent_p,
            "proto_g_dead_ratio": dead_g,
            "proto_p_dead_ratio": dead_p,
            "proto_g_entropy_a": st_g_a["entropy_mean"],
            "proto_g_entropy_b": st_g_b["entropy_mean"],
            "proto_g_maxprob_a": st_g_a["maxprob_mean"],
            "proto_g_maxprob_b": st_g_b["maxprob_mean"],
            "proto_g_active_ratio_a": st_g_a["active_ratio"],
            "proto_g_active_ratio_b": st_g_b["active_ratio"],
            "proto_p_entropy_a": st_p_a["entropy_mean"],
            "proto_p_entropy_b": st_p_b["entropy_mean"],
            "proto_p_maxprob_a": st_p_a["maxprob_mean"],
            "proto_p_maxprob_b": st_p_b["maxprob_mean"],
            "proto_p_active_ratio_a": st_p_a["active_ratio"],
            "proto_p_active_ratio_b": st_p_b["active_ratio"],
            "proto_consistency_temperature": torch.tensor(float(self.consistency_temperature), device=cons.device, dtype=cons.dtype),
            "proto_consistency_conf_thresh": torch.tensor(float(self.consistency_conf_thresh), device=cons.device, dtype=cons.dtype),
            "proto_consistency_part_weight": torch.tensor(float(self.consistency_part_weight), device=cons.device, dtype=cons.dtype),
            "proto_consistency_use_part": torch.tensor(float(1.0 if self.consistency_use_part else 0.0), device=cons.device, dtype=cons.dtype),
            "proto_part_geom_weight_mean": (
                part_geom_weight_flat.mean().detach() if part_geom_weight_flat is not None else torch.tensor(1.0, device=cons.device, dtype=cons.dtype)
            ),
        }

        if self.return_assignments:
            out["qg_a"] = qg_a.detach()
            out["qg_b"] = qg_b.detach()
            out["qp_a"] = qp_a.detach()
            out["qp_b"] = qp_b.detach()
            out["qg_a_t"] = qg_a_t.detach()
            out["qg_b_t"] = qg_b_t.detach()
            out["qp_a_t"] = qp_a_t.detach()
            out["qp_b_t"] = qp_b_t.detach()

        return out
