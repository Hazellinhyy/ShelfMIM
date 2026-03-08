# -*- coding: utf-8 -*-
"""
models/branch_proto/prototype_bank.py

鎻愪緵 PrototypeBank / MultiPrototypeBank锛屼緵棰勮缁?Branch-II (prototype) 浣跨敤銆?

淇锛?
- ImportError: cannot import name 'MultiPrototypeBank'
鍘熷洜锛氬師鏂囦欢琚浛鎹㈠悗缂哄皯 MultiPrototypeBank 瀹氫箟銆?

澧炲己绋冲畾鎬э細
- softmax 鍓嶅噺 max锛坙ogits stability锛?
- EMA 鏇存柊鏃跺仛 all_reduce锛堝垎甯冨紡鍙敤锛屼笉褰卞搷鍗曟満锛?
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@torch.no_grad()
def _all_reduce_sum_(x: torch.Tensor) -> torch.Tensor:
    if _is_dist_avail_and_initialized():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


class PrototypeBank(nn.Module):
    """
    A prototype bank with soft assignment and EMA update.
    prototypes: (P, D)
    """

    def __init__(
        self,
        num_prototypes: int,
        dim: int,
        momentum: float = 0.99,
        eps: float = 1e-6,
        init_std: float = 0.02,
        logit_scale: float = 1.0,
        enable_revive: bool = True,
        revive_after_steps: int = 100,
        revive_noise_std: float = 0.01,
    ):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.logit_scale = float(max(logit_scale, 1e-3))
        self.enable_revive = bool(enable_revive)
        self.revive_after_steps = int(max(1, revive_after_steps))
        self.revive_noise_std = float(max(0.0, revive_noise_std))

        proto = torch.randn(self.num_prototypes, self.dim) * float(init_std)
        proto = F.normalize(proto, dim=-1)
        self.register_buffer("prototypes", proto)
        self.register_buffer("inactive_steps", torch.zeros(self.num_prototypes, dtype=torch.long))

    @torch.no_grad()
    def reset_parameters(self):
        proto = torch.randn(
            self.num_prototypes,
            self.dim,
            device=self.prototypes.device,
            dtype=self.prototypes.dtype,
        )
        proto = F.normalize(proto, dim=-1)
        self.prototypes.copy_(proto)

    def assign(
        self,
        emb: torch.Tensor,            # (..., D)
        temperature: float,
        detach_emb: bool = False,
        return_logits: bool = False,
    ):
        """
        Return soft assignment q over prototypes.
        q: (..., P)
        """
        if emb.size(-1) != self.dim:
            raise ValueError(f"emb last dim {emb.size(-1)} != bank dim {self.dim}")

        x = emb.detach() if detach_emb else emb
        proto_dtype = self.prototypes.dtype
        x = x.to(dtype=proto_dtype)
        x = F.normalize(x, dim=-1)
        c = F.normalize(self.prototypes, dim=-1)

        logits = (torch.matmul(x, c.t()) / float(temperature)) * self.logit_scale
        logits = logits - logits.max(dim=-1, keepdim=True).values  # 鉁呮暟鍊肩ǔ瀹?
        q = F.softmax(logits, dim=-1)

        if return_logits:
            return q, logits
        return q

    @torch.no_grad()
    def ema_update(self, emb: torch.Tensor, q: torch.Tensor):
        """
        EMA update prototypes using assignments.
        emb: (..., D)
        q:   (..., P)
        """
        if emb.size(-1) != self.dim:
            raise ValueError(f"emb last dim {emb.size(-1)} != dim {self.dim}")
        if q.size(-1) != self.num_prototypes:
            raise ValueError(f"q last dim {q.size(-1)} != num_prototypes {self.num_prototypes}")

        proto_dtype = self.prototypes.dtype
        x = emb.reshape(-1, self.dim).detach().to(dtype=proto_dtype)
        w = q.reshape(-1, self.num_prototypes).detach().to(dtype=proto_dtype)

        x = F.normalize(x, dim=-1)

        num = torch.matmul(w.t(), x)         # (P, D)
        den = w.sum(dim=0)                   # (P,)

        _all_reduce_sum_(num)
        _all_reduce_sum_(den)

        den_safe = den.clamp_min(self.eps)
        new_proto = num / den_safe.unsqueeze(-1)  # (P, D)

        # Only update active prototypes; keep inactive ones from drifting to noise.
        active = den > (self.eps * 10.0)
        proto_updated = self.prototypes.clone()
        if active.any():
            upd = self.prototypes[active] * self.momentum + new_proto[active] * (1.0 - self.momentum)
            proto_updated[active] = F.normalize(upd, dim=-1)

        # Conservative revive: trigger only after persistent inactivity.
        self.inactive_steps[active] = 0
        self.inactive_steps[~active] = self.inactive_steps[~active] + 1
        if self.enable_revive and x.size(0) > 0:
            revive_mask = (~active) & (self.inactive_steps >= self.revive_after_steps)
            if revive_mask.any():
                take = int(revive_mask.sum().item())
                ridx = torch.randint(0, x.size(0), (take,), device=x.device)
                seeds = x[ridx]
                if self.revive_noise_std > 0:
                    seeds = F.normalize(seeds + torch.randn_like(seeds) * self.revive_noise_std, dim=-1)
                proto_updated[revive_mask] = seeds
                self.inactive_steps[revive_mask] = 0

        self.prototypes.copy_(F.normalize(proto_updated, dim=-1))



class MultiPrototypeBank(nn.Module):
    """
    Container for two banks:
    - global prototypes (kg)
    - part prototypes (kp)
    """

    def __init__(
        self,
        dim: int,
        kg: int,
        kp: int,
        momentum: float = 0.99,
        eps: float = 1e-6,
        init_std: float = 0.02,
        logit_scale: float = 1.0,
        enable_revive: bool = True,
        revive_after_steps: int = 100,
        revive_noise_std: float = 0.01,
    ):
        super().__init__()
        self.global_bank = PrototypeBank(
            num_prototypes=kg,
            dim=dim,
            momentum=momentum,
            eps=eps,
            init_std=init_std,
            logit_scale=logit_scale,
            enable_revive=enable_revive,
            revive_after_steps=revive_after_steps,
            revive_noise_std=revive_noise_std,
        )
        self.part_bank = PrototypeBank(
            num_prototypes=kp,
            dim=dim,
            momentum=momentum,
            eps=eps,
            init_std=init_std,
            logit_scale=logit_scale,
            enable_revive=enable_revive,
            revive_after_steps=revive_after_steps,
            revive_noise_std=revive_noise_std,
        )

    @torch.no_grad()
    def ema_update_global(self, u: torch.Tensor, qg: torch.Tensor):
        self.global_bank.ema_update(u, qg)

    @torch.no_grad()
    def ema_update_part(self, z: torch.Tensor, qp: torch.Tensor):
        self.part_bank.ema_update(z, qp)
