# # -*- coding: utf-8 -*-
# """
# Optimizer / Scheduler utilities:
# - AdamW
# - Cosine decay with warmup
# - Gradient clipping handled in engine
# - AMP scaler helper
# """
#
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Iterable, List, Optional, Dict, Any, Tuple
#
# import math
# import torch
# import torch.nn as nn
#
#
# def _is_norm_layer(m: nn.Module) -> bool:
#     return isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
#                           nn.GroupNorm, nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d))
#
#
# def build_adamw(
#     model: nn.Module,
#     base_lr: float,
#     weight_decay: float,
#     betas: Tuple[float, float] = (0.9, 0.95),
# ) -> torch.optim.Optimizer:
#     """
#     AdamW with typical ViT-style weight decay exclusion for bias & norm parameters.
#     """
#     decay_params: List[torch.nn.Parameter] = []
#     no_decay_params: List[torch.nn.Parameter] = []
#
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         # exclude biases
#         if name.endswith(".bias"):
#             no_decay_params.append(p)
#             continue
#         # exclude norm layers
#         # (best-effort check by name; robust check happens below via module types)
#         if "norm" in name.lower() or "ln" in name.lower():
#             no_decay_params.append(p)
#             continue
#         decay_params.append(p)
#
#     # second pass: ensure params inside actual norm modules are excluded
#     norm_param_ids = set()
#     for module_name, m in model.named_modules():
#         if _is_norm_layer(m):
#             for p in m.parameters(recurse=False):
#                 norm_param_ids.add(id(p))
#
#     decay_params2, no_decay_params2 = [], []
#     for p in decay_params:
#         (no_decay_params2 if id(p) in norm_param_ids else decay_params2).append(p)
#     no_decay_params2.extend([p for p in no_decay_params if id(p) not in set(id(x) for x in no_decay_params2)])
#
#     param_groups = [
#         {"params": decay_params2, "weight_decay": float(weight_decay), "lr": float(base_lr)},
#         {"params": no_decay_params2, "weight_decay": 0.0, "lr": float(base_lr)},
#     ]
#     optim = torch.optim.AdamW(param_groups, lr=float(base_lr), betas=betas)
#     return optim
#
#
# class WarmupCosineLR:
#     """
#     Per-iteration warmup + cosine decay.
#     """
#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         total_steps: int,
#         warmup_steps: int,
#         base_lr: float,
#         min_lr: float = 0.0,
#     ):
#         self.optimizer = optimizer
#         self.total_steps = int(total_steps)
#         self.warmup_steps = int(warmup_steps)
#         self.base_lr = float(base_lr)
#         self.min_lr = float(min_lr)
#         self.step_num = 0
#
#     def state_dict(self) -> Dict[str, Any]:
#         return {
#             "total_steps": self.total_steps,
#             "warmup_steps": self.warmup_steps,
#             "base_lr": self.base_lr,
#             "min_lr": self.min_lr,
#             "step_num": self.step_num,
#         }
#
#     def load_state_dict(self, sd: Dict[str, Any]):
#         self.total_steps = int(sd["total_steps"])
#         self.warmup_steps = int(sd["warmup_steps"])
#         self.base_lr = float(sd["base_lr"])
#         self.min_lr = float(sd["min_lr"])
#         self.step_num = int(sd["step_num"])
#
#     def _lr_at(self, t: int) -> float:
#         if t < self.warmup_steps:
#             return self.base_lr * (t + 1) / max(1, self.warmup_steps)
#         # cosine from base_lr -> min_lr
#         progress = (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
#         cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
#         return self.min_lr + (self.base_lr - self.min_lr) * cosine
#
#     def step(self) -> float:
#         lr = self._lr_at(self.step_num)
#         for pg in self.optimizer.param_groups:
#             pg["lr"] = lr
#         self.step_num += 1
#         return lr
#
#
# def build_amp_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
#     return torch.cuda.amp.GradScaler(enabled=bool(enabled))
#
#
# def compute_total_steps(num_epochs: int, steps_per_epoch: int) -> int:
#     return int(num_epochs) * int(steps_per_epoch)
# -*- coding: utf-8 -*-
"""
Optimizer / Scheduler utilities:
- AdamW
- Cosine decay with warmup
- Gradient clipping handled in engine
- AMP scaler helper
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple

import math
import torch
import torch.nn as nn


def _is_norm_layer(m: nn.Module) -> bool:
    return isinstance(
        m,
        (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.SyncBatchNorm, nn.InstanceNorm1d,
            nn.InstanceNorm2d, nn.InstanceNorm3d,
        ),
    )


def build_adamw(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    norm_param_ids = set()
    for _, m in model.named_modules():
        if _is_norm_layer(m):
            for p in m.parameters(recurse=False):
                norm_param_ids.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or id(p) in norm_param_ids or "norm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": float(weight_decay), "lr": float(base_lr)},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": float(base_lr)},
    ]
    return torch.optim.AdamW(param_groups, lr=float(base_lr), betas=betas)


class WarmupCosineLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.step_num = 0
        self._sync_optimizer_lrs(self._lr_at(0))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "step_num": self.step_num,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.total_steps = int(sd["total_steps"])
        self.warmup_steps = int(sd["warmup_steps"])
        self.base_lr = float(sd["base_lr"])
        self.min_lr = float(sd["min_lr"])
        self.step_num = int(sd["step_num"])
        self._sync_optimizer_lrs(self._lr_at(self.step_num))

    def _lr_at(self, t: int) -> float:
        if self.total_steps <= 0:
            return self.base_lr
        if t < self.warmup_steps:
            return self.base_lr * (t + 1) / max(1, self.warmup_steps)
        progress = (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def _sync_optimizer_lrs(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> float:
        lr = self._lr_at(self.step_num)
        self._sync_optimizer_lrs(lr)
        self.step_num += 1
        return lr


def build_amp_scaler(enabled: bool):
    enabled = bool(enabled) and torch.cuda.is_available()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_total_steps(num_epochs: int, steps_per_epoch: int) -> int:
    return int(num_epochs) * int(steps_per_epoch)
