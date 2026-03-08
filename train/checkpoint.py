# # -*- coding: utf-8 -*-
# """
# Checkpoint save/load:
# - model + optimizer + scheduler + amp scaler
# - includes prototype bank buffers (since they are buffers in model state_dict)
# - saves RNG states (python/random, numpy, torch, cuda)
# """
#
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
#
# import os
# import random
# import numpy as np
# import torch
#
#
# def _is_dist() -> bool:
#     return torch.distributed.is_available() and torch.distributed.is_initialized()
#
#
# def _rank() -> int:
#     return torch.distributed.get_rank() if _is_dist() else 0
#
#
# def _save_rng_state() -> Dict[str, Any]:
#     state = {
#         "python_random": random.getstate(),
#         "numpy_random": np.random.get_state(),
#         "torch_cpu": torch.get_rng_state(),
#     }
#     if torch.cuda.is_available():
#         state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
#     return state
#
#
# def _load_rng_state(state: Dict[str, Any]):
#     random.setstate(state["python_random"])
#     np.random.set_state(state["numpy_random"])
#     torch.set_rng_state(state["torch_cpu"])
#     if torch.cuda.is_available() and "torch_cuda_all" in state:
#         torch.cuda.set_rng_state_all(state["torch_cuda_all"])
#
#
# @dataclass
# class CheckpointManager:
#     out_dir: str
#
#     def __post_init__(self):
#         os.makedirs(self.out_dir, exist_ok=True)
#
#     def save(
#         self,
#         *,
#         name: str,
#         model: torch.nn.Module,
#         optimizer: Optional[torch.optim.Optimizer] = None,
#         scheduler: Optional[Any] = None,
#         scaler: Optional[torch.cuda.amp.GradScaler] = None,
#         epoch: int = 0,
#         step: int = 0,
#         extra: Optional[Dict[str, Any]] = None,
#     ) -> str:
#         # only rank0 saves
#         if _rank() != 0:
#             return ""
#
#         path = os.path.join(self.out_dir, name)
#         ckpt = {
#             "epoch": int(epoch),
#             "step": int(step),
#             "model": model.state_dict(),
#             "rng": _save_rng_state(),
#         }
#         if optimizer is not None:
#             ckpt["optimizer"] = optimizer.state_dict()
#         if scheduler is not None and hasattr(scheduler, "state_dict"):
#             ckpt["scheduler"] = scheduler.state_dict()
#         if scaler is not None:
#             ckpt["scaler"] = scaler.state_dict()
#         if extra is not None:
#             ckpt["extra"] = extra
#
#         torch.save(ckpt, path)
#         return path
#
#     def load(
#         self,
#         path: str,
#         *,
#         model: torch.nn.Module,
#         optimizer: Optional[torch.optim.Optimizer] = None,
#         scheduler: Optional[Any] = None,
#         scaler: Optional[torch.cuda.amp.GradScaler] = None,
#         map_location: str = "cpu",
#         strict: bool = True,
#         load_rng: bool = True,
#     ) -> Dict[str, Any]:
#         ckpt = torch.load(path, map_location=map_location)
#         model.load_state_dict(ckpt["model"], strict=strict)
#
#         if optimizer is not None and "optimizer" in ckpt:
#             optimizer.load_state_dict(ckpt["optimizer"])
#         if scheduler is not None and "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
#             scheduler.load_state_dict(ckpt["scheduler"])
#         if scaler is not None and "scaler" in ckpt:
#             scaler.load_state_dict(ckpt["scaler"])
#
#         if load_rng and "rng" in ckpt:
#             _load_rng_state(ckpt["rng"])
#
#         return {
#             "epoch": int(ckpt.get("epoch", 0)),
#             "step": int(ckpt.get("step", 0)),
#             "extra": ckpt.get("extra", {}),
#         }
# -*- coding: utf-8 -*-
"""
Checkpoint save/load:
- model + optimizer + scheduler + amp scaler
- includes prototype bank buffers (since they are buffers in model state_dict)
- saves RNG states (python/random, numpy, torch, cuda)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import random
import numpy as np
import torch


def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _is_dist() else 0


def _save_rng_state() -> Dict[str, Any]:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state(state: Dict[str, Any]):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


@dataclass
class CheckpointManager:
    out_dir: str

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)

    def save(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if _rank() != 0:
            return ""

        path = os.path.join(self.out_dir, name)
        tmp_path = path + ".tmp"
        ckpt = {
            "epoch": int(epoch),
            "step": int(step),
            "model": model.state_dict(),
            "rng": _save_rng_state(),
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            ckpt["scheduler"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            ckpt["scaler"] = scaler.state_dict()
        if extra is not None:
            ckpt["extra"] = extra

        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load(
        self,
        path: str,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        map_location: str = "cpu",
        strict: bool = True,
        load_rng: bool = True,
    ) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model"], strict=strict)

        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(ckpt["scaler"])

        if load_rng and "rng" in ckpt:
            _load_rng_state(ckpt["rng"])

        return {
            "epoch": int(ckpt.get("epoch", 0)),
            "step": int(ckpt.get("step", 0)),
            "extra": ckpt.get("extra", {}),
        }
