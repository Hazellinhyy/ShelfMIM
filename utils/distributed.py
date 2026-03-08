# -*- coding: utf-8 -*-
"""
Distributed helpers for DDP training.
Works on single-GPU/CPU too.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict

import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def setup_for_distributed(is_master: bool):
    """
    Disable printing when not master.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_seconds: int = 1800,
):
    """
    Initialize distributed mode using env vars:
      RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        setup_for_distributed(True)
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.timedelta(seconds=timeout_seconds) if hasattr(torch.distributed, "timedelta") else None,
    )
    dist.barrier()
    setup_for_distributed(rank == 0)


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """
    All-reduce mean for scalar tensors.
    """
    if not is_dist_avail_and_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= float(get_world_size())
    return y


@torch.no_grad()
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


@torch.no_grad()
def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a picklable python object from src to all ranks.
    """
    if not is_dist_avail_and_initialized():
        return obj
    obj_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
