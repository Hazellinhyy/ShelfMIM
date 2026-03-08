# -*- coding: utf-8 -*-
"""
Misc utilities:
- seed fixing
- stopgrad wrapper
- cosine similarity
- shape checks
- average meter
- simple timer
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a,b: (...,D)
    returns: (...) cosine similarity
    """
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


def assert_shape(t: torch.Tensor, shape: Tuple[Optional[int], ...], name: str = "tensor"):
    """
    shape: tuple with ints or None (None means don't check that dim)
    """
    if t.dim() != len(shape):
        raise ValueError(f"{name} dim mismatch: expected {len(shape)}D, got {t.dim()}D shape={tuple(t.shape)}")
    for i, s in enumerate(shape):
        if s is None:
            continue
        if t.shape[i] != s:
            raise ValueError(f"{name} shape mismatch at dim {i}: expected {s}, got {t.shape[i]} full={tuple(t.shape)}")


@dataclass
class AverageMeter:
    """
    Track running average for scalars.
    """
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class SmoothedDict:
    """
    Store multiple AverageMeters.
    """
    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(float(v), n=1)

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}


@dataclass
class Timer:
    start: float = 0.0

    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        import time
        self.end = time.time()
        self.elapsed = self.end - self.start
        return False
