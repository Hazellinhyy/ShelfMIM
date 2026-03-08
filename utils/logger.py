# -*- coding: utf-8 -*-
"""
Logger utilities:
- TensorBoard SummaryWriter
- Optional Weights & Biases (wandb)
- DDP-safe: only rank0 writes
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import os
import time

from utils.distributed import is_main_process

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    SummaryWriter = None
    _HAS_TB = False


class TBLogger:
    def __init__(self, log_dir: str):
        if not _HAS_TB:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, values: Dict[str, float], step: int):
        self.writer.add_scalars(main_tag, values, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class WandBLogger:
    def __init__(self, project: str, name: str, config: Optional[Dict[str, Any]] = None, dir: str = "./wandb"):
        import wandb
        os.makedirs(dir, exist_ok=True)
        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config, dir=dir)

    def log(self, values: Dict[str, float], step: int):
        self.wandb.log(values, step=step)

    def finish(self):
        self.wandb.finish()


@dataclass
class LoggerConfig:
    output_dir: str
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "ShelfMIM"
    wandb_name: str = "run"
    wandb_dir: str = "./wandb"


class MetricLogger:
    """
    Minimal scalar logger for console + TB/WandB.
    """
    def __init__(self, cfg: LoggerConfig, run_config: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.tb = None
        self.wb = None

        self.log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        if is_main_process():
            if cfg.use_tensorboard:
                self.tb = TBLogger(self.log_dir)
            if cfg.use_wandb:
                self.wb = WandBLogger(
                    project=cfg.wandb_project,
                    name=cfg.wandb_name,
                    config=run_config,
                    dir=cfg.wandb_dir,
                )

        self.start_time = time.time()

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = ""):
        if not is_main_process():
            return
        # console
        msg = f"[step {step}] " + " ".join([f"{prefix}{k}={v:.4f}" for k, v in scalars.items()])
        print(msg)

        # tensorboard
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(prefix + k, float(v), step)
            self.tb.flush()

        # wandb
        if self.wb is not None:
            wb_vals = {prefix + k: float(v) for k, v in scalars.items()}
            self.wb.log(wb_vals, step=step)

    def close(self):
        if not is_main_process():
            return
        if self.tb is not None:
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()
