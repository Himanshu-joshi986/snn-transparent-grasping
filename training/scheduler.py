"""
training/scheduler.py

Learning Rate Schedulers
=========================
Cosine annealing with linear warmup — standard for SNN training where
early epochs need stability for membrane potential initialisation.
"""

from __future__ import annotations
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    Args:
        optimizer:      PyTorch optimizer.
        warmup_epochs:  Number of linear warmup epochs.
        T_max:          Total training epochs (for cosine period).
        eta_min:        Minimum learning rate.
        last_epoch:     Resume epoch (-1 = start fresh).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        T_max: int,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * factor for base_lr in self.base_lrs]

        # Cosine annealing after warmup
        progress = (epoch - self.warmup_epochs) / max(
            self.T_max - self.warmup_epochs, 1
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer: Optimizer, cfg: dict) -> _LRScheduler:
    """
    Build scheduler from config.

    Supports: 'cosine_warmup' | 'cosine' | 'step' | 'plateau'
    """
    sched_type = cfg["scheduler"]["type"]

    if sched_type == "cosine_warmup":
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs = cfg["scheduler"].get("warmup_epochs", 5),
            T_max         = cfg["scheduler"].get("T_max", cfg["training"]["epochs"]),
            eta_min       = cfg["scheduler"].get("eta_min", 1e-7),
        )
    elif sched_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max   = cfg["scheduler"].get("T_max", cfg["training"]["epochs"]),
            eta_min = cfg["scheduler"].get("eta_min", 1e-7),
        )
    elif sched_type == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif sched_type == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")