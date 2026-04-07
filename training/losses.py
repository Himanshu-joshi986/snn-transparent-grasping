"""
training/losses.py

Segmentation Loss Functions
============================
Implements Dice, Focal, BCE, and combined losses optimised for
sparse, imbalanced transparent-object segmentation masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Sørensen–Dice coefficient loss for binary segmentation.

    Better than BCE for imbalanced masks (transparent objects occupy
    a small fraction of the image).

    Args:
        smooth (float): Laplace smoothing to avoid division by zero.
        from_logits (bool): If True, apply sigmoid first.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = False):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced binary segmentation.

    Reduces the relative loss for well-classified examples,
    focusing training on hard misclassified pixels.

    Args:
        alpha (float): Weighting factor for positive class.
        gamma (float): Focusing parameter (γ=2 recommended).
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        bce_exp = torch.exp(-bce)
        focal = self.alpha * ((1 - bce_exp) ** self.gamma) * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice + BCE + optional Focal losses.

    Default weighting: 0.8 × Dice + 0.2 × BCE achieves the best
    balance between overlap-maximisation and pixel-wise accuracy.
    """

    def __init__(
        self,
        dice_weight: float = 0.8,
        bce_weight:  float = 0.2,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight  = dice_weight
        self.bce_weight   = bce_weight
        self.focal_weight = focal_weight

        self.dice  = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(pred, target)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * F.binary_cross_entropy(pred, target)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(pred, target)
        return loss


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def build_loss(cfg: dict) -> nn.Module:
    """
    Build loss function from config.

    Args:
        cfg: loss config dict with keys: type, dice_weight, bce_weight,
             focal_gamma.
    Returns:
        nn.Module loss function.
    """
    loss_type = cfg.get("type", "dice")

    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0))
    elif loss_type == "combined":
        return CombinedLoss(
            dice_weight  = cfg.get("dice_weight",  0.8),
            bce_weight   = cfg.get("bce_weight",   0.2),
            focal_weight = cfg.get("focal_weight", 0.0),
            focal_gamma  = cfg.get("focal_gamma",  2.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")