"""
utils/metrics.py

Evaluation Metrics
==================
IoU, F1, Precision, Recall for binary segmentation.
"""

from __future__ import annotations
from collections import defaultdict
import torch
import numpy as np


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute binary segmentation metrics.

    Args:
        pred:   (B, 1, H, W) float probability map [0, 1].
        target: (B, 1, H, W) float binary ground truth.
        threshold: binarisation threshold.

    Returns:
        dict with keys: iou, f1, precision, recall, accuracy
    """
    pred_bin = (pred > threshold).float()
    target   = target.float()

    TP = (pred_bin * target).sum().item()
    FP = (pred_bin * (1 - target)).sum().item()
    FN = ((1 - pred_bin) * target).sum().item()
    TN = ((1 - pred_bin) * (1 - target)).sum().item()

    eps = 1e-7
    iou       = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / (TP + FP + FN + TN + eps)

    return {
        "iou":       iou,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "accuracy":  accuracy,
    }


class MetricsTracker:
    """Accumulates and averages metrics across batches."""

    def __init__(self):
        self._sums   = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, metrics: dict, count: int = 1):
        for k, v in metrics.items():
            self._sums[k]   += v * count
            self._counts[k] += count

    def mean(self) -> dict:
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self):
        self._sums.clear()
        self._counts.clear()