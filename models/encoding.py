"""
models/encoding.py

Temporal Correlation Encoding (TCE) — Patent-Pending Component
===============================================================
Converts raw event voxel grids into spike trains that emphasize
transparent-object edge structures by exploiting temporal correlations
between adjacent event bins.

Key idea: transparent object boundaries produce burst-like event patterns
that correlate strongly across adjacent time windows. TCE captures this
by computing normalised cross-correlations between successive time bins,
boosting edge-like activations while suppressing background noise.

Copyright (c) 2024. Patent Pending — All rights reserved for commercial use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TemporalCorrelationEncoder(nn.Module):
    """
    Temporal Correlation Encoding for Event Voxel Grids.

    Input shape:  (B, C, T, H, W)  — C=2 for pos/neg polarity
    Output shape: (T, B, C', H, W) — spike tensor for SNN processing

    where C' = C * (1 + window_size - 1) correlation channels.

    Args:
        in_channels (int):    Input event channels (default 2: pos + neg).
        out_channels (int):   Output encoding channels.
        time_steps (int):     Number of temporal bins T.
        window_size (int):    Number of adjacent bins to correlate.
        threshold_ratio (float): Fraction of max value used for spike threshold.
        normalize_polarity (bool): L2-normalize across polarity dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        time_steps: int = 10,
        window_size: int = 3,
        threshold_ratio: float = 0.1,
        normalize_polarity: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_steps = time_steps
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio
        self.normalize_polarity = normalize_polarity

        # Learnable temporal mixing kernel
        self.temporal_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(window_size, 1, 1),
            padding=(window_size // 2, 0, 0),
            bias=False,
        )
        nn.init.kaiming_normal_(self.temporal_conv.weight, mode="fan_out")

        # Learnable spatial edge enhancement (Sobel-initialised)
        self.edge_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self._init_edge_conv()

        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.SiLU()

    # ------------------------------------------------------------------
    def _init_edge_conv(self):
        """Initialise depthwise conv with Sobel-like kernels."""
        with torch.no_grad():
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            )
            sobel_y = sobel_x.T
            for i in range(self.out_channels):
                k = sobel_x if i % 2 == 0 else sobel_y
                self.edge_conv.weight[i, 0] = k

    # ------------------------------------------------------------------
    def _compute_temporal_correlation(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalised cross-correlation across adjacent time bins.

        Args:
            x: (B, C, T, H, W)
        Returns:
            corr: (B, C, T, H, W) correlation-enhanced tensor
        """
        B, C, T, H, W = x.shape
        # Pad along time dimension for boundary handling
        x_pad = F.pad(x, (0, 0, 0, 0, 1, 1))  # pad T dim

        corr = torch.zeros_like(x)
        for t in range(T):
            # Current, previous, next bins
            curr = x_pad[:, :, t + 1]         # (B, C, H, W)
            prev = x_pad[:, :, t]
            nxt  = x_pad[:, :, t + 2]

            # Normalised correlation: E[curr * prev] / (sigma_curr * sigma_prev + eps)
            eps = 1e-6
            c_prev = (curr * prev).mean(dim=(-2, -1), keepdim=True) / (
                curr.std(dim=(-2, -1), keepdim=True)
                * prev.std(dim=(-2, -1), keepdim=True)
                + eps
            )
            c_next = (curr * nxt).mean(dim=(-2, -1), keepdim=True) / (
                curr.std(dim=(-2, -1), keepdim=True)
                * nxt.std(dim=(-2, -1), keepdim=True)
                + eps
            )
            corr[:, :, t] = curr * (1.0 + 0.5 * c_prev + 0.5 * c_next)

        return corr

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Event voxel grid (B, C, T, H, W), values in [-1, 1].
        Returns:
            spikes: Spike tensor (T, B, C', H, W).
        """
        B, C, T, H, W = x.shape

        # 1. Optional L2 normalisation across polarity channels
        if self.normalize_polarity:
            x = F.normalize(x, p=2, dim=1)

        # 2. Temporal correlation enhancement
        x = self._compute_temporal_correlation(x)  # (B, C, T, H, W)

        # 3. Learnable temporal mixing
        x = self.temporal_conv(x)                  # (B, C', T, H, W)
        x = self.bn(x)
        x = self.activation(x)

        # 4. Per-timestep spatial edge enhancement
        x_out = []
        for t in range(T):
            frame = x[:, :, t]                     # (B, C', H, W)
            frame = self.edge_conv(frame)
            x_out.append(frame)
        x = torch.stack(x_out, dim=2)             # (B, C', T, H, W)

        # 5. Threshold to generate spikes (hard threshold with STE)
        threshold = self.threshold_ratio * x.abs().amax(dim=(2, 3, 4), keepdim=True)
        spikes = (x.abs() >= threshold).float() * x.sign()

        # 6. Rearrange to SNN format (T, B, C', H, W)
        spikes = rearrange(spikes, "b c t h w -> t b c h w")
        return spikes


# ──────────────────────────────────────────────────────────────────────
class SimpleVoxelEncoder(nn.Module):
    """
    Lightweight fallback encoder: directly reshape voxel grid to (T,B,C,H,W).
    Used for ablation studies.
    """

    def __init__(self, in_channels: int = 2, time_steps: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) or (B, C*T, H, W)
        Returns:
            (T, B, C, H, W)
        """
        if x.dim() == 4:
            B, CT, H, W = x.shape
            T = CT // self.in_channels
            x = x.view(B, self.in_channels, T, H, W)
        return rearrange(x, "b c t h w -> t b c h w")