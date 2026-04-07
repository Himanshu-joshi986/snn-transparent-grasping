"""
models/attention.py

Dual Temporal-Channel Attention (DTA) — PATENT PENDING
=======================================================
Novel attention mechanism for Spiking Neural Networks that operates
jointly over temporal spike sequences and feature channels.

Invention Summary:
    Standard channel attention (SE-Net) ignores temporal spike dynamics.
    Standard temporal attention ignores inter-channel feature correlations.
    DTA uniquely couples both axes through a joint spatio-temporal gating
    mechanism designed specifically for spike-based neural representations.

    The DTA module:
    1. Temporal Gate: attends over the T time-step axis to identify
       which spike epochs contain boundary-relevant activity.
    2. Channel Gate: attends over the C feature axis to weight channels
       by their discriminative contribution to transparent object edges.
    3. Cross-coupling: temporal and channel gates are coupled through
       a learned interaction matrix, capturing spatio-temporal correlations
       that are unique to event-camera transparent object signatures.

Legal Notice:
    This module is covered by a provisional patent application filed
    under 35 U.S.C. § 111(b). Commercial reproduction or use without
    an explicit license from the inventors is prohibited.

Copyright (c) 2024. All rights reserved.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


# ══════════════════════════════════════════════════════════════════════
# Sub-modules
# ══════════════════════════════════════════════════════════════════════

class TemporalGate(nn.Module):
    """
    Multi-head attention over the time dimension of spike tensors.

    Computes a soft gate g_t ∈ [0,1]^T indicating which time steps
    carry boundary-discriminative information for transparent objects.

    Input:  (T, B, C, H, W) spike tensor
    Output: (T, B, 1, 1, 1) temporal gate weights
    """

    def __init__(self, time_steps: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_steps = time_steps
        self.num_heads = num_heads
        head_dim = max(1, time_steps // num_heads)

        # Project flattened spatial spike statistics to query/key/value
        self.qkv = nn.Linear(3, num_heads * head_dim * 3, bias=False)
        self.proj = nn.Linear(num_heads * head_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C, H, W)
        Returns:
            gate: (T, B, 1, 1, 1) — normalised temporal weights
        """
        T, B, C, H, W = x.shape

        # Aggregate spatial info per time step → (T, B, 3)
        # Statistics: mean, std, firing rate (fraction of non-zero spikes)
        x_mean = x.mean(dim=(2, 3, 4))                    # (T, B)
        x_std  = x.std(dim=(2, 3, 4))                     # (T, B)
        x_rate = (x.abs() > 0).float().mean(dim=(2, 3, 4))  # (T, B)
        stats  = torch.stack([x_mean, x_std, x_rate], dim=-1)  # (T, B, 3)

        # Rearrange to (B, T, 3)
        stats = rearrange(stats, "t b s -> b t s")

        # Multi-head attention (self-attention over time steps)
        qkv = self.qkv(stats)                              # (B, T, 3*H*d)
        q, k, v = qkv.chunk(3, dim=-1)                    # each (B, T, H*d)

        # Scaled dot-product attention
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out  = torch.bmm(attn, v)                          # (B, T, H*d)

        # Project to scalar gate per time step
        gate = torch.sigmoid(self.proj(out))               # (B, T, 1)
        gate = rearrange(gate, "b t 1 -> t b 1 1 1")
        return gate


class ChannelGate(nn.Module):
    """
    Squeeze-and-Excitation style channel gate with temporal aggregation.

    Aggregates across the time axis before computing channel importance,
    making it spike-temporal-aware unlike standard SE blocks.

    Input:  (T, B, C, H, W) spike tensor
    Output: (1, B, C, 1, 1) channel gate weights
    """

    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C, H, W)
        Returns:
            gate: (1, B, C, 1, 1)
        """
        # Global average pool across T, H, W → (B, C)
        z = reduce(x, "t b c h w -> b c", "mean")

        # Excitation
        s = self.act(self.fc1(z))                          # (B, mid)
        s = self.dropout(s)
        s = torch.sigmoid(self.fc2(s))                     # (B, C)
        gate = rearrange(s, "b c -> 1 b c 1 1")
        return gate


class CrossCouplingLayer(nn.Module):
    """
    Cross-coupling matrix between temporal and channel gates.

    This is the core novel component: instead of applying temporal and
    channel gates independently, they are coupled through a learned
    T×C interaction matrix M, so that:

        combined_gate = softplus(g_t ⊗ g_c + M * (g_t × g_c^T))

    where ⊗ is element-wise broadcast and × is outer product.
    This allows the model to learn that certain combinations of
    time steps AND channels are jointly important — a capability
    neither pure temporal nor pure channel attention provides.

    Args:
        time_steps (int): T
        channels (int):   C
    """

    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels

        # Learnable T×C interaction matrix
        self.M = nn.Parameter(
            torch.zeros(time_steps, channels), requires_grad=True
        )
        nn.init.xavier_uniform_(self.M.unsqueeze(0)).squeeze(0)

        self.bias = nn.Parameter(torch.ones(1))

    def forward(
        self, g_t: torch.Tensor, g_c: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            g_t: (T, B, 1, 1, 1) — temporal gate
            g_c: (1, B, C, 1, 1) — channel gate
        Returns:
            gate: (T, B, C, 1, 1) — coupled gate
        """
        T, B = g_t.shape[:2]
        C    = g_c.shape[2]

        # Independent product: (T, B, C, 1, 1)
        indep = g_t * g_c                                  # broadcast

        # Outer product: g_t (T,B) × g_c (B,C) → (T, B, C)
        g_t_flat = g_t.squeeze(-1).squeeze(-1).squeeze(-1)    # (T, B)
        g_c_flat = g_c.squeeze(0).squeeze(-1).squeeze(-1)     # (B, C)
        outer = torch.einsum("tb,bc->tbc", g_t_flat, g_c_flat)  # (T, B, C)

        # Interaction term: M (T,C) element-wise with outer
        interact = self.M.unsqueeze(1) * outer             # (T, B, C)
        interact = rearrange(interact, "t b c -> t b c 1 1")

        # Coupled gate
        gate = F.softplus(indep + interact + self.bias)
        # Normalise to [0, 1] range
        gate = gate / (gate.amax(dim=(0, 2), keepdim=True) + 1e-6)
        return gate


# ══════════════════════════════════════════════════════════════════════
# Main Module
# ══════════════════════════════════════════════════════════════════════

class DualTemporalChannelAttention(nn.Module):
    """
    Dual Temporal-Channel Attention (DTA) — Patent Pending.

    Full module combining TemporalGate, ChannelGate, and CrossCouplingLayer.

    Input:  (T, B, C, H, W) spike feature tensor
    Output: (T, B, C, H, W) attended spike feature tensor (same shape)

    Usage:
        dta = DualTemporalChannelAttention(
            time_steps=10, channels=512, num_heads=4, reduction=16
        )
        x_attended = dta(x_spike)

    Reference:
        "Dual Temporal-Channel Attention for Event-Based Transparent
         Object Grasping with Spiking Neural Networks"
        [Paper in preparation, ICRA 2025]
    """

    def __init__(
        self,
        time_steps: int,
        channels: int,
        num_heads: int = 4,
        reduction: int = 16,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.residual = residual

        self.temporal_gate = TemporalGate(
            time_steps=time_steps,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.channel_gate = ChannelGate(
            channels=channels,
            reduction=reduction,
            dropout=dropout,
        )
        self.cross_coupling = CrossCouplingLayer(
            time_steps=time_steps,
            channels=channels,
        )

        # Layer norm for stability (applied over C, H, W per time step)
        self.norm = nn.GroupNorm(
            num_groups=min(8, channels),
            num_channels=channels,
        )

        # Output projection to restore original feature distribution
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.eye_(
            self.out_proj.weight.view(channels, channels)
            if self.out_proj.weight.shape[-2:] == (1, 1)
            else self.out_proj.weight.view(channels, -1)[:channels]
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C, H, W) spike feature tensor.
        Returns:
            out: (T, B, C, H, W) attention-modulated spike tensor.
        """
        T, B, C, H, W = x.shape
        assert T == self.time_steps, (
            f"Expected T={self.time_steps}, got {T}"
        )

        # Compute gates
        g_t = self.temporal_gate(x)                        # (T, B, 1, 1, 1)
        g_c = self.channel_gate(x)                         # (1, B, C, 1, 1)

        # Cross-coupled gate
        gate = self.cross_coupling(g_t, g_c)               # (T, B, C, 1, 1)

        # Apply gate
        x_attended = x * gate                              # (T, B, C, H, W)

        # Per-timestep normalisation + projection
        x_out = []
        for t in range(T):
            frame = x_attended[t]                          # (B, C, H, W)
            frame = self.norm(frame)
            frame = self.out_proj(frame)
            x_out.append(frame)
        x_attended = torch.stack(x_out, dim=0)

        # Residual connection
        if self.residual:
            return x + x_attended
        return x_attended

    # ------------------------------------------------------------------
    def get_attention_maps(self, x: torch.Tensor):
        """
        Returns intermediate gate tensors for visualisation / paper figures.

        Returns:
            dict with keys: 'temporal_gate', 'channel_gate', 'coupled_gate'
        """
        with torch.no_grad():
            g_t = self.temporal_gate(x)
            g_c = self.channel_gate(x)
            gate = self.cross_coupling(g_t, g_c)
        return {
            "temporal_gate": g_t.squeeze(),   # (T,)
            "channel_gate":  g_c.squeeze(),   # (C,)
            "coupled_gate":  gate.squeeze(),  # (T, C)
        }