"""
models/spiking_unet.py

Spiking U-Net Backbone
======================
Standard U-Net architecture with all ReLU activations replaced by
LIF (Leaky Integrate-and-Fire) spiking neurons from SpikingJelly.

Architecture:
  Encoder: 4 spiking convolutional blocks with max pooling
  Bridge:  Deepest feature extraction
  Decoder: 4 upsampling blocks with skip connections
  Head:    1×1 convolution → sigmoid (or softmax for multi-class)

Temporal dimension handling:
  The SNN processes T time steps independently in a time-loop.
  The final segmentation mask is obtained by averaging across T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.activation_based import neuron, layer, functional
    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False
    print("[WARNING] SpikingJelly not installed. Using mock LIF neurons.")


# ──────────────────────────────────────────────────────────────────────
# LIF neuron wrapper (graceful fallback if SpikingJelly not installed)
# ──────────────────────────────────────────────────────────────────────

def make_lif(cfg: dict) -> nn.Module:
    """Return a LIF neuron module with the given config."""
    if SPIKINGJELLY_AVAILABLE:
        return neuron.LIFNode(
            tau=cfg.get("tau", 2.0),
            v_threshold=cfg.get("v_threshold", 1.0),
            v_reset=cfg.get("v_reset", 0.0),
            surrogate_function=_get_surrogate(cfg.get("surrogate", "atan")),
            detach_reset=cfg.get("detach_reset", True),
        )
    else:
        # Fallback: simple ReLU (for environments without SpikingJelly)
        return nn.ReLU()


def _get_surrogate(name: str):
    """Return SpikingJelly surrogate gradient function by name."""
    if not SPIKINGJELLY_AVAILABLE:
        return None
    from spikingjelly.activation_based import surrogate
    surrogates = {
        "atan": surrogate.ATan(),
        "sigmoid": surrogate.Sigmoid(),
        "piecewise_quadratic": surrogate.PiecewiseQuadratic(),
    }
    return surrogates.get(name.lower(), surrogate.ATan())


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────

class SpikingConvBlock(nn.Module):
    """
    Double spiking convolution block.
    BN → Conv → LIF → BN → Conv → LIF

    This ordering (BN before conv) is preferred for SNNs to stabilise
    membrane potential accumulation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        lif_cfg: dict,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_lif(lif_cfg),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_lif(lif_cfg),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class SpikingEncoderBlock(nn.Module):
    """Encoder stage: SpikingConvBlock + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, lif_cfg: dict):
        super().__init__()
        self.conv = SpikingConvBlock(in_ch, out_ch, lif_cfg)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class SpikingDecoderBlock(nn.Module):
    """Decoder stage: Upsample + concat skip + SpikingConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, lif_cfg: dict):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = SpikingConvBlock(in_ch + skip_ch, out_ch, lif_cfg)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────
# Spiking U-Net
# ──────────────────────────────────────────────────────────────────────

class SpikingUNet(nn.Module):
    """
    Spiking U-Net for event-based segmentation.

    Processes a spike tensor (T, B, C, H, W) by unrolling T time steps
    through a shared U-Net, then averaging the output spike maps.

    Args:
        cfg (dict): Model configuration dict. Expected keys:
            - in_channels (int)
            - out_channels (int)
            - time_steps (int)
            - encoder_channels (list)
            - decoder_channels (list)
            - lif (dict)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        in_ch   = cfg.get("in_channels", 2)
        out_ch  = cfg.get("out_channels", 1)
        T       = cfg.get("time_steps", 10)
        enc_chs = cfg.get("encoder_channels", [64, 128, 256, 512])
        dec_chs = cfg.get("decoder_channels", [256, 128, 64, 32])
        lif_cfg = cfg.get("lif", {})

        self.time_steps = T

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = SpikingEncoderBlock(in_ch,      enc_chs[0], lif_cfg)
        self.enc2 = SpikingEncoderBlock(enc_chs[0], enc_chs[1], lif_cfg)
        self.enc3 = SpikingEncoderBlock(enc_chs[1], enc_chs[2], lif_cfg)
        self.enc4 = SpikingEncoderBlock(enc_chs[2], enc_chs[3], lif_cfg)

        # ── Bridge ───────────────────────────────────────────
        self.bridge = SpikingConvBlock(enc_chs[3], enc_chs[3], lif_cfg)

        # ── Decoder ──────────────────────────────────────────
        self.dec4 = SpikingDecoderBlock(enc_chs[3], enc_chs[3], dec_chs[0], lif_cfg)
        self.dec3 = SpikingDecoderBlock(dec_chs[0], enc_chs[2], dec_chs[1], lif_cfg)
        self.dec2 = SpikingDecoderBlock(dec_chs[1], enc_chs[1], dec_chs[2], lif_cfg)
        self.dec1 = SpikingDecoderBlock(dec_chs[2], enc_chs[0], dec_chs[3], lif_cfg)

        # ── Segmentation Head ────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[3], out_ch, kernel_size=1),
        )

    # ------------------------------------------------------------------
    def _reset_neurons(self):
        """Reset SNN neuron states between batches."""
        if SPIKINGJELLY_AVAILABLE:
            functional.reset_net(self)

    # ------------------------------------------------------------------
    def _forward_single(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for a single time step.

        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, out_ch, H, W)
            skips: tuple of encoder skip tensors
        """
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bridge(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x), (s1, s2, s3, s4)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spike tensor (T, B, C, H, W).
        Returns:
            mask: Segmentation probability map (B, out_ch, H, W).
        """
        self._reset_neurons()
        T, B, C, H, W = x.shape
        outputs = []

        for t in range(T):
            out, _ = self._forward_single(x[t])
            outputs.append(out)

        # Average firing-rate decoding across time steps
        out = torch.stack(outputs, dim=0).mean(dim=0)  # (B, out_ch, H, W)
        return torch.sigmoid(out)

    # ------------------------------------------------------------------
    def forward_with_features(self, x: torch.Tensor):
        """
        Forward pass returning intermediate feature maps.
        Used for attention insertion in DTA-SNN.

        Args:
            x: (T, B, C, H, W)
        Returns:
            bridge_features: (T, B, C_bridge, H/16, W/16)
            outputs: list of (B, out_ch, H, W) per time step
        """
        self._reset_neurons()
        T = x.shape[0]
        bridge_feats = []
        outputs = []

        for t in range(T):
            xt = x[t]
            xt, s1 = self.enc1(xt)
            xt, s2 = self.enc2(xt)
            xt, s3 = self.enc3(xt)
            xt, s4 = self.enc4(xt)
            xt = self.bridge(xt)
            bridge_feats.append(xt)

            xt = self.dec4(xt, s4)
            xt = self.dec3(xt, s3)
            xt = self.dec2(xt, s2)
            xt = self.dec1(xt, s1)
            outputs.append(self.head(xt))

        bridge_feats = torch.stack(bridge_feats, dim=0)  # (T, B, C, h, w)
        return bridge_feats, outputs