"""
models/cnn_baseline.py

CNN U-Net Baseline
==================
Standard convolutional U-Net baseline for transparent object segmentation.
Uses frame-averaged event voxel grids as input (no temporal processing).

Purpose:
  - Validate that the data pipeline and dataset are learnable.
  - Provide a performance upper-bound for standard approaches.
  - Confirm the advantage of the SNN temporal processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class CNNUNet(nn.Module):
    """
    Standard U-Net baseline.

    Input:  (B, C*T, H, W)  — time-collapsed event voxel grid
            OR (B, C, H, W) — single-frame event frame
    Output: (B, 1, H, W)    — segmentation probability map

    Args:
        cfg (dict): Model config. Expected keys:
            - in_channels (int): number of input channels (C or C*T)
            - out_channels (int)
            - encoder_channels (list)
            - decoder_channels (list)
            - dropout (float)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        in_ch   = cfg.get("in_channels", 2)
        out_ch  = cfg.get("out_channels", 1)
        enc_chs = cfg.get("encoder_channels", [64, 128, 256, 512])
        dec_chs = cfg.get("decoder_channels", [256, 128, 64, 32])
        dropout = cfg.get("dropout", 0.2)

        # Collapse time dimension if needed
        # The CNN baseline takes mean over T before processing
        self.time_steps = cfg.get("time_steps", 10)
        self.in_channels = in_ch

        # Encoder
        self.enc1 = EncoderBlock(in_ch,      enc_chs[0])
        self.enc2 = EncoderBlock(enc_chs[0], enc_chs[1])
        self.enc3 = EncoderBlock(enc_chs[1], enc_chs[2])
        self.enc4 = EncoderBlock(enc_chs[2], enc_chs[3])

        # Bridge
        self.bridge = ConvBlock(enc_chs[3], enc_chs[3], dropout)

        # Decoder
        self.dec4 = DecoderBlock(enc_chs[3], enc_chs[3], dec_chs[0])
        self.dec3 = DecoderBlock(dec_chs[0], enc_chs[2], dec_chs[1])
        self.dec2 = DecoderBlock(dec_chs[1], enc_chs[1], dec_chs[2])
        self.dec1 = DecoderBlock(dec_chs[2], enc_chs[0], dec_chs[3])

        # Head
        self.head = nn.Conv2d(dec_chs[3], out_ch, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C, H, W) OR (B, C, H, W)
        Returns:
            mask: (B, 1, H, W)
        """
        # Accept spike tensor format (T, B, C, H, W) and collapse time
        if x.dim() == 5:
            x = x.mean(dim=0)  # (B, C, H, W) — temporal average

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bridge(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return torch.sigmoid(self.head(x))