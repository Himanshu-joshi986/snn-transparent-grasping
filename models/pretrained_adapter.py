"""
models/pretrained_adapter.py

Pretrained SNN Backbone Adapter
================================
Adapts weights from pretrained SNN models (SpikingYOLOX, SpikeSMOKE)
for initialisation of the DTASNN encoder backbone.

Usage:
    adapter = PretrainedAdapter(pretrained_type="spikingyolox")
    dta_model = adapter.load_and_adapt(
        dta_model,
        checkpoint_path="path/to/spikingyolox.pth",
        freeze_epochs=5,
    )

SpikingYOLOX Reference:
    "SpikingYOLOX: Object Detection with Spiking Neural Networks"
    https://arxiv.org/abs/2306.xxxxx
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PretrainedAdapter:
    """
    Adapter for loading pretrained SNN encoder weights.

    Handles the weight name mapping between SpikingYOLOX / SpikeSMOKE
    backbone layers and the DTASNN encoder layers.

    Args:
        pretrained_type: "spikingyolox" | "spikesmoke" | "custom"
        strict: Whether to require all keys to match (default False).
    """

    # Map from SpikingYOLOX layer names → DTASNN encoder layer names
    YOLOX_TO_DTA = {
        "backbone.stem.":          "enc1.conv.block.",
        "backbone.dark2.":         "enc2.conv.block.",
        "backbone.dark3.":         "enc3.conv.block.",
        "backbone.dark4.":         "enc4.conv.block.",
    }

    SMOKE_TO_DTA = {
        "backbone.layer1.":        "enc1.conv.block.",
        "backbone.layer2.":        "enc2.conv.block.",
        "backbone.layer3.":        "enc3.conv.block.",
        "backbone.layer4.":        "enc4.conv.block.",
    }

    def __init__(self, pretrained_type: str = "spikingyolox", strict: bool = False):
        self.pretrained_type = pretrained_type
        self.strict = strict
        self._mapping = (
            self.YOLOX_TO_DTA if pretrained_type == "spikingyolox"
            else self.SMOKE_TO_DTA
        )

    # ------------------------------------------------------------------
    def _remap_state_dict(
        self, pretrained_sd: dict, model_sd: dict
    ) -> dict:
        """
        Remap pretrained state dict keys to DTA-SNN naming convention.

        Returns:
            adapted_sd: filtered + renamed state dict ready for load_state_dict
            stats: dict with n_loaded / n_total counts
        """
        adapted = {}
        n_total = len(pretrained_sd)
        n_loaded = 0

        for old_key, tensor in pretrained_sd.items():
            new_key = old_key
            for prefix_old, prefix_new in self._mapping.items():
                if old_key.startswith(prefix_old):
                    new_key = old_key.replace(prefix_old, prefix_new, 1)
                    break

            if new_key in model_sd:
                if model_sd[new_key].shape == tensor.shape:
                    adapted[new_key] = tensor
                    n_loaded += 1
                else:
                    logger.debug(
                        f"Shape mismatch: {new_key} "
                        f"({tensor.shape} vs {model_sd[new_key].shape}). Skipping."
                    )
            else:
                logger.debug(f"Key not found in target model: {new_key}. Skipping.")

        logger.info(
            f"Pretrained adapter: loaded {n_loaded}/{n_total} parameters "
            f"({100 * n_loaded / max(n_total, 1):.1f}% coverage)."
        )
        return adapted, {"n_loaded": n_loaded, "n_total": n_total}

    # ------------------------------------------------------------------
    def load_and_adapt(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str],
        freeze_epochs: int = 5,
    ) -> nn.Module:
        """
        Load pretrained weights into the model encoder.

        Args:
            model: DTASNN instance.
            checkpoint_path: Path to .pth checkpoint file.
            freeze_epochs: Number of training epochs to freeze encoder.
                           Set to 0 to not freeze.
        Returns:
            model with pretrained encoder weights loaded.
        """
        if checkpoint_path is None:
            logger.warning("No checkpoint path provided. Skipping pretrained loading.")
            return model

        path = Path(checkpoint_path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {path}. Skipping.")
            return model

        ckpt = torch.load(path, map_location="cpu")
        pretrained_sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

        model_sd = model.state_dict()
        adapted_sd, stats = self._remap_state_dict(pretrained_sd, model_sd)

        # Load adapted weights
        missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
        logger.info(f"Missing keys: {len(missing)}. Unexpected keys: {len(unexpected)}.")

        # Optionally freeze encoder
        if freeze_epochs > 0:
            self._freeze_encoder(model)
            logger.info(
                f"Encoder frozen for {freeze_epochs} epochs. "
                "Call adapter.unfreeze_encoder(model) when ready."
            )

        return model

    # ------------------------------------------------------------------
    @staticmethod
    def freeze_encoder(model: nn.Module):
        """Freeze encoder parameters (enc1–enc4 + bridge)."""
        for name in ["enc1", "enc2", "enc3", "enc4", "bridge_conv"]:
            module = getattr(model, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
        logger.info("Encoder parameters frozen.")

    @staticmethod
    def unfreeze_encoder(model: nn.Module):
        """Unfreeze all parameters for full fine-tuning."""
        for p in model.parameters():
            p.requires_grad = True
        logger.info("All parameters unfrozen.")

    # ------------------------------------------------------------------
    @staticmethod
    def print_parameter_stats(model: nn.Module):
        """Print trainable / frozen parameter counts per module."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n{'─'*50}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  Frozen parameters:    {frozen:>12,}")
        print(f"{'─'*50}\n")