"""
models/__init__.py
Model registry for SNN-DTA project.
"""

from .cnn_baseline import CNNUNet
from .spiking_unet import SpikingUNet
from .dta_snn import DTASNN
from .attention import DualTemporalChannelAttention
from .encoding import TemporalCorrelationEncoder


MODEL_REGISTRY = {
    "cnn": CNNUNet,
    "snn": SpikingUNet,
    "dta": DTASNN,
}


def build_model(name: str, cfg: dict):
    """
    Factory function. Returns an instantiated model.

    Args:
        name: one of 'cnn', 'snn', 'dta'
        cfg:  model config dict (from YAML)
    Returns:
        nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](cfg)


__all__ = [
    "CNNUNet",
    "SpikingUNet",
    "DTASNN",
    "DualTemporalChannelAttention",
    "TemporalCorrelationEncoder",
    "build_model",
    "MODEL_REGISTRY",
]