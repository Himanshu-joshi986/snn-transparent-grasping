"""
utils/visualization.py

Visualization Utilities
=======================
Visualization functions for spike rasters, attention maps, and segmentation masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

logger = logging.getLogger(__name__)


def plot_spike_raster(
    spikes: np.ndarray,
    x_idx: int = 0,
    y_idx: int = 0,
    output_path: Optional[Path] = None,
    title: str = "Spike Raster",
) -> None:
    """
    Plot spike raster for a single pixel.

    Args:
        spikes: Spike tensor (T, H, W)
        x_idx: X pixel coordinate
        y_idx: Y pixel coordinate
        output_path: Optional path to save figure
        title: Plot title
    """
    if plt is None:
        logger.warning("Matplotlib not installed, skipping visualization")
        return

    spike_times = np.where(spikes[:, y_idx, x_idx] > 0)[0]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(spike_times, [1] * len(spike_times), marker='|', s=100, color='black')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Neuron')
    ax.set_title(title)
    ax.set_ylim([0.5, 1.5])

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved spike raster to {output_path}")
    else:
        plt.show()

    plt.close()


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[float, float, float] = (0, 0, 1),
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Overlay segmentation mask on image.

    Args:
        image: Input image (H, W, 3), values in [0, 1]
        mask: Segmentation mask (H, W), values in [0, 1]
        alpha: Mask transparency
        color: RGB color for mask (0-1 range)
        output_path: Optional path to save figure

    Returns:
        Overlay image (H, W, 3)
    """
    if plt is None:
        logger.warning("Matplotlib not installed, skipping visualization")
        return image

    # Normalize inputs
    image = np.clip(image, 0, 1).astype(np.float32)
    mask = np.clip(mask, 0, 1).astype(np.float32)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]

    # Blend
    overlay = image.copy()
    overlay_mask_bool = (mask > 0.5).astype(np.float32)[:, :, np.newaxis]
    overlay = (1 - alpha) * image + alpha * colored_mask
    overlay = np.where(overlay_mask_bool > 0, overlay, image)

    if output_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Segmentation Mask")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved overlay to {output_path}")
        plt.close()

    return overlay.astype(np.uint8) if image.max() <= 1 else overlay.astype(np.uint8)


def visualize_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Attention Map",
) -> None:
    """
    Visualize attention map heatmap on image.

    Args:
        image: Input image (H, W, 3)
        attention: Attention map (H, W)
        output_path: Optional path to save figure
        title: Plot title
    """
    if plt is None:
        logger.warning("Matplotlib not installed")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis('off')

    im = axes[1].imshow(attention, cmap='hot')
    axes[1].set_title(f"{title}")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved attention map to {output_path}")
    plt.close()
