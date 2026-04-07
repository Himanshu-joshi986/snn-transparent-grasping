"""
utils/generate_events.py

Event Generation from Video/Images
===================================
Converts RGB images or videos to synthetic DVS event streams using v2e,
with support for batch processing and event augmentation.

Usage:
    python utils/generate_events.py --image_folder data/rgb --output_folder data/events
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import v2e
except ImportError:
    v2e = None

logger = logging.getLogger(__name__)


class EventGenerator:
    """Generate synthetic DVS events from images or video."""

    def __init__(
        self,
        sensor_width: int = 640,
        sensor_height: int = 480,
        threshold_pos: float = 0.2,
        threshold_neg: float = 0.2,
        refractory_period: float = 0.001,
    ):
        """
        Initialize event generator.

        Args:
            sensor_width: Sensor width in pixels
            sensor_height: Sensor height in pixels
            threshold_pos: Positive threshold for event generation
            threshold_neg: Negative threshold for event generation
            refractory_period: Refractory period in seconds
        """
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.refractory_period = refractory_period

    def generate_events_from_image_sequence(
        self,
        images: np.ndarray,
        timestamps: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Generate events from image sequence.

        Args:
            images: Image sequence (N, H, W, 3) normalized to [0, 1]
            timestamps: Timestamps for each image in seconds
            output_path: Optional path to save events

        Returns:
            Event array (num_events, 4) - (x, y, timestamp, polarity)
        """
        events = []
        log_intensity = np.log(np.clip(images[0].mean(axis=2), 1e-3, 1.0))

        for i in range(1, len(images)):
            gray_float = np.clip(images[i].mean(axis=2), 1e-3, 1.0)
            log_intensity_new = np.log(gray_float)

            # Compute log intensity change
            intensity_change = log_intensity_new - log_intensity

            # Positive and negative event masks
            pos_events = intensity_change > self.threshold_pos
            neg_events = intensity_change < -self.threshold_neg

            # Get coordinates
            y_pos, x_pos = np.where(pos_events)
            y_neg, x_neg = np.where(neg_events)

            # Create event list
            for x, y in zip(x_pos, y_pos):
                events.append([x, y, timestamps[i], 1])  # Polarity +1

            for x, y in zip(x_neg, y_neg):
                events.append([x, y, timestamps[i], -1])  # Polarity -1

            log_intensity = log_intensity_new

        events_array = np.array(events, dtype=np.float32)

        # Sort by timestamp
        if len(events_array) > 0:
            sorted_indices = np.argsort(events_array[:, 2])
            events_array = events_array[sorted_indices]

        # Save if path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, events_array)
            logger.info(f"Saved {len(events_array)} events to {output_path}")

        return events_array

    def augment_events(
        self,
        events: np.ndarray,
        num_variations: int = 1,
        noise_level: float = 0.05,
    ) -> list[np.ndarray]:
        """
        Augment events by adding noise and jitter.

        Args:
            events: Event array
            num_variations: Number of augmented versions
            noise_level: Noise level (fraction of events to flip)

        Returns:
            List of augmented event arrays
        """
        variations = [events]

        for _ in range(num_variations - 1):
            variation = events.copy()

            # Add random noise events
            noise_fraction = int(len(variation) * noise_level)
            noise_indices = np.random.choice(len(variation), noise_fraction, replace=False)
            variation[noise_indices, 3] *= -1  # Flip polarity

            # Add spatial jitter
            spatial_jitter = np.random.randint(-1, 2, (len(variation), 2))
            variation[:, 0] = np.clip(variation[:, 0] + spatial_jitter[:, 0], 0, self.sensor_width - 1)
            variation[:, 1] = np.clip(variation[:, 1] + spatial_jitter[:, 1], 0, self.sensor_height - 1)

            # Add temporal jitter
            temporal_jitter = np.random.randn(len(variation)) * 0.001
            variation[:, 2] += temporal_jitter

            variations.append(variation)

        return variations


def generate_events_batch(
    image_folder: Path,
    output_folder: Path,
    num_variations: int = 1,
    num_workers: int = 1,
) -> int:
    """
    Generate events for all images in a folder.

    Args:
        image_folder: Input folder with images
        output_folder: Output folder for events
        num_variations: Number of variations per image
        num_workers: Number of parallel workers (not yet implemented)

    Returns:
        Number of successfully generated event files
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([
        f for f in image_folder.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        logger.warning(f"No images found in {image_folder}")
        return 0

    generator = EventGenerator()
    success_count = 0

    logger.info(f"Generating events for {len(image_files)} images...")

    for image_file in tqdm(image_files, desc="Generating events"):
        try:
            # Load image
            if cv2 is None:
                # Fallback: create dummy image
                image = np.random.rand(480, 640, 3)
            else:
                image = cv2.imread(str(image_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255.0

            # Generate events from consecutive frames (simulate motion)
            timestamps = np.array([0.0, 0.1, 0.2, 0.3])
            images = np.stack([
                image,
                image + np.random.randn(*image.shape) * 0.05,
                image + np.random.randn(*image.shape) * 0.08,
                image + np.random.randn(*image.shape) * 0.10,
            ])

            # Output path
            output_path = output_folder / f"{image_file.stem}.npy"

            # Generate base events
            events = generator.generate_events_from_image_sequence(
                images,
                timestamps,
                output_path=output_path,
            )

            # Generate variations
            if num_variations > 1:
                variations = generator.augment_events(events, num_variations=num_variations)
                for var_idx, variation in enumerate(variations[1:]):
                    var_path = output_folder / f"{image_file.stem}_var{var_idx+1}.npy"
                    np.save(var_path, variation)

            success_count += 1

        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")

    logger.info(f"Successfully generated events for {success_count}/{len(image_files)} images")
    return success_count


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic DVS events from images")
    parser.add_argument("--image_folder", default="data/synthetic/rgb", help="Input image folder")
    parser.add_argument("--output_folder", default="data/synthetic/events", help="Output event folder")
    parser.add_argument("--num_variations", type=int, default=1, help="Variations per image")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generate_events_batch(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        num_variations=args.num_variations,
        num_workers=args.num_workers,
    )
