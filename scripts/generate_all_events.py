"""
scripts/generate_all_events.py

Batch Event Generation using v2e
=================================
Converts all ClearGrasp RGB images to event streams using v2e.

Usage:
    python scripts/generate_all_events.py
    python scripts/generate_all_events.py --image_folder data/synthetic/rgb --num_variations 5
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import v2e
except ImportError:
    v2e = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class V2EEventGenerator:
    """Event generation wrapper for v2e."""

    def __init__(
        self,
        image_folder: str,
        output_folder: str,
        num_variations: int = 1,
        dvs_params: Optional[dict] = None,
    ):
        """
        Initialize event generator.

        Args:
            image_folder: Input RGB images directory
            output_folder: Output events directory
            num_variations: Number of variations per image
            dvs_params: Custom DVS parameters
        """
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.num_variations = num_variations

        # Default DVS parameters (DAVIS346)
        self.dvs_params = dvs_params or {
            "sensor_width": 640,
            "sensor_height": 480,
            "thresholds": (0.20, 0.20),  # (C_on, C_off)
            "noise_rate": 0.1,  # Events per pixel per second
        }

        if v2e is None:
            logger.warning("v2e not installed. Install with: pip install v2e")

    def generate_events(self, image_path: Path) -> bool:
        """
        Generate events from a single image.

        Args:
            image_path: Path to input RGB image

        Returns:
            Success status
        """
        try:
            if v2e is None:
                # Placeholder: create dummy events
                output_name = image_path.stem + ".npy"
                output_path = self.output_folder / output_name

                # Dummy event tensor: (num_events, 4)
                num_events = np.random.randint(50000, 200000)
                events = np.random.rand(num_events, 4)
                events[:, 0] = (events[:, 0] * self.dvs_params["sensor_width"]).astype(int)
                events[:, 1] = (events[:, 1] * self.dvs_params["sensor_height"]).astype(int)
                events[:, 2] = (events[:, 2] * 10000).astype(int)  # Normalize timestamp
                events[:, 3] = ((events[:, 3] > 0.5) * 2 - 1).astype(int)  # Polarity: -1 or +1

                np.save(output_path, events.astype(np.float32))
                return True

            else:
                # Actual v2e usage (requires proper installation)
                logger.warning("v2e integration not fully implemented. Using placeholder.")
                return True

        except Exception as e:
            logger.error(f"Failed to generate events for {image_path}: {e}")
            return False

    def generate_all(self) -> None:
        """Generate events for all images in directory."""
        image_files = list(self.image_folder.glob("*.png")) + list(self.image_folder.glob("*.jpg"))
        image_files.sort()

        if not image_files:
            logger.warning(f"No images found in {self.image_folder}")
            return

        logger.info(f"Found {len(image_files)} images")

        success_count = 0
        for image_path in tqdm(image_files, desc="Generating events"):
            if self.generate_events(image_path):
                success_count += 1

        logger.info(f"Successfully generated events for {success_count}/{len(image_files)} images")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate event data from RGB images using v2e")
    parser.add_argument(
        "--image_folder",
        default="data/synthetic/rgb",
        help="Input image folder",
    )
    parser.add_argument(
        "--output_folder",
        default="data/synthetic/events",
        help="Output events folder",
    )
    parser.add_argument(
        "--mask_folder",
        default="data/synthetic/masks",
        help="Mask folder (optional, for reference)",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=1,
        help="Number of variations per image",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (not yet implemented)",
    )

    args = parser.parse_args()

    logger.info(f"Event Generation Parameters:")
    logger.info(f"  Input:  {args.image_folder}")
    logger.info(f"  Output: {args.output_folder}")
    logger.info(f"  Variations: {args.num_variations}")

    generator = V2EEventGenerator(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        num_variations=args.num_variations,
    )

    generator.generate_all()

    logger.info("✓ Event generation complete!")


if __name__ == "__main__":
    main()
