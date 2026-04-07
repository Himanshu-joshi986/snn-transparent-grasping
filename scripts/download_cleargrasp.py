"""
scripts/download_cleargrasp.py

ClearGrasp Dataset Downloader
=============================
Automates downloading the ClearGrasp dataset for transparent object segmentation.

Reference: https://github.com/Shreeyak/cleargrasp

Usage:
    python scripts/download_cleargrasp.py --output_dir data/ --split all
    python scripts/download_cleargrasp.py --output_dir data/ --split synthetic
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CLEARGRASP_REPO = "https://github.com/Shreeyak/cleargrasp.git"
CLEARGRASP_DOWNLOAD_SCRIPT = "download_cleargrasp_dataset.sh"


def download_dataset(
    output_dir: str,
    split: str = "all",
    workers: int = 4,
) -> bool:
    """
    Download ClearGrasp dataset.

    Args:
        output_dir: Directory to save dataset
        split: "synthetic", "real", or "all"
        workers: Number of parallel download workers

    Returns:
        Success status
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading ClearGrasp dataset to {output_dir}")
    logger.info(f"Split: {split}")

    try:
        # Clone ClearGrasp repo (if not already present)
        cleargrasp_dir = output_path / "cleargrasp_repo"
        if not cleargrasp_dir.exists():
            logger.info("Cloning ClearGrasp repository...")
            subprocess.run(
                ["git", "clone", CLEARGRASP_REPO, str(cleargrasp_dir)],
                check=True,
                capture_output=True,
            )

        # Run download script
        download_script = cleargrasp_dir / CLEARGRASP_DOWNLOAD_SCRIPT
        if not download_script.exists():
            logger.warning(f"Download script not found at {download_script}")
            logger.info("Please follow manual instructions at:")
            logger.info(f"  {CLEARGRASP_REPO}#download-the-dataset")
            return False

        # Make script executable
        os.chmod(download_script, 0o755)

        # Run download
        env = os.environ.copy()
        env["OUTPUT_DIR"] = str(output_path / split)
        env["NUM_WORKERS"] = str(workers)

        logger.info(f"Running download script with {workers} workers...")
        result = subprocess.run(
            [str(download_script), split],
            env=env,
            cwd=str(cleargrasp_dir),
        )

        if result.returncode == 0:
            logger.info("✓ Dataset download successful!")
            return True
        else:
            logger.warning("Download script returned non-zero exit code")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        logger.info("Try manual download from:")
        logger.info(f"  {CLEARGRASP_REPO}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def create_directory_structure(base_dir: str) -> bool:
    """Create standard ClearGrasp directory structure."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    dirs = [
        "synthetic/rgb",
        "synthetic/masks",
        "synthetic/depth",
        "synthetic/events",
        "real/rgb",
        "real/masks",
        "real/events",
    ]

    for dir_name in dirs:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created {dir_path}")

    return True


def verify_download(base_dir: str) -> None:
    """Verify dataset integrity and print statistics."""
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.warning(f"Dataset directory not found: {base_dir}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 60)

    for split in ["synthetic", "real"]:
        split_path = base_path / split
        if split_path.exists():
            rgb_dir = split_path / "rgb"
            mask_dir = split_path / "masks"

            rgb_count = len(list(rgb_dir.glob("*.png") or rgb_dir.glob("*.jpg"))) if rgb_dir.exists() else 0
            mask_count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0

            logger.info(f"\n{split.upper()} split:")
            logger.info(f"  Images: {rgb_count}")
            logger.info(f"  Masks: {mask_count}")

            if rgb_dir.exists():
                size_mb = sum(f.stat().st_size for f in rgb_dir.iterdir()) / (1024**2)
                logger.info(f"  Size: {size_mb:.1f} MB")

    logger.info("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download ClearGrasp dataset for transparent object segmentation"
    )
    parser.add_argument(
        "--output_dir",
        default="data/",
        help="Output directory for dataset (default: data/)",
    )
    parser.add_argument(
        "--split",
        choices=["synthetic", "real", "all"],
        default="all",
        help="Dataset split to download (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--create_structure",
        action="store_true",
        help="Create directory structure only",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset",
    )

    args = parser.parse_args()

    if args.verify:
        verify_download(args.output_dir)
        return

    if args.create_structure:
        logger.info(f"Creating directory structure in {args.output_dir}")
        if create_directory_structure(args.output_dir):
            logger.info("✓ Directory structure created!")
            verify_download(args.output_dir)
        return

    # Download dataset
    success = download_dataset(
        output_dir=args.output_dir,
        split=args.split,
        workers=args.workers,
    )

    if success:
        verify_download(args.output_dir)
    else:
        logger.error("Dataset download may have failed. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
