"""
utils/event_loader.py

Event Data Loader
=================
Loads event data from various formats (.aedat2, .npy, CSV)
and converts to usable tensor representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)


class EventLoader:
    """Loads event data from various formats."""

    @staticmethod
    def load_events_npy(npy_path: Path) -> np.ndarray:
        """
        Load events from .npy file.

        Args:
            npy_path: Path to .npy file

        Returns:
            Event array (num_events, 4) - (x, y, t, polarity)
        """
        events = np.load(npy_path)
        return events.astype(np.float32)

    @staticmethod
    def load_events_aedat2(aedat2_path: Path) -> np.ndarray:
        """
        Load events from AEDAT2 binary file.

        Args:
            aedat2_path: Path to .aedat2 file

        Returns:
            Event array (num_events, 4)
        """
        # AEDAT2: header + events
        # Each event: 40-bit timestamp, 16-bit x, 16-bit y, 1-bit polarity
        try:
            with open(aedat2_path, 'rb') as f:
                # Skip header
                while f.read(1) == b'#':
                    f.readline()
                f.seek(0)

                # Read binary events
                events = []
                while True:
                    data = f.read(8)
                    if len(data) < 8:
                        break

                    timestamp = int.from_bytes(data[:4], byteorder='little')
                    xy = int.from_bytes(data[4:8], byteorder='little')
                    polarity = (xy >> 31) & 1
                    y = (xy >> 16) & 0x7FFF
                    x = xy & 0xFFFF

                    events.append([x, y, timestamp, 2 * polarity - 1])

            return np.array(events, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to load AEDAT2: {e}")
            return np.array([], dtype=np.float32)

    @staticmethod
    def load_events_csv(csv_path: Path) -> np.ndarray:
        """Load events from CSV file."""
        try:
            events = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
            return events
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return np.array([], dtype=np.float32)

    @staticmethod
    def load_events(path: Path, format: Optional[str] = None) -> np.ndarray:
        """
        Auto-detect and load events from file.

        Args:
            path: File path
            format: Optional format hint ('npy', 'aedat2', 'csv')

        Returns:
            Event array
        """
        path = Path(path)

        if format is None:
            suffix = path.suffix.lower()
            format = {'.npy': 'npy', '.aedat2': 'aedat2', '.csv': 'csv'}.get(suffix, 'npy')

        if format == 'npy':
            return EventLoader.load_events_npy(path)
        elif format == 'aedat2':
            return EventLoader.load_events_aedat2(path)
        elif format == 'csv':
            return EventLoader.load_events_csv(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def events_to_tensor(
        events: np.ndarray,
        sensor_height: int = 480,
        sensor_width: int = 640,
        num_bins: int = 4,
    ) -> np.ndarray:
        """
        Convert event list to tensor representation.

        Args:
            events: Event array (num_events, 4)
            sensor_height: Sensor height
            sensor_width: Sensor width
            num_bins: Number of temporal bins

        Returns:
            Tensor (num_bins, height, width)
        """
        if len(events) == 0:
            return np.zeros((num_bins, sensor_height, sensor_width), dtype=np.float32)

        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2]
        p = events[:, 3].astype(np.int32)

        # Normalize time
        if t.max() > t.min():
            t_norm = (t - t.min()) / (t.max() - t.min())
        else:
            t_norm = np.zeros_like(t)

        bin_indices = np.clip((t_norm * (num_bins - 1)).astype(np.int32), 0, num_bins - 1)

        # Create tensor
        tensor = np.zeros((num_bins, sensor_height, sensor_width), dtype=np.float32)

        for idx in range(len(events)):
            xi = np.clip(int(x[idx]), 0, sensor_width - 1)
            yi = np.clip(int(y[idx]), 0, sensor_height - 1)
            bi = bin_indices[idx]
            tensor[bi, yi, xi] += (1 if p[idx] > 0 else -1)

        # Normalize
        tensor_max = np.abs(tensor).max()
        if tensor_max > 0:
            tensor = tensor / tensor_max

        return tensor.astype(np.float32)
