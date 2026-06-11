"""FramePacket — internal transport type for captured frames.

This is NOT a §4 canonical schema. It is a pipeline-internal envelope
that carries a raw frame from ingestion to detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    """A captured video frame with metadata.

    Attributes:
        camera_id: Source camera identifier.
        frame_id: Monotonic frame counter (1-indexed, per camera).
        timestamp_ns: Unix nanoseconds at capture time.
        frame: BGR image as numpy array (HxWxC).
        resolution: (width, height) of the frame.
    """

    camera_id: str
    frame_id: int
    timestamp_ns: int
    frame: np.ndarray
    resolution: tuple[int, int]
