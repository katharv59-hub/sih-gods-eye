"""Detection schemas — §4 Canonical Data Schemas.

These are inter-module contracts. All modules must consume and produce
these types exactly. No module may invent its own representation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: float  # Top-left x (pixels, absolute)
    y1: float  # Top-left y (pixels, absolute)
    x2: float  # Bottom-right x (pixels, absolute)
    y2: float  # Bottom-right y (pixels, absolute)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Detection:
    """A single detection from a single frame of a single camera.

    Attributes:
        detection_id: UUID, unique per detection.
        camera_id: Source camera identifier.
        frame_id: Monotonic frame counter for this camera.
        timestamp_ns: Unix nanoseconds — do NOT use float seconds.
        bbox: Bounding box in absolute pixel coordinates.
        confidence: Detection confidence in [0.0, 1.0].
        class_label: Detected class (e.g. "person").
        source_resolution: (width, height) of source frame.
    """

    detection_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    class_label: str
    source_resolution: tuple[int, int]
