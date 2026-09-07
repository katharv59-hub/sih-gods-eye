"""Vehicle Perception Schemas — Phase 8.1 (SIH 26187).

Defines canonical data contracts for vehicle detection and tracking:
- VehicleDetection: mirrors Detection with vehicle_class
- VehicleTrack: mirrors Track with vehicle_class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from gods_eye.schemas.detection import BoundingBox


VEHICLE_CLASSES = ("car", "motorcycle", "truck", "bus", "van", "other")

VehicleClassType = Literal["car", "motorcycle", "truck", "bus", "van", "other"]


class VehicleTrackState(Enum):
    """Lifecycle state of a vehicle track."""

    ACTIVE = "active"
    LOST = "lost"
    DEAD = "dead"


@dataclass
class VehicleDetection:
    """A single vehicle detection from a single frame of a single camera.

    Mirrors the person Detection schema (§4) with the addition of vehicle_class.

    Attributes:
        detection_id: UUID, unique per detection.
        camera_id: Source camera identifier.
        frame_id: Monotonic frame counter for this camera.
        timestamp_ns: Unix nanoseconds — never float seconds.
        bbox: Bounding box in absolute pixel coordinates.
        confidence: Detection confidence in [0.0, 1.0].
        vehicle_class: Detected vehicle class.
        source_resolution: (width, height) of source frame.
    """

    detection_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    vehicle_class: VehicleClassType
    source_resolution: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate VehicleDetection invariants."""
        if not self.detection_id or not self.detection_id.strip():
            raise ValueError("detection_id MUST be a non-empty string")
        if not self.camera_id or not self.camera_id.strip():
            raise ValueError("camera_id MUST be a non-empty string")
        if self.frame_id < 0:
            raise ValueError(f"frame_id MUST be non-negative, got {self.frame_id}")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")
        if self.vehicle_class not in VEHICLE_CLASSES:
            raise ValueError(
                f"vehicle_class MUST be one of {VEHICLE_CLASSES}, got '{self.vehicle_class}'"
            )


@dataclass
class VehicleTrack:
    """A tracked vehicle across frames within a single camera.

    Mirrors the person Track schema (§4) with vehicle_class.

    Attributes:
        track_id: Tracker-local ID (not global — see ADR-004).
        camera_id: Source camera identifier.
        state: Current lifecycle state.
        bbox: Most recent bounding box.
        velocity: (dx, dy) pixels/frame from consecutive bbox centers.
        vehicle_class: Detected vehicle class.
        first_frame_id: Frame where this track was first seen.
        last_frame_id: Frame where this track was last seen.
        lost_frame_count: Consecutive frames without a detection match.
        detection_history: List of detection_ids (capped at N).
        plate_text: Associated license plate text, if detected.
    """

    track_id: str
    camera_id: str
    state: VehicleTrackState
    bbox: BoundingBox
    velocity: tuple[float, float]
    vehicle_class: VehicleClassType
    first_frame_id: int
    last_frame_id: int
    lost_frame_count: int
    detection_history: list[str] = field(default_factory=list)
    plate_text: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate VehicleTrack invariants."""
        if not self.track_id or not str(self.track_id).strip():
            raise ValueError("track_id MUST be a non-empty string")
        if not self.camera_id or not self.camera_id.strip():
            raise ValueError("camera_id MUST be a non-empty string")
        if not isinstance(self.state, VehicleTrackState):
            raise ValueError(f"state MUST be a VehicleTrackState, got {type(self.state)}")
        if self.vehicle_class not in VEHICLE_CLASSES:
            raise ValueError(
                f"vehicle_class MUST be one of {VEHICLE_CLASSES}, got '{self.vehicle_class}'"
            )
