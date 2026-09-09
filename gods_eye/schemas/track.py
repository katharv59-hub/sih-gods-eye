"""Track schemas — §4 Canonical Data Schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gods_eye.schemas.detection import BoundingBox


class TrackState(Enum):
    """Lifecycle state of a track."""

    ACTIVE = "active"
    LOST = "lost"    # Not seen for < TRACK_LOST_TIMEOUT frames
    DEAD = "dead"    # Not seen for >= TRACK_DEAD_TIMEOUT frames — evict


@dataclass
class Track:
    """A tracked object across frames within a single camera.

    Attributes:
        track_id: Tracker-local ID (not global — see ADR-004).
        camera_id: Source camera identifier.
        state: Current lifecycle state.
        bbox: Most recent bounding box.
        velocity: (dx, dy) pixels/frame from consecutive bbox centers.
        first_frame_id: Frame where this track was first seen.
        last_frame_id: Frame where this track was last seen.
        lost_frame_count: Consecutive frames without a detection match.
        detection_history: List of detection_ids (capped at N).
    """

    track_id: str
    camera_id: str
    state: TrackState
    bbox: BoundingBox
    velocity: tuple[float, float]  # (dx, dy) pixels/frame — v3 §4
    first_frame_id: int
    last_frame_id: int
    lost_frame_count: int
    detection_history: list[str] = field(default_factory=list)
    class_label: str = "person"

