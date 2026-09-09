"""Vehicle Perception Schemas — Phase 8.1 (SIH 26187).

Defines canonical data contracts for vehicle detection and tracking:
- VehicleDetection: mirrors Detection with vehicle_class
- VehicleTrack: mirrors Track with vehicle_class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

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
class VehiclePlateAssociation:
    """Explicit association between a vehicle track and a license plate observation.

    Attributes:
        track_id: Vehicle track identifier.
        plate_text: Associated license plate string, or 'uncertain'.
        confidence: Association confidence score in [0.0, 1.0].
        camera_id: Source camera identifier where observation occurred.
        frame_id: Frame number of the plate observation.
        timestamp_ns: Unix timestamp in nanoseconds of the observation.
        is_uncertain: True if the association or OCR reading is uncertain.
        plate_region_id: Optional reference to PlateRegion identifier.
        ocr_confidence: Optional raw OCR confidence score.
        spatial_confidence: Optional spatial containment/geometry score.
    """

    track_id: str
    plate_text: str
    confidence: float
    camera_id: Optional[str] = None
    frame_id: Optional[int] = None
    timestamp_ns: Optional[int] = None
    is_uncertain: bool = False
    plate_region_id: Optional[str] = None
    ocr_confidence: Optional[float] = None
    spatial_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate VehiclePlateAssociation invariants."""
        if not self.track_id or not str(self.track_id).strip():
            raise ValueError("track_id MUST be a non-empty string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")

    def __eq__(self, other: object) -> bool:
        """Support equality check with strings for legacy compatibility."""
        if isinstance(other, str):
            return self.plate_text == other
        if isinstance(other, VehiclePlateAssociation):
            return (
                self.track_id == other.track_id
                and self.plate_text == other.plate_text
                and self.confidence == other.confidence
                and self.camera_id == other.camera_id
                and self.frame_id == other.frame_id
            )
        return False

    def to_event(self, event_id: Optional[str] = None) -> Any:
        """Adapt this vehicle-plate association into a canonical Event(ANPR_READING)."""
        from gods_eye.schemas.event import Event, EventType

        conf = self.ocr_confidence if self.ocr_confidence is not None else self.confidence
        ev_id = event_id or f"ev_anpr_{self.track_id}_{self.timestamp_ns or 0}"
        explanation = (
            f"Plate '{self.plate_text}' read with OCR confidence {conf:.2f} "
            f"(associated to vehicle {self.track_id} with confidence {self.confidence:.2f})"
            if not self.is_uncertain
            else f"Uncertain plate reading '{self.plate_text}' (confidence {conf:.2f}) "
            f"associated to vehicle {self.track_id}"
        )
        return Event(
            event_id=ev_id,
            event_type=EventType.ANPR_READING,
            global_id=self.track_id,
            camera_id=self.camera_id or "cam-01",
            zone_id=None,
            timestamp_ns=self.timestamp_ns or 0,
            frame_id=self.frame_id or 1,
            confidence=conf,
            explanation=explanation,
            metadata={
                "plate_text": self.plate_text,
                "confidence": conf,
                "ocr_confidence": self.ocr_confidence,
                "is_uncertain": self.is_uncertain,
                "vehicle_track_id": self.track_id,
                "association_confidence": self.confidence,
                "plate_region_id": self.plate_region_id,
                "spatial_confidence": self.spatial_confidence,
            },
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
        association_confidence: Association confidence score in [0.0, 1.0].
        plate_association: Structured vehicle-to-plate association record.
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
    association_confidence: Optional[float] = None
    plate_association: Optional[VehiclePlateAssociation] = None

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
        if self.association_confidence is not None:
            if not (0.0 <= self.association_confidence <= 1.0):
                raise ValueError(
                    f"association_confidence MUST be between 0.0 and 1.0, got {self.association_confidence}"
                )

    def to_event(
        self,
        timestamp_ns: int,
        frame_id: Optional[int] = None,
        confidence: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> Any:
        """Adapt this vehicle track into a canonical Event(VEHICLE_DETECTED)."""
        from gods_eye.schemas.event import Event, EventType

        conf = float(confidence) if confidence is not None else 0.90
        ev_id = event_id or f"ev_veh_{self.track_id}_{timestamp_ns}"
        f_id = frame_id if frame_id is not None else self.last_frame_id
        explanation = (
            f"Vehicle {self.track_id} ({self.vehicle_class}) detected on camera {self.camera_id}"
        )

        metadata: dict[str, Any] = {
            "vehicle_track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "confidence": conf,
            "state": self.state.value,
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "velocity": list(self.velocity),
            "plate_text": self.plate_text,
            "association_confidence": self.association_confidence,
        }
        if self.plate_association is not None:
            metadata["is_uncertain"] = self.plate_association.is_uncertain
            metadata["ocr_confidence"] = self.plate_association.ocr_confidence

        return Event(
            event_id=ev_id,
            event_type=EventType.VEHICLE_DETECTED,
            global_id=self.track_id,
            camera_id=self.camera_id,
            zone_id=None,
            timestamp_ns=timestamp_ns,
            frame_id=f_id,
            confidence=conf,
            explanation=explanation,
            metadata=metadata,
        )

