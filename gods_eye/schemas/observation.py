"""Canonical Temporal Observation Data Contract — Phase 4.1 (§4 Canonical Data Schemas).

Defines canonical TemporalObservation dataclass, observation types, priority
classification, and schema boundary validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gods_eye.schemas.event import Event


class TemporalObservationType(Enum):
    """Categories of temporal observations emitted across the temporal boundary."""

    IDENTITY_PRESENCE = "identity_presence"
    CAMERA_TRANSITION = "camera_transition"
    ZONE_OCCUPANCY = "zone_occupancy"
    ENVIRONMENTAL_STATE = "environmental_state"
    SYSTEM_EVENT = "system_event"


class ObservationPriority(Enum):
    """Priority classification governing admission control and thinning policy."""

    CRITICAL = "critical"
    PERIODIC = "periodic"


@dataclass
class TemporalObservation:
    """Canonical temporal observation record for Phase 4 temporal boundary.

    Decouples temporal persistence workers from upstream algorithm objects.
    """

    # Mandatory identity & category attributes
    observation_id: str
    observation_type: TemporalObservationType
    priority: ObservationPriority
    timestamp_ns: int

    # Camera association (Optional for global SYSTEM_EVENT, Required for all other types)
    camera_id: Optional[str] = None

    # Optional spatial-temporal & identity attributes
    global_id: Optional[str] = None
    zone_id: Optional[str] = None
    confidence: float = 1.0
    dwell_s: Optional[float] = None
    event: Optional[Event] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate camera_id invariant at the schema boundary."""
        if not self.observation_id or not self.observation_id.strip():
            raise ValueError("observation_id MUST be a non-empty string")

        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")

        # Enforce camera_id rule across observation types
        if self.observation_type != TemporalObservationType.SYSTEM_EVENT:
            if self.camera_id is None or not self.camera_id.strip():
                raise ValueError(
                    f"camera_id is REQUIRED for observation_type '{self.observation_type.value}'"
                )
