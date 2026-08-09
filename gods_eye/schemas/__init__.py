"""Canonical data schemas — §4 of GODS_EYE_MASTER_SPEC.

This package defines ALL inter-module contracts.
No module may invent its own representation of these concepts.
Schema changes require an ADR entry.

IMPORTANT: This package must NEVER import from other gods_eye modules.
"""

from gods_eye.schemas.behavioral import (
    BehavioralAnomalyResult,
    PredictedDestination,
    PredictionResult,
    TrajectoryCluster,
)
from gods_eye.schemas.camera import CameraNode, Zone
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.environment import (
    BackgroundModel,
    LightingCondition,
    OccupancyBaseline,
    SceneState,
    SystemMode,
)
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.identity import Identity, IdentityStatus
from gods_eye.schemas.observation import (
    ObservationPriority,
    TemporalObservation,
    TemporalObservationType,
)
from gods_eye.schemas.reasoning import (
    Evidence,
    QueryStatus,
    ReasoningResult,
    ToolCall,
    ToolResult,
    UserQuery,
)
from gods_eye.schemas.timeline import (
    CameraTimeline,
    IdentityTimeline,
    ReplayFrame,
    VisitSegment,
    ZoneTimeline,
)
from gods_eye.schemas.track import Track, TrackState

__all__ = [
    "BoundingBox",
    "Detection",
    "Track",
    "TrackState",
    "Identity",
    "IdentityStatus",
    "Event",
    "EventType",
    "CameraNode",
    "Zone",
    "BackgroundModel",
    "LightingCondition",
    "OccupancyBaseline",
    "SceneState",
    "SystemMode",
    "ObservationPriority",
    "TemporalObservation",
    "TemporalObservationType",
    "VisitSegment",
    "IdentityTimeline",
    "CameraTimeline",
    "ZoneTimeline",
    "ReplayFrame",
    "QueryStatus",
    "UserQuery",
    "ToolCall",
    "Evidence",
    "ToolResult",
    "ReasoningResult",
    "TrajectoryCluster",
    "PredictedDestination",
    "PredictionResult",
    "BehavioralAnomalyResult",
]
