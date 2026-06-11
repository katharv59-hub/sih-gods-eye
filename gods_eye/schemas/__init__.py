"""Canonical data schemas — §4 of GODS_EYE_MASTER_SPEC.

This package defines ALL inter-module contracts.
No module may invent its own representation of these concepts.
Schema changes require an ADR entry.

IMPORTANT: This package must NEVER import from other gods_eye modules.
"""

from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import Track, TrackState
from gods_eye.schemas.identity import Identity, IdentityStatus
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.camera import CameraNode

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
]
