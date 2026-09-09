"""Temporal Adapter — Phase 4.1 Conversion Boundary (§14).

Converts upstream pipeline objects (IdentityResult, SceneState, Event) into
canonical TemporalObservation records and classifies observation priority (CRITICAL vs PERIODIC).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.observation import (
    ObservationPriority,
    TemporalObservation,
    TemporalObservationType,
)

if TYPE_CHECKING:
    from gods_eye.reid.identity_mapper import IdentityResult
    from gods_eye.schemas.environment import SceneState

_CRITICAL_EVENT_TYPES: set[EventType] = {
    EventType.CROSS_CAMERA_TRANSITION,
    EventType.IDENTITY_CONFIRMED,
    EventType.IDENTITY_LOST,
    EventType.IDENTITY_PURGED,
    EventType.ENVIRONMENTAL_ANOMALY,
    EventType.SYSTEM_MODE_CHANGED,
    EventType.MODEL_DRIFT_DETECTED,
}


class TemporalAdapter:
    """Adapter converting pipeline objects to TemporalObservation records."""

    @staticmethod
    def classify_priority(
        obs_type: TemporalObservationType,
        event_type: Optional[EventType] = None,
    ) -> ObservationPriority:
        """Classify observation priority deterministically as CRITICAL or PERIODIC."""
        if obs_type == TemporalObservationType.CAMERA_TRANSITION:
            return ObservationPriority.CRITICAL

        if event_type is not None and event_type in _CRITICAL_EVENT_TYPES:
            return ObservationPriority.CRITICAL

        return ObservationPriority.PERIODIC

    @classmethod
    def adapt_identity_result(cls, res: IdentityResult) -> list[TemporalObservation]:
        """Convert IdentityResult into a list of TemporalObservation records."""
        observations: list[TemporalObservation] = []
        camera_id = res.tracking.packet.camera_id
        timestamp_ns = res.tracking.packet.timestamp_ns

        for track_id, identity in res.identities.items():
            obs = TemporalObservation(
                observation_id=str(uuid.uuid4()),
                observation_type=TemporalObservationType.IDENTITY_PRESENCE,
                priority=ObservationPriority.PERIODIC,
                timestamp_ns=timestamp_ns,
                camera_id=camera_id,
                global_id=identity.global_id,
                confidence=identity.confidence,
                metadata={
                    "track_id": track_id,
                    "frame_id": res.tracking.packet.frame_id,
                },
            )
            observations.append(obs)

        for trans in getattr(res, "transitions", []):
            if trans.transition_type.value == "cross_camera_transition":
                ident = res.identities.get(trans.track_id) if trans.track_id else None
                from_cam = "unknown"
                if ident and len(ident.camera_history) >= 2:
                    from_cam = ident.camera_history[-2]
                obs_trans = TemporalObservation(
                    observation_id=str(uuid.uuid4()),
                    observation_type=TemporalObservationType.CAMERA_TRANSITION,
                    priority=ObservationPriority.CRITICAL,
                    timestamp_ns=trans.timestamp_ns,
                    camera_id=trans.camera_id,
                    global_id=trans.global_id,
                    confidence=trans.confidence,
                    metadata={
                        "from_camera_id": from_cam,
                        "track_id": trans.track_id,
                        "frame_id": trans.frame_id,
                        "transition_type": trans.transition_type.value,
                    },
                )
                observations.append(obs_trans)

        return observations

    @classmethod
    def persist_identity_to_graph(
        cls,
        identity: Any,
        graph_store: Any,
        camera_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist an Identity node into SQLiteGraphStore."""
        cam_id = camera_id or getattr(identity, "last_camera_id", None) or (
            identity.camera_history[-1] if getattr(identity, "camera_history", None) else "cam-01"
        )
        state_str = identity.state.value if hasattr(identity.state, "value") else str(identity.state)
        first_seen = getattr(identity, "first_seen_ns", None) or getattr(identity, "created_ns", 0)
        last_seen = getattr(identity, "last_seen_ns", first_seen)
        meta = metadata or {}
        if hasattr(identity, "camera_history"):
            meta.setdefault("camera_history", list(identity.camera_history))
        if hasattr(identity, "confidence"):
            meta.setdefault("confidence", identity.confidence)
        graph_store.upsert_identity_node(
            global_id=identity.global_id,
            first_seen_ns=first_seen,
            last_seen_ns=last_seen,
            primary_camera_id=cam_id,
            state=state_str,
            metadata=meta,
        )

    @classmethod
    def adapt_event(cls, event: Event) -> TemporalObservation:
        """Convert a standalone Event into a TemporalObservation record."""
        obs_type = TemporalObservationType.SYSTEM_EVENT
        if event.event_type == EventType.CROSS_CAMERA_TRANSITION:
            obs_type = TemporalObservationType.CAMERA_TRANSITION

        priority = cls.classify_priority(obs_type, event.event_type)

        return TemporalObservation(
            observation_id=event.event_id,
            observation_type=obs_type,
            priority=priority,
            timestamp_ns=event.timestamp_ns,
            camera_id=event.camera_id if event.camera_id else None,
            global_id=event.global_id,
            zone_id=event.zone_id,
            confidence=event.confidence,
            event=event,
            metadata=event.metadata,
        )

    @classmethod
    def adapt_environmental_state(cls, scene_state: SceneState) -> TemporalObservation:
        """Convert a SceneState into a TemporalObservation record."""
        return TemporalObservation(
            observation_id=str(uuid.uuid4()),
            observation_type=TemporalObservationType.ENVIRONMENTAL_STATE,
            priority=ObservationPriority.PERIODIC,
            timestamp_ns=scene_state.timestamp_ns,
            camera_id=scene_state.camera_id,
            confidence=1.0,
            metadata={
                "lighting_condition": scene_state.lighting_condition.value,
                "occupancy_count": scene_state.occupancy_count,
                "occupancy_density": scene_state.occupancy_density,
                "foreground_ratio": scene_state.foreground_ratio,
                "is_anomalous": scene_state.is_anomalous,
                "anomaly_confidence": scene_state.anomaly_confidence,
            },
        )
