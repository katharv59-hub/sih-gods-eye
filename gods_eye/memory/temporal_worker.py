"""Temporal Worker — Phase 4.5 (§14).

Consumes canonical TemporalObservation records from TemporalAdmissionControl queue,
applies routing rules, and dispatches to EventWriter and GraphWriter.
Runs as a single background thread to preserve strict submission sequence order.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from gods_eye.events.event_writer import EventWriter
from gods_eye.memory.admission_control import TemporalAdmissionControl
from gods_eye.memory.graph_writer import GraphRecord, GraphWriter
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.observation import (
    TemporalObservation,
    TemporalObservationType,
)

_log = get_logger("memory.temporal_worker")


class TemporalWorker:
    """Background worker processing TemporalObservations and routing to writers."""

    def __init__(
        self,
        admission_control: TemporalAdmissionControl,
        event_writer: EventWriter,
        graph_writer: GraphWriter,
        metrics: Optional[MetricsRegistry] = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self._admission_control = admission_control
        self._event_writer = event_writer
        self._graph_writer = graph_writer
        self._metrics = metrics
        self._poll_interval_s = poll_interval_s

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="TemporalWorkerThread", daemon=True
        )
        self._thread.start()
        _log.info("temporal_worker_started")

    def _run(self) -> None:
        while self._running:
            obs = self._admission_control.get_nowait()
            if obs is None:
                time.sleep(self._poll_interval_s)
                continue

            self._process_observation(obs)

        # Drain remaining items on shutdown
        while True:
            obs = self._admission_control.get_nowait()
            if obs is None:
                break
            self._process_observation(obs)

    def _process_observation(self, obs: TemporalObservation) -> None:
        """Route single observation to EventWriter and/or GraphWriter."""
        try:
            obs_type = obs.observation_type

            # 1. Identity Presence -> GraphStore (Node & Edge)
            if obs_type == TemporalObservationType.IDENTITY_PRESENCE:
                if obs.global_id is not None and obs.camera_id is not None:
                    # Identity Node
                    self._graph_writer.emit(
                        GraphRecord(
                            "identity_node",
                            {
                                "global_id": obs.global_id,
                                "first_seen_ns": obs.timestamp_ns,
                                "last_seen_ns": obs.timestamp_ns,
                                "primary_camera_id": obs.camera_id,
                                "state": "ACTIVE",
                                "metadata": obs.metadata,
                            },
                        )
                    )
                    # Camera Node
                    self._graph_writer.emit(
                        GraphRecord(
                            "camera_node",
                            {
                                "camera_id": obs.camera_id,
                                "status": "ACTIVE",
                            },
                        )
                    )
                    # Observation Edge
                    track_id = str(obs.metadata.get("track_id")) if obs.metadata.get("track_id") else None
                    self._graph_writer.emit(
                        GraphRecord(
                            "obs_edge",
                            {
                                "edge_id": obs.observation_id,
                                "global_id": obs.global_id,
                                "camera_id": obs.camera_id,
                                "timestamp_ns": obs.timestamp_ns,
                                "confidence": obs.confidence,
                                "track_id": track_id,
                                "metadata": obs.metadata,
                            },
                        )
                    )

            # 2. Camera Transition -> Both EventStore (Event) and GraphStore (TransitionEdge)
            elif obs_type == TemporalObservationType.CAMERA_TRANSITION:
                from_cam = str(obs.metadata.get("from_camera_id", "unknown"))
                to_cam = obs.camera_id or "unknown"
                duration_s = float(obs.metadata.get("transition_duration_s", 0.0))

                # EventStore
                evt = Event(
                    event_id=obs.observation_id,
                    event_type=EventType.CROSS_CAMERA_TRANSITION,
                    global_id=obs.global_id,
                    camera_id=to_cam,
                    timestamp_ns=obs.timestamp_ns,
                    frame_id=int(obs.metadata.get("frame_id", 0)),
                    confidence=obs.confidence,
                    explanation=f"Transition from {from_cam} to {to_cam}",
                    metadata=obs.metadata,
                )
                self._event_writer.emit(evt)

                # GraphStore Transition Edge
                if obs.global_id is not None:
                    self._graph_writer.emit(
                        GraphRecord(
                            "trans_edge",
                            {
                                "edge_id": obs.observation_id,
                                "global_id": obs.global_id,
                                "from_camera_id": from_cam,
                                "to_camera_id": to_cam,
                                "timestamp_ns": obs.timestamp_ns,
                                "transition_duration_s": duration_s,
                                "confidence": obs.confidence,
                                "metadata": obs.metadata,
                            },
                        )
                    )

            # 3. Zone Occupancy -> GraphStore (ZoneNode & ZoneOccupancyEdge)
            elif obs_type == TemporalObservationType.ZONE_OCCUPANCY:
                zone_id = obs.zone_id or str(obs.metadata.get("zone_id", "unknown_zone"))
                if obs.global_id is not None:
                    self._graph_writer.emit(
                        GraphRecord(
                            "zone_node",
                            {"zone_id": zone_id, "name": f"Zone {zone_id}"},
                        )
                    )
                    self._graph_writer.emit(
                        GraphRecord(
                            "zone_edge",
                            {
                                "edge_id": obs.observation_id,
                                "global_id": obs.global_id,
                                "zone_id": zone_id,
                                "timestamp_ns": obs.timestamp_ns,
                                "dwell_s": obs.dwell_s,
                                "metadata": obs.metadata,
                            },
                        )
                    )

            # 4. Environmental State -> GraphStore CameraNode status update
            elif obs_type == TemporalObservationType.ENVIRONMENTAL_STATE:
                if obs.camera_id is not None:
                    self._graph_writer.emit(
                        GraphRecord(
                            "camera_node",
                            {
                                "camera_id": obs.camera_id,
                                "status": "ACTIVE",
                                "metadata": obs.metadata,
                            },
                        )
                    )

            # 5. System Event / Anomaly -> EventStore
            elif obs_type == TemporalObservationType.SYSTEM_EVENT:
                evt_type = EventType.SYSTEM_MODE_CHANGED
                if obs.event is not None:
                    evt = obs.event
                else:
                    raw_type = str(obs.metadata.get("event_type", "system_mode_changed"))
                    try:
                        evt_type = EventType(raw_type)
                    except ValueError:
                        pass
                    evt = Event(
                        event_id=obs.observation_id,
                        event_type=evt_type,
                        global_id=obs.global_id,
                        camera_id=obs.camera_id or "system",
                        timestamp_ns=obs.timestamp_ns,
                        frame_id=int(obs.metadata.get("frame_id", 0)),
                        confidence=obs.confidence,
                        explanation=str(obs.metadata.get("explanation", "System event")),
                        metadata=obs.metadata,
                    )
                self._event_writer.emit(evt)

        except Exception as exc:
            _log.error("temporal_worker_processing_error", error=str(exc), observation_id=obs.observation_id)

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop worker thread cleanly."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

        _log.info("temporal_worker_stopped")
