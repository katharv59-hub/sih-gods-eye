"""Unit tests for Phase 4.1 Temporal Observation Boundary.

Tests schema validation, priority classification, TemporalAdapter conversions,
non-blocking admission control thinning, and Prometheus metric tracking.
"""

from __future__ import annotations

import time
import uuid
import numpy as np
import pytest

from gods_eye.memory.adapter import TemporalAdapter
from gods_eye.memory.admission_control import TemporalAdmissionControl
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.reid.identity_mapper import IdentityResult
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.environment import LightingCondition, SceneState
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.observation import (
    ObservationPriority,
    TemporalObservation,
    TemporalObservationType,
)
from gods_eye.schemas.track import Track


def _make_obs(
    obs_type: TemporalObservationType = TemporalObservationType.IDENTITY_PRESENCE,
    camera_id: str | None = "cam_01",
    priority: ObservationPriority = ObservationPriority.PERIODIC,
    timestamp_ns: int = 1_000_000_000,
) -> TemporalObservation:
    return TemporalObservation(
        observation_id=str(uuid.uuid4()),
        observation_type=obs_type,
        priority=priority,
        timestamp_ns=timestamp_ns,
        camera_id=camera_id,
        global_id="global_001",
        confidence=0.9,
    )


class TestTemporalObservationSchema:

    def test_valid_observation_creation(self) -> None:
        obs = _make_obs()
        assert obs.observation_id is not None
        assert obs.observation_type == TemporalObservationType.IDENTITY_PRESENCE
        assert obs.priority == ObservationPriority.PERIODIC
        assert obs.camera_id == "cam_01"

    @pytest.mark.parametrize(
        "obs_type",
        [
            TemporalObservationType.IDENTITY_PRESENCE,
            TemporalObservationType.CAMERA_TRANSITION,
            TemporalObservationType.ZONE_OCCUPANCY,
            TemporalObservationType.ENVIRONMENTAL_STATE,
        ],
    )
    def test_missing_camera_id_rejection(self, obs_type: TemporalObservationType) -> None:
        """Requirement 2 & 4: Missing or empty camera_id rejected for non-SYSTEM_EVENT types."""
        with pytest.raises(ValueError, match="camera_id is REQUIRED"):
            TemporalObservation(
                observation_id="obs_1",
                observation_type=obs_type,
                priority=ObservationPriority.PERIODIC,
                timestamp_ns=1000,
                camera_id=None,
            )

        with pytest.raises(ValueError, match="camera_id is REQUIRED"):
            TemporalObservation(
                observation_id="obs_1",
                observation_type=obs_type,
                priority=ObservationPriority.PERIODIC,
                timestamp_ns=1000,
                camera_id="   ",
            )

    def test_missing_camera_id_accepted_for_system_event(self) -> None:
        """Requirement 3: camera_id is Optional for SYSTEM_EVENT."""
        obs = TemporalObservation(
            observation_id="obs_sys",
            observation_type=TemporalObservationType.SYSTEM_EVENT,
            priority=ObservationPriority.CRITICAL,
            timestamp_ns=1000,
            camera_id=None,
        )
        assert obs.camera_id is None


class TestTemporalAdapter:

    def test_priority_classification(self) -> None:
        """Requirement 5: Deterministic priority classification."""
        # Critical event types
        assert TemporalAdapter.classify_priority(TemporalObservationType.SYSTEM_EVENT, EventType.CROSS_CAMERA_TRANSITION) == ObservationPriority.CRITICAL
        assert TemporalAdapter.classify_priority(TemporalObservationType.SYSTEM_EVENT, EventType.IDENTITY_CONFIRMED) == ObservationPriority.CRITICAL
        assert TemporalAdapter.classify_priority(TemporalObservationType.SYSTEM_EVENT, EventType.ENVIRONMENTAL_ANOMALY) == ObservationPriority.CRITICAL
        assert TemporalAdapter.classify_priority(TemporalObservationType.CAMERA_TRANSITION) == ObservationPriority.CRITICAL

        # Periodic observation types
        assert TemporalAdapter.classify_priority(TemporalObservationType.IDENTITY_PRESENCE) == ObservationPriority.PERIODIC
        assert TemporalAdapter.classify_priority(TemporalObservationType.ENVIRONMENTAL_STATE) == ObservationPriority.PERIODIC
        assert TemporalAdapter.classify_priority(TemporalObservationType.SYSTEM_EVENT, EventType.PERSON_ENTERED_FRAME) == ObservationPriority.PERIODIC

    def test_adapt_identity_result(self) -> None:
        """Requirement 15: TemporalAdapter converts IdentityResult."""
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.pipeline import TrackingResult
        from gods_eye.reid.identity import create_identity

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = FramePacket(camera_id="cam_01", frame_id=1, timestamp_ns=1000, frame=frame, resolution=(10, 10))
        det = Detection("d1", "cam_01", 1, 1000, BoundingBox(0, 0, 10, 10), 0.9, "person", source_resolution=(100, 100))
        track = Track("t1", "cam_01", 1, 1000, BoundingBox(0, 0, 10, 10), det, last_frame_id=1, lost_frame_count=0)
        tr_res = TrackingResult(packet, [det], [track])
        identity = create_identity(np.zeros(512, dtype=np.float32), 1000, confidence=0.95, global_id="gid_100", camera_id="cam_01")
        res = IdentityResult(tracking=tr_res, identities={"t1": identity})

        observations = TemporalAdapter.adapt_identity_result(res)
        assert len(observations) == 1
        obs = observations[0]
        assert obs.observation_type == TemporalObservationType.IDENTITY_PRESENCE
        assert obs.priority == ObservationPriority.PERIODIC
        assert obs.global_id == "gid_100"
        assert obs.camera_id == "cam_01"

    def test_adapt_event(self) -> None:
        """Requirement 15: TemporalAdapter converts Event."""
        evt = Event(
            event_id="evt_01",
            event_type=EventType.CROSS_CAMERA_TRANSITION,
            global_id="gid_01",
            camera_id="cam_02",
            timestamp_ns=5000,
            frame_id=20,
            confidence=0.92,
            explanation="Transition detected",
        )
        obs = TemporalAdapter.adapt_event(evt)
        assert obs.observation_type == TemporalObservationType.CAMERA_TRANSITION
        assert obs.priority == ObservationPriority.CRITICAL
        assert obs.global_id == "gid_01"
        assert obs.camera_id == "cam_02"

    def test_adapt_environmental_state(self) -> None:
        """Requirement 15: TemporalAdapter converts SceneState."""
        state = SceneState(
            camera_id="cam_01",
            timestamp_ns=10000,
            lighting_condition=LightingCondition.DAY_BRIGHT,
            occupancy_count=5,
            occupancy_density=0.1,
            background_model_id="bg_1",
            foreground_ratio=0.05,
            is_anomalous=False,
            anomaly_confidence=0.0,
            anomaly_explanation="",
        )
        obs = TemporalAdapter.adapt_environmental_state(state)
        assert obs.observation_type == TemporalObservationType.ENVIRONMENTAL_STATE
        assert obs.priority == ObservationPriority.PERIODIC
        assert obs.camera_id == "cam_01"


class TestAdmissionControl:

    def test_periodic_admitted_below_80(self) -> None:
        """Requirement 6: Periodic admitted below 80% watermark limit."""
        ac = TemporalAdmissionControl(maxsize=10, high_watermark_pct=0.80)
        # Fill queue to 7 items (70%)
        for _ in range(7):
            obs = _make_obs(priority=ObservationPriority.PERIODIC)
            assert ac.submit(obs) is True
        assert ac.qsize() == 7

    def test_periodic_rejected_at_or_above_80(self) -> None:
        """Requirement 7: Periodic rejected at/above 80% watermark limit."""
        ac = TemporalAdmissionControl(maxsize=10, high_watermark_pct=0.80)
        # Fill queue to 8 items (80%)
        for _ in range(8):
            ac.submit(_make_obs(priority=ObservationPriority.PERIODIC))
        assert ac.qsize() == 8

        # 9th periodic observation must be thinned/rejected BEFORE enqueue
        obs_thinned = _make_obs(priority=ObservationPriority.PERIODIC)
        assert ac.evaluate_admission(obs_thinned) is False
        assert ac.submit(obs_thinned) is False
        assert ac.qsize() == 8  # Queue depth untouched

    def test_critical_admitted_above_80(self) -> None:
        """Requirement 8: Critical admitted above 80% watermark limit."""
        ac = TemporalAdmissionControl(maxsize=10, high_watermark_pct=0.80)
        # Fill queue to 8 items (80%) with periodic observations
        for _ in range(8):
            ac.submit(_make_obs(priority=ObservationPriority.PERIODIC))

        # Critical observation submitted when depth is 80%
        critical_obs = _make_obs(obs_type=TemporalObservationType.CAMERA_TRANSITION, priority=ObservationPriority.CRITICAL)
        assert ac.evaluate_admission(critical_obs) is True
        assert ac.submit(critical_obs) is True
        assert ac.qsize() == 9

    def test_critical_rejected_only_when_queue_full(self) -> None:
        """Requirement 9: Critical rejected only when queue depth reaches 100% (10)."""
        ac = TemporalAdmissionControl(maxsize=10, high_watermark_pct=0.80)
        # Fill queue to 10 items using critical observations
        for _ in range(10):
            obs_crit = _make_obs(obs_type=TemporalObservationType.CAMERA_TRANSITION, priority=ObservationPriority.CRITICAL)
            assert ac.submit(obs_crit) is True

        assert ac.qsize() == 10

        # 11th critical observation rejected non-blockingly
        obs_overflow = _make_obs(obs_type=TemporalObservationType.CAMERA_TRANSITION, priority=ObservationPriority.CRITICAL)
        assert ac.evaluate_admission(obs_overflow) is False
        assert ac.submit(obs_overflow) is False
        assert ac.qsize() == 10

    def test_admission_control_non_blocking_behavior(self) -> None:
        """Requirement 10 & 11: Admission control occurs before enqueue and never blocks."""
        ac = TemporalAdmissionControl(maxsize=5, high_watermark_pct=0.80)
        for _ in range(5):
            ac.submit(_make_obs(priority=ObservationPriority.CRITICAL))

        start_t = time.time()
        # Submit to full queue
        res = ac.submit(_make_obs(priority=ObservationPriority.CRITICAL))
        elapsed_s = time.time() - start_t

        assert res is False
        assert elapsed_s < 0.01  # Immediate non-blocking execution

    def test_metrics_tracking(self) -> None:
        """Requirement 12, 13, 14: Prometheus metric tracking for drops and watermark."""
        from prometheus_client import CollectorRegistry
        metrics = MetricsRegistry(registry=CollectorRegistry())
        ac = TemporalAdmissionControl(maxsize=5, high_watermark_pct=0.80, metrics=metrics)

        # Fill to watermark (4 items = 80%)
        for _ in range(4):
            ac.submit(_make_obs(priority=ObservationPriority.PERIODIC))

        # 5th periodic item thinned
        ac.submit(_make_obs(priority=ObservationPriority.PERIODIC))

        # Fill queue to 100% (5 items)
        ac.submit(_make_obs(priority=ObservationPriority.CRITICAL))

        # Critical overflow
        ac.submit(_make_obs(priority=ObservationPriority.CRITICAL))

        # Check metric values directly via prometheus registry
        periodic_drops = metrics.temporal_queue_drops_total.labels(priority="periodic")._value.get()
        critical_drops = metrics.temporal_queue_drops_total.labels(priority="critical")._value.get()
        watermark_active = metrics.temporal_queue_watermark_active._value.get()

        assert periodic_drops >= 1.0
        assert critical_drops >= 1.0
        assert watermark_active == 1.0
