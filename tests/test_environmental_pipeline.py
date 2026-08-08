"""Integration tests for Environmental Pipeline & Multi-Camera — Phase 3.5 Component 6."""

from __future__ import annotations

import time
import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.environmental.environmental_worker import (
    EnvironmentalResult,
    EnvironmentalWorker,
)
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.pipeline import DetectionResult, PipelineQueue
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.event import EventType


class TestEnvironmentalPipelineIntegration:

    def test_environmental_worker_suppresses_alerts_in_learning_mode(self) -> None:
        """Rule 9: System must NEVER produce anomaly alerts during LEARNING_MODE."""
        settings = Settings(
            cpu_only=True,
            warmup_bg_frames=100,  # stays in LEARNING_MODE
            warmup_occupancy_samples=50,
        )

        in_q: PipelineQueue[DetectionResult] = PipelineQueue(
            maxsize=10, queue_name="test/det", camera_id="cam_test"
        )
        out_q: PipelineQueue[EnvironmentalResult] = PipelineQueue(
            maxsize=10, queue_name="test/env", camera_id="cam_test"
        )

        results: list[EnvironmentalResult] = []

        def callback(res: EnvironmentalResult) -> None:
            results.append(res)

        worker = EnvironmentalWorker(
            "cam_test", settings, in_q, out_q, callback=callback
        )
        worker.start()

        det = Detection(
            detection_id="d1",
            camera_id="cam_test",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            confidence=0.9,
            class_label="person",
            source_resolution=(100, 100),
        )

        # Send 5 frames (large crowd to try triggering anomaly)
        for i in range(5):
            packet = FramePacket(
                camera_id="cam_test",
                frame_id=i + 1,
                timestamp_ns=(i + 1) * 1_000_000_000,
                frame=np.ones((100, 100, 3), dtype=np.uint8) * 100,
                resolution=(100, 100),
            )
            in_q.put(DetectionResult(packet=packet, detections=[det] * 10))

        in_q.close()
        worker.join(timeout=2.0)

        assert len(results) == 5
        for res in results:
            assert res.scene_state.is_anomalous is False or res.events == []
            # Rule 9: Zero ENVIRONMENTAL_ANOMALY events emitted in LEARNING_MODE
            anomaly_events = [
                e for e in res.events if e.event_type is EventType.ENVIRONMENTAL_ANOMALY
            ]
            assert len(anomaly_events) == 0

    def test_environmental_worker_transitions_and_emits_warmup(self) -> None:
        """System emits WARM_UP_COMPLETE when criteria are met."""
        settings = Settings(
            cpu_only=True,
            warmup_bg_frames=2,  # fast warm-up
            warmup_bg_confidence=0.10,
            warmup_occupancy_samples=1,
        )

        in_q: PipelineQueue[DetectionResult] = PipelineQueue(
            maxsize=10, queue_name="test/det", camera_id="cam_test"
        )
        results: list[EnvironmentalResult] = []

        worker = EnvironmentalWorker(
            "cam_test",
            settings,
            in_q,
            callback=lambda res: results.append(res),
        )
        worker.start()

        det = Detection(
            detection_id="d1",
            camera_id="cam_test",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            confidence=0.9,
            class_label="person",
            source_resolution=(100, 100),
        )

        for i in range(3):
            packet = FramePacket(
                camera_id="cam_test",
                frame_id=i + 1,
                timestamp_ns=(i + 1) * 1_000_000_000,
                frame=np.ones((100, 100, 3), dtype=np.uint8) * 120,
                resolution=(100, 100),
            )
            in_q.put(DetectionResult(packet=packet, detections=[det]))

        in_q.close()
        worker.join(timeout=2.0)

        assert len(results) == 3
        all_events = [e for res in results for e in res.events]
        warmup_events = [
            e for e in all_events if e.event_type is EventType.WARM_UP_COMPLETE
        ]
        assert len(warmup_events) == 1
        assert warmup_events[0].explanation != ""
