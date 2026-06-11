"""Tests for the pipeline module."""

from __future__ import annotations

import threading
import time

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.pipeline import (
    DetectionResult,
    OutputWorker,
    Pipeline,
    PipelineQueue,
    TrackingResult,
)
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import Track, TrackState

from tests.test_ingestion import MockFrameSource


def _make_packet(frame_id: int = 1) -> FramePacket:
    return FramePacket(
        camera_id="test",
        frame_id=frame_id,
        timestamp_ns=frame_id * 33_000_000,
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        resolution=(2, 2),
    )


class TestPipelineQueue:
    def test_put_get(self) -> None:
        q: PipelineQueue[int] = PipelineQueue(5, "test", "cam")
        q.put(42)
        assert q.get(timeout=1.0) == 42

    def test_drop_oldest(self) -> None:
        q: PipelineQueue[int] = PipelineQueue(2, "test", "cam")
        q.put(1)
        q.put(2)
        q.put(3)  # drops 1
        assert q.drop_count == 1
        assert q.get(timeout=0.1) == 2

    def test_close_drain(self) -> None:
        q: PipelineQueue[str] = PipelineQueue(5, "test", "cam")
        q.put("a")
        q.close()
        assert q.get(timeout=0.1) == "a"
        assert q.get(timeout=0.1) is None


class TestOutputWorker:
    def test_delivers_results(self) -> None:
        q: PipelineQueue[TrackingResult] = PipelineQueue(5, "test", "cam")
        results: list[TrackingResult] = []

        worker = OutputWorker("cam", q, results.append)
        worker.start()

        pkt = _make_packet()
        q.put(TrackingResult(packet=pkt, detections=[], tracks=[]))
        time.sleep(0.1)
        q.close()
        worker.join(timeout=2.0)

        assert len(results) == 1
        assert worker.delivered_count == 1


class TestPipelineE2E:
    def test_full_pipeline_with_mock_source(self) -> None:
        settings = Settings(
            detection_confidence_threshold=0.25,
            track_lost_timeout=5,
            track_dead_timeout=10,
            frame_queue_size=10,
            detection_queue_size=10,
            track_queue_size=10,
        )
        source = MockFrameSource(num_frames=5)
        results: list[TrackingResult] = []
        lock = threading.Lock()

        def on_result(r: TrackingResult) -> None:
            with lock:
                results.append(r)

        # Use a no-op detector for unit testing
        from gods_eye.detection.detector import Detector
        from gods_eye.tracking.tracker import Tracker

        class StubDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Detection]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class StubTracker(Tracker):
            def update(self, detections: list[Detection],
                       frame_id: int, camera_id: str) -> list[Track]:
                return []
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 0
            @property
            def total_track_count(self) -> int:
                return 0

        pipeline = Pipeline(
            camera_id="test",
            source=source,
            detector=StubDetector(),
            tracker=StubTracker(),
            settings=settings,
            on_result=on_result,
        )

        pipeline.start()
        time.sleep(1.0)
        pipeline.stop(timeout=3.0)

        with lock:
            assert len(results) == 5
        assert pipeline.stats["delivered"] == 5
