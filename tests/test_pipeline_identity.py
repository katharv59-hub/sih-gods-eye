"""Pipeline integration tests for IdentityWorker — Phase 2, Task 8.

Tests verify:
    - IdentityWorker passthrough mode (no mapper)
    - IdentityWorker with a mock identity mapper
    - IdentityWorker error handling (mapper raises)
    - IdentityWorker shutdown propagation (queue close chain)
    - Full pipeline E2E with identity mapper wired in
    - Pipeline stats include identity count
    - Pipeline E2E with passthrough (backward compat)
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.detection.detector import Detector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.pipeline import (
    IdentityWorker,
    Pipeline,
    PipelineQueue,
    TrackingResult,
)
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.identity import LifecycleState, create_identity
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.identity_mapper import (
    IdentityMapper,
    IdentityResult,
    TransitionType,
)
from gods_eye.reid.matcher import Matcher
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import Track, TrackState
from gods_eye.tracking.tracker import Tracker
from tests.test_ingestion import MockFrameSource


# ── Helpers ────────────────────────────────────────────────────────────────


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_packet(frame_id: int = 1) -> FramePacket:
    return FramePacket(
        camera_id="test",
        frame_id=frame_id,
        timestamp_ns=frame_id * 33_000_000,
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        resolution=(640, 480),
    )


def _make_track(
    track_id: str, state: TrackState = TrackState.ACTIVE
) -> Track:
    return Track(
        track_id=track_id,
        camera_id="test",
        state=state,
        bbox=BoundingBox(x1=10, y1=20, x2=110, y2=220),
        velocity=(0.0, 0.0),
        first_frame_id=1,
        last_frame_id=5,
        lost_frame_count=0 if state is TrackState.ACTIVE else 10,
    )


def _make_detection() -> Detection:
    return Detection(
        detection_id="det-1",
        camera_id="test",
        frame_id=1,
        timestamp_ns=1000,
        bbox=BoundingBox(x1=10, y1=20, x2=110, y2=220),
        confidence=0.9,
        class_label="person",
        source_resolution=(640, 480),
    )


def _make_tracking_result(
    frame_id: int = 1,
    tracks: list[Track] | None = None,
) -> TrackingResult:
    return TrackingResult(
        packet=_make_packet(frame_id),
        detections=[_make_detection()] if tracks else [],
        tracks=tracks or [],
    )


# ── Stub Components ───────────────────────────────────────────────────────


class _StubDetector(Detector):
    """Returns a single detection with a fixed bbox per frame."""

    def detect(self, packet: FramePacket) -> list[Detection]:
        return [
            Detection(
                detection_id=f"det-{packet.frame_id}",
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                timestamp_ns=packet.timestamp_ns,
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300),
                confidence=0.9,
                class_label="person",
                source_resolution=packet.resolution,
            )
        ]

    def warmup(self) -> None:
        pass

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "stub"


class _NoDetectionsDetector(Detector):
    """Returns no detections — exercises passthrough path."""

    def detect(self, packet: FramePacket) -> list[Detection]:
        return []

    def warmup(self) -> None:
        pass

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "stub-empty"


class _StubTracker(Tracker):
    """Returns one stable track per detection."""

    def __init__(self) -> None:
        self._track_count = 0

    def update(
        self, detections: list[Detection], frame_id: int, camera_id: str
    ) -> list[Track]:
        tracks: list[Track] = []
        for det in detections:
            tracks.append(
                Track(
                    track_id="T1",
                    camera_id=camera_id,
                    state=TrackState.ACTIVE,
                    bbox=det.bbox,
                    velocity=(0.0, 0.0),
                    first_frame_id=1,
                    last_frame_id=frame_id,
                    lost_frame_count=0,
                )
            )
        return tracks

    def reset(self) -> None:
        pass

    @property
    def active_track_count(self) -> int:
        return 0

    @property
    def total_track_count(self) -> int:
        return 0


class _MockExtractor(EmbeddingExtractor):
    """Returns deterministic embeddings by call count."""

    def __init__(self) -> None:
        self._call_count = 0

    def extract(
        self, frame: np.ndarray, bboxes: list[BoundingBox]
    ) -> list[np.ndarray]:
        self._call_count += 1
        return [_unit_vec(i + self._call_count * 1000) for i in range(len(bboxes))]

    def warmup(self) -> None:
        pass

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "mock"


class _FailingMapper:
    """Mock mapper that always raises."""

    def process(self, result: TrackingResult) -> IdentityResult:
        raise RuntimeError("Simulated GPU OOM")


# ── IdentityWorker Unit Tests ────────────────────────────────────────────


class TestIdentityWorkerPassthrough:
    """IdentityWorker with mapper=None — pure passthrough mode."""

    def test_wraps_tracking_result_in_identity_result(self) -> None:
        """Passthrough mode wraps TrackingResult in bare IdentityResult."""
        in_q: PipelineQueue[TrackingResult] = PipelineQueue(5, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(5, "out", "test")

        worker = IdentityWorker("test", None, in_q, out_q)
        worker.start()

        tr = _make_tracking_result(1, [_make_track("T1")])
        in_q.put(tr)
        time.sleep(0.1)

        ir = out_q.get(timeout=1.0)
        assert ir is not None
        assert isinstance(ir, IdentityResult)
        assert ir.tracking is tr
        assert ir.identities == {}
        assert ir.transitions == []

        in_q.close()
        worker.join(timeout=2.0)
        assert worker.processed_count == 1

    def test_multiple_frames_pass_through(self) -> None:
        """Multiple frames pass through with correct ordering."""
        in_q: PipelineQueue[TrackingResult] = PipelineQueue(10, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(10, "out", "test")

        worker = IdentityWorker("test", None, in_q, out_q)
        worker.start()

        for i in range(5):
            in_q.put(_make_tracking_result(i + 1))

        time.sleep(0.3)
        in_q.close()
        worker.join(timeout=2.0)

        assert worker.processed_count == 5
        results = []
        while True:
            r = out_q.get(timeout=0.1)
            if r is None:
                break
            results.append(r)
        assert len(results) == 5


class TestIdentityWorkerWithMapper:
    """IdentityWorker with a real IdentityMapper."""

    def test_resolves_identities(self) -> None:
        """Mapper produces IdentityResult with resolved identities."""
        settings = Settings()
        gallery = IdentityGallery()
        lifecycle = LifecycleManager(settings)
        matcher = Matcher(settings)
        mapper = IdentityMapper(
            extractor=_MockExtractor(),
            gallery=gallery,
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="test",
        )

        in_q: PipelineQueue[TrackingResult] = PipelineQueue(5, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(5, "out", "test")

        worker = IdentityWorker("test", mapper, in_q, out_q)
        worker.start()

        tr = _make_tracking_result(1, [_make_track("T1")])
        in_q.put(tr)
        time.sleep(0.2)

        ir = out_q.get(timeout=1.0)
        assert ir is not None
        assert "T1" in ir.identities
        assert ir.identities["T1"].state is LifecycleState.ACTIVE
        assert len(ir.transitions) == 1
        assert ir.transitions[0].transition_type is TransitionType.CONFIRMED_NEW

        in_q.close()
        worker.join(timeout=2.0)


class TestIdentityWorkerErrorHandling:
    """IdentityWorker error handling — mapper failures."""

    def test_mapper_exception_produces_bare_result(self) -> None:
        """When mapper.process() raises, worker falls back to bare IdentityResult."""
        in_q: PipelineQueue[TrackingResult] = PipelineQueue(5, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(5, "out", "test")

        worker = IdentityWorker("test", _FailingMapper(), in_q, out_q)  # type: ignore[arg-type]
        worker.start()

        tr = _make_tracking_result(1, [_make_track("T1")])
        in_q.put(tr)
        time.sleep(0.2)

        ir = out_q.get(timeout=1.0)
        assert ir is not None
        assert ir.tracking is tr
        assert ir.identities == {}  # no identities due to error
        assert ir.transitions == []

        in_q.close()
        worker.join(timeout=2.0)
        assert worker.processed_count == 1  # still counted

    def test_error_does_not_kill_worker(self) -> None:
        """Worker survives mapper errors and processes subsequent frames."""
        in_q: PipelineQueue[TrackingResult] = PipelineQueue(5, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(5, "out", "test")

        worker = IdentityWorker("test", _FailingMapper(), in_q, out_q)  # type: ignore[arg-type]
        worker.start()

        for i in range(3):
            in_q.put(_make_tracking_result(i + 1))

        time.sleep(0.3)
        in_q.close()
        worker.join(timeout=2.0)
        assert worker.processed_count == 3


class TestIdentityWorkerShutdown:
    """IdentityWorker shutdown behavior."""

    def test_closes_output_queue_on_input_close(self) -> None:
        """When input queue closes, worker closes output queue."""
        in_q: PipelineQueue[TrackingResult] = PipelineQueue(5, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(5, "out", "test")

        worker = IdentityWorker("test", None, in_q, out_q)
        worker.start()

        in_q.close()
        worker.join(timeout=2.0)

        assert out_q.is_closed
        assert not worker.is_alive()

    def test_drains_remaining_items_before_close(self) -> None:
        """Worker processes all queued items before exiting."""
        in_q: PipelineQueue[TrackingResult] = PipelineQueue(10, "in", "test")
        out_q: PipelineQueue[IdentityResult] = PipelineQueue(10, "out", "test")

        worker = IdentityWorker("test", None, in_q, out_q)

        # Put items before starting worker
        for i in range(3):
            in_q.put(_make_tracking_result(i + 1))

        worker.start()
        time.sleep(0.3)
        in_q.close()
        worker.join(timeout=2.0)

        assert worker.processed_count == 3


# ── Full Pipeline E2E Tests ──────────────────────────────────────────────


class TestPipelineE2EWithIdentity:
    """Full pipeline E2E with IdentityMapper wired in."""

    def test_pipeline_with_identity_mapper(self) -> None:
        """Pipeline produces IdentityResults with resolved identities."""
        settings = Settings(
            frame_queue_size=10,
            detection_queue_size=10,
            track_queue_size=10,
        )
        source = MockFrameSource(num_frames=5)
        gallery = IdentityGallery()
        lifecycle = LifecycleManager(settings)
        matcher = Matcher(settings)
        mapper = IdentityMapper(
            extractor=_MockExtractor(),
            gallery=gallery,
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="test",
        )

        results: list[IdentityResult] = []
        lock = threading.Lock()

        def on_result(r: IdentityResult) -> None:
            with lock:
                results.append(r)

        pipeline = Pipeline(
            camera_id="test",
            source=source,
            detector=_StubDetector(),
            tracker=_StubTracker(),
            settings=settings,
            on_result=on_result,
            identity_mapper=mapper,
        )

        pipeline.start()
        time.sleep(2.0)
        pipeline.stop(timeout=3.0)

        with lock:
            assert len(results) == 5

        # Each result should have the tracking data
        for r in results:
            assert isinstance(r, IdentityResult)
            assert r.tracking is not None
            assert r.tracking.packet is not None

        # First result should have registered a new identity
        assert len(results[0].identities) == 1
        first_tid = list(results[0].identities.keys())[0]
        assert results[0].identities[first_tid].state is LifecycleState.ACTIVE

    def test_pipeline_passthrough_backward_compat(self) -> None:
        """Pipeline with no identity_mapper still works (Phase 1 compat)."""
        settings = Settings(
            frame_queue_size=10,
            detection_queue_size=10,
            track_queue_size=10,
        )
        source = MockFrameSource(num_frames=3)
        results: list[IdentityResult] = []
        lock = threading.Lock()

        def on_result(r: IdentityResult) -> None:
            with lock:
                results.append(r)

        pipeline = Pipeline(
            camera_id="test",
            source=source,
            detector=_NoDetectionsDetector(),
            tracker=_StubTracker(),
            settings=settings,
            on_result=on_result,
            # identity_mapper=None (default)
        )

        pipeline.start()
        time.sleep(1.0)
        pipeline.stop(timeout=3.0)

        with lock:
            assert len(results) == 3

        # All results should be bare IdentityResults
        for r in results:
            assert isinstance(r, IdentityResult)
            assert r.identities == {}
            assert r.transitions == []

    def test_pipeline_stats_include_identity_count(self) -> None:
        """Pipeline.stats includes 'identified' count."""
        settings = Settings(
            frame_queue_size=10,
            detection_queue_size=10,
            track_queue_size=10,
        )
        source = MockFrameSource(num_frames=3)

        pipeline = Pipeline(
            camera_id="test",
            source=source,
            detector=_NoDetectionsDetector(),
            tracker=_StubTracker(),
            settings=settings,
            on_result=lambda r: None,
        )

        pipeline.start()
        time.sleep(1.0)
        pipeline.stop(timeout=3.0)

        stats = pipeline.stats
        assert "identified" in stats
        assert stats["identified"] == 3

    def test_pipeline_graceful_shutdown_chain(self) -> None:
        """Shutdown propagates: capture → det → track → identity → output."""
        settings = Settings(
            frame_queue_size=10,
            detection_queue_size=10,
            track_queue_size=10,
        )
        source = MockFrameSource(num_frames=2)

        pipeline = Pipeline(
            camera_id="test",
            source=source,
            detector=_NoDetectionsDetector(),
            tracker=_StubTracker(),
            settings=settings,
            on_result=lambda r: None,
        )

        pipeline.start()
        time.sleep(1.0)
        pipeline.stop(timeout=3.0)

        # All queues should be closed after shutdown
        assert pipeline._frame_q.is_closed
        assert pipeline._det_q.is_closed
        assert pipeline._track_q.is_closed
        assert pipeline._identity_q.is_closed

        # All workers should have exited
        assert not pipeline._detection.is_alive()
        assert not pipeline._tracking.is_alive()
        assert not pipeline._identity.is_alive()
        assert not pipeline._output.is_alive()
