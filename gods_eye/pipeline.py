"""End-to-end pipeline — §8 Concurrency Model.

CaptureThread → [FrameQueue] → DetectionWorker → [DetectionQueue] →
TrackingWorker → [TrackQueue] → IdentityWorker → [IdentityQueue] →
OutputWorker

Rules (§8):
- Every queue: one producer, one consumer
- Overflow: drop oldest, log reason="queue_overflow"
- Shutdown: producers close first, consumers drain before exit
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.detection.detector import Detector
from gods_eye.ingestion.capture_thread import CameraCaptureThread
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.frame_queue import FrameQueue
from gods_eye.ingestion.source import FrameSource
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.detection import Detection
from gods_eye.schemas.track import Track
from gods_eye.tracking.tracker import Tracker

if TYPE_CHECKING:
    from gods_eye.events.event_store import BaseEventStore
    from gods_eye.evidence.evidence_manager import EvidenceManager
    from gods_eye.face.live_engine import LiveFaceEngine
    from gods_eye.face.recognizer import FaceRecognizer
    from gods_eye.memory.graph_store import BaseGraphStore
    from gods_eye.reid.identity_mapper import IdentityMapper, IdentityResult

# ─── Pipeline Data Types ─────────────────────────────────────────────────────

T = TypeVar("T")


@dataclass
class DetectionResult:
    """Detection worker output: frame + detections."""

    packet: FramePacket
    detections: list[Detection]


@dataclass
class TrackingResult:
    """Tracking worker output: frame + detections + tracks."""

    packet: FramePacket
    detections: list[Detection]
    tracks: list[Track]


# ─── Generic Pipeline Queue ──────────────────────────────────────────────────


class PipelineQueue(Generic[T]):
    """Thread-safe queue with drop-oldest overflow (§8)."""

    def __init__(
        self,
        maxsize: int,
        queue_name: str,
        camera_id: str,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._maxsize = maxsize
        self._queue_name = queue_name
        self._camera_id = camera_id
        self._metrics = metrics
        self._deque: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self._drop_count = 0
        self._log = get_logger("pipeline.queue", camera_id)

    def put(self, item: T) -> None:
        with self._lock:
            if self._closed:
                return
            if len(self._deque) >= self._maxsize:
                self._deque.popleft()
                self._drop_count += 1
                self._log.warning(
                    "item_dropped",
                    reason="queue_overflow",
                    queue=self._queue_name,
                )
                if self._metrics is not None:
                    self._metrics.frame_drops_total.labels(
                        camera_id=self._camera_id, reason="queue_overflow"
                    ).inc()
            self._deque.append(item)
            if self._metrics is not None:
                self._metrics.queue_depth.labels(
                    queue_name=self._queue_name
                ).set(len(self._deque))
            self._not_empty.notify()

    def get(self, timeout: float | None = None) -> T | None:
        with self._not_empty:
            while len(self._deque) == 0:
                if self._closed:
                    return None
                if not self._not_empty.wait(timeout=timeout):
                    return None
            item = self._deque.popleft()
            if self._metrics is not None:
                self._metrics.queue_depth.labels(
                    queue_name=self._queue_name
                ).set(len(self._deque))
            return item

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def drop_count(self) -> int:
        return self._drop_count


# ─── Worker Threads ──────────────────────────────────────────────────────────


class DetectionWorker(threading.Thread):
    """Consumes FramePackets, runs detection, produces DetectionResults."""

    def __init__(
        self,
        camera_id: str,
        detector: Detector,
        input_queue: FrameQueue,
        output_queue: PipelineQueue[DetectionResult],
        *,
        secondary_output_queue: PipelineQueue[DetectionResult] | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"detection-{camera_id}")
        self._camera_id = camera_id
        self._detector = detector
        self._input = input_queue
        self._output = output_queue
        self._secondary_output = secondary_output_queue
        self._metrics = metrics
        self._log = get_logger("detection.worker", camera_id)
        self._processed = 0

    def run(self) -> None:
        self._log.info("detection_worker_started")
        fps_timer = time.perf_counter()
        fps_count = 0

        while True:
            packet = self._input.get(timeout=0.5)
            if packet is None:
                if self._input.is_closed:
                    break
                continue

            t0 = time.perf_counter()
            try:
                dets = self._detector.detect(packet)
            except Exception as exc:
                self._log.error("detection_error", error=str(exc))
                continue
            latency_ms = (time.perf_counter() - t0) * 1000

            if self._metrics is not None:
                self._metrics.latency_ms.labels(
                    subsystem="detection", camera_id=self._camera_id
                ).observe(latency_ms)

            det_result = DetectionResult(packet=packet, detections=dets)
            self._output.put(det_result)
            if self._secondary_output is not None:
                self._secondary_output.put(det_result)

            self._processed += 1
            fps_count += 1

            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0 and self._metrics is not None:
                self._metrics.fps.labels(
                    subsystem="detection", camera_id=self._camera_id
                ).set(fps_count / elapsed)
                fps_timer = time.perf_counter()
                fps_count = 0

        self._output.close()
        if self._secondary_output is not None:
            self._secondary_output.close()
        self._log.info("detection_worker_stopped", processed=self._processed)

    @property
    def processed_count(self) -> int:
        return self._processed


class TrackingWorker(threading.Thread):
    """Consumes DetectionResults, runs tracking, produces TrackingResults."""

    def __init__(
        self,
        camera_id: str,
        tracker: Tracker,
        input_queue: PipelineQueue[DetectionResult],
        output_queue: PipelineQueue[TrackingResult],
        *,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"tracking-{camera_id}")
        self._camera_id = camera_id
        self._tracker = tracker
        self._input = input_queue
        self._output = output_queue
        self._metrics = metrics
        self._log = get_logger("tracking.worker", camera_id)
        self._processed = 0

    def run(self) -> None:
        self._log.info("tracking_worker_started")
        fps_timer = time.perf_counter()
        fps_count = 0

        while True:
            det_result = self._input.get(timeout=0.5)
            if det_result is None:
                if self._input.is_closed:
                    break
                continue

            t0 = time.perf_counter()
            try:
                tracks = self._tracker.update(
                    det_result.detections,
                    det_result.packet.frame_id,
                    self._camera_id,
                )
            except Exception as exc:
                self._log.error("tracking_error", error=str(exc))
                tracks = []
            latency_ms = (time.perf_counter() - t0) * 1000

            if self._metrics is not None:
                self._metrics.latency_ms.labels(
                    subsystem="tracking", camera_id=self._camera_id
                ).observe(latency_ms)

            self._output.put(
                TrackingResult(
                    packet=det_result.packet,
                    detections=det_result.detections,
                    tracks=tracks,
                )
            )
            self._processed += 1
            fps_count += 1

            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0 and self._metrics is not None:
                self._metrics.fps.labels(
                    subsystem="tracking", camera_id=self._camera_id
                ).set(fps_count / elapsed)
                fps_timer = time.perf_counter()
                fps_count = 0

        self._output.close()
        self._log.info("tracking_worker_stopped", processed=self._processed)

    @property
    def processed_count(self) -> int:
        return self._processed


class IdentityWorker(threading.Thread):
    """Consumes TrackingResults, resolves identities, produces IdentityResults.

    Inserted between TrackingWorker and OutputWorker.  When no
    ``IdentityMapper`` is provided, passes TrackingResults through
    wrapped in a bare ``IdentityResult`` (backward-compatible).
    """

    def __init__(
        self,
        camera_id: str,
        mapper: IdentityMapper | None = None,
        input_queue: PipelineQueue[TrackingResult] | None = None,
        output_queue: PipelineQueue[IdentityResult] | None = None,
        *,
        identity_mapper: IdentityMapper | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"identity-{camera_id}")
        self._camera_id = camera_id
        self._mapper = mapper if mapper is not None else identity_mapper
        if input_queue is None or output_queue is None:
            raise ValueError("input_queue and output_queue are required")
        self._input = input_queue
        self._output = output_queue
        self._metrics = metrics
        self._log = get_logger("identity.worker", camera_id)
        self._processed = 0

    def run(self) -> None:
        from gods_eye.reid.identity_mapper import IdentityResult  # lazy import

        self._log.info("identity_worker_started")
        fps_timer = time.perf_counter()
        fps_count = 0

        while True:
            track_result = self._input.get(timeout=0.5)
            if track_result is None:
                if self._input.is_closed:
                    break
                continue

            t0 = time.perf_counter()
            if self._mapper is not None:
                try:
                    id_result = self._mapper.process(track_result)
                except Exception as exc:
                    self._log.error("identity_error", error=str(exc))
                    # Graceful fallback: pass through without identities
                    id_result = IdentityResult(tracking=track_result)
            else:
                # Passthrough mode: wrap tracking result as-is
                id_result = IdentityResult(tracking=track_result)

            latency_ms = (time.perf_counter() - t0) * 1000

            if self._metrics is not None:
                self._metrics.latency_ms.labels(
                    subsystem="identity", camera_id=self._camera_id
                ).observe(latency_ms)

            self._output.put(id_result)
            self._processed += 1
            fps_count += 1

            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0 and self._metrics is not None:
                self._metrics.fps.labels(
                    subsystem="identity", camera_id=self._camera_id
                ).set(fps_count / elapsed)
                fps_timer = time.perf_counter()
                fps_count = 0

        self._output.close()
        self._log.info("identity_worker_stopped", processed=self._processed)

    @property
    def processed_count(self) -> int:
        return self._processed


class OutputWorker(threading.Thread):
    """Consumes IdentityResults and delivers them via callback."""

    def __init__(
        self,
        camera_id: str,
        input_queue: PipelineQueue[IdentityResult],
        callback: Callable[[IdentityResult], None],
    ) -> None:
        super().__init__(daemon=True, name=f"output-{camera_id}")
        self._camera_id = camera_id
        self._input = input_queue
        self._callback = callback
        self._log = get_logger("output.worker", camera_id)
        self._delivered = 0

    def run(self) -> None:
        self._log.info("output_worker_started")
        while True:
            result = self._input.get(timeout=0.5)
            if result is None:
                if self._input.is_closed:
                    break
                continue
            self._callback(result)
            self._delivered += 1

        self._log.info("output_worker_stopped", delivered=self._delivered)

    @property
    def delivered_count(self) -> int:
        return self._delivered


# ─── Pipeline Orchestrator ───────────────────────────────────────────────────


class Pipeline:
    """End-to-end single-camera pipeline.

    Wires: CaptureThread → DetectionWorker → TrackingWorker →
    IdentityWorker → OutputWorker with bounded queues and graceful
    shutdown.

    When ``identity_mapper`` is None the IdentityWorker operates in
    passthrough mode — tracking results flow through wrapped in bare
    ``IdentityResult`` objects.  This keeps the pipeline backward-
    compatible with Phase 1 callers.
    """

    def __init__(
        self,
        camera_id: str,
        source: FrameSource,
        detector: Detector,
        tracker: Tracker,
        settings: Settings,
        on_result: Callable[[IdentityResult], None],
        *,
        identity_mapper: IdentityMapper | None = None,
        metrics: MetricsRegistry | None = None,
        live_face_engine: LiveFaceEngine | None = None,
        enable_face_recognition: bool = True,
        event_store: BaseEventStore | None = None,
        graph_store: BaseGraphStore | None = None,
        evidence_manager: EvidenceManager | None = None,
        face_recognizer: FaceRecognizer | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._log = get_logger("pipeline", camera_id)

        # ── Live Face Engine Integration (§10 SIH 26187) ──────────────
        self._live_face_engine = live_face_engine
        if self._live_face_engine is None and enable_face_recognition:
            try:
                from pathlib import Path
                from gods_eye.events.event_store import SQLiteEventStore
                from gods_eye.evidence.evidence_manager import EvidenceManager
                from gods_eye.face.live_engine import LiveFaceEngine
                from gods_eye.face.recognizer import FaceRecognizer
                from gods_eye.memory.graph_store import SQLiteGraphStore

                f_rec = face_recognizer
                if f_rec is None:
                    det_p = Path(settings.face_detection_model_path)
                    rec_p = Path(settings.face_recognition_model_path)
                    if det_p.exists() and rec_p.exists():
                        f_rec = FaceRecognizer(
                            detection_model_path=str(det_p),
                            recognition_model_path=str(rec_p),
                            recognition_threshold=settings.face_recognition_threshold,
                            detection_threshold=settings.face_detection_threshold,
                        )
                if f_rec is not None:
                    ev_store = (
                        event_store
                        if event_store is not None
                        else SQLiteEventStore(settings.event_store_path)
                    )
                    gr_store = (
                        graph_store
                        if graph_store is not None
                        else SQLiteGraphStore(settings.graph_store_path)
                    )
                    ev_mgr = (
                        evidence_manager
                        if evidence_manager is not None
                        else EvidenceManager(base_path=settings.evidence_base_path)
                    )
                    self._live_face_engine = LiveFaceEngine(
                        recognizer=f_rec,
                        graph_store=gr_store,
                        event_store=ev_store,
                        evidence_manager=ev_mgr,
                        throttle_window_s=settings.face_recognition_throttle_s,
                    )
            except Exception as exc:
                self._log.warning("face_engine_auto_init_failed", error=str(exc))
                self._live_face_engine = None

        # Queues (§8)
        self._frame_q = FrameQueue(
            maxsize=settings.frame_queue_size,
            queue_name=f"{camera_id}/frames",
            camera_id=camera_id,
            metrics=metrics,
        )
        self._det_q: PipelineQueue[DetectionResult] = PipelineQueue(
            maxsize=settings.detection_queue_size,
            queue_name=f"{camera_id}/detections",
            camera_id=camera_id,
            metrics=metrics,
        )
        self._track_q: PipelineQueue[TrackingResult] = PipelineQueue(
            maxsize=settings.track_queue_size,
            queue_name=f"{camera_id}/tracks",
            camera_id=camera_id,
            metrics=metrics,
        )
        self._identity_q: PipelineQueue[IdentityResult] = PipelineQueue(
            maxsize=settings.track_queue_size,
            queue_name=f"{camera_id}/identities",
            camera_id=camera_id,
            metrics=metrics,
        )

        # Workers
        self._capture = CameraCaptureThread(
            camera_id, source, self._frame_q, metrics=metrics
        )
        self._detection = DetectionWorker(
            camera_id, detector, self._frame_q, self._det_q, metrics=metrics
        )
        self._tracking = TrackingWorker(
            camera_id, tracker, self._det_q, self._track_q, metrics=metrics
        )
        self._identity = IdentityWorker(
            camera_id, identity_mapper, self._track_q, self._identity_q,
            metrics=metrics,
        )

        def _pipeline_on_result(res: IdentityResult) -> None:
            if self._live_face_engine is not None and res.tracking is not None:
                try:
                    self._live_face_engine.process_frame(
                        frame=res.tracking.packet.frame,
                        tracks=res.tracking.tracks,
                        camera_id=self._camera_id,
                        frame_id=res.tracking.packet.frame_id,
                        timestamp_ns=res.tracking.packet.timestamp_ns,
                        identities=res.identities,
                    )
                except Exception as exc:
                    self._log.error("live_face_engine_error", error=str(exc))
            on_result(res)

        self._output = OutputWorker(camera_id, self._identity_q, _pipeline_on_result)

    def start(self) -> None:
        """Start all pipeline threads."""
        self._log.info("pipeline_starting")
        self._capture.start()
        self._detection.start()
        self._tracking.start()
        self._identity.start()
        self._output.start()
        self._log.info("pipeline_started")

    def stop(self, timeout: float = 5.0) -> None:
        """Graceful shutdown: producers close first, consumers drain."""
        self._log.info("pipeline_stopping")
        self._capture.stop()
        self._capture.join(timeout=timeout)
        self._detection.join(timeout=timeout)
        self._tracking.join(timeout=timeout)
        self._identity.join(timeout=timeout)
        self._output.join(timeout=timeout)
        self._log.info(
            "pipeline_stopped",
            captured=self._capture.frame_count,
            detected=self._detection.processed_count,
            tracked=self._tracking.processed_count,
            identified=self._identity.processed_count,
            delivered=self._output.delivered_count,
            frame_q_drops=self._frame_q_drops,
            det_q_drops=self._det_q.drop_count,
            track_q_drops=self._track_q.drop_count,
            identity_q_drops=self._identity_q.drop_count,
        )

    @property
    def is_alive(self) -> bool:
        return self._capture.is_alive()

    @property
    def _frame_q_drops(self) -> int:
        # FrameQueue doesn't expose drop_count, count from the deque
        return 0  # tracked via metrics

    @property
    def stats(self) -> dict[str, int]:
        return {
            "captured": self._capture.frame_count,
            "detected": self._detection.processed_count,
            "tracked": self._tracking.processed_count,
            "identified": self._identity.processed_count,
            "delivered": self._output.delivered_count,
        }

    @property
    def live_face_engine(self) -> LiveFaceEngine | None:
        """Attached LiveFaceEngine instance, or None if disabled/unavailable."""
        return self._live_face_engine
