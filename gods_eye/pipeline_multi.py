"""Multi-camera pipeline — Phase 3 §9 Concurrency Model.

Manages N per-camera capture→detect→track chains that fan-in to a
single shared IdentityWorker with one IdentityGallery. This preserves
the single-writer guarantee from §9.

Architecture::

    Camera A: Capture → Detect → Track ──┐
    Camera B: Capture → Detect → Track ──┼──▶ [Fan-In TrackQueue] → IdentityWorker → OutputWorker
    Camera C: Capture → Detect → Track ──┘

The existing single-camera ``Pipeline`` class is left unchanged for
backward compatibility. ``MultiCameraPipeline`` composes the same
worker classes but wires them into a fan-in topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gods_eye.evidence.evidence_manager import EvidenceManager
    from gods_eye.face.live_engine import LiveFaceEngine
    from gods_eye.face.recognizer import FaceRecognizer

from gods_eye.camera_graph.camera_graph import CameraGraph
from gods_eye.config.settings import Settings
from gods_eye.detection.detector import Detector
from gods_eye.ingestion.capture_thread import CameraCaptureThread
from gods_eye.ingestion.frame_queue import FrameQueue
from gods_eye.ingestion.source import FrameSource
from gods_eye.observability.logger import get_logger
from gods_eye.environmental.environmental_worker import (
    EnvironmentalResult,
    EnvironmentalWorker,
)
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.pipeline import (
    DetectionResult,
    DetectionWorker,
    IdentityWorker,
    OutputWorker,
    PipelineQueue,
    TrackingResult,
    TrackingWorker,
)
from gods_eye.reid.identity_mapper import IdentityMapper, IdentityResult
from gods_eye.schemas.camera import CameraNode
from gods_eye.tracking.tracker import Tracker


@dataclass
class CameraPipelineChain:
    """A per-camera capture→detect→track + environmental processing chain.

    Each chain has its own threads for capture, detection, tracking, and
    environmental intelligence. Tracking output is written to a shared
    fan-in queue consumed by a single IdentityWorker.
    """

    camera_id: str
    capture: CameraCaptureThread
    detection: DetectionWorker
    tracking: TrackingWorker
    environmental: EnvironmentalWorker
    frame_queue: FrameQueue
    det_queue: PipelineQueue  # type: ignore[type-arg]
    env_queue: PipelineQueue  # type: ignore[type-arg]
    track_queue_local: PipelineQueue  # type: ignore[type-arg]


from gods_eye.events.event_store import BaseEventStore
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.memory.temporal_worker import TemporalWorker


class MultiCameraPipeline:
    """Multi-camera pipeline with shared identity & temporal memory processing.

    Wires N per-camera chains (capture→detect→track) into a single
    shared IdentityWorker via a fan-in TrackQueue and TemporalWorker
    for async graph & event persistence.
    """

    def __init__(
        self,
        camera_configs: list[CameraNode],
        source_factory: Callable[[str], FrameSource],
        detector: Detector,
        tracker_factory: Callable[[str], Tracker],
        identity_mapper: IdentityMapper,
        camera_graph: CameraGraph,
        settings: Settings,
        on_result: Callable[[IdentityResult], None],
        *,
        on_environmental_result: Callable[[EnvironmentalResult], None] | None = None,
        event_store: BaseEventStore | None = None,
        graph_store: BaseGraphStore | None = None,
        evidence_manager: EvidenceManager | None = None,
        face_recognizer: FaceRecognizer | None = None,
        live_face_engine: LiveFaceEngine | None = None,
        enable_face_recognition: bool = True,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._log = get_logger("pipeline.multi")
        self._camera_graph = camera_graph
        self._settings = settings
        self._on_env_result = on_environmental_result
        self._metrics = metrics
        self._user_on_result = on_result

        # ── Live Face Engine integration (§10 SIH 26187) ─────────────
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

        # ── Temporal memory integration (§14 Phase 4) ────────────────
        self._temporal_worker: TemporalWorker | None = None
        self._event_writer: EventWriter | None = None
        self._graph_writer: GraphWriter | None = None
        self._admission_control: TemporalAdmissionControl | None = None

        if event_store is not None and graph_store is not None:
            self._event_writer = EventWriter(event_store, metrics=metrics, batch_size=settings.temporal_batch_size)
            self._graph_writer = GraphWriter(graph_store, metrics=metrics, batch_size=settings.temporal_batch_size)
            self._admission_control = TemporalAdmissionControl(
                maxsize=settings.event_queue_size,
                high_watermark_pct=0.80,
                metrics=metrics,
            )
            self._temporal_worker = TemporalWorker(
                admission_control=self._admission_control,
                event_writer=self._event_writer,
                graph_writer=self._graph_writer,
                metrics=metrics,
            )

        def _combined_on_result(result: IdentityResult) -> None:
            if self._admission_control is not None:
                observations = TemporalAdapter.adapt_identity_result(result)
                for obs in observations:
                    self._admission_control.submit(obs)
            if self._live_face_engine is not None and result.tracking is not None:
                try:
                    self._live_face_engine.process_frame(
                        frame=result.tracking.packet.frame,
                        tracks=result.tracking.tracks,
                        camera_id=result.tracking.packet.camera_id,
                        frame_id=result.tracking.packet.frame_id,
                        timestamp_ns=result.tracking.packet.timestamp_ns,
                        identities=result.identities,
                    )
                except Exception as exc:
                    self._log.error("live_face_engine_error", error=str(exc))
            self._user_on_result(result)

        def _combined_on_env_result(env_res: EnvironmentalResult) -> None:
            if self._admission_control is not None:
                env_obs = TemporalAdapter.adapt_environmental_state(env_res.scene_state)
                self._admission_control.submit(env_obs)
                if env_res.events:
                    for evt in env_res.events:
                        evt_obs = TemporalAdapter.adapt_event(evt)
                        self._admission_control.submit(evt_obs)
            if self._on_env_result is not None:
                self._on_env_result(env_res)

        self._on_env_result = _combined_on_env_result

        # ── Shared fan-in queue: all tracking workers output here ─────
        self._fan_in_queue: PipelineQueue[TrackingResult] = PipelineQueue(
            maxsize=settings.track_queue_size * len(camera_configs),
            queue_name="fan_in/tracks",
            camera_id="multi",
            metrics=metrics,
        )

        # ── Shared identity processing (single-writer §9) ───────────
        self._identity_q: PipelineQueue[IdentityResult] = PipelineQueue(
            maxsize=settings.track_queue_size,
            queue_name="shared/identities",
            camera_id="multi",
            metrics=metrics,
        )

        self._identity_worker = IdentityWorker(
            camera_id="multi",
            identity_mapper=identity_mapper,
            input_queue=self._fan_in_queue,
            output_queue=self._identity_q,
            metrics=metrics,
        )
        self._output_worker = OutputWorker(
            camera_id="multi",
            input_queue=self._identity_q,
            callback=_combined_on_result,
        )

        # ── Per-camera chains ────────────────────────────────────────
        self._chains: list[CameraPipelineChain] = []

        for cam in camera_configs:
            chain = self._create_chain(
                cam, source_factory, detector, tracker_factory,
            )
            self._chains.append(chain)

        self._log.info(
            "multi_camera_pipeline_created",
            camera_count=len(self._chains),
            camera_ids=[c.camera_id for c in self._chains],
        )

    def _create_chain(
        self,
        cam: CameraNode,
        source_factory: Callable[[str], FrameSource],
        detector: Detector,
        tracker_factory: Callable[[str], Tracker],
    ) -> CameraPipelineChain:
        """Create a per-camera capture→detect→track + environmental chain."""
        cam_id = cam.camera_id
        source = source_factory(cam.source_uri)

        frame_q = FrameQueue(
            maxsize=self._settings.frame_queue_size,
            queue_name=f"{cam_id}/frames",
            camera_id=cam_id,
            metrics=self._metrics,
        )
        det_q: PipelineQueue[DetectionResult] = PipelineQueue(
            maxsize=self._settings.detection_queue_size,
            queue_name=f"{cam_id}/detections",
            camera_id=cam_id,
            metrics=self._metrics,
        )
        env_q: PipelineQueue[DetectionResult] = PipelineQueue(
            maxsize=self._settings.detection_queue_size,
            queue_name=f"{cam_id}/environmental",
            camera_id=cam_id,
            metrics=self._metrics,
        )

        capture = CameraCaptureThread(
            cam_id, source, frame_q, metrics=self._metrics,
        )
        detection = DetectionWorker(
            cam_id,
            detector,
            frame_q,
            det_q,
            secondary_output_queue=env_q,
            metrics=self._metrics,
        )
        tracking = TrackingWorker(
            cam_id,
            tracker_factory(cam_id),
            det_q,
            self._fan_in_queue,  # Fan-in: output goes to shared queue
            metrics=self._metrics,
        )
        environmental = EnvironmentalWorker(
            cam_id,
            self._settings,
            env_q,
            camera_graph=self._camera_graph,
            callback=self._on_env_result,
            metrics=self._metrics,
        )

        return CameraPipelineChain(
            camera_id=cam_id,
            capture=capture,
            detection=detection,
            tracking=tracking,
            environmental=environmental,
            frame_queue=frame_q,
            det_queue=det_q,
            env_queue=env_q,
            track_queue_local=det_q,
        )

    def start(self) -> None:
        """Start all camera chains, shared identity/output workers, and temporal memory workers."""
        self._log.info("multi_pipeline_starting")

        # Start temporal memory workers first
        if self._event_writer is not None:
            self._event_writer.start()
        if self._graph_writer is not None:
            self._graph_writer.start()
        if self._temporal_worker is not None:
            self._temporal_worker.start()

        # Start per-camera chains
        for chain in self._chains:
            chain.capture.start()
            chain.detection.start()
            chain.tracking.start()
            chain.environmental.start()

        # Start shared workers
        self._identity_worker.start()
        self._output_worker.start()

        self._log.info(
            "multi_pipeline_started",
            cameras=[c.camera_id for c in self._chains],
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Graceful shutdown: stop cameras first, drain shared queue, flush temporal writers."""
        self._log.info("multi_pipeline_stopping")

        # Phase 1: Stop all capture threads (producers)
        for chain in self._chains:
            chain.capture.stop()

        # Phase 2: Wait for capture threads to finish
        for chain in self._chains:
            chain.capture.join(timeout=timeout)

        # Phase 3: Wait for detection workers to drain and stop
        for chain in self._chains:
            chain.detection.join(timeout=timeout)

        # Phase 4: Wait for tracking & environmental workers to drain and stop
        for chain in self._chains:
            chain.tracking.join(timeout=timeout)
            chain.environmental.join(timeout=timeout)

        # Phase 5: Close fan-in queue, drain identity worker
        self._fan_in_queue.close()
        self._identity_worker.join(timeout=timeout)

        # Phase 6: Drain output worker
        self._output_worker.join(timeout=timeout)

        # Phase 7: Stop temporal worker & flush writers cleanly
        if self._temporal_worker is not None:
            self._temporal_worker.stop(timeout_s=timeout)
        if self._event_writer is not None:
            self._event_writer.stop(timeout_s=timeout)
        if self._graph_writer is not None:
            self._graph_writer.stop(timeout_s=timeout)

        self._log.info(
            "multi_pipeline_stopped",
            chains_stopped=len(self._chains),
        )

    @property
    def camera_count(self) -> int:
        """Number of camera chains in this pipeline."""
        return len(self._chains)

    @property
    def is_alive(self) -> bool:
        """True if at least one camera chain is still running."""
        return any(c.capture.is_alive() for c in self._chains)

    @property
    def stats(self) -> dict[str, dict[str, int]]:
        """Per-camera and shared statistics."""
        result: dict[str, dict[str, int]] = {}
        for chain in self._chains:
            result[chain.camera_id] = {
                "captured": chain.capture.frame_count,
                "detected": chain.detection.processed_count,
                "tracked": chain.tracking.processed_count,
                "environmental": chain.environmental.processed_count,
            }
        result["shared"] = {
            "identified": self._identity_worker.processed_count,
            "delivered": self._output_worker.delivered_count,
        }
        return result

    @property
    def live_face_engine(self) -> LiveFaceEngine | None:
        """Attached LiveFaceEngine instance, or None if disabled/unavailable."""
        return self._live_face_engine
