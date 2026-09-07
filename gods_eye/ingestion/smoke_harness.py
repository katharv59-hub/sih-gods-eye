"""Single-Camera Pipeline Smoke Test Harness — Sub-Phase 7.5.1.

Provides automated smoke testing of the end-to-end perception, tracking, event,
and memory ingestion pipeline against an authorized FrameSource (RTSP stream,
video file, or synthetic stream).

Measures runtime engineering metrics only (FPS, latency, frame counts, active tracks,
dropped frames, errors) without claiming empirical accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.events.event_store import BaseEventStore
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.source import FrameSource
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.track import Track, TrackState

if TYPE_CHECKING:
    from gods_eye.detection.detector import Detector
    from gods_eye.tracking.tracker import Tracker

_log = get_logger("ingestion.smoke_harness")


def _get_smoke_mock_detector_class():
    from gods_eye.detection.detector import Detector

    class SmokeMockDetector(Detector):
        def detect(self, packet: FramePacket) -> list[Detection]:
            bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
            det = Detection(
                detection_id=f"det_{packet.frame_id}_0",
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                timestamp_ns=packet.timestamp_ns,
                bbox=bbox,
                confidence=0.88,
                class_label="person",
                source_resolution=packet.source_resolution,
            )
            return [det]

        def warmup(self) -> None:
            pass

        @property
        def device(self) -> str:
            return "cpu"

        @property
        def model_name(self) -> str:
            return "smoke_mock_detector"

    return SmokeMockDetector


def _get_smoke_mock_tracker_class():
    from gods_eye.tracking.tracker import Tracker

    class SmokeMockTracker(Tracker):
        def __init__(self) -> None:
            self._tracks: list[Track] = []

        def update(
            self,
            detections: list[Detection],
            frame_id: int,
            camera_id: str,
        ) -> list[Track]:
            tracks: list[Track] = []
            for idx, det in enumerate(detections, start=1):
                trk = Track(
                    track_id=idx,
                    camera_id=camera_id,
                    state=TrackState.TRACKED,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=0,
                    first_seen_ns=det.timestamp_ns,
                    last_seen_ns=det.timestamp_ns,
                    hits=1,
                    age=1,
                )
                tracks.append(trk)
            self._tracks = tracks
            return tracks

        def reset(self) -> None:
            self._tracks = []

        @property
        def active_track_count(self) -> int:
            return len(self._tracks)

        @property
        def total_track_count(self) -> int:
            return len(self._tracks)

    return SmokeMockTracker


@dataclass
class SmokeHarnessMetrics:
    """Engineering metrics collected during pipeline smoke test run."""

    camera_id: str
    source_type: str
    duration_seconds: float
    total_frames_read: int = 0
    total_frames_processed: int = 0
    dropped_frames: int = 0
    input_fps: float = 0.0
    processed_fps: float = 0.0
    total_detections: int = 0
    total_active_tracks: int = 0
    track_creations: int = 0
    track_terminations: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    events_generated: int = 0
    reconnect_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize engineering metrics to dictionary representation."""
        return {
            "camera_id": self.camera_id,
            "source_type": self.source_type,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_frames_read": self.total_frames_read,
            "total_frames_processed": self.total_frames_processed,
            "dropped_frames": self.dropped_frames,
            "input_fps": round(self.input_fps, 2),
            "processed_fps": round(self.processed_fps, 2),
            "total_detections": self.total_detections,
            "total_active_tracks": self.total_active_tracks,
            "track_creations": self.track_creations,
            "track_terminations": self.track_terminations,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "error_count": self.error_count,
            "events_generated": self.events_generated,
            "reconnect_count": self.reconnect_count,
        }


class SingleCameraSmokeHarness:
    """Smoke test orchestrator for single-camera pipeline integration."""

    def __init__(
        self,
        camera_id: str,
        source: FrameSource,
        detector: Optional[Detector] = None,
        tracker: Optional[Tracker] = None,
        settings: Optional[Settings] = None,
        event_store: Optional[BaseEventStore] = None,
        graph_store: Optional[BaseGraphStore] = None,
        metrics_registry: Optional[MetricsRegistry] = None,
    ) -> None:
        self._camera_id = camera_id
        self._source = source
        self._detector = detector or _get_smoke_mock_detector_class()()
        self._tracker = tracker or _get_smoke_mock_tracker_class()()
        self._settings = settings or Settings()
        self._event_store = event_store
        self._graph_store = graph_store
        self._metrics_registry = metrics_registry

        self._active_track_ids: set[int] = set()
        self._seen_track_ids: set[int] = set()

    def run_smoke_test(
        self,
        duration_seconds: float = 5.0,
        max_frames: Optional[int] = None,
    ) -> SmokeHarnessMetrics:
        """Run single-camera pipeline smoke test and return engineering metrics."""
        from gods_eye.pipeline import Pipeline  # lazy import

        metrics = SmokeHarnessMetrics(
            camera_id=self._camera_id,
            source_type=self._source.source_type,
            duration_seconds=duration_seconds,
        )

        latencies: list[float] = []
        start_time = time.perf_counter()

        def _on_result(id_result) -> None:
            t_proc = time.perf_counter()
            metrics.total_frames_processed += 1
            trk_res = id_result.tracking
            metrics.total_detections += len(trk_res.detections)

            current_tracks = {t.track_id for t in trk_res.tracks}
            new_tracks = current_tracks - self._seen_track_ids
            self._seen_track_ids.update(new_tracks)
            metrics.track_creations += len(new_tracks)

            metrics.total_active_tracks = len(current_tracks)
            self._active_track_ids = current_tracks

            if self._event_store is not None and len(trk_res.tracks) > 0:
                for trk in trk_res.tracks:
                    ev = Event(
                        event_id=f"ev_smoke_{self._camera_id}_{trk_res.packet.frame_id}_{trk.track_id}",
                        event_type=EventType.PERSON_DETECTION,
                        timestamp_ns=trk_res.packet.timestamp_ns,
                        camera_id=self._camera_id,
                        global_id=f"sub_{trk.track_id}",
                        confidence=trk.confidence,
                        payload={"track_id": trk.track_id, "bbox": [trk.bbox.x1, trk.bbox.y1, trk.bbox.x2, trk.bbox.y2]},
                    )
                    try:
                        self._event_store.append_event(ev)
                        metrics.events_generated += 1
                    except Exception as exc:
                        metrics.error_count += 1

            proc_time_ms = (t_proc - start_time) * 1000 / max(1, metrics.total_frames_processed)
            latencies.append(proc_time_ms)

        pipeline = Pipeline(
            camera_id=self._camera_id,
            source=self._source,
            detector=self._detector,
            tracker=self._tracker,
            settings=self._settings,
            on_result=_on_result,
            metrics=self._metrics_registry,
        )

        pipeline.start()

        run_start = time.perf_counter()
        while (time.perf_counter() - run_start) < duration_seconds:
            if max_frames is not None and metrics.total_frames_processed >= max_frames:
                break
            time.sleep(0.05)

        pipeline.stop()

        elapsed = time.perf_counter() - run_start
        metrics.duration_seconds = elapsed
        metrics.total_frames_read = pipeline.stats.get("captured", 0)
        metrics.dropped_frames = max(0, metrics.total_frames_read - metrics.total_frames_processed)
        metrics.input_fps = metrics.total_frames_read / elapsed if elapsed > 0 else 0.0
        metrics.processed_fps = metrics.total_frames_processed / elapsed if elapsed > 0 else 0.0
        metrics.avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0

        return metrics
