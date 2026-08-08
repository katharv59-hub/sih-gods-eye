"""Environmental Pipeline Worker Thread — Phase 3.5 (§8, §9).

Consumes DetectionResults from a dedicated pipeline queue, executes the full
Environmental Intelligence workflow (background modeling, lighting classification,
occupancy tracking, operational mode governance), and outputs SceneState and Events.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from gods_eye.camera_graph.camera_graph import CameraGraph
from gods_eye.config.settings import Settings
from gods_eye.environmental.background_model import BackgroundModeler
from gods_eye.environmental.lighting_classifier import LightingClassifier
from gods_eye.environmental.occupancy_tracker import OccupancyTracker
from gods_eye.environmental.operational_mode import OperationalModeManager
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.pipeline import DetectionResult, PipelineQueue
from gods_eye.schemas.camera import Zone
from gods_eye.schemas.environment import (
    BackgroundModel,
    OccupancyBaseline,
    SceneState,
    SystemMode,
)
from gods_eye.schemas.event import Event, EventType


@dataclass
class EnvironmentalResult:
    """Output envelope from EnvironmentalWorker per processed frame."""

    camera_id: str
    frame_id: int
    timestamp_ns: int
    scene_state: SceneState
    background_model: BackgroundModel
    baselines: dict[Optional[str], OccupancyBaseline]
    events: list[Event]


class EnvironmentalWorker(threading.Thread):
    """Worker thread running environmental intelligence for a single camera.

    Not thread-safe — owns all environmental state for its camera (§9 single-writer).
    """

    def __init__(
        self,
        camera_id: str,
        settings: Settings,
        input_queue: PipelineQueue[DetectionResult],
        output_queue: Optional[PipelineQueue[EnvironmentalResult]] = None,
        *,
        camera_graph: Optional[CameraGraph] = None,
        callback: Optional[Callable[[EnvironmentalResult], None]] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        super().__init__(daemon=True, name=f"environmental-{camera_id}")
        self._camera_id = camera_id
        self._settings = settings
        self._input = input_queue
        self._output = output_queue
        self._callback = callback
        self._metrics = metrics
        self._log = get_logger("environmental.worker", camera_id)

        # ── Initialize Environmental Subsystems ────────────────────────────────
        self._bg_modeler = BackgroundModeler(camera_id, settings)
        self._lighting_classifier = LightingClassifier()
        self._occupancy_tracker = OccupancyTracker(camera_id, settings)
        self._mode_manager = OperationalModeManager(camera_id, settings)

        # Fetch zones for this camera from graph if available
        self._zones: list[Zone] = []
        if camera_graph is not None:
            node = camera_graph.get_node(camera_id)
            if node is not None:
                self._zones = node.zones

        self._processed = 0

    def run(self) -> None:
        self._log.info("environmental_worker_started", camera_id=self._camera_id)
        fps_timer = time.perf_counter()
        fps_count = 0

        while True:
            det_result = self._input.get(timeout=0.5)
            if det_result is None:
                if self._input.is_closed:
                    break
                continue

            t0 = time.perf_counter()
            frame_packet = det_result.packet
            timestamp_ns = frame_packet.timestamp_ns
            frame_id = frame_packet.frame_id

            try:
                # ── 1. Background Modeling ─────────────────────────────────────
                fg_mask, bg_model_record = self._bg_modeler.apply(
                    frame_packet.frame, timestamp_ns
                )

                # ── 2. Lighting Classification ─────────────────────────────────
                lighting = self._lighting_classifier.classify(
                    frame_packet.frame, timestamp_ns
                )

                # ── 3. Occupancy Tracking & Baselines ──────────────────────────
                scene_state, baselines = self._occupancy_tracker.update(
                    det_result.detections,
                    self._zones,
                    timestamp_ns,
                    lighting,
                    bg_model_record,
                    fg_mask,
                )

                # ── 4. Operational Mode Governance & Warm-up ───────────────────
                mode, system_events = self._mode_manager.evaluate(
                    bg_model_record, baselines, timestamp_ns, frame_id
                )

                # Collect environmental anomaly events (Rule 9: only in OPERATIONAL_MODE)
                events: list[Event] = list(system_events)

                if (
                    mode is SystemMode.OPERATIONAL_MODE
                    and scene_state.is_anomalous
                    and scene_state.anomaly_confidence
                    >= self._settings.anomaly_min_confidence
                ):
                    events.append(
                        Event(
                            event_id=uuid.uuid4().hex,
                            event_type=EventType.ENVIRONMENTAL_ANOMALY,
                            global_id=None,
                            camera_id=self._camera_id,
                            timestamp_ns=timestamp_ns,
                            frame_id=frame_id,
                            confidence=scene_state.anomaly_confidence,
                            explanation=scene_state.anomaly_explanation,
                            zone_id=None,
                            metadata={
                                "occupancy_count": scene_state.occupancy_count,
                                "lighting": lighting.value,
                            },
                        )
                    )

                env_result = EnvironmentalResult(
                    camera_id=self._camera_id,
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    scene_state=scene_state,
                    background_model=bg_model_record,
                    baselines=baselines,
                    events=events,
                )

                if self._output is not None:
                    self._output.put(env_result)

                if self._callback is not None:
                    try:
                        self._callback(env_result)
                    except Exception as cb_exc:
                        self._log.error(
                            "environmental_callback_error", error=str(cb_exc)
                        )

            except Exception as exc:
                self._log.error(
                    "environmental_worker_error",
                    frame_id=frame_id,
                    error=str(exc),
                    exc_info=True,
                )

            latency_ms = (time.perf_counter() - t0) * 1000
            self._processed += 1
            fps_count += 1

            # ── Prometheus Metrics (§7) ────────────────────────────────────────
            if self._metrics is not None:
                self._metrics.latency_ms.labels(
                    subsystem="environmental", camera_id=self._camera_id
                ).observe(latency_ms)

                mode_map = {
                    SystemMode.LEARNING_MODE: 0,
                    SystemMode.OPERATIONAL_MODE: 1,
                    SystemMode.DEGRADED_MODE: 2,
                }
                self._metrics.operational_mode.labels(
                    camera_id=self._camera_id
                ).set(mode_map.get(mode, 0))

                self._metrics.background_model_confidence.labels(
                    camera_id=self._camera_id
                ).set(bg_model_record.confidence)

                self._metrics.scene_anomaly_score.labels(
                    camera_id=self._camera_id
                ).set(scene_state.anomaly_confidence)

                # Scene occupancy gauge per camera
                self._metrics.scene_occupancy.labels(
                    camera_id=self._camera_id, zone_id="all"
                ).set(scene_state.occupancy_count)

            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0 and self._metrics is not None:
                self._metrics.fps.labels(
                    subsystem="environmental", camera_id=self._camera_id
                ).set(fps_count / elapsed)
                fps_timer = time.perf_counter()
                fps_count = 0

        self._log.info(
            "environmental_worker_stopped", processed=self._processed
        )

    @property
    def processed_count(self) -> int:
        """Total frames processed by environmental worker."""
        return self._processed
