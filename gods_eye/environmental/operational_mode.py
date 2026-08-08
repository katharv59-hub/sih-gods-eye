"""Operational Mode State Machine — Phase 3.5 (§6, Rule 9).

Governs transitions between LEARNING_MODE, OPERATIONAL_MODE, and DEGRADED_MODE.
Evaluates warm-up criteria before permitting alert emission, and monitors
drift triggers to drop to DEGRADED_MODE if models break.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.environment import (
    BackgroundModel,
    OccupancyBaseline,
    SystemMode,
)
from gods_eye.schemas.event import Event, EventType

_log = get_logger("environmental.operational_mode")


class OperationalModeManager:
    """State machine governing operational modes and warm-up criteria per camera stream.

    Not thread-safe — owned exclusively by ``EnvironmentalWorker`` (§9 single-writer).
    """

    def __init__(self, camera_id: str, settings: Settings) -> None:
        self._camera_id = camera_id
        self._warmup_bg_frames = settings.warmup_bg_frames
        self._warmup_bg_confidence = settings.warmup_bg_confidence
        self._warmup_occupancy_samples = settings.warmup_occupancy_samples
        self._force_degraded = settings.force_degraded_mode

        self._mode = (
            SystemMode.DEGRADED_MODE
            if self._force_degraded
            else SystemMode.LEARNING_MODE
        )
        self._warmup_complete = False

    @property
    def current_mode(self) -> SystemMode:
        """Return the current system operational mode."""
        return self._mode

    @property
    def is_operational(self) -> bool:
        """Return True if system is in OPERATIONAL_MODE (alerts enabled)."""
        return self._mode is SystemMode.OPERATIONAL_MODE

    def evaluate(
        self,
        bg_model: BackgroundModel,
        baselines: dict[Optional[str], OccupancyBaseline],
        timestamp_ns: int,
        frame_id: int,
    ) -> tuple[SystemMode, list[Event]]:
        """Evaluate warm-up progress or drift triggers, return current mode & generated events.

        Args:
            bg_model: Latest BackgroundModel record.
            baselines: Dict of latest OccupancyBaseline records.
            timestamp_ns: Frame timestamp.
            frame_id: Monotonic frame counter.

        Returns:
            Tuple of (current SystemMode, list of state transition Events).
        """
        events: list[Event] = []

        if self._force_degraded and self._mode is not SystemMode.DEGRADED_MODE:
            old_mode = self._mode
            self._mode = SystemMode.DEGRADED_MODE
            events.append(
                self._make_event(
                    EventType.SYSTEM_MODE_CHANGED,
                    timestamp_ns,
                    frame_id,
                    confidence=1.0,
                    explanation=f"Forced transition from {old_mode.value} to DEGRADED_MODE via configuration override.",
                    metadata={"from_mode": old_mode.value, "to_mode": self._mode.value},
                )
            )

        # ── 1. LEARNING_MODE → OPERATIONAL_MODE Warm-up Evaluation ────────────
        if self._mode is SystemMode.LEARNING_MODE:
            bg_frames_pass = bg_model.frame_count >= self._warmup_bg_frames
            bg_conf_pass = bg_model.confidence >= self._warmup_bg_confidence

            scene_baseline = baselines.get(None)
            occ_samples_pass = (
                scene_baseline is not None
                and scene_baseline.sample_count >= self._warmup_occupancy_samples
            )

            if bg_frames_pass and bg_conf_pass and occ_samples_pass:
                self._mode = SystemMode.OPERATIONAL_MODE
                self._warmup_complete = True

                events.append(
                    self._make_event(
                        EventType.WARM_UP_COMPLETE,
                        timestamp_ns,
                        frame_id,
                        confidence=bg_model.confidence,
                        explanation=f"All warm-up criteria passed for camera {self._camera_id}. System transitioning to OPERATIONAL_MODE.",
                        metadata={
                            "bg_frames": bg_model.frame_count,
                            "bg_confidence": bg_model.confidence,
                            "occupancy_samples": scene_baseline.sample_count if scene_baseline else 0,
                        },
                    )
                )
                events.append(
                    self._make_event(
                        EventType.SYSTEM_MODE_CHANGED,
                        timestamp_ns,
                        frame_id,
                        confidence=1.0,
                        explanation=f"System mode changed from LEARNING_MODE to OPERATIONAL_MODE for camera {self._camera_id}.",
                        metadata={
                            "from_mode": SystemMode.LEARNING_MODE.value,
                            "to_mode": SystemMode.OPERATIONAL_MODE.value,
                        },
                    )
                )
                _log.info(
                    "warmup_complete_operational_mode_started",
                    camera_id=self._camera_id,
                    bg_frames=bg_model.frame_count,
                    bg_confidence=bg_model.confidence,
                )

        # ── 2. OPERATIONAL_MODE → DEGRADED_MODE Drift Detection ──────────────
        elif self._mode is SystemMode.OPERATIONAL_MODE:
            drift_trigger: Optional[str] = None

            if bg_model.confidence < 0.60:
                drift_trigger = f"Background model confidence collapsed to {bg_model.confidence:.2f} (< 0.60)"

            if drift_trigger is not None:
                self._mode = SystemMode.DEGRADED_MODE
                events.append(
                    self._make_event(
                        EventType.MODEL_DRIFT_DETECTED,
                        timestamp_ns,
                        frame_id,
                        confidence=1.0,
                        explanation=f"Model drift detected for camera {self._camera_id}: {drift_trigger}.",
                        metadata={"trigger": drift_trigger},
                    )
                )
                events.append(
                    self._make_event(
                        EventType.SYSTEM_MODE_CHANGED,
                        timestamp_ns,
                        frame_id,
                        confidence=1.0,
                        explanation=f"System mode dropped from OPERATIONAL_MODE to DEGRADED_MODE due to model drift.",
                        metadata={
                            "from_mode": SystemMode.OPERATIONAL_MODE.value,
                            "to_mode": SystemMode.DEGRADED_MODE.value,
                            "trigger": drift_trigger,
                        },
                    )
                )
                _log.warning(
                    "model_drift_degraded_mode_entered",
                    camera_id=self._camera_id,
                    trigger=drift_trigger,
                )

        return self._mode, events

    def force_mode(self, mode: SystemMode, reason: str) -> None:
        """Manually override system mode."""
        old_mode = self._mode
        self._mode = mode
        _log.info(
            "force_mode_override",
            camera_id=self._camera_id,
            from_mode=old_mode.value,
            to_mode=mode.value,
            reason=reason,
        )

    def _make_event(
        self,
        event_type: EventType,
        timestamp_ns: int,
        frame_id: int,
        confidence: float,
        explanation: str,
        metadata: dict[str, object],
    ) -> Event:
        """Construct a §4 Event object."""
        return Event(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            global_id=None,
            camera_id=self._camera_id,
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            confidence=confidence,
            explanation=explanation,
            zone_id=None,
            metadata=metadata,
        )
