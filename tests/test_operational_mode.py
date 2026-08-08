"""Unit tests for operational mode state machine — Phase 3.5 Component 6."""

from __future__ import annotations

from gods_eye.config.settings import Settings
from gods_eye.environmental.operational_mode import OperationalModeManager
from gods_eye.schemas.environment import (
    BackgroundModel,
    OccupancyBaseline,
    SystemMode,
)
from gods_eye.schemas.event import EventType


class TestOperationalModeManager:

    def test_initial_mode_learning(self) -> None:
        settings = Settings(warmup_bg_frames=10, warmup_occupancy_samples=5)
        mgr = OperationalModeManager("cam_test", settings)
        assert mgr.current_mode is SystemMode.LEARNING_MODE
        assert mgr.is_operational is False

    def test_warmup_transition_to_operational(self) -> None:
        settings = Settings(
            warmup_bg_frames=10,
            warmup_bg_confidence=0.80,
            warmup_occupancy_samples=3,
        )
        mgr = OperationalModeManager("cam_test", settings)

        bg_model = BackgroundModel(
            camera_id="cam_test",
            model_type="gmm",
            version=1,
            created_ns=1_000_000_000,
            last_updated_ns=1_000_000_000,
            warm_up_complete=True,
            frame_count=10,
            confidence=0.85,
        )

        baseline = OccupancyBaseline(
            camera_id="cam_test",
            zone_id=None,
            time_bucket="MON_10",
            mean_occupancy=2.0,
            std_occupancy=0.5,
            sample_count=3,
            last_updated_ns=1_000_000_000,
            is_mature=True,
        )

        mode, events = mgr.evaluate(bg_model, {None: baseline}, 1_000_000_000, 10)

        assert mode is SystemMode.OPERATIONAL_MODE
        assert mgr.is_operational is True

        event_types = [e.event_type for e in events]
        assert EventType.WARM_UP_COMPLETE in event_types
        assert EventType.SYSTEM_MODE_CHANGED in event_types

    def test_drift_detection_drops_to_degraded(self) -> None:
        settings = Settings(
            warmup_bg_frames=10,
            warmup_bg_confidence=0.80,
            warmup_occupancy_samples=3,
        )
        mgr = OperationalModeManager("cam_test", settings)

        bg_good = BackgroundModel(
            camera_id="cam_test",
            model_type="gmm",
            version=1,
            created_ns=1_000_000_000,
            last_updated_ns=1_000_000_000,
            warm_up_complete=True,
            frame_count=10,
            confidence=0.85,
        )
        baseline = OccupancyBaseline(
            camera_id="cam_test",
            zone_id=None,
            time_bucket="MON_10",
            mean_occupancy=2.0,
            std_occupancy=0.5,
            sample_count=3,
            last_updated_ns=1_000_000_000,
            is_mature=True,
        )

        mgr.evaluate(bg_good, {None: baseline}, 1_000_000_000, 10)
        assert mgr.current_mode is SystemMode.OPERATIONAL_MODE

        # Low confidence trigger
        bg_bad = BackgroundModel(
            camera_id="cam_test",
            model_type="gmm",
            version=1,
            created_ns=2_000_000_000,
            last_updated_ns=2_000_000_000,
            warm_up_complete=False,
            frame_count=11,
            confidence=0.40,  # drops below 0.60
        )

        mode, events = mgr.evaluate(bg_bad, {None: baseline}, 2_000_000_000, 11)

        assert mode is SystemMode.DEGRADED_MODE
        assert mgr.is_operational is False

        event_types = [e.event_type for e in events]
        assert EventType.MODEL_DRIFT_DETECTED in event_types
        assert EventType.SYSTEM_MODE_CHANGED in event_types
