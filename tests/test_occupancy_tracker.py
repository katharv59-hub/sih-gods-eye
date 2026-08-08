"""Unit tests for occupancy tracker & baselines — Phase 3.5 Component 6."""

from __future__ import annotations

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.environmental.background_model import BackgroundModeler
from gods_eye.environmental.occupancy_tracker import (
    OccupancyTracker,
    get_time_bucket,
    point_in_polygon,
)
from gods_eye.schemas.camera import Zone
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.environment import LightingCondition


class TestOccupancyTracker:

    def test_point_in_polygon(self) -> None:
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert point_in_polygon(0.5, 0.5, poly) is True
        assert point_in_polygon(1.5, 1.5, poly) is False

    def test_time_bucket_format(self) -> None:
        # Timestamp for 2026-08-07 10:00 UTC (Friday)
        ts = 1786096800 * 1_000_000_000
        bucket = get_time_bucket(ts)
        assert bucket.startswith("FRI_")

    def test_occupancy_tracker_updates_and_baselines(self) -> None:
        settings = Settings(warmup_occupancy_samples=3)
        tracker = OccupancyTracker("cam_01", settings)
        bg_modeler = BackgroundModeler("cam_01", settings)

        zone = Zone(
            zone_id="z1",
            camera_id="cam_01",
            display_name="Door",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            zone_type="entrance",
            expected_dwell_s=(1.0, 10.0),
        )

        det = Detection(
            detection_id="d1",
            camera_id="cam_01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            confidence=0.9,
            class_label="person",
            source_resolution=(100, 100),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Feed 3 samples
        for i in range(3):
            ts = (i + 1) * 1_000_000_000
            fg_mask, bg_model_record = bg_modeler.apply(frame, ts)
            scene_state, baselines = tracker.update(
                [det],
                [zone],
                ts,
                LightingCondition.DAY_NORMAL,
                bg_model_record,
                fg_mask,
            )

        assert scene_state.occupancy_count == 1
        assert "z1" in baselines
        z_base = baselines["z1"]
        assert z_base.sample_count == 3
        assert z_base.is_mature is True
