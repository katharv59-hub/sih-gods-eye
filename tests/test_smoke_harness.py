"""Unit test suite for Sub-Phase 7.5.1 SingleCameraSmokeHarness."""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.ingestion.smoke_harness import (
    SingleCameraSmokeHarness,
    SmokeHarnessMetrics,
)
from gods_eye.ingestion.source import FrameSource


class MockSyntheticFrameSource(FrameSource):
    """Synthetic FrameSource for smoke test harness verification."""

    def __init__(self, fps: float = 30.0, resolution: tuple[int, int] = (640, 480)) -> None:
        self._fps = fps
        self._resolution = resolution
        self._open = False

    def open(self) -> bool:
        self._open = True
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._open:
            return False, None
        frame = np.zeros((self._resolution[1], self._resolution[0], 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self._open = False

    def is_opened(self) -> bool:
        return self._open

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps_nominal(self) -> float:
        return self._fps

    @property
    def source_type(self) -> str:
        return "synthetic"

    @property
    def is_live(self) -> bool:
        return True


class TestSmokeHarness:
    def test_metrics_serialization(self) -> None:
        m = SmokeHarnessMetrics(
            camera_id="cam_smoke_1",
            source_type="file",
            duration_seconds=5.0,
            total_frames_read=150,
            total_frames_processed=150,
            input_fps=30.0,
            processed_fps=30.0,
        )
        d = m.to_dict()
        assert d["camera_id"] == "cam_smoke_1"
        assert d["source_type"] == "file"
        assert d["total_frames_read"] == 150
        assert d["input_fps"] == 30.0

    def test_smoke_harness_execution(self, tmp_path) -> None:
        db_path = str(tmp_path / "smoke_events.db")
        event_store = SQLiteEventStore(db_path)
        source = MockSyntheticFrameSource(fps=30.0)

        harness = SingleCameraSmokeHarness(
            camera_id="cam_test_01",
            source=source,
            event_store=event_store,
        )

        metrics = harness.run_smoke_test(duration_seconds=0.5)

        assert metrics.camera_id == "cam_test_01"
        assert metrics.source_type == "synthetic"
        assert metrics.total_frames_read >= 0
        assert metrics.total_frames_processed >= 0
        assert metrics.error_count == 0

        event_store.close()
