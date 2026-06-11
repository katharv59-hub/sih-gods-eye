"""Tests for the ingestion subsystem.

Tests FramePacket, FrameQueue (overflow, close, drain), and
CameraCaptureThread (lifecycle with mock source).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.frame_queue import FrameQueue
from gods_eye.ingestion.capture_thread import CameraCaptureThread
from gods_eye.ingestion.source import FrameSource


# ─── Mock Source ──────────────────────────────────────────────────────────────


class MockFrameSource(FrameSource):
    """Generates synthetic frames for testing."""

    def __init__(
        self,
        num_frames: int = 10,
        *,
        is_live: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self._num_frames = num_frames
        self._is_live_flag = is_live
        self._fail_after = fail_after
        self._opened = False
        self._read_count = 0
        self._resolution: tuple[int, int] = (640, 480)

    def open(self) -> bool:
        self._opened = True
        self._read_count = 0
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened:
            return False, None
        if self._fail_after is not None and self._read_count >= self._fail_after:
            return False, None
        if self._read_count >= self._num_frames:
            return False, None
        self._read_count += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps_nominal(self) -> float:
        return 30.0

    @property
    def source_type(self) -> str:
        return "mock"

    @property
    def is_live(self) -> bool:
        return self._is_live_flag


# ─── FramePacket Tests ───────────────────────────────────────────────────────


class TestFramePacket:
    def test_construction(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            frame=frame,
            resolution=(640, 480),
        )
        assert packet.camera_id == "cam-01"
        assert packet.frame_id == 1
        assert packet.frame.shape == (480, 640, 3)
        assert packet.resolution == (640, 480)


# ─── FrameQueue Tests ────────────────────────────────────────────────────────


class TestFrameQueue:
    @staticmethod
    def _make_packet(frame_id: int) -> FramePacket:
        return FramePacket(
            camera_id="cam-01",
            frame_id=frame_id,
            timestamp_ns=frame_id * 1_000_000,
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            resolution=(2, 2),
        )

    def test_put_get(self) -> None:
        q = FrameQueue(maxsize=5, queue_name="test_q", camera_id="cam-01")
        pkt = self._make_packet(1)
        q.put(pkt)
        result = q.get(timeout=1.0)
        assert result is not None
        assert result.frame_id == 1

    def test_drop_oldest_on_overflow(self) -> None:
        q = FrameQueue(maxsize=3, queue_name="test_q", camera_id="cam-01")
        # Fill queue
        q.put(self._make_packet(1))
        q.put(self._make_packet(2))
        q.put(self._make_packet(3))
        assert q.qsize == 3
        # This should drop frame_id=1 (oldest)
        q.put(self._make_packet(4))
        assert q.qsize == 3
        # First out should be frame_id=2 (oldest after drop)
        result = q.get(timeout=1.0)
        assert result is not None
        assert result.frame_id == 2

    def test_get_timeout_returns_none(self) -> None:
        q = FrameQueue(maxsize=5, queue_name="test_q", camera_id="cam-01")
        result = q.get(timeout=0.05)
        assert result is None

    def test_close_and_drain(self) -> None:
        q = FrameQueue(maxsize=5, queue_name="test_q", camera_id="cam-01")
        q.put(self._make_packet(1))
        q.put(self._make_packet(2))
        q.close()
        assert q.is_closed
        # Can still drain remaining items
        r1 = q.get(timeout=0.1)
        assert r1 is not None and r1.frame_id == 1
        r2 = q.get(timeout=0.1)
        assert r2 is not None and r2.frame_id == 2
        # Empty and closed → None
        r3 = q.get(timeout=0.1)
        assert r3 is None

    def test_close_unblocks_waiting_consumer(self) -> None:
        q = FrameQueue(maxsize=5, queue_name="test_q", camera_id="cam-01")
        result_holder: list[FramePacket | None] = [None]

        def consumer() -> None:
            result_holder[0] = q.get(timeout=5.0)

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.05)
        q.close()
        t.join(timeout=1.0)
        assert result_holder[0] is None

    def test_put_after_close_is_noop(self) -> None:
        q = FrameQueue(maxsize=5, queue_name="test_q", camera_id="cam-01")
        q.close()
        q.put(self._make_packet(1))
        assert q.qsize == 0


# ─── CameraCaptureThread Tests ───────────────────────────────────────────────


class TestCameraCaptureThread:
    def test_captures_all_frames_from_file_source(self) -> None:
        source = MockFrameSource(num_frames=5, is_live=False)
        q = FrameQueue(maxsize=10, queue_name="test_q", camera_id="cam-01")
        thread = CameraCaptureThread("cam-01", source, q)

        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert q.is_closed  # producer closed first
        assert thread.frame_count == 5

        # Drain and verify
        frames: list[FramePacket] = []
        while True:
            pkt = q.get(timeout=0.1)
            if pkt is None:
                break
            frames.append(pkt)
        assert len(frames) == 5
        assert frames[0].frame_id == 1
        assert frames[4].frame_id == 5

    def test_stop_terminates_thread(self) -> None:
        source = MockFrameSource(num_frames=1_000_000, is_live=False)
        q = FrameQueue(maxsize=10, queue_name="test_q", camera_id="cam-01")
        thread = CameraCaptureThread("cam-01", source, q)

        thread.start()
        time.sleep(0.05)
        thread.stop()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert q.is_closed
        assert thread.frame_count > 0

    def test_queue_overflow_drops_oldest(self) -> None:
        source = MockFrameSource(num_frames=10, is_live=False)
        q = FrameQueue(maxsize=3, queue_name="test_q", camera_id="cam-01")
        thread = CameraCaptureThread("cam-01", source, q)

        thread.start()
        thread.join(timeout=2.0)

        assert thread.frame_count == 10
        # Queue should have last 3 frames (oldest were dropped)
        frames: list[FramePacket] = []
        while True:
            pkt = q.get(timeout=0.1)
            if pkt is None:
                break
            frames.append(pkt)
        assert len(frames) == 3
        assert frames[0].frame_id == 8
        assert frames[2].frame_id == 10

    def test_source_open_failure(self) -> None:
        """Thread exits gracefully when source fails to open."""

        class FailSource(MockFrameSource):
            def open(self) -> bool:
                return False

        source = FailSource(num_frames=5)
        q = FrameQueue(maxsize=10, queue_name="test_q", camera_id="cam-01")
        thread = CameraCaptureThread("cam-01", source, q)

        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert thread.frame_count == 0
