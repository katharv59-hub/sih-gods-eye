"""CameraCaptureThread — one thread per video source.

Per §8 Concurrency Model:
- Camera Thread (1 per source) → FrameQueue
- Capture thread owns VideoCapture
- Capture thread never performs inference
- Capture thread only acquires frames and publishes to queue
"""

from __future__ import annotations

import threading
import time

from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.frame_queue import FrameQueue
from gods_eye.ingestion.source import FrameSource
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry


# Reconnection constants (§12: "Reconnect within 5s of stream drop")
_RECONNECT_INITIAL_DELAY_S: float = 0.5
_RECONNECT_MAX_DELAY_S: float = 5.0
_RECONNECT_BACKOFF_FACTOR: float = 2.0


class CameraCaptureThread(threading.Thread):
    """Captures frames from a FrameSource and publishes them to a FrameQueue.

    Lifecycle:
        1. Thread starts → opens source
        2. Loop: read frame → wrap in FramePacket → put into queue
        3. On failure (live source): reconnect with exponential backoff
        4. On stop() or source exhaustion: release source, close queue
    """

    def __init__(
        self,
        camera_id: str,
        source: FrameSource,
        frame_queue: FrameQueue,
        *,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"capture-{camera_id}")
        self._camera_id = camera_id
        self._source = source
        self._frame_queue = frame_queue
        self._metrics = metrics
        self._stop_event = threading.Event()
        self._frame_counter = 0
        self._log = get_logger("ingestion", camera_id)

    def run(self) -> None:
        """Main capture loop. Runs until stop() is called or source exhausted."""
        self._log.info(
            "capture_thread_started",
            source_type=self._source.source_type,
            is_live=self._source.is_live,
        )

        if not self._source.open():
            self._log.error("source_open_failed", source_type=self._source.source_type)
            return

        self._log.info(
            "source_opened",
            resolution=list(self._source.resolution),
            fps_nominal=self._source.fps_nominal,
        )

        fps_timer = time.perf_counter()
        fps_frame_count = 0

        while not self._stop_event.is_set():
            ret, frame = self._source.read()

            if not ret or frame is None:
                if self._source.is_live:
                    if not self._handle_reconnect():
                        break  # stop was requested during reconnect
                    fps_timer = time.perf_counter()
                    fps_frame_count = 0
                    continue
                else:
                    self._log.info(
                        "source_exhausted", total_frames=self._frame_counter
                    )
                    break

            self._frame_counter += 1
            timestamp_ns = time.time_ns()

            packet = FramePacket(
                camera_id=self._camera_id,
                frame_id=self._frame_counter,
                timestamp_ns=timestamp_ns,
                frame=frame,
                resolution=self._source.resolution,
            )
            self._frame_queue.put(packet)

            # FPS measurement (update every second)
            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                if self._metrics is not None:
                    self._metrics.fps.labels(
                        subsystem="ingestion", camera_id=self._camera_id
                    ).set(current_fps)
                fps_timer = time.perf_counter()
                fps_frame_count = 0

        # Shutdown: producer closes first (§8)
        self._source.release()
        self._frame_queue.close()
        self._log.info(
            "capture_thread_stopped", total_frames=self._frame_counter
        )

    def _handle_reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff.

        Returns True if reconnected, False if stop was requested.
        Per §12: "Reconnect with backoff; log ERROR; continue"
        Per Phase 1 gate: "RTSP reconnect within 5s of stream drop"
        """
        self._log.error(
            "source_disconnected", source_type=self._source.source_type
        )
        self._source.release()

        if self._metrics is not None:
            self._metrics.frame_drops_total.labels(
                camera_id=self._camera_id, reason="source_disconnected"
            ).inc()

        delay = _RECONNECT_INITIAL_DELAY_S

        while not self._stop_event.is_set():
            self._log.info("reconnect_attempt", delay_s=delay)
            self._stop_event.wait(delay)

            if self._stop_event.is_set():
                return False

            if self._source.open():
                self._log.info(
                    "reconnect_success",
                    resolution=list(self._source.resolution),
                )
                return True

            self._log.warning("reconnect_failed", next_delay_s=delay)
            delay = min(delay * _RECONNECT_BACKOFF_FACTOR, _RECONNECT_MAX_DELAY_S)

        return False

    def stop(self) -> None:
        """Request graceful shutdown of the capture thread."""
        self._log.info("capture_stop_requested")
        self._stop_event.set()

    @property
    def frame_count(self) -> int:
        """Total frames captured so far."""
        return self._frame_counter

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is alive and not stopped."""
        return self.is_alive() and not self._stop_event.is_set()
