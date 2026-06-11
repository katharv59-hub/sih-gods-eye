"""FrameQueue — thread-safe queue with drop-oldest overflow behavior.

Per §8: Queue overflow behavior is 'drop oldest frame (not newest)'.
Log every drop with reason="queue_overflow".
"""

from __future__ import annotations

import threading
from collections import deque

from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry


class FrameQueue:
    """Thread-safe frame queue with drop-oldest backpressure.

    Spec §8 rules:
    - Exactly one producer, exactly one consumer.
    - Overflow: drop oldest frame, not newest. Log every drop.
    - Shutdown: producer closes first, consumer drains before exit.
    """

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
        self._deque: deque[FramePacket] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self._log = get_logger("ingestion.frame_queue", camera_id)

    def put(self, item: FramePacket) -> None:
        """Add a frame. If full, drop the oldest frame first."""
        with self._lock:
            if self._closed:
                return
            if len(self._deque) >= self._maxsize:
                dropped = self._deque.popleft()
                self._log.warning(
                    "frame_dropped",
                    reason="queue_overflow",
                    dropped_frame_id=dropped.frame_id,
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

    def get(self, timeout: float | None = None) -> FramePacket | None:
        """Get the next frame. Returns None on timeout or if closed and empty."""
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
        """Signal that no more frames will be produced.

        Consumers can still drain remaining frames via get().
        """
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def qsize(self) -> int:
        """Current number of frames in the queue."""
        with self._lock:
            return len(self._deque)

    @property
    def is_closed(self) -> bool:
        """Whether the producer has signaled completion."""
        return self._closed
