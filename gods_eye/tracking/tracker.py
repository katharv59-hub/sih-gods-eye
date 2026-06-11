"""Tracker interface — abstract base class for all trackers.

Input: List[Detection] (§4) + frame context
Output: List[Track] (§4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gods_eye.schemas.detection import Detection
from gods_eye.schemas.track import Track


class Tracker(ABC):
    """Abstract tracker interface.

    Every tracker implementation must:
    - Accept a list of Detection objects per frame
    - Maintain state across frames
    - Return a list of Track objects per §4
    - Be single-threaded (not thread-safe)
    """

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        camera_id: str,
    ) -> list[Track]:
        """Process detections for one frame and return updated tracks.

        Args:
            detections: Detections from the current frame.
            frame_id: Monotonic frame counter.
            camera_id: Source camera identifier.

        Returns:
            List of Track objects (ACTIVE + LOST). DEAD tracks are evicted.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all tracker state."""
        ...

    @property
    @abstractmethod
    def active_track_count(self) -> int:
        """Number of currently ACTIVE tracks."""
        ...

    @property
    @abstractmethod
    def total_track_count(self) -> int:
        """Number of ACTIVE + LOST tracks (excludes DEAD/evicted)."""
        ...
