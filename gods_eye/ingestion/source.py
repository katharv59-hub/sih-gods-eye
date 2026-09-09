"""Video source abstraction — FrameSource ABC and concrete implementations.

Supported sources (Phase 1):
- Video files (VideoFileSource)
- Webcams (WebcamSource)
- RTSP streams (RTSPSource)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameSource(ABC):
    """Abstract base class for all video sources.

    Each FrameSource wraps a single cv2.VideoCapture and is owned
    by exactly one CameraCaptureThread. No other thread may call
    these methods.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame. Returns (success, bgr_frame_or_none)."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release the underlying VideoCapture."""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the source is currently open and readable."""
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height) of the source frames."""
        ...

    @property
    @abstractmethod
    def fps_nominal(self) -> float:
        """Expected frames per second from this source."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type identifier: 'file', 'webcam', or 'rtsp'."""
        ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for live sources (webcam, RTSP) that support reconnection."""
        ...


# ─── Concrete Implementations ────────────────────────────────────────────────


class VideoFileSource(FrameSource):
    """Reads frames from a video file on disk."""

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._cap: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] = (0, 0)
        self._fps: float = 30.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._file_path)
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (w, h)
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = float(fps) if fps and fps > 0 else 30.0
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        ret_val = self._cap.read()
        ret = bool(ret_val[0])
        if ret:
            return True, ret_val[1]
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps_nominal(self) -> float:
        return self._fps

    @property
    def source_type(self) -> str:
        return "file"

    @property
    def is_live(self) -> bool:
        return False


class WebcamSource(FrameSource):
    """Reads frames from a local webcam device."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] = (0, 0)
        self._fps: float = 30.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._device_index)
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (w, h)
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = float(fps) if fps and fps > 0 else 30.0
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        ret_val = self._cap.read()
        ret = bool(ret_val[0])
        if ret:
            return True, ret_val[1]
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps_nominal(self) -> float:
        return self._fps

    @property
    def device_index(self) -> int:
        """Device capture index."""
        return self._device_index

    @property
    def source_type(self) -> str:
        return "webcam"

    @property
    def is_live(self) -> bool:
        return True


class RTSPSource(FrameSource):
    """Reads frames from an RTSP network stream."""

    def __init__(self, rtsp_url: str) -> None:
        self._rtsp_url = rtsp_url
        self._cap: cv2.VideoCapture | None = None
        self._resolution: tuple[int, int] = (0, 0)
        self._fps: float = 30.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._rtsp_url)
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (w, h)
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = float(fps) if fps and fps > 0 else 30.0
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        ret_val = self._cap.read()
        ret = bool(ret_val[0])
        if ret:
            return True, ret_val[1]
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps_nominal(self) -> float:
        return self._fps

    @property
    def source_type(self) -> str:
        return "rtsp"

    @property
    def is_live(self) -> bool:
        return True
