"""Ingestion layer — Camera threads, frame queues.

Phase 1 deliverable: webcam, RTSP, and file video ingestion
with threaded capture and queue backpressure.
"""

from gods_eye.ingestion.capture_thread import CameraCaptureThread
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.frame_queue import FrameQueue
from gods_eye.ingestion.smoke_harness import (
    SingleCameraSmokeHarness,
    SmokeHarnessMetrics,
)
from gods_eye.ingestion.source import (
    FrameSource,
    RTSPSource,
    VideoFileSource,
    WebcamSource,
)

__all__ = [
    "CameraCaptureThread",
    "FramePacket",
    "FrameQueue",
    "FrameSource",
    "RTSPSource",
    "SingleCameraSmokeHarness",
    "SmokeHarnessMetrics",
    "VideoFileSource",
    "WebcamSource",
]
