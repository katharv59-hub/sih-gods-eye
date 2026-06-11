"""Detector interface — abstract base class for all detectors.

Input: FramePacket (from ingestion)
Output: List[Detection] (§4 canonical schema)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.schemas.detection import Detection


class Detector(ABC):
    """Abstract detector interface.

    Every detector implementation must:
    - Accept a FramePacket
    - Return a list of Detection objects per §4
    - Be single-threaded (not thread-safe)
    """

    @abstractmethod
    def detect(self, packet: FramePacket) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            packet: Captured frame with metadata from ingestion.

        Returns:
            List of Detection objects conforming to §4 schema.
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy inference to warm up the model (JIT, CUDA kernels)."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Current execution device: 'cuda' or 'cpu'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/path of the loaded model."""
        ...
