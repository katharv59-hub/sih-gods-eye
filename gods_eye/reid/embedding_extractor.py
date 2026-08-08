"""Embedding extractor interface — abstract base class for all Re-ID models.

Input:  Full BGR frame + list of bounding boxes
Output: List of L2-normalized embedding vectors (one per bbox)

This interface is model-agnostic. Any embedding model (OSNet, ResNet,
EfficientNet) can be swapped in behind this ABC without changing downstream
identity matching or gallery code (spec Rule 4 — modularity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from gods_eye.schemas.detection import BoundingBox


class EmbeddingExtractor(ABC):
    """Abstract embedding extraction interface.

    Every extractor implementation must:
    - Accept a full frame and a list of bounding boxes
    - Crop persons from the frame using the bounding boxes
    - Return one L2-normalized embedding vector per crop
    - Return a zero-vector for any crop that fails extraction
    - Be single-threaded (not thread-safe)
    """

    @abstractmethod
    def extract(
        self, frame: np.ndarray, bboxes: list[BoundingBox]
    ) -> list[np.ndarray]:
        """Extract appearance embeddings for person crops.

        Args:
            frame: Full BGR image as numpy array (H×W×C, uint8).
            bboxes: List of bounding boxes to crop from the frame.
                    Coordinates are in absolute pixel space.

        Returns:
            List of L2-normalized embedding vectors, one per bbox.
            Each vector has shape (embedding_dim,).
            Returns a zero-vector for any crop that fails extraction
            (e.g. zero-area bbox, out-of-bounds coordinates).
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Run dummy inference to warm up model (JIT, CUDA kernels)."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Current execution device: 'cuda' or 'cpu'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded embedding model."""
        ...
