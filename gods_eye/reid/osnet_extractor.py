"""OSNet embedding extractor — concrete implementation via torchreid.

Model:     osnet_x1_0 pretrained on ImageNet (ADR-001)
Output:    512-dimensional L2-normalized embedding vectors
Device:    CUDA with AMP (FP16 autocast) if available, CPU FP32 fallback
Batching:  All crops from one frame processed as a single batch

Performance optimizations:
- CUDA AMP autocast for mixed-precision inference (~1.5-2x speedup)
  Note: Task 0 found *full* FP16 casting slower; AMP selectively uses
  FP16 only for safe ops (convs, matmuls) and keeps others in FP32.
- Vectorized batch preprocessing (NumPy SIMD over per-crop Python loop)
- 512-dim output
- Optimal batch size ~10 (1.76ms per image at FP32, faster with AMP)
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
import torchreid

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.schemas.detection import BoundingBox

# OSNet standard input size: height × width (from torchreid training config)
_OSNET_INPUT_H: int = 256
_OSNET_INPUT_W: int = 128
_OSNET_EMBEDDING_DIM: int = 512

# ImageNet normalization constants (used by torchreid during training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Minimum crop dimension in pixels — below this, crop is too small for
# meaningful feature extraction and will produce unreliable embeddings.
_MIN_CROP_PIXELS: int = 10


class OSNetExtractor(EmbeddingExtractor):
    """OSNet-based person Re-ID embedding extractor.

    Loads ``osnet_x1_0`` via torchreid, runs inference on person crops
    extracted from bounding boxes, and returns L2-normalized 512-dim
    embedding vectors.

    Not thread-safe — must be called from a single thread.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = get_logger("reid")

        # Resolve device — AMP autocast on CUDA, FP32 on CPU
        use_cuda = torch.cuda.is_available() and not settings.cpu_only
        self._device_str: str = "cuda" if use_cuda else "cpu"
        self._torch_device = torch.device(self._device_str)
        self._use_amp: bool = use_cuda  # AMP only beneficial on CUDA

        self._log.info(
            "reid_extractor_init",
            model=settings.reid_model,
            device=self._device_str,
            amp_enabled=self._use_amp,
        )

        # Load model
        try:
            pretrained = settings.reid_weights_path is None
            self._model: Any = torchreid.models.build_model(
                name=settings.reid_model,
                num_classes=1,  # Dummy — we extract features, not classify
                pretrained=pretrained,
            )
            if settings.reid_weights_path:
                torchreid.utils.load_pretrained_weights(
                    self._model, settings.reid_weights_path
                )
            self._model.eval()
            self._model.to(self._torch_device)

            param_count = sum(p.numel() for p in self._model.parameters())
            self._log.info(
                "reid_model_loaded",
                model=settings.reid_model,
                device=self._device_str,
                parameters=param_count,
                embedding_dim=_OSNET_EMBEDDING_DIM,
            )
        except Exception as exc:
            self._log.error(
                "reid_model_load_failed",
                model=settings.reid_model,
                error=str(exc),
            )
            raise

    def extract(
        self, frame: np.ndarray, bboxes: list[BoundingBox]
    ) -> list[np.ndarray]:
        """Extract embeddings for person crops from a full frame.

        Args:
            frame: Full BGR image (H×W×C, uint8).
            bboxes: Bounding boxes in absolute pixel coordinates.

        Returns:
            List of L2-normalized 512-dim embedding vectors.
            Zero-vector returned for invalid crops.
        """
        if not bboxes:
            return []

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Crop and track validity (collect raw crops for batch preprocess)
        crops: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, bbox in enumerate(bboxes):
            crop = self._crop_and_validate(frame, bbox, h, w)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(i)

        # Initialize all results as zero vectors
        results: list[np.ndarray] = [
            np.zeros(_OSNET_EMBEDDING_DIM, dtype=np.float32)
            for _ in range(len(bboxes))
        ]

        if not crops:
            self._log.debug(
                "reid_no_valid_crops",
                total_bboxes=len(bboxes),
            )
            return results

        # Vectorized batch preprocessing — avoids per-crop Python loop
        batch = self._preprocess_batch(crops)  # (N, 3, 256, 128)
        tensor = torch.from_numpy(batch).to(self._torch_device)

        # Batch inference with optional AMP autocast (CUDA only)
        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    features = self._model(tensor)  # (N, 512)
            else:
                features = self._model(tensor)  # (N, 512)

        embeddings = features.float().cpu().numpy()  # ensure FP32 output

        # Vectorized L2 normalization — batch NumPy op
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)  # avoid division by zero
        normalized = (embeddings / norms).astype(np.float32)

        for idx, valid_i in enumerate(valid_indices):
            results[valid_i] = normalized[idx]

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log.debug(
            "reid_extract_complete",
            total_crops=len(bboxes),
            valid_crops=len(valid_indices),
            latency_ms=round(latency_ms, 2),
            per_crop_ms=round(latency_ms / len(valid_indices), 2)
            if valid_indices
            else 0.0,
        )

        return results

    def warmup(self) -> None:
        """Run dummy inference to warm up CUDA kernels and JIT.

        Runs with the same AMP context as production inference to ensure
        CUDA kernel cache is populated for the actual precision path.
        """
        self._log.info(
            "reid_warmup_start",
            device=self._device_str,
            amp=self._use_amp,
        )
        dummy = torch.randn(
            1, 3, _OSNET_INPUT_H, _OSNET_INPUT_W,
            device=self._torch_device,
        )
        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    self._model(dummy)
            else:
                self._model(dummy)
        if self._device_str == "cuda":
            torch.cuda.synchronize()
        self._log.info("reid_warmup_complete")

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality: 512 for OSNet."""
        return _OSNET_EMBEDDING_DIM

    @property
    def device(self) -> str:
        """Current execution device."""
        return self._device_str

    @property
    def model_name(self) -> str:
        """Loaded model name."""
        return self._settings.reid_model

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _crop_and_validate(
        frame: np.ndarray,
        bbox: BoundingBox,
        frame_h: int,
        frame_w: int,
    ) -> np.ndarray | None:
        """Crop a person region from the frame with bounds checking.

        Returns None if the crop is invalid (out of bounds, too small).
        """
        # Clamp coordinates to frame boundaries
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(frame_w, int(bbox.x2))
        y2 = min(frame_h, int(bbox.y2))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < _MIN_CROP_PIXELS or crop_h < _MIN_CROP_PIXELS:
            return None

        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Preprocess a single person crop for OSNet inference.

        Kept for backward compatibility and testing. Production path
        uses ``_preprocess_batch`` for vectorized processing.

        1. Resize to 256×128 (H×W)
        2. Convert BGR → RGB
        3. Normalize to [0, 1]
        4. Apply ImageNet normalization
        5. Transpose to CHW format
        """
        resized = cv2.resize(
            crop, (_OSNET_INPUT_W, _OSNET_INPUT_H),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - _MEAN) / _STD
        chw = normalized.transpose(2, 0, 1)  # HWC → CHW
        return chw

    @staticmethod
    def _preprocess_batch(crops: list[np.ndarray]) -> np.ndarray:
        """Vectorized batch preprocessing — all crops in one pass.

        Steps per crop (resize + color convert must remain per-crop due
        to variable input sizes), then batch normalization + transpose:

        1. Resize each crop to 256×128
        2. Convert each BGR → RGB
        3. Stack into (N, H, W, 3) array
        4. Batch normalize: float32 / 255, ImageNet mean/std
        5. Batch transpose: (N, H, W, C) → (N, C, H, W)
        """
        n = len(crops)
        # Pre-allocate output array for resized RGB crops
        batch_hwc = np.empty(
            (n, _OSNET_INPUT_H, _OSNET_INPUT_W, 3), dtype=np.float32,
        )
        for i, crop in enumerate(crops):
            resized = cv2.resize(
                crop, (_OSNET_INPUT_W, _OSNET_INPUT_H),
                interpolation=cv2.INTER_LINEAR,
            )
            batch_hwc[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Vectorized normalization over entire batch (NumPy SIMD)
        batch_hwc /= 255.0
        batch_hwc -= _MEAN  # broadcasts (3,) over (N, H, W, 3)
        batch_hwc /= _STD

        # Batch transpose: (N, H, W, C) → (N, C, H, W)
        batch_chw = batch_hwc.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(batch_chw)
