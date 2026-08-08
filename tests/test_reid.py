"""Tests for the Re-ID embedding extraction layer (Phase 2, Task 1).

Tests validate:
- EmbeddingExtractor ABC interface contract
- OSNetExtractor initialization and model loading
- Embedding dimension (512)
- L2 normalization of output vectors
- Invalid/edge-case crop handling (zero-area, out-of-bounds, empty list)
- Batch processing correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.osnet_extractor import OSNetExtractor
from gods_eye.schemas.detection import BoundingBox


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def settings() -> Settings:
    """Default settings for tests."""
    return Settings()


@pytest.fixture(scope="module")
def extractor(settings: Settings) -> OSNetExtractor:
    """Shared OSNetExtractor instance (expensive to load — reuse across tests)."""
    ext = OSNetExtractor(settings)
    ext.warmup()
    return ext


@pytest.fixture()
def dummy_frame() -> np.ndarray:
    """A synthetic BGR image (480×640)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ── ABC Interface Tests ──────────────────────────────────────────────────────

def test_abc_cannot_instantiate() -> None:
    """EmbeddingExtractor ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EmbeddingExtractor()  # type: ignore[abstract]


def test_osnet_is_embedding_extractor(extractor: OSNetExtractor) -> None:
    """OSNetExtractor is a valid EmbeddingExtractor subclass."""
    assert isinstance(extractor, EmbeddingExtractor)


# ── Initialization Tests ────────────────────────────────────────────────────

def test_model_loads_successfully(extractor: OSNetExtractor) -> None:
    """OSNet model loads without error."""
    assert extractor is not None


def test_embedding_dim_is_512(extractor: OSNetExtractor) -> None:
    """Embedding dimension must be 512 (OSNet architecture output)."""
    assert extractor.embedding_dim == 512


def test_model_name(extractor: OSNetExtractor) -> None:
    """Model name must match configured setting."""
    assert extractor.model_name == "osnet_x1_0"


def test_device_is_string(extractor: OSNetExtractor) -> None:
    """Device must be 'cuda' or 'cpu'."""
    assert extractor.device in ("cuda", "cpu")


# ── Embedding Extraction Tests ───────────────────────────────────────────────

def test_single_crop_produces_correct_shape(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """Single bounding box produces one 512-dim embedding."""
    bboxes = [BoundingBox(x1=100, y1=50, x2=200, y2=300)]
    results = extractor.extract(dummy_frame, bboxes)
    assert len(results) == 1
    assert results[0].shape == (512,)


def test_embedding_is_l2_normalized(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """Output embedding must be L2-normalized (unit length)."""
    bboxes = [BoundingBox(x1=100, y1=50, x2=200, y2=300)]
    results = extractor.extract(dummy_frame, bboxes)
    norm = float(np.linalg.norm(results[0]))
    assert abs(norm - 1.0) < 1e-4, f"L2 norm is {norm}, expected ~1.0"


def test_batch_produces_correct_count(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """Multiple bounding boxes produce one embedding per box."""
    bboxes = [
        BoundingBox(x1=10, y1=10, x2=80, y2=200),
        BoundingBox(x1=100, y1=50, x2=200, y2=300),
        BoundingBox(x1=300, y1=100, x2=400, y2=400),
    ]
    results = extractor.extract(dummy_frame, bboxes)
    assert len(results) == 3
    for emb in results:
        assert emb.shape == (512,)


def test_empty_bbox_list_returns_empty(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """No bounding boxes produces an empty result list."""
    results = extractor.extract(dummy_frame, [])
    assert results == []


def test_different_crops_produce_different_embeddings(
    extractor: OSNetExtractor,
) -> None:
    """Two visually distinct crops should produce different embeddings."""
    # Create two distinct synthetic images
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:300, 100:200] = [255, 0, 0]   # Blue region
    frame[50:300, 300:400] = [0, 255, 0]    # Green region

    bboxes = [
        BoundingBox(x1=100, y1=50, x2=200, y2=300),
        BoundingBox(x1=300, y1=50, x2=400, y2=300),
    ]
    results = extractor.extract(frame, bboxes)

    similarity = float(np.dot(results[0], results[1]))
    # Different crops should have similarity significantly less than 1.0
    assert similarity < 0.99, f"Similarity {similarity} is too high for distinct crops"


# ── Invalid Crop Handling ────────────────────────────────────────────────────

def test_zero_area_bbox_returns_zero_vector(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """A zero-area bounding box produces a zero embedding vector."""
    bboxes = [BoundingBox(x1=100, y1=100, x2=100, y2=100)]
    results = extractor.extract(dummy_frame, bboxes)
    assert len(results) == 1
    assert np.allclose(results[0], 0.0), "Zero-area bbox should produce zero vector"


def test_tiny_bbox_returns_zero_vector(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """A bounding box smaller than MIN_CROP_PIXELS produces a zero vector."""
    bboxes = [BoundingBox(x1=100, y1=100, x2=105, y2=105)]  # 5×5 < 10×10 min
    results = extractor.extract(dummy_frame, bboxes)
    assert len(results) == 1
    assert np.allclose(results[0], 0.0), "Tiny bbox should produce zero vector"


def test_out_of_bounds_bbox_is_clamped(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """Bounding box extending beyond frame is clamped, not crashed."""
    h, w = dummy_frame.shape[:2]
    bboxes = [BoundingBox(x1=-50, y1=-50, x2=w + 50, y2=h + 50)]
    results = extractor.extract(dummy_frame, bboxes)
    assert len(results) == 1
    assert results[0].shape == (512,)
    norm = float(np.linalg.norm(results[0]))
    assert norm > 0.5, "Clamped bbox should still produce a valid embedding"


def test_mixed_valid_and_invalid_bboxes(
    extractor: OSNetExtractor, dummy_frame: np.ndarray
) -> None:
    """Mix of valid and invalid bboxes: valid get embeddings, invalid get zeros."""
    bboxes = [
        BoundingBox(x1=100, y1=50, x2=200, y2=300),   # Valid
        BoundingBox(x1=50, y1=50, x2=55, y2=55),      # Too small
        BoundingBox(x1=200, y1=100, x2=350, y2=400),  # Valid
    ]
    results = extractor.extract(dummy_frame, bboxes)
    assert len(results) == 3

    # First and third should be valid (non-zero, normalized)
    assert float(np.linalg.norm(results[0])) > 0.5
    assert float(np.linalg.norm(results[2])) > 0.5

    # Second should be zero vector
    assert np.allclose(results[1], 0.0)
