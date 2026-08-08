# Task 1 — Embedding Extraction Layer Validation Report

**Status:** PASS
**Date:** 2026-06-23
**Phase:** 2 — Identity Persistence Layer
**Task:** 1 — EmbeddingExtractor ABC + OSNetExtractor

---

## 1. Architecture Summary

Task 1 implements the embedding extraction layer — the component responsible for converting a person crop (bounding box region from a video frame) into a stable 512-dimensional appearance vector using OSNet.

### Data Flow

```
Full Frame (np.ndarray, BGR) + list[BoundingBox]
    │
    ▼
EmbeddingExtractor.extract()
    │
    ├── Crop each bbox from frame (with bounds clamping)
    ├── Validate crop size (≥ 10×10 pixels)
    ├── Preprocess: resize 256×128, BGR→RGB, /255, ImageNet normalize, HWC→CHW
    ├── Batch all valid crops into single tensor
    ├── OSNet forward pass (CUDA FP32)
    ├── L2-normalize each output vector
    │
    ▼
list[np.ndarray]  (one 512-dim vector per bbox; zero-vector for invalid crops)
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ABC pattern | `EmbeddingExtractor` ABC | Matches `Detector` and `Tracker` ABCs (spec Rule 4 — modularity) |
| Precision | FP32 | Task 0 finding: FP16 is slower on OSNet architecture |
| Normalization | L2-normalize all outputs | Required for cosine similarity in Task 2 |
| Invalid crop handling | Zero-vector fallback | Downstream gallery search will never match a zero-vector |
| Minimum crop size | 10×10 pixels | Below this, ResNet features are meaningless noise |
| Preprocessing | ImageNet normalization | Matches OSNet training protocol in torchreid |

---

## 2. Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `gods_eye/reid/__init__.py` | 14 | Package init with public exports |
| `gods_eye/reid/embedding_extractor.py` | 71 | ABC defining model-agnostic extraction interface |
| `gods_eye/reid/osnet_extractor.py` | 222 | Concrete OSNet implementation via torchreid |
| `tests/test_reid.py` | 178 | 15 unit tests covering all requirements |

### Files Modified

| File | Change |
|------|--------|
| `gods_eye/config/settings.py` | Added `reid_model: str = "osnet_x1_0"` config key + env var resolution |
| `pyproject.toml` | Added `torchreid` to mypy overrides for missing import stubs |

---

## 3. Test Results

### Re-ID Tests (15 tests)

```
tests/test_reid.py::test_abc_cannot_instantiate PASSED
tests/test_reid.py::test_osnet_is_embedding_extractor PASSED
tests/test_reid.py::test_model_loads_successfully PASSED
tests/test_reid.py::test_embedding_dim_is_512 PASSED
tests/test_reid.py::test_model_name PASSED
tests/test_reid.py::test_device_is_string PASSED
tests/test_reid.py::test_single_crop_produces_correct_shape PASSED
tests/test_reid.py::test_embedding_is_l2_normalized PASSED
tests/test_reid.py::test_batch_produces_correct_count PASSED
tests/test_reid.py::test_empty_bbox_list_returns_empty PASSED
tests/test_reid.py::test_different_crops_produce_different_embeddings PASSED
tests/test_reid.py::test_zero_area_bbox_returns_zero_vector PASSED
tests/test_reid.py::test_tiny_bbox_returns_zero_vector PASSED
tests/test_reid.py::test_out_of_bounds_bbox_is_clamped PASSED
tests/test_reid.py::test_mixed_valid_and_invalid_bboxes PASSED
```

### Full Suite (58 tests — no regressions)

```
58 passed, 3 warnings in 12.23s
```

### Type Checking

```
mypy --strict gods_eye/ → Success: no issues found in 33 source files
```

(Up from 31 files in Phase 1 — 2 new files: `embedding_extractor.py`, `osnet_extractor.py`)

---

## 4. Inference Verification

### Embedding Properties Verified

| Property | Expected | Actual | Status |
|----------|----------|--------|--------|
| Embedding dimension | 512 | 512 | ✅ |
| L2 norm of valid embedding | 1.0 | 1.0 ± 1e-4 | ✅ |
| Zero-vector for invalid crop | all zeros | all zeros | ✅ |
| Different crops → different embeddings | similarity < 0.99 | Verified | ✅ |
| Batch of N bboxes → N embeddings | 1:1 correspondence | Verified | ✅ |

### Timing (from Task 0 benchmark)

| Batch Size | Per-Image Latency | Total Latency |
|------------|------------------|---------------|
| 1 | 22.72 ms | 22.72 ms |
| 5 | 3.62 ms | 18.11 ms |
| 10 | 1.76 ms | 17.62 ms |
| 20 | 1.46 ms | 29.26 ms |

---

## 5. Known Limitations

1. **torchreid `torch.load` FutureWarning:** torchreid uses `weights_only=False` internally. This will require a torchreid update when PyTorch changes the default to `weights_only=True`. Not blocking.

2. **Cython evaluation unavailable:** torchreid's rank metric computation falls back to pure Python. Not relevant for inference — only affects benchmark evaluation speed.

3. **No crop caching:** Each call to `extract()` recrops from the full frame. If the same frame is processed multiple times (unlikely in the pipeline), crops are recomputed. Acceptable for Phase 2.

4. **Single-threaded only:** `OSNetExtractor` is not thread-safe. Must be called from a single thread (the future IdentityWorker). This matches the `Detector` and `Tracker` contract.

5. **No model variant benchmarking yet:** Only `osnet_x1_0` has been validated. Task 3 (Market-1501 benchmark) will determine if this variant meets the Rank-1 ≥ 0.88 gate.

---

## 6. Verdict

| Requirement | Status |
|-------------|--------|
| Abstract extraction interface created | ✅ |
| OSNet model loads and runs on CUDA | ✅ |
| Embedding dimension = 512 | ✅ |
| Outputs are L2-normalized | ✅ |
| Invalid crops return zero-vectors | ✅ |
| Configuration via settings (no hardcoded paths) | ✅ |
| Structured logging (no print statements) | ✅ |
| 15 unit tests passing | ✅ |
| mypy --strict: 0 errors (33 files) | ✅ |
| Full test suite: 58 passed, 0 regressions | ✅ |

**TASK 1: PASS**

*Awaiting approval to proceed to Task 2.*
