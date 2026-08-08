# Phase 2 — Identity Persistence Layer: Implementation Plan

**Status:** PLANNING
**Document Version:** 1.0
**Source of Truth:** GODS_EYE_MASTER_SPEC.md v3.0
**Prerequisite:** Phase 1 GATE CLOSED (commit `180d6c7a`, tag `phase1-final`)
**Author:** Technical Architect / AI Assistant
**Date:** 2026-06-22

---
### Task 0: TorchReID Feasibility Validation

Purpose:
Verify that the selected Re-ID stack can execute on the target hardware before architecture implementation begins.

Validation Steps:
- Install torchreid and dependencies
- Verify compatibility with current PyTorch version
- Load OSNet pretrained weights
- Execute inference on a sample pedestrian crop
- Verify CUDA execution on RTX 4050
- Measure single-image inference latency

Success Criteria:
- Model loads successfully
- CUDA execution functions correctly
- No dependency conflicts
- Inference completes without errors

Failure Criteria:
- torchreid incompatible with environment
- OSNet model fails to load
- CUDA execution unavailable

Deliverable:
docs/phases/phase_2/TASK_0_FEASIBILITY_REPORT.md

Exit Criteria:
Proceed to Task 1 only after feasibility validation passes.

### Phase 2 Go / No-Go Review

After Task 0 and Task 3:

Review:
- TorchReID compatibility
- OSNet loading
- CUDA execution
- Market1501 Rank-1 accuracy

If any Phase 2 gate appears unattainable,
implementation pauses and architecture is reassessed.

Deliverable:
GO_NO_GO_REVIEW.md

## 1. PHASE OBJECTIVE

### 1.1 Problem Statement

Phase 1 tracks are ephemeral: ByteTrack assigns integer IDs that reset every session, have no appearance model, and cannot survive occlusion longer than `track_dead_timeout` (120 frames / 4 seconds). When a person walks behind a pillar and reappears, they receive a new track ID. Phase 1 logged 102 ID switches on MOT17-04 — all caused by spatial-only association failing during crossings and occlusions.

Phase 2 solves this by adding a persistent identity layer that maps short-lived tracker IDs to stable global UUIDs using appearance-based Re-Identification (Re-ID).

### 1.2 Success Definition

Phase 2 is complete when all 5 gate criteria from §14 of the master spec are satisfied with measured values:

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | Re-ID Rank-1 on Market-1501 | ≥ 0.88 |
| 2 | Identity continuity through 2s occlusion | ≥ 85% re-link rate |
| 3 | Gallery operation latency | ≤ 5ms p99 |
| 4 | False merge rate | < 1% |
| 5 | FPS regression vs Phase 1 | < 10% |
### Additional Success Metric

Phase 1 baseline:

- ID Switches: 102

Phase 2 objective:

- Reduce ID switches by at least 30%

Target:

- ≤ 70 ID switches on MOT17-04

Rationale:

This provides a direct measurable improvement over the Phase 1 baseline and validates the effectiveness of appearance-based identity persistence.

### 1.3 Architecture Summary (from spec §14)

```
ByteTrack ID → IML → OSNet embedding → cosine similarity → gallery → EMA update → Global Identity UUID
```

**IML** = Identity Mapping Layer (ADR-004). This is the core abstraction that separates tracker-local ephemeral IDs from persistent global identities.

---
### 1.4 Scope Boundary

Phase 2 is strictly limited to appearance-based identity persistence.

Included:
- Re-ID embeddings
- Identity Mapping Layer (IML)
- Identity gallery
- Identity lifecycle management
- Identity persistence through occlusions

Explicitly Excluded:
- Multi-camera identity matching
- Cross-camera handoff
- Spatial graph reasoning
- Temporal memory systems
- Behavioral analysis
- Anomaly detection
- Natural language querying

These capabilities belong to later phases and must not be implemented during Phase 2.

## 2. DESIGN DECISIONS (PRE-LOCKED BY SPEC)

The master spec pre-locks four critical decisions via ADRs:

| ADR | Decision | Choice | Rejected Alternatives |
|-----|----------|--------|----------------------|
| ADR-001 | Embedding model | OSNet via torchreid | MobileNetV3 (below Rank-1 gate), EfficientNet Re-ID (3x cost) |
| ADR-002 | Similarity metric | Cosine similarity | Euclidean (magnitude-sensitive), Siamese (requires training) |
| ADR-003 | Embedding update | EMA with α=0.9 | Static template (brittle), simple mean (drowns recent signal) |
| ADR-004 | Identity-track separation | IML (permanent) | Direct track ID reuse (breaks on tracker reset) |

### 2.1 Remaining Decisions To Make During Implementation

| Decision | Options | Evaluation Method |
|----------|---------|-------------------|
| OSNet variant (osnet_x1_0 vs osnet_ain_x1_0 vs osnet_x0_75) | Rank-1 accuracy vs inference latency benchmark | Run `benchmarks/benchmark_reid_model.py` on Market-1501 |
| Person crop size (128×256 vs 256×128 vs 128×128) | OSNet default training resolution | Use torchreid default (128×256 height×width) unless benchmark shows regression |
| Gallery search structure (brute-force vs approximate NN) | Latency at gallery sizes 100, 1000, 10000 | Benchmark p99 latency. Brute-force first; switch to FAISS only if p99 > 5ms |
| Re-ID integration point (synchronous in pipeline vs async worker) | FPS regression measurement | Start synchronous (simpler); add async worker only if FPS regression > 10% |
| Identity event emission (inline vs batched) | Queue depth impact | Inline first; batch if EventQueue depth exceeds alert threshold |

---

## 3. COMPONENTS TO BUILD

### 3.1 Module Map

```
gods_eye/
├── reid/                          # NEW — Phase 2
│   ├── __init__.py                # (exists, empty stub)
│   ├── extractor.py               # EmbeddingExtractor ABC
│   ├── osnet_extractor.py         # OSNet concrete implementation
│   └── similarity.py              # Cosine similarity + threshold logic
├── identity/                      # NEW — Phase 2
│   ├── __init__.py                # (exists, empty stub)
│   ├── gallery.py                 # Identity gallery (store + search)
│   ├── identity_mapper.py         # IML: Track → Identity mapping
│   └── state_machine.py           # Identity lifecycle (ACTIVE/LOST/PURGED)
├── schemas/
│   └── identity.py                # (exists, complete — no changes needed)
├── pipeline.py                    # MODIFY — add Re-ID stage after tracking
├── observability/
│   └── metrics.py                 # (exists — identity metrics already registered)
└── config/
    └── settings.py                # (exists — Re-ID config keys already present)
```

### 3.2 Component Specifications

#### 3.2.1 `reid/extractor.py` — EmbeddingExtractor ABC

```python
class EmbeddingExtractor(ABC):
    @abstractmethod
    def extract(self, frame: np.ndarray, bboxes: list[BoundingBox]) -> list[np.ndarray]:
        """Extract embeddings for person crops from a frame.
        
        Args:
            frame: Full BGR frame (HxWxC).
            bboxes: List of bounding boxes to crop.
        
        Returns:
            List of L2-normalized embedding vectors (one per bbox).
            Returns zero-vector for any crop that fails extraction.
        """
        ...

    @abstractmethod
    def warmup(self) -> None: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> str: ...
```

**Design rationale:** Follows the same ABC pattern as `Detector` and `Tracker` (spec Rule 4 — modularity). Allows future swap to a different embedding model without changing downstream code.

#### 3.2.2 `reid/osnet_extractor.py` — OSNet Implementation

- **Model:** `osnet_x1_0` pretrained on Market-1501+MSMT17 via `torchreid` (ADR-001)
- **Input:** Person crop resized to 256×128 (H×W), normalized
- **Output:** 512-dim L2-normalized embedding vector
- **Device:** CUDA FP16 if available, CPU FP32 fallback
- **Batching:** Process all crops from one frame as a single batch (typically 5–30 crops)
- **Error handling:** Invalid/empty crops → zero vector + warning log

**Key method signatures:**
```python
class OSNetExtractor(EmbeddingExtractor):
    def __init__(self, settings: Settings) -> None: ...
    def extract(self, frame: np.ndarray, bboxes: list[BoundingBox]) -> list[np.ndarray]: ...
    def warmup(self) -> None: ...
    @property
    def embedding_dim(self) -> int: ...  # Returns 512
    @property
    def device(self) -> str: ...
```

#### 3.2.3 `reid/similarity.py` — Similarity Computation

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""

def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Compute similarity between one query and N gallery vectors.
    Returns: 1D array of shape (N,) with similarity scores in [-1, 1].
    """
```

**Design rationale:** Keeping similarity computation as pure functions (not a class) because ADR-002 locks cosine similarity. No polymorphism needed here.

#### 3.2.4 `identity/gallery.py` — Identity Gallery

The gallery is the core data structure: a searchable store of `Identity` objects indexed by their current EMA embedding.

```python
class IdentityGallery:
    def __init__(self, settings: Settings, metrics: MetricsRegistry | None = None) -> None: ...

    def search(self, embedding: np.ndarray) -> tuple[Identity | None, float]:
        """Find the best matching identity above REID_MATCH_THRESHOLD.
        Returns: (matching_identity, similarity_score) or (None, 0.0).
        """

    def register(self, embedding: np.ndarray, camera_id: str, track_id: str,
                 timestamp_ns: int) -> Identity:
        """Create a new Identity with a fresh global_id UUID."""

    def update(self, identity: Identity, new_embedding: np.ndarray,
               camera_id: str, track_id: str, timestamp_ns: int) -> None:
        """EMA-update the identity's embedding and refresh metadata."""

    def tick(self, current_ns: int) -> list[Identity]:
        """Advance lifecycle: ACTIVE→LOST, LOST→PURGED based on timeouts.
        Returns list of newly purged identities.
        """

    def get(self, global_id: str) -> Identity | None: ...
    
    @property
    def size(self) -> int: ...
    
    @property
    def active_count(self) -> int: ...
```

**LRU eviction:** When `size >= GALLERY_MAX_SIZE` (default 10,000), evict the identity with the oldest `last_seen_ns` that is in LOST status. If no LOST identities exist, log CRITICAL and reject registration.

**Thread safety:** Gallery is single-writer (Identity Mapping Thread). No locking needed if accessed only from that thread. If Phase 3 requires cross-thread reads, add read-write lock (defer to Phase 3).

#### 3.2.5 `identity/identity_mapper.py` — Identity Mapping Layer (IML)

This is the core Phase 2 orchestrator. It receives `TrackingResult` objects and produces identity-enriched results.

```python
@dataclass
class IdentityResult:
    """Phase 2 pipeline output: tracking result + identity mappings."""
    packet: FramePacket
    detections: list[Detection]
    tracks: list[Track]
    identities: dict[str, Identity]     # track_id → Identity mapping
    events: list[Event]                 # Identity events generated this frame

class IdentityMapper:
    def __init__(self, extractor: EmbeddingExtractor, gallery: IdentityGallery,
                 settings: Settings, metrics: MetricsRegistry | None = None) -> None: ...

    def process(self, result: TrackingResult) -> IdentityResult:
        """Core IML logic for one frame:
        
        1. Filter ACTIVE tracks only
        2. Crop person regions from frame using Track.bbox
        3. Batch-extract OSNet embeddings
        4. For each track:
           a. If track_id is in active mapping → update existing Identity (EMA)
           b. Else → search gallery for match
              - Match found (similarity ≥ threshold) → re-link Identity
              - No match → register new Identity
        5. Tick gallery lifecycle (ACTIVE→LOST→PURGED)
        6. Generate identity events (CONFIRMED, LOST, PURGED)
        7. Update metrics
        """
```

**Track-to-identity mapping cache:** Maintain a `dict[str, str]` mapping `(camera_id, track_id) → global_id` for ACTIVE tracks. This avoids redundant gallery searches for tracks already associated. When ByteTrack emits a track with a known `(camera_id, track_id)`, skip the gallery search and directly update the linked identity via EMA.

**Event generation rules:**
- `IDENTITY_CONFIRMED`: New identity registered OR re-linked after LOST
- `IDENTITY_LOST`: Identity transitions ACTIVE → LOST (timeout)
- `IDENTITY_PURGED`: Identity transitions LOST → PURGED (TTL expired)

#### 3.2.6 `identity/state_machine.py` — Identity Lifecycle

```python
class IdentityStateMachine:
    """Manages Identity state transitions per §4 spec:
    
    ACTIVE → LOST:   last_seen_ns older than IDENTITY_LOST_TIMEOUT_S
    LOST   → ACTIVE: Re-ID match >= REID_MATCH_THRESHOLD
    LOST   → PURGED: current_time > purge_at_ns
    PURGED → (none):  Terminal. Embeddings zeroed.
    """

    @staticmethod
    def should_transition_to_lost(identity: Identity, current_ns: int,
                                   timeout_s: int) -> bool: ...

    @staticmethod
    def should_purge(identity: Identity, current_ns: int) -> bool: ...

    @staticmethod
    def transition_to_lost(identity: Identity, current_ns: int,
                            ttl_s: int) -> Identity: ...

    @staticmethod
    def transition_to_active(identity: Identity, current_ns: int) -> Identity: ...

    @staticmethod
    def purge(identity: Identity) -> Identity: ...
```

---

## 4. PIPELINE INTEGRATION

### 4.1 Threading Model Change

**Phase 1 (current):**
```
CaptureThread →[FrameQueue]→ DetectionWorker →[DetQ]→ TrackingWorker →[TrackQ]→ OutputWorker
```

**Phase 2 (proposed — Option A: Synchronous):**
```
CaptureThread →[FrameQueue]→ DetectionWorker →[DetQ]→ TrackingWorker →[TrackQ]→ IdentityWorker →[IdentQ]→ OutputWorker
```

A new `IdentityWorker` thread is inserted between `TrackingWorker` and `OutputWorker`. It consumes `TrackingResult` objects and produces `IdentityResult` objects.

**Why synchronous first:** The spec Rule 2 (Simplicity First) mandates the simplest working solution. Adding a thread is simpler than async I/O for a CPU/GPU-bound task. If FPS regression exceeds 10%, we can optimize by batching crops across frames or moving Re-ID to a separate GPU stream.

### 4.2 New Pipeline Queue

```python
self._identity_q: PipelineQueue[IdentityResult] = PipelineQueue(
    maxsize=settings.track_queue_size,
    queue_name=f"{camera_id}/identities",
    camera_id=camera_id,
    metrics=metrics,
)
```

### 4.3 OutputWorker Change

`OutputWorker` callback signature changes from `Callable[[TrackingResult], None]` to `Callable[[IdentityResult], None]`. This is a **breaking change** to the pipeline API. `run_demo.py` and `scripts/run_demo_suite.py` must be updated.

### 4.4 IdentityWorker Specification

```python
class IdentityWorker(threading.Thread):
    """Consumes TrackingResults, runs Re-ID + IML, produces IdentityResults."""

    def __init__(self, camera_id: str, mapper: IdentityMapper,
                 input_queue: PipelineQueue[TrackingResult],
                 output_queue: PipelineQueue[IdentityResult],
                 *, metrics: MetricsRegistry | None = None) -> None: ...

    def run(self) -> None:
        # Same pattern as DetectionWorker and TrackingWorker:
        # get from input, process, put to output, measure latency
```

---

## 5. BENCHMARK PLAN

### 5.1 Re-ID Model Accuracy (`benchmarks/benchmark_reid_market1501.py`)

- **Dataset:** Market-1501 (standard Re-ID benchmark)
- **Metric:** Rank-1 accuracy, mAP
- **Gate:** Rank-1 ≥ 0.88
- **Procedure:** Load pretrained OSNet, extract gallery+query embeddings, compute CMC curve
- **Output:** `benchmarks/results/reid_market1501.json`

### 5.2 Gallery Operation Latency (`benchmarks/benchmark_gallery_latency.py`)

- **Procedure:** Populate gallery to 100, 1000, 5000, 10000 identities. Measure search, register, update latency.
- **Gate:** search p99 ≤ 5ms at gallery size 10,000
- **Output:** `benchmarks/results/gallery_latency.json`

### 5.3 Occlusion Re-link Rate (`benchmarks/benchmark_occlusion_relink.py`)

- **Dataset:** MOT17-09 and MOT17-04 ground truth (tracks with known occlusion gaps)
- **Procedure:** Simulate track loss for 2 seconds (60 frames at 30fps). Measure whether IML re-links the same global_id when the person reappears.
- **Gate:** ≥ 85% re-link rate
- **Output:** `benchmarks/results/occlusion_relink.json`

### 5.4 False Merge Rate (`benchmarks/benchmark_false_merge.py`)

- **Dataset:** MOT17-04 (83 unique pedestrians)
- **Procedure:** Run full pipeline. Count cases where two distinct ground-truth identities are assigned the same global_id.
- **Gate:** < 1%
- **Output:** `benchmarks/results/false_merge.json`

### 5.5 FPS Regression (`benchmarks/benchmark_fps_regression.py`)

- **Procedure:** Run Phase 2 pipeline on MOT17-04 video. Compare sustained FPS against Phase 1 baseline (12.6 FPS from `demo_suite_results.json`).
- **Gate:** < 10% regression (i.e., ≥ 11.3 FPS)
- **Output:** `benchmarks/results/fps_regression_phase2.json`

---

## 6. TEST PLAN

### 6.1 Unit Tests (new)

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `tests/test_reid.py` | ~8 tests | Embedding extraction: shape, normalization, batch, empty input, GPU/CPU device |
| `tests/test_similarity.py` | ~5 tests | Cosine similarity: identical vectors=1.0, orthogonal=0.0, matrix shape |
| `tests/test_gallery.py` | ~10 tests | Gallery: register, search (match/no-match), update (EMA), tick (LOST/PURGE), LRU eviction, size limits |
| `tests/test_identity_mapper.py` | ~8 tests | IML: new track creates identity, known track updates identity, re-link after occlusion, event generation |
| `tests/test_state_machine.py` | ~6 tests | All state transitions: ACTIVE→LOST, LOST→ACTIVE, LOST→PURGED, PURGED terminal, timing |

**Estimated new test count:** ~37 tests
**Expected total after Phase 2:** 43 (Phase 1) + ~37 = ~80 tests

### 6.2 Integration Tests

| Test | Purpose |
|------|---------|
| `tests/test_pipeline.py::test_identity_worker_e2e` | Full pipeline with mock extractor: frame → detection → track → identity |
| `tests/test_pipeline.py::test_identity_worker_relink` | Track disappears and reappears → same global_id |

### 6.3 Type Checking

- `mypy --strict gods_eye/` must pass with 0 errors after all Phase 2 modules added
- New `[[tool.mypy.overrides]]` entry for `torchreid` (missing stubs)

---

## 7. IMPLEMENTATION ORDER

Strict dependency-ordered sequence. Each task produces a testable, commitable unit.

### Task 1: EmbeddingExtractor ABC + OSNetExtractor
- **Files:** `reid/extractor.py`, `reid/osnet_extractor.py`
- **Tests:** `tests/test_reid.py`
- **Depends on:** Nothing (standalone module)
- **Validation:** `pytest tests/test_reid.py` + manual crop extraction check

### Task 2: Cosine Similarity Functions
- **Files:** `reid/similarity.py`
- **Tests:** `tests/test_similarity.py`
- **Depends on:** Nothing
- **Validation:** `pytest tests/test_similarity.py`

### Task 3: Re-ID Model Benchmark (Market-1501)
- **Files:** `benchmarks/benchmark_reid_market1501.py`
- **Depends on:** Task 1
- **Gate check:** Rank-1 ≥ 0.88
- **Decision point:** If Rank-1 < 0.88, evaluate `osnet_ain_x1_0` variant before proceeding

### Task 4: Identity State Machine
- **Files:** `identity/state_machine.py`
- **Tests:** `tests/test_state_machine.py`
- **Depends on:** Existing `schemas/identity.py`
- **Validation:** Pure logic — no GPU, no external deps

### Task 5: Identity Gallery
- **Files:** `identity/gallery.py`
- **Tests:** `tests/test_gallery.py`
- **Depends on:** Task 2, Task 4
- **Validation:** Gallery latency benchmark at 10,000 entries

### Task 6: Gallery Latency Benchmark
- **Files:** `benchmarks/benchmark_gallery_latency.py`
- **Depends on:** Task 5
- **Gate check:** p99 ≤ 5ms at 10,000 entries

### Task 7: Identity Mapper (IML)
- **Files:** `identity/identity_mapper.py`, `IdentityResult` dataclass
- **Tests:** `tests/test_identity_mapper.py`
- **Depends on:** Task 1, Task 5
- **Validation:** Unit tests with mock extractor

### Task 8: Pipeline Integration (IdentityWorker)
- **Files:** Modify `pipeline.py`, update `run_demo.py`, update `scripts/run_demo_suite.py`
- **Tests:** Update `tests/test_pipeline.py`
- **Depends on:** Task 7
- **Validation:** Full E2E with demo video

### Task 9: Occlusion Re-link Benchmark
- **Files:** `benchmarks/benchmark_occlusion_relink.py`
- **Depends on:** Task 8
- **Gate check:** ≥ 85% re-link rate through 2s occlusion

### Task 10: False Merge + FPS Regression Benchmarks
- **Files:** `benchmarks/benchmark_false_merge.py`, `benchmarks/benchmark_fps_regression.py`
- **Depends on:** Task 8
- **Gate checks:** False merge < 1%, FPS regression < 10%

### Task 11: Final Validation + Documentation
- **mypy --strict:** 0 errors
- **pytest:** All tests pass
- **Documentation:** Phase 2 reports (IMPLEMENTATION, BENCHMARKS, AUDIT, LESSONS_LEARNED)
- **Git tag:** `phase2-complete`

---

## 8. RISK ANALYSIS

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OSNet Rank-1 < 0.88 on Market-1501 | Low (published: 0.946) | Blocks gate | Try `osnet_ain_x1_0` variant; if still failing, revisit ADR-001 |
| FPS regression > 10% from Re-ID overhead | Medium | Blocks gate | Batch crops, async GPU stream, skip Re-ID on high-confidence tracked IDs |
| Gallery search > 5ms p99 at 10K entries | Low | Blocks gate | Switch from brute-force to FAISS approximate NN |
| False merges from similar-looking pedestrians | Medium | Blocks gate | Lower `REID_MATCH_THRESHOLD` from 0.75 to 0.80; add top-K verification |
| torchreid compatibility with current torch version | Medium | Delays start | Test import + model load before starting Task 1 |
| VRAM exhaustion with YOLO + OSNet simultaneously | Low-Medium | Crashes | OSNet is lightweight (~6MB). Total VRAM: YOLO(38MB) + OSNet(~24MB) = ~62MB — well within 6GB budget |

---

## 9. CONFIGURATION (ALREADY IN settings.py)

All Phase 2 configuration keys already exist in `gods_eye/config/settings.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `reid_match_threshold` | 0.75 | Minimum cosine similarity for identity match |
| `reid_ema_alpha` | 0.9 | EMA decay for embedding updates |
| `identity_lost_timeout_s` | 300 | Seconds before ACTIVE → LOST |
| `identity_ttl_s` | 86400 | Seconds before LOST → PURGED (24h) |
| `embedding_history_len` | 50 | Max embeddings retained per identity |
| `gallery_max_size` | 10000 | Maximum gallery capacity before LRU eviction |

No new settings need to be added for Phase 2.

---

## 10. INTEGRATION CONTRACTS

### 10.1 What Phase 2 Consumes from Phase 1

| Contract | Type | Source |
|----------|------|--------|
| `TrackingResult` | dataclass | `pipeline.py` line 48 |
| `TrackingResult.packet.frame` | `np.ndarray` (BGR HxWxC) | Used for person crop extraction |
| `TrackingResult.tracks` | `list[Track]` | ACTIVE tracks provide bboxes for cropping |
| `Track.bbox` | `BoundingBox` (absolute pixels) | Crop coordinates |
| `Track.track_id` | `str` | Key for track→identity mapping cache |
| `MetricsRegistry` | class | Identity metrics already registered (lines 62-88) |
| `Settings` | frozen dataclass | Re-ID config keys already present |

### 10.2 What Phase 2 Exposes for Phase 3

| Contract | Type | Consumer |
|----------|------|----------|
| `IdentityResult` | new dataclass | Phase 3 cross-camera linking |
| `IdentityResult.identities` | `dict[str, Identity]` | Phase 3 matches identities across cameras |
| `Identity.embedding` | `np.ndarray` (512-dim) | Phase 3 cross-camera Re-ID |
| `Identity.camera_history` | `list[str]` | Phase 3 transition modeling |
| `IdentityGallery.search()` | method | Phase 3 cross-camera gallery lookup |
| `Event` (IDENTITY_CONFIRMED etc.) | schema | Phase 4 event store |

---

## 11. DEMO VIDEO VALIDATION STRATEGY

Phase 2 will be validated primarily against the MOT17 demo videos established in Phase 1:

| Video | Use Case |
|-------|----------|
| `mot17_09_low_density.mp4` (26 peds) | Baseline Re-ID accuracy — few targets, low occlusion |
| `mot17_04_medium_density.mp4` (83 peds) | Primary benchmark — ID switch reduction, false merge measurement |
| `mot17_05_high_density.mp4` (166 tracks) | Stress test — embedding extraction throughput under high target count |

The `scripts/run_demo_suite.py` will be updated to display global identity UUIDs instead of tracker-local IDs in the HUD overlay.

---

## 12. PHASE 2 DELIVERABLES CHECKLIST

| Deliverable | Status |
|-------------|--------|
| `gods_eye/reid/extractor.py` | Planned |
| `gods_eye/reid/osnet_extractor.py` | Planned |
| `gods_eye/reid/similarity.py` | Planned |
| `gods_eye/identity/gallery.py` | Planned |
| `gods_eye/identity/identity_mapper.py` | Planned |
| `gods_eye/identity/state_machine.py` | Planned |
| Modified `gods_eye/pipeline.py` (IdentityWorker) | Planned |
| Modified `run_demo.py` (identity HUD) | Planned |
| `benchmarks/benchmark_reid_market1501.py` | Planned |
| `benchmarks/benchmark_gallery_latency.py` | Planned |
| `benchmarks/benchmark_occlusion_relink.py` | Planned |
| `benchmarks/benchmark_false_merge.py` | Planned |
| `benchmarks/benchmark_fps_regression.py` | Planned |
| `tests/test_reid.py` | Planned |
| `tests/test_similarity.py` | Planned |
| `tests/test_gallery.py` | Planned |
| `tests/test_identity_mapper.py` | Planned |
| `tests/test_state_machine.py` | Planned |
| `docs/phases/phase_2/PHASE_2_IMPLEMENTATION.md` | Planned |
| `docs/phases/phase_2/PHASE_2_BENCHMARKS.md` | Planned |
| `docs/phases/phase_2/PHASE_2_FINAL_AUDIT.md` | Planned |
| `docs/phases/phase_2/PHASE_2_LESSONS_LEARNED.md` | Planned |
| Git tag `phase2-complete` | Planned |
| mypy --strict: 0 errors | Required |
| pytest: all tests pass (~80 total) | Required |
