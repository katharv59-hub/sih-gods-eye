# God's Eye — Phase 1 Engineering Report

**Document Version:** 1.0  
**Date:** 2026-06-11  
**Spec Reference:** GODS_EYE_MASTER_SPEC.md v2.0  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), Python 3.11.9, Windows  

---

## 1. Executive Summary

Phase 1 implemented the complete single-camera person tracking pipeline: frame ingestion from file/webcam/RTSP sources, YOLOv8n person detection on CUDA with FP16, ByteTrack multi-object tracking, and a threaded pipeline with bounded queues and backpressure.

**Implemented:** Foundation, Schemas, Config, Observability, Ingestion, Detection, Tracking, Pipeline Integration, Visual Demo.

**Intentionally not implemented:** Re-ID, Identity Mapping, Multi-Camera, Camera Graph, Memory, Reasoning, Dashboard, API.

**Outcome:** All subsystems pass `mypy --strict` (31 source files, 0 errors), `pytest` (38 tests, 0 failures), and real-world webcam validation. End-to-end pipeline delivers 24.9 FPS on webcam with zero queue drops.

---

## 2. Scope

### Implemented (Phase 1)

| Subsystem | Module | Status |
|-----------|--------|--------|
| Foundation | `gods_eye/` package structure | Complete |
| Schemas | `gods_eye/schemas/` (Detection, Track, Identity, Event, CameraNode, BoundingBox) | Complete |
| Config | `gods_eye/config/settings.py` (24 fields, env-var resolution) | Complete |
| Observability | `gods_eye/observability/` (structlog JSON, prometheus_client 14 metrics) | Complete |
| Ingestion | `gods_eye/ingestion/` (FrameSource ABC, 3 sources, FrameQueue, CameraCaptureThread) | Complete |
| Detection | `gods_eye/detection/` (Detector ABC, YOLODetector) | Complete |
| Tracking | `gods_eye/tracking/` (Tracker ABC, ByteTrackTracker) | Complete |
| Pipeline | `gods_eye/pipeline.py` (PipelineQueue, Workers, Orchestrator) | Complete |
| Visual Demo | `run_demo.py` | Complete |

### Not Implemented (Phase 2+)

| Subsystem | Reason |
|-----------|--------|
| Re-ID (`gods_eye/reid/`) | Phase 2 — requires OSNet embedding extractor |
| Identity Mapping (`gods_eye/identity/`) | Phase 2 — requires Re-ID and gallery |
| Multi-Camera | Phase 2 — requires Camera Graph |
| Camera Graph (`gods_eye/camera_graph/`) | Phase 2 — requires topology definition |
| Memory (`gods_eye/memory/`) | Phase 3 — requires Identity layer |
| Reasoning (`gods_eye/reasoning/`) | Phase 3 — requires Memory layer |
| Dashboard | Phase 4 — presentation layer |
| API | Phase 4 — external interfaces |

---

## 3. Final Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                     │
│                    (gods_eye/pipeline.py)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CameraCaptureThread ──[FrameQueue]──► DetectionWorker      │
│       (Thread 1)          maxsize=32       (Thread 2)       │
│       owns: FrameSource                   owns: Detector    │
│                                                             │
│  DetectionWorker ──[DetectionQueue]──► TrackingWorker        │
│                       maxsize=16          (Thread 3)        │
│                                           owns: Tracker     │
│                                                             │
│  TrackingWorker ──[TrackQueue]──► OutputWorker               │
│                     maxsize=16      (Thread 4)              │
│                                     owns: callback          │
└─────────────────────────────────────────────────────────────┘
```

### Subsystem Responsibilities

**Ingestion** — Acquires frames from video sources. One thread per source. Never performs inference. Publishes `FramePacket` into `FrameQueue`. Supports reconnection with exponential backoff (0.5s–5s).

**Detection** — Runs YOLOv8n inference on `FramePacket`. Filters for person class (COCO class 0). Emits `List[Detection]` per §4 schema. CUDA FP16 when available, CPU FP32 fallback.

**Tracking** — Runs ByteTrack on `List[Detection]` per frame. Maintains per-track lifecycle (ACTIVE → LOST → DEAD). Emits `List[Track]` per §4 schema. IoU-based detection-to-track matching.

**Pipeline** — Wires all subsystems with bounded queues. Enforces backpressure via drop-oldest overflow. Implements graceful shutdown: producers close first, consumers drain before exit.

---

## 4. Canonical Data Contracts

All schemas defined in `gods_eye/schemas/` as frozen `@dataclass` objects per §4.

### BoundingBox
```
x1, y1, x2, y2: float  (top-left, bottom-right)
Derived: width, height, area
```

### Detection
```
detection_id: str (UUID4)     camera_id: str
frame_id: int                 timestamp_ns: int
bbox: BoundingBox             confidence: float [0.0–1.0]
class_label: str ("person")   source_resolution: tuple[int, int]
```

### Track
```
track_id: str                 camera_id: str
state: TrackState             bbox: BoundingBox
first_frame_id: int           last_frame_id: int
lost_frame_count: int         detection_history: list[str]
```
`TrackState`: `ACTIVE | LOST | DEAD`

### Identity, Event, CameraNode
Defined per §4 but not exercised in Phase 1. Schemas pass construction and field tests. Will be activated in Phase 2.

---

## 5. Implementation Details

### 5.1 Ingestion

**Files:** `gods_eye/ingestion/source.py`, `frame_packet.py`, `frame_queue.py`, `capture_thread.py`

**FrameSource ABC:** `open() → bool`, `read() → tuple[bool, ndarray | None]`, `release()`, `is_live → bool`

| Source | Backend | `is_live` | Notes |
|--------|---------|-----------|-------|
| `VideoFileSource` | `cv2.VideoCapture(path)` | `False` | Terminates on exhaustion |
| `WebcamSource` | `cv2.VideoCapture(index)` | `True` | Index-based device |
| `RTSPSource` | `cv2.VideoCapture(url)` | `True` | URL-based stream |

**FrameQueue:** Thread-safe deque, `maxsize` configurable via `GODS_EYE_FRAME_QUEUE_SIZE`. Drop-oldest on overflow. Logs every drop with `reason="queue_overflow"` and `dropped_frame_id`.

**CameraCaptureThread:** Daemon thread. Owns `VideoCapture` lifecycle. Exponential backoff reconnection (0.5s base, 5s cap). On source exhaustion, closes queue and exits.

### 5.2 Detection

**Files:** `gods_eye/detection/detector.py`, `yolo_detector.py`

**Detector ABC:** `detect(FramePacket) → list[Detection]`, `warmup()`, `device`, `model_name`

**YOLODetector:**
- Model: `ultralytics.YOLO("yolov8n.pt")` — auto-downloads 6.2 MB weights
- Device resolution: `torch.cuda.is_available() and not settings.cpu_only`
- CUDA path: FP16 (`half=True`), GPU inference
- CPU path: FP32 (`half=False`), fallback
- Filtering: `classes=[0]` (person only), `conf=settings.detection_confidence_threshold`
- Output: Each YOLO box → `Detection(detection_id=uuid4(), class_label="person", ...)`
- Warmup: Single dummy 480×640 inference to JIT CUDA kernels

### 5.3 Tracking

**Files:** `gods_eye/tracking/tracker.py`, `bytetrack_tracker.py`

**Tracker ABC:** `update(detections, frame_id, camera_id) → list[Track]`, `reset()`, `active_track_count`, `total_track_count`

**ByteTrackTracker:**
- Backend: `supervision.ByteTrack`
- Internal state: `dict[int, _TrackInfo]` mapping ByteTrack IDs to our track metadata
- Detection matching: IoU-based (threshold 0.3) between tracked bbox and input detections
- Detection history: capped at 50 entries per track
- State transitions: ACTIVE (matched) → LOST (unmatched < dead_timeout) → DEAD (evicted)
- LOST tracks returned with last-known bbox. DEAD tracks silently evicted with structured log.

### 5.4 Pipeline

**File:** `gods_eye/pipeline.py`

**PipelineQueue[T]:** Generic thread-safe queue. Same drop-oldest semantics as FrameQueue. Used for DetectionResult and TrackingResult transport.

**Workers:**

| Worker | Consumes | Produces | Owns |
|--------|----------|----------|------|
| `DetectionWorker` | `FrameQueue → FramePacket` | `PipelineQueue → DetectionResult` | Detector |
| `TrackingWorker` | `PipelineQueue → DetectionResult` | `PipelineQueue → TrackingResult` | Tracker |
| `OutputWorker` | `PipelineQueue → TrackingResult` | callback invocation | — |

**Shutdown sequence:** `Pipeline.stop()` → `CaptureThread.stop()` → FrameQueue closes → DetectionWorker drains & closes DetectionQueue → TrackingWorker drains & closes TrackQueue → OutputWorker drains → all threads join.

**Error isolation:** Detection/tracking exceptions on individual frames are logged and skipped. Pipeline continues processing subsequent frames.

---

## 6. Validation History

### 6.1 Static Analysis

| Tool | Scope | Result |
|------|-------|--------|
| `mypy --strict` | 31 source files in `gods_eye/` | 0 errors |
| `pytest` | 38 tests across 5 test files | 38 passed, 0 failed |

### 6.2 Unit Tests

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_schemas.py` | 10 | BoundingBox, Detection, Track, Identity, Event, CameraNode |
| `tests/test_ingestion.py` | 10 | FramePacket, FrameQueue (put/get/overflow/close/drain), CameraCaptureThread (capture/stop/overflow/failure) |
| `tests/test_detection.py` | 5 | Detection fields, UUID uniqueness, BoundingBox math, empty/multi results |
| `tests/test_tracking.py` | 7 | Empty input, single track, continuity, LOST transition, multi-track, schema fields, reset |
| `tests/test_pipeline.py` | 6 | PipelineQueue (put/get/drop/close), OutputWorker delivery, full E2E with stub detector/tracker |

### 6.3 Integration Validations

| Validation | Script | Source | Outcome |
|------------|--------|--------|---------|
| Ingestion (file) | `scripts/validate_ingestion.py` | Synthetic 150-frame video | 150/150 delivered, 0 drops, 1638 FPS |
| Ingestion (webcam) | `scripts/validate_ingestion.py` | Webcam 5s | 111/111 delivered, 0 drops, 20.6 FPS |
| Detection (synthetic) | `scripts/validate_detection.py` | Synthetic video | 88.7 FPS, 11.3ms avg latency |
| Detection (real person) | `scripts/validate_detection_quality.py` | Webcam 8s | 48.2 FPS, 225/231 frames with detections |
| Tracking (webcam) | `scripts/validate_tracking.py` | Webcam 10s | 47.5 FPS, 1 unique ID, 292-frame lifetime |
| Tracking (stress) | `scripts/stress_test_tracking.py` | Synthetic 8-person, 300 frames | 928 FPS tracker-only, 8/8 IDs, 0 switches |
| Pipeline (file) | `scripts/validate_pipeline.py` | Synthetic video | 43.9 FPS, 31 delivered |
| Pipeline (webcam) | `scripts/validate_pipeline.py` | Webcam 10s | 24.9 FPS, 258/258 delivered, 0 drops |
| Visual demo | `run_demo.py` | Webcam | 689 frames, 0 drops, screenshot captured |

### 6.4 Debug Audit

**Trigger:** Visual demo showed bounding box appearing larger than expected.

**Investigation:** `scripts/debug_bbox.py` — side-by-side raw YOLO vs tracker coordinates on 30 frames.

**Finding:** YOLO and tracker boxes match exactly (`MATCH` on every frame). Box size (430×312px on 640×480) is correct — person is close to webcam, occupying 67% of frame. No bug. No code changes.

---

## 7. Benchmark Results

### 7.1 Detection Benchmarks (RTX 4050, CUDA, FP16)

| Metric | Synthetic Video | Real Webcam |
|--------|----------------|-------------|
| Detection FPS | 88.7 | 48.2 |
| Avg Latency (ms) | 11.3 | 20.7 |
| P95 Latency (ms) | 16.7 | — |
| VRAM Allocated | 38.1 MB | — |
| VRAM Reserved | 68.0 MB | — |
| Model Load Time | 28.85s (incl. download) | — |
| Warmup Time | 1.32s | — |
| Avg Confidence | — | 0.496 |
| Min Confidence | — | 0.251 |
| Max Confidence | — | 0.692 |

### 7.2 Tracking Benchmarks

| Metric | Single Person (webcam) | Stress Test (8 synthetic) |
|--------|----------------------|--------------------------|
| Tracking FPS | 47.5 (det+track) | 928 (tracker only) |
| Avg Active Tracks/Frame | 1.00 | 5.54 |
| Max Active Tracks | 1 | 7 |
| Unique Track IDs | 1 | 8 |
| ID Switches | 0 | 0 |
| Avg Track Lifetime | 292 frames | 207.8 frames |
| Max Track Lifetime | 292 | 299 |
| ACTIVE→LOST Transitions | — | 8 |
| DEAD Evictions | — | 1 |
| Tracker Latency | 0.2 ms | 1.08 ms |

### 7.3 Pipeline Benchmarks (E2E)

| Metric | Video File | Webcam |
|--------|-----------|--------|
| End-to-End FPS | 43.9 | 24.9 |
| Frames Captured | 150 | 258 |
| Frames Delivered | 31 | 258 |
| Avg E2E Latency (ms) | 254.1 | 35.0 |
| P95 E2E Latency (ms) | 447.1 | 62.1 |
| Det Queue Drops | 0 | 0 |
| Track Queue Drops | 0 | 0 |
| Frame Queue Drops | 119 (expected) | 0 |

**Note:** Video file drops are expected — file reads at >1000 FPS while detection processes at ~50 FPS. Backpressure (drop-oldest) correctly discards stale frames.

---

## 8. Screenshots & Evidence

All screenshots stored in `tests/data/`.

| Path | Content |
|------|---------|
| `tests/data/detection_samples/sample_1.jpg` | Raw YOLO detection with green bounding box and confidence label on real person |
| `tests/data/tracking_samples/track_sample_1.jpg` | ByteTrack overlay with `ID:1 [active]` label on tracked person |
| `tests/data/screenshots/screenshot_*.jpg` | 17 frames from visual demo showing full HUD overlay (FPS, detections, tracks, queue depths, drops) |
| `tests/data/debug_bbox/debug_1.jpg` | Debug audit — green (YOLO) and blue (TRACK) boxes overlaid, demonstrating exact coordinate match |

---

## 9. Known Limitations

| Limitation | Impact | Mitigation (Phase 2+) |
|-----------|--------|----------------------|
| Motion-only tracking | ID switches on heavy occlusion or crossing at similar velocity | OSNet appearance embeddings (Phase 2) |
| No cross-camera persistence | Track IDs are per-camera, per-session | Identity Mapping Layer + Camera Graph (Phase 2) |
| No appearance embeddings | Cannot re-identify person after track loss | Re-ID module with gallery matching (Phase 2) |
| Single-camera only | Pipeline supports one source at a time | Multi-camera Pipeline with shared Identity layer (Phase 2) |
| `supervision.ByteTrack` deprecated | Will be removed in supervision v0.30.0 | Migrate to replacement API before upgrading |
| YOLO bbox size at close range | Box covers 60-70% of frame for nearby subjects | Expected behavior, not a bug |
| No temporal memory | System forgets all tracks on restart | Memory subsystem with SQLite/Redis (Phase 3) |
| Webcam FPS variance | 20-30 FPS depending on lighting and USB bandwidth | Acceptable for Phase 1 targets |

---

## 10. Rejected Approaches

| Approach | Reason for Rejection |
|----------|---------------------|
| Face recognition for identification | Violates §2 Rule 5 (Privacy Aware). Unreliable at distance and with occlusion. Spec mandates full-body appearance embeddings. |
| Global IDs from ByteTrack alone | ByteTrack IDs are local to one tracker instance. Cannot persist across camera boundaries or session restarts. Spec requires Identity Mapping Layer. |
| `model.track()` from ultralytics | Couples detection and tracking into one call. Violates §8 modularity (separate threads, separate queues). Prevents independent scaling. |
| Motion-only long-term persistence | Kalman prediction degrades rapidly beyond 1-2 seconds. Spec requires Re-ID for robust re-association. |
| Direct `queue.Queue` for pipeline | stdlib `queue.Queue` blocks on full. Spec §8 mandates drop-oldest overflow for backpressure. Custom `PipelineQueue[T]` implemented. |
| Shared detector across threads | `YOLODetector` is not thread-safe (GPU state). Each pipeline owns its detector instance on a single thread. |

---

## 11. Lessons Learned

### Detection is the pipeline bottleneck
Tracker-only FPS: 928. Detection FPS: 48-88. Detection consumes >95% of per-frame latency. Any pipeline optimization must target detection throughput (batching, model pruning, TensorRT).

### Tracking overhead is negligible
ByteTrack adds ~0.2ms per frame (~1ms under 8-person stress). It can be treated as essentially free in pipeline timing calculations.

### Backpressure works correctly under real conditions
Video file ingestion at >1000 FPS correctly overwhelms the detection queue. Drop-oldest discards stale frames without blocking the producer. Webcam ingestion is naturally balanced with detection throughput.

### Visual validation caught what unit tests could not
The "oversized bounding box" concern from visual inspection led to a debug audit. While the finding was "no bug" (correct YOLO behavior at close range), the investigation validated the entire data path from YOLO output through schema conversion through ByteTrack to overlay rendering.

### Structured logging is essential for debugging threaded systems
JSON-formatted logs with `subsystem`, `camera_id`, and `event` fields made it possible to trace frame flow through 4 concurrent threads without ambiguity.

### mypy strict prevents entire classes of bugs
Strict mode caught `structlog` API issues, missing return types, and incorrect type annotations that would have caused silent runtime failures.

---

## 12. Phase 2 Readiness Assessment

### Prerequisites Satisfied

| Prerequisite | Status |
|-------------|--------|
| Canonical Detection schema with UUID detection_id | ✅ |
| Canonical Track schema with detection_history | ✅ |
| Identity schema defined (§4) | ✅ (not yet exercised) |
| Stable single-camera tracking pipeline | ✅ |
| FramePacket carries raw frame for embedding extraction | ✅ |
| Track bbox available for person crop extraction | ✅ |
| Pipeline supports callback-based output delivery | ✅ |
| Structured logging for all subsystems | ✅ |
| Prometheus metrics infrastructure | ✅ |
| mypy strict compliance | ✅ |

### Phase 2 Targets

| Component | Description |
|-----------|------------|
| OSNet Re-ID | Extract 512-d appearance embeddings from person crops |
| Identity Gallery | Store and match embeddings against known identities |
| Identity Mapping Layer | Bridge Track → Identity using Re-ID similarity |

### Integration Points

Re-ID module will receive `TrackingResult` (containing `packet.frame` + `Track.bbox`), crop the person region, extract an embedding, and query the Identity Gallery. The existing `OutputWorker` callback is the natural integration point.

---

## 13. Final Status

```
Foundation      [x]  Package structure, schemas, config, observability
Ingestion       [x]  FrameSource ABC, 3 sources, FrameQueue, CameraCaptureThread
Detection       [x]  Detector ABC, YOLODetector (CUDA FP16 / CPU FP32)
Tracking        [x]  Tracker ABC, ByteTrackTracker (supervision.ByteTrack)
Pipeline        [x]  PipelineQueue[T], 4 worker threads, graceful shutdown
Visual Demo     [x]  run_demo.py with HUD overlay and screenshot support
mypy --strict   [x]  31 source files, 0 errors
pytest          [x]  38 tests, 0 failures
Webcam E2E      [x]  24.9 FPS, 0 drops, stable tracking
Debug Audit     [x]  No bugs found

Phase 1         [COMPLETE]
```
