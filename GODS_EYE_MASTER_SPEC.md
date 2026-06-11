# GODS_EYE_MASTER_SPEC.md
**Version:** 2.0  
**Status:** Active — Source of Truth  
**Last Revised:** 2026-06-10  
**Supersedes:** v1.0

---

## Table of Contents

1. [Project Name & Mission](#1-project-name--mission)
2. [Core Philosophy & Commandments](#2-core-philosophy--commandments)
3. [What God's Eye Is NOT](#3-what-gods-eye-is-not)
4. [Canonical Data Schemas](#4-canonical-data-schemas)
5. [Architecture Overview](#5-architecture-overview)
6. [Execution Environment](#6-execution-environment)
7. [Observability Contract](#7-observability-contract)
8. [Concurrency Model](#8-concurrency-model)
9. [Security & Compliance Architecture](#9-security--compliance-architecture)
10. [Engineering Standards](#10-engineering-standards)
11. [Development Workflow Template](#11-development-workflow-template)
12. [Testing Specification](#12-testing-specification)
13. [Project Roadmap](#13-project-roadmap)
14. [Re-ID Design Decisions](#14-re-id-design-decisions)
15. [Architectural Horizon (Locked Future)](#15-architectural-horizon-locked-future)
16. [Relationship to Sentinel AI](#16-relationship-to-sentinel-ai)
17. [Architecture Decision Records (ADR)](#17-architecture-decision-records-adr)
18. [Changelog](#18-changelog)

---

## 1. Project Name & Mission

**Project Name:** God's Eye

**Mission Statement:**  
God's Eye is a real-time situational awareness platform designed to transform raw video streams into persistent spatial and temporal intelligence.

The objective is not merely object detection.  
The objective is to answer:

- Who is present?
- Where did they come from?
- Where did they go?
- What path did they follow?
- Which camera observed them?
- What events occurred over time?
- What is happening right now?

The system progressively evolves from **perception** → **memory** → **reasoning**.

At its core, God's Eye is a **graph construction engine**: cameras are sensors, detections are nodes, Re-ID matches are edges, and time is the axis. Everything is built to serve that graph.

---

## 2. Core Philosophy & Commandments

### Design Rules

**Rule 1 — Benchmarks Over Assumptions**  
No optimization without measurement. No architecture without validation. Measured results override intuition. Every performance claim must reference a benchmark result with hardware, dataset, and metric specified.

**Rule 2 — Simplicity First**  
The simplest working solution is always preferred. Avoid unnecessary frameworks, abstractions, and patterns. Complexity must be justified by a benchmark or a documented requirement.

**Rule 3 — Incremental Development**  
Build strictly in order:

```
Detection → Tracking → Identity Persistence → Multi-Camera Intelligence → Temporal Memory → Reasoning
```

Never implement future-phase components. Never skip a layer. Phase N cannot begin until Phase N-1 passes its gate criteria.

**Rule 4 — Modularity**  
Every subsystem must be independently replaceable without modifying adjacent subsystems. Replaceable means: same typed interface in, same typed interface out. Interface contracts are defined in §4 and must not be altered without an ADR entry.

**Rule 5 — Observability**  
No black boxes. Every subsystem exposes its metrics in the format specified in §7. Silence from a subsystem is treated as a failure signal.

**Rule 6 — Explicit Failure Modes**  
Every subsystem must define its failure behavior. Silent degradation is prohibited. Unknown states must be logged and surfaced, never swallowed.

### Commandments

1. Benchmark first. Optimize second.
2. Never trust assumptions — measure them.
3. Never build future phases early.
4. Never overengineer. Complexity must earn its place.
5. Keep every module replaceable.
6. Measure everything that moves.
7. Document every important decision in the ADR log.
8. Prefer clarity over cleverness.
9. Build one layer at a time.
10. Failure modes are features. Define them explicitly.

---

## 3. What God's Eye Is NOT

God's Eye is NOT:

- A face recognition project
- A dashboard project
- A React project
- A FastAPI project
- A database project
- A chatbot project

These may exist *around* the system. They are not the system itself.

God's Eye produces structured intelligence. Other systems consume it.

---

## 4. Canonical Data Schemas

These are the inter-module contracts. All modules must consume and produce these types exactly. No module may invent its own representation of these concepts. Schema changes require an ADR entry.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


# ─── DETECTION ───────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: float          # Top-left x (pixels, absolute)
    y1: float          # Top-left y (pixels, absolute)
    x2: float          # Bottom-right x (pixels, absolute)
    y2: float          # Bottom-right y (pixels, absolute)

@dataclass
class Detection:
    detection_id: str            # UUID, unique per detection
    camera_id: str               # Source camera identifier
    frame_id: int                # Monotonic frame counter for this camera
    timestamp_ns: int            # Unix nanoseconds — do NOT use float seconds
    bbox: BoundingBox
    confidence: float            # [0.0, 1.0]
    class_label: str             # e.g. "person"
    source_resolution: tuple[int, int]  # (width, height) of source frame


# ─── TRACK ────────────────────────────────────────────────────────────────────

class TrackState(Enum):
    ACTIVE = "active"
    LOST = "lost"       # Not seen for < TRACK_LOST_TIMEOUT frames
    DEAD = "dead"       # Not seen for >= TRACK_DEAD_TIMEOUT frames — evict

@dataclass
class Track:
    track_id: str                         # Tracker-local ID (not global)
    camera_id: str
    state: TrackState
    bbox: BoundingBox                     # Most recent bounding box
    first_frame_id: int
    last_frame_id: int
    lost_frame_count: int                 # Consecutive frames without detection
    detection_history: list[str]          # List of detection_ids (capped at N)


# ─── IDENTITY ─────────────────────────────────────────────────────────────────

class IdentityStatus(Enum):
    ACTIVE = "active"           # Currently visible in at least one camera
    LOST = "lost"               # Not seen for > IDENTITY_LOST_TIMEOUT seconds
    PURGED = "purged"           # TTL expired — record tombstoned, embeddings deleted

@dataclass
class Identity:
    global_id: str                         # UUID — stable across cameras and sessions
    status: IdentityStatus
    embedding: np.ndarray                  # Current EMA-updated embedding (OSNet dim)
    embedding_history: list[np.ndarray]    # Rolling window, max EMBEDDING_HISTORY_LEN
    first_seen_ns: int                     # Unix nanoseconds
    last_seen_ns: int
    last_camera_id: str
    camera_history: list[str]              # Ordered list of camera_ids visited
    track_id_history: list[tuple[str, str]] # [(camera_id, track_id), ...]
    confidence: float                      # Mean match confidence over lifetime
    purge_at_ns: Optional[int]             # Set when status transitions to LOST


# Identity State Transition Conditions:
#
#   ACTIVE  → LOST    : last_seen_ns older than IDENTITY_LOST_TIMEOUT_S seconds
#   LOST    → ACTIVE  : Re-ID match above REID_MATCH_THRESHOLD
#   LOST    → PURGED  : current_time > purge_at_ns (set at IDENTITY_TTL_S from last_seen)
#   PURGED  → (none)  : terminal state — embeddings zeroed, record tombstoned


# ─── EVENT ────────────────────────────────────────────────────────────────────

class EventType(Enum):
    PERSON_ENTERED_FRAME = "person_entered_frame"
    PERSON_LEFT_FRAME = "person_left_frame"
    IDENTITY_CONFIRMED = "identity_confirmed"
    IDENTITY_LOST = "identity_lost"
    CROSS_CAMERA_TRANSITION = "cross_camera_transition"
    IDENTITY_PURGED = "identity_purged"

@dataclass
class Event:
    event_id: str
    event_type: EventType
    global_id: Optional[str]     # None if identity not yet resolved
    camera_id: str
    timestamp_ns: int
    frame_id: int
    metadata: dict                # Event-type-specific payload (keep flat, JSON-serializable)


# ─── CAMERA NODE ──────────────────────────────────────────────────────────────

@dataclass
class CameraNode:
    camera_id: str
    display_name: str
    source_uri: str               # RTSP URL, device index, or file path
    location_label: str           # Human-readable physical location
    resolution: tuple[int, int]   # (width, height)
    fps_nominal: float
    adjacent_cameras: list[str]   # camera_ids reachable by direct physical path
    # Phase 3 fields — optional until Phase 3 gate
    transition_priors: dict[str, float]  # {target_camera_id: probability}
    overlap_regions: list[BoundingBox]   # Shared FOV regions with adjacent cameras
```

---

## 5. Architecture Overview

```
Video Sources (RTSP / Webcam / File)
          │
          ▼
    [Ingestion Layer]          ← Thread per source; frame queues with backpressure
          │
          ▼
    [Detection]                ← YOLOv8; outputs List[Detection]
          │
          ▼
    [Tracking]                 ← ByteTrack; outputs List[Track]
          │
          ▼
    [Identity Mapping Layer]   ← OSNet Re-ID; maps Track → Identity
          │
          ▼
    [Identity Gallery]         ← In-memory + persisted; thread-safe; TTL-managed
          │
          ▼
    [Camera Graph]             ← CameraNode topology; transition modeling (Phase 3)
          │
          ▼
    [Event Store]              ← Append-only Event log (Phase 4)
          │
          ▼
    [Temporal Memory]          ← Movement history, timeline reconstruction (Phase 4)
          │
          ▼
    [Situational Reasoning]    ← Query engine over graph + memory (Phase 5)
          │
          ▼
    [NLQ Interface]            ← LLM + structured tool calls into graph (Phase 5)
```

**Data flow rule:** Each layer consumes the schema from §4 and produces the schema from §4. No layer invents private data representations that cross module boundaries.

---

## 6. Execution Environment

### Reference Hardware (Development Target)

| Component | Specification |
|-----------|---------------|
| CPU | 8-core x86-64, 3.0 GHz+ |
| RAM | 16 GB minimum |
| GPU | NVIDIA GPU with 8 GB VRAM (CUDA 12.x) |
| Storage | SSD, 50 GB working space |
| OS | Ubuntu 22.04 LTS |

### Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.11.x |
| CUDA | 12.1 |
| cuDNN | 8.9.x |
| PyTorch | 2.2.x |

### CPU-Only Degraded Mode

The system must remain operable without a GPU, with the following capability reductions:

| Capability | GPU Mode | CPU-Only Mode |
|------------|----------|---------------|
| Detection FPS | ≥ 25 FPS @ 1080p | ≥ 6 FPS @ 720p |
| Re-ID | Full OSNet | Lightweight MobileNet backbone |
| Multi-stream | Up to N streams | 1 stream only |
| Embedding update | Real-time EMA | Batch, every 10 frames |

CPU-only mode is activated by the env flag `GODS_EYE_CPU_ONLY=1`.  
CPU-only mode does **not** disable observability. All metrics still emit.

### Dependency Manifest (Baseline)

```toml
# pyproject.toml — core dependencies
[project]
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "ultralytics>=8.0.0",        # YOLOv8
    "torchreid>=0.2.5",          # OSNet / Re-ID
    "numpy>=1.26.0",
    "opencv-python>=4.9.0",
    "prometheus-client>=0.20.0",
    "structlog>=24.0.0",
    "pydantic>=2.6.0",
]
```

Pinned versions live in `requirements.lock`. The lock file is the authoritative install source.

---

## 7. Observability Contract

### Required Metrics — All Subsystems

Every subsystem **must** expose these metrics. No exceptions.

| Metric | Type | Labels |
|--------|------|--------|
| `gods_eye_fps` | Gauge | `subsystem`, `camera_id` |
| `gods_eye_latency_ms` | Histogram | `subsystem`, `camera_id` |
| `gods_eye_cpu_utilization` | Gauge | `subsystem` |
| `gods_eye_ram_bytes` | Gauge | `subsystem` |
| `gods_eye_gpu_utilization` | Gauge | `subsystem` |
| `gods_eye_vram_bytes` | Gauge | `subsystem` |
| `gods_eye_queue_depth` | Gauge | `queue_name` |
| `gods_eye_frame_drops_total` | Counter | `camera_id`, `reason` |

### Additional Metrics — Identity Layer

| Metric | Type | Labels |
|--------|------|--------|
| `gods_eye_identities_active` | Gauge | — |
| `gods_eye_identities_lost` | Gauge | — |
| `gods_eye_identities_purged_total` | Counter | — |
| `gods_eye_reid_match_confidence` | Histogram | `camera_id` |
| `gods_eye_reid_false_merge_total` | Counter | — |
| `gods_eye_gallery_size` | Gauge | — |

### Exposure Format

Metrics are exposed via a **Prometheus `/metrics` HTTP endpoint** on `localhost:9090` by default, configurable via `GODS_EYE_METRICS_PORT`.

Structured logs use **JSON format via `structlog`**. Every log entry must include:

```json
{
  "timestamp": "<ISO8601>",
  "level": "info|warning|error",
  "subsystem": "<module_name>",
  "camera_id": "<id or null>",
  "event": "<description>",
  "data": {}
}
```

No print statements in production code. No unstructured logging.

### Alerting Thresholds (Phase 1 Baseline)

| Condition | Threshold | Action |
|-----------|-----------|--------|
| FPS drops below target | < 20 FPS for > 5s | Log WARNING + emit alert event |
| Queue depth exceeds limit | > 50 frames | Log WARNING + emit backpressure signal |
| Frame drop rate | > 5% over 30s window | Log ERROR |
| GPU memory | > 90% VRAM | Log WARNING |
| Identity gallery overflow | > MAX_GALLERY_SIZE | Begin LRU eviction + log WARNING |

---

## 8. Concurrency Model

### Pipeline Architecture

```
Camera Thread (1 per source)
    │  [FrameQueue — maxsize=FRAME_QUEUE_SIZE]
    ▼
Detection Worker Thread
    │  [DetectionQueue — maxsize=DETECTION_QUEUE_SIZE]
    ▼
Tracking Worker Thread (1 per camera)
    │  [TrackQueue — maxsize=TRACK_QUEUE_SIZE]
    ▼
Identity Mapping Thread (shared, serialized)
    │  [EventQueue — maxsize=EVENT_QUEUE_SIZE]
    ▼
Event Writer Thread
```

### Shared State Rules

| Shared Resource | Owner Thread | Access Rule |
|----------------|--------------|-------------|
| Identity Gallery | Identity Mapping Thread | Single-writer. All reads/writes go through this thread via queue messages — no direct cross-thread access. |
| Camera Graph | Main thread (init-time) | Immutable after init. Read-only access is safe without locks. |
| Metrics registry | Prometheus client (thread-safe) | Direct write permitted from any thread. |
| Event log | Event Writer Thread | Append-only. All writes go through EventQueue. |

### Queue Ownership Rules

- Every queue has exactly **one producer** and exactly **one consumer**. No fan-out without an explicit router.
- Queue overflow behavior: **drop oldest frame** (not newest). Log every drop with `reason="queue_overflow"`.
- Shutdown sequence: producers close first, consumers drain queues before exit.

### Thread Safety Contract

The Identity Gallery is **not thread-safe by default**. All gallery mutations (insert, update, purge, query) **must** be performed by the Identity Mapping Thread. External modules request gallery operations by posting messages to the `IdentityRequestQueue`.

---

## 9. Security & Compliance Architecture

God's Eye processes biometric-adjacent data (appearance embeddings + movement histories of individuals). The following requirements are non-optional regardless of deployment context.

### Data Minimization

- Raw video frames are **never persisted** by God's Eye. Only derived data (detections, embeddings, events) is stored.
- Embeddings are stored as opaque vectors. No names, faces, or PII are attached to identities at the God's Eye layer.

### Identity Lifecycle & Retention

| Parameter | Default | Config Key |
|-----------|---------|------------|
| Identity active timeout | 300s | `IDENTITY_LOST_TIMEOUT_S` |
| Identity TTL (lost → purge) | 86400s (24h) | `IDENTITY_TTL_S` |
| Embedding history window | 50 embeddings | `EMBEDDING_HISTORY_LEN` |
| Gallery max size | 10,000 identities | `MAX_GALLERY_SIZE` |

When an identity is **purged**: embedding vector is zeroed, embedding history is cleared, and only a tombstone record (global_id, first_seen, last_seen, purge_timestamp) is retained for audit purposes.

### Access Control

- The metrics endpoint and any future query API must support token-based authentication (`GODS_EYE_API_TOKEN`).
- All API access is logged in the audit trail with timestamp, caller identity, and query.

### Audit Trail

An append-only audit log records:

- Every identity creation
- Every identity purge
- Every cross-camera transition event
- Every external query against the identity or event store

Audit log format: structured JSON, written to a separate `audit.jsonl` file. Audit log is never auto-rotated without archival.

### Compliance Scope

God's Eye is designed to be compliant with the *data minimization and storage limitation principles* of GDPR (Art. 5), CCPA, and BIPA. Deployers are responsible for ensuring their deployment configuration (retention periods, access controls) meets their regional legal requirements. The system provides the mechanisms; the deployer sets the policy.

---

## 10. Engineering Standards

All code must:

- Use type hints (enforced via `mypy --strict`)
- Be modular — one class, one responsibility
- Be independently testable with mocked interfaces
- Be benchmarkable — every module has a `benchmark_<module>.py` counterpart
- Use structured logging via `structlog` (no print statements)
- Avoid hidden globals — all state is passed explicitly or held in well-defined singletons
- Prefer composition over inheritance
- Import only from `gods_eye.*` namespaces — no circular imports

### File Structure

```
gods_eye/
├── schemas/          # Canonical data schemas (§4) — never import from other gods_eye modules
├── ingestion/        # Camera threads, frame queues
├── detection/        # YOLO wrapper, detection pipeline
├── tracking/         # ByteTrack wrapper, track management
├── reid/             # OSNet wrapper, embedding extraction
├── identity/         # Identity Mapping Layer, gallery, state machine
├── camera_graph/     # CameraNode topology (Phase 3)
├── events/           # Event store, event writer (Phase 4)
├── memory/           # Temporal memory, timeline (Phase 4)
├── reasoning/        # Query engine, NLQ interface (Phase 5)
├── observability/    # Prometheus metrics, structured logger
├── config/           # Config loader, defaults, env var resolution
└── benchmarks/       # Benchmark harnesses per module
```

---

## 11. Development Workflow Template

Every task — every module, every fix, every optimization — must be documented using this template before implementation begins. No silent architectural decisions.

```markdown
## Task: [Name]

### Goal
What outcome does this task achieve?

### Implementation Plan
Step-by-step description of what will be built.

### Interface
- Input: [typed schema consumed]
- Output: [typed schema produced]
- Config keys introduced: [list]

### Risks
What could go wrong? What is uncertain?

### Tradeoffs
What was considered and rejected? Why was this approach chosen?

### Implementation
[Code / changes]

### Changes Made
Summary of what was actually built (may differ from plan — document the delta).

### Testing
- Unit tests: [what is tested]
- Integration tests: [what is tested]
- Benchmark: [what is measured, on what hardware, expected result]
```

---

## 12. Testing Specification

### Test Categories

| Category | Scope | Tool |
|----------|-------|------|
| Unit | Single function / class in isolation | `pytest` |
| Integration | Two adjacent modules interacting via schema | `pytest` |
| Pipeline | End-to-end from frame ingestion to identity output | `pytest` + synthetic stream |
| Benchmark | Performance characteristics on reference hardware | Custom harness in `benchmarks/` |

### Test Data Sources

| Source | Use Case |
|--------|----------|
| MOT17 dataset | Tracking accuracy (MOTA, IDF1, ID switches) |
| Market-1501 dataset | Re-ID accuracy (Rank-1, mAP) |
| Synthetic RTSP streams | Pipeline stability, concurrency, failure injection |
| Recorded real-world clips | Regression testing, demo reproducibility |

All test datasets must be stored in `tests/data/` with a `README.md` documenting source, license, and download instructions.

### Coverage Requirements

| Module | Minimum Unit Test Coverage |
|--------|--------------------------|
| `schemas/` | 100% — these are contracts |
| `identity/` | 90% |
| `detection/` | 80% |
| `tracking/` | 80% |
| `reid/` | 80% |
| `observability/` | 70% |

### Failure Injection Tests (Required — Phase 1)

| Failure Scenario | Expected Behavior |
|-----------------|-------------------|
| RTSP stream drops mid-run | Reconnect with backoff; log ERROR; continue |
| GPU OOM during inference | Fallback to CPU mode; log CRITICAL; continue at degraded FPS |
| Identity gallery reaches MAX_GALLERY_SIZE | LRU eviction; log WARNING; no crash |
| Re-ID match below threshold for extended period | Identity transitions to LOST; log INFO |
| Queue overflow | Drop oldest frame; log WARNING with reason |

---

## 13. Project Roadmap

### Phase 1 — Perception Layer

**Goal:** Stable, real-time, single-camera detection and tracking.

**Deliverables:**
- Video ingestion: webcam, RTSP, file
- YOLOv8 detection (outputs `List[Detection]`)
- ByteTrack tracking (outputs `List[Track]`)
- Threaded pipeline with queue backpressure
- Prometheus metrics endpoint
- Structured JSON logging
- CPU-only degraded mode
- Benchmark harness
- Failure injection test suite
- Reproducible demo (Docker Compose)

**Phase Gate Criteria (all must pass on reference hardware):**

| Criterion | Threshold |
|-----------|-----------|
| FPS — 1080p RTSP, GPU mode | ≥ 25 FPS sustained over 10 minutes |
| FPS — 720p, CPU-only mode | ≥ 6 FPS sustained over 5 minutes |
| Zero crashes | 10-minute continuous run, 0 unhandled exceptions |
| MOTA on MOT17-04 | ≥ 0.60 |
| ID switches on MOT17-04 | ≤ 150 |
| Frame drop rate | < 2% over any 60s window |
| Metrics endpoint | All 8 required metrics present and updating |
| Stream recovery | RTSP reconnect within 5s of stream drop |

Phase 2 does not begin until all Phase 1 gate criteria pass with documented benchmark results.

---

### Phase 2 — Identity Persistence Layer

**Goal:** Maintain identity through occlusion and reappearance within a single camera.

**Core Conclusion:**  
Motion-only tracking (ByteTrack) loses identity on occlusion. Appearance-based Re-ID is required.

**Architecture:**

```
ByteTrack Track ID
        │
        ▼
Identity Mapping Layer (IML)
        │
  OSNet Embedding Extraction
        │
  Cosine Similarity → Gallery Lookup
        │
  EMA Embedding Update
        │
        ▼
Global Identity (UUID)
```

**Components:**
- OSNet embedding extractor (torchreid)
- Gallery manager: insert, update (EMA), query, evict (LRU), purge (TTL)
- Similarity engine: cosine similarity with configurable threshold
- Identity state machine (see §4 state transition conditions)
- Identity Mapping Layer (IML): separates tracker IDs from global IDs

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Re-ID Rank-1 on Market-1501 | ≥ 0.88 |
| Identity continuity through 2s occlusion | ≥ 85% correct re-link rate |
| Gallery operations latency | ≤ 5ms p99 |
| False merge rate (same ID assigned to 2 people) | < 1% |
| IML throughput | No FPS degradation > 10% vs Phase 1 baseline |

---

### Phase 3 — Multi-Camera Intelligence

**Goal:** Track identities seamlessly across cameras.

**Components:**
- Camera graph (populated `CameraNode` topology)
- Transition modeling: empirical prior probabilities between adjacent cameras
- Cross-camera identity persistence: Re-ID matching across camera galleries
- Temporal synchronization: timestamp alignment across streams
- Predictive transition hints: "Identity X likely to appear on Camera Y in ~T seconds"

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Cross-camera identity linking accuracy | ≥ 80% on held-out multi-cam test set |
| Cross-camera transition detection latency | ≤ 2s from appearance to confirmed link |
| No regression on Phase 2 single-camera metrics | All Phase 2 gates still pass |

---

### Phase 4 — Temporal Memory

**Goal:** Persist and reconstruct the full history of events and movements.

**Storage Choice:** Graph database (e.g., Neo4j or equivalent) or a graph-structured document store. **This choice must be made in an ADR before Phase 4 begins.** The NLQ layer in Phase 5 (LLM + structured tool calls) informs this decision — the query patterns must be prototyped before storage is finalized.

**Components:**
- Append-only Event log (structured, queryable)
- Movement history per identity: ordered `(camera_id, timestamp, bbox)` sequence
- Timeline reconstruction: generate movement narrative from Event log
- Replay capability: reconstruct scene state at any past timestamp

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Event write throughput | ≥ 1,000 events/sec sustained |
| Timeline query latency (last 1h) | ≤ 200ms p95 |
| Replay accuracy | 100% event ordering preserved |
| Storage growth rate | ≤ 1 GB per 24h per camera (before TTL purge) |

---

### Phase 5 — Situational Reasoning

**Goal:** Transform memory into queryable intelligence.

**NLQ Architecture Decision (Provisional):**  
The NLQ interface uses an **LLM (Claude claude-sonnet-4-20250514) with structured tool calls** into the graph and event store. The LLM does not hold state — all queries are stateless calls with context assembled from the graph. This provisional decision is subject to revision via ADR when Phase 4 storage is finalized.

**Components:**
- Structured query API over the temporal graph
- Tool definitions: `get_identity_timeline`, `search_events_by_type`, `get_camera_path`, `list_active_identities`, `query_location_at_time`
- NLQ interface: translates natural language to tool calls and assembles narrative response
- Timeline search and event reconstruction

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Tool call accuracy on 50-query benchmark | ≥ 85% correct tool selection |
| Query response latency | ≤ 3s p95 for 1h history queries |
| NLQ correctness on factual questions | ≥ 90% on held-out QA set |

---

## 14. Re-ID Design Decisions

The following decisions are **currently accepted**. Each has an ADR entry with rejection rationale for alternatives.

| Decision | Choice | ADR |
|----------|--------|-----|
| Embedding model | OSNet (torchreid) | ADR-001 |
| Similarity metric | Cosine similarity | ADR-002 |
| Embedding update strategy | EMA (Exponential Moving Average) | ADR-003 |
| Identity-track separation | Identity Mapping Layer (IML) | ADR-004 |

### Configuration Keys (Re-ID)

```python
REID_MATCH_THRESHOLD: float = 0.75        # Cosine similarity minimum for match
REID_EMA_ALPHA: float = 0.9               # EMA weight: new = alpha * old + (1-alpha) * current
IDENTITY_LOST_TIMEOUT_S: int = 300        # Seconds before ACTIVE → LOST
IDENTITY_TTL_S: int = 86400               # Seconds before LOST → PURGED
EMBEDDING_HISTORY_LEN: int = 50           # Rolling embedding history window
GALLERY_MAX_SIZE: int = 10000             # LRU eviction above this
```

All thresholds must be empirically validated. Never hardcode without a benchmark result to justify the value. Threshold changes require a documented re-benchmark.

---

## 15. Architectural Horizon (Locked Future)

These capabilities are **not in scope for current phases** but are named here so that present data structure decisions do not foreclose them.

**Anomaly Detection:**  
Once baseline behavioral patterns exist in temporal memory, statistical deviation becomes detectable automatically. Event types and movement history schemas are designed to support this. No anomaly logic in Phase 1–4, but schemas must not prevent it.

**Platform API:**  
God's Eye will eventually expose a query API for external consumer systems (Sentinel AI and others). Phase 4 event and graph stores must be queryable by an authenticated external caller. This shapes the Phase 4 storage choice — the store must support external reads, not just internal pipeline writes.

**Online Identity Refinement:**  
EMA embedding updates are the first step. The architectural horizon includes more sophisticated online learning for long-lived identities. Phase 2 EMA implementation must preserve embedding history in a structure compatible with future model fine-tuning.

**Predictive Tracking:**  
Phase 3 camera graph transition priors enable prediction ("likely to appear on Camera 4 in ~8s"). This is not implemented in Phase 3 but the `CameraNode.transition_priors` field must be populated and maintained.

---

## 16. Relationship to Sentinel AI

Sentinel AI and God's Eye are separate, independently developed systems.

| System | Focus |
|--------|-------|
| Sentinel AI | Surveillance operations, face recognition, alerting, dashboards, monitoring |
| God's Eye | Identity persistence, spatial awareness, temporal memory, situational reasoning |

**Integration Protocol (Provisional):**

Integration occurs only after both systems are independently mature and have passed their respective phase gate criteria.

The provisional integration contract:

```
Sentinel AI → God's Eye:
  Pushes: Track events, Face recognition results, Alert triggers
  Format: Events conforming to God's Eye Event schema
  Transport: Message queue (specific technology: ADR pending)

God's Eye → Sentinel AI:
  Exposes: Identity graph queries, Timeline queries, Active identity list
  Format: Structured JSON via authenticated REST API
  Transport: HTTP/2 with token auth
```

This contract is provisional. A formal integration ADR must be written before implementation begins.

---

## 17. Architecture Decision Records (ADR)

Each ADR is immutable once closed. Revisions create a new ADR that supersedes the old one.

---

**ADR-001 — Embedding Model: OSNet**

- **Status:** Accepted
- **Decision:** Use OSNet (via torchreid) as the baseline Re-ID embedding model.
- **Alternatives Considered:** MobileNetV3 (lighter but lower Rank-1), EfficientNet-based Re-ID (higher accuracy but 3x inference cost), AGW baseline (strong but requires heavier torchreid setup).
- **Rationale:** OSNet achieves Rank-1 ≥ 0.88 on Market-1501 at < 5ms inference per crop on reference GPU. It is natively supported by torchreid, maintaining Rule 2 (Simplicity). MobileNetV3 was rejected due to accuracy below Phase 2 gate threshold.
- **Revisit Trigger:** If Phase 2 gate criteria cannot be met with OSNet, or if a model achieves ≥ 2% Rank-1 improvement at equal or lower inference cost.

---

**ADR-002 — Similarity Metric: Cosine Similarity**

- **Status:** Accepted
- **Decision:** Use cosine similarity for embedding matching.
- **Alternatives Considered:** Euclidean distance (sensitive to embedding magnitude), learned metric (Siamese network — high accuracy but requires training infrastructure).
- **Rationale:** Cosine similarity is magnitude-invariant and computationally cheap (dot product after L2 norm). Performs within 1% of learned metrics on standard Re-ID benchmarks when using normalized OSNet embeddings. Euclidean rejected due to sensitivity to EMA update magnitude drift.
- **Revisit Trigger:** If false merge rate exceeds 1% threshold in Phase 2 gate testing.

---

**ADR-003 — Embedding Update: EMA**

- **Status:** Accepted
- **Decision:** Update identity embeddings using Exponential Moving Average: `embedding = α * embedding + (1-α) * new_embedding`
- **Alternatives Considered:** Static template (first embedding only — brittle to appearance change), simple mean (equal weight to all observations — drowns recent signal), online learning (high accuracy but requires training infrastructure and GPU overhead).
- **Rationale:** EMA balances stability (prevents single-frame drift) with adaptability (tracks gradual appearance changes like lighting or clothing). α=0.9 is the starting configuration; must be tuned empirically in Phase 2.
- **Revisit Trigger:** Identity drift under significant appearance change (clothing change, lighting shift) produces false negative rate > 15%.

---

**ADR-004 — Identity-Track Separation: Identity Mapping Layer**

- **Status:** Accepted
- **Decision:** Maintain a strict separation between tracker-local IDs (ByteTrack) and global identity UUIDs (God's Eye). All translation occurs in the Identity Mapping Layer (IML).
- **Rationale:** ByteTrack IDs are ephemeral — they reset on track loss. Global identities must persist across occlusions, camera transitions, and session restarts. Conflating the two namespaces produces silent identity fragmentation.
- **Revisit Trigger:** Never. This separation is architectural and must be maintained.

---

## 18. Changelog

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026-06-01 | Initial spec — vision, philosophy, phase roadmap, Re-ID decisions |
| 2.0 | 2026-06-10 | Major revision: added canonical data schemas, execution environment, observability contract, concurrency model, security architecture, testing specification, numerical phase gate criteria, ADR log, architectural horizon, development workflow template, NLQ architecture decision, Sentinel AI integration protocol |
