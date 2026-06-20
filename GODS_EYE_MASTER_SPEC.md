# GODS_EYE_MASTER_SPEC.md
**Version:** 3.0  
**Status:** Active — Source of Truth  
**Last Revised:** 2026-06-11  
**Supersedes:** v2.0

---

## Table of Contents

1. [Project Name & Mission](#1-project-name--mission)
2. [Core Philosophy & Commandments](#2-core-philosophy--commandments)
3. [What God's Eye Is NOT](#3-what-gods-eye-is-not)
4. [Canonical Data Schemas](#4-canonical-data-schemas)
5. [Architecture Overview](#5-architecture-overview)
6. [Operational Modes](#6-operational-modes)
7. [Execution Environment](#7-execution-environment)
8. [Observability Contract](#8-observability-contract)
9. [Concurrency Model](#9-concurrency-model)
10. [Security, Privacy & Compliance Architecture](#10-security-privacy--compliance-architecture)
11. [Engineering Standards](#11-engineering-standards)
12. [Development Workflow Template](#12-development-workflow-template)
13. [Testing Specification](#13-testing-specification)
14. [Project Roadmap](#14-project-roadmap)
15. [Re-ID Design Decisions](#15-re-id-design-decisions)
16. [Environmental Intelligence Design Decisions](#16-environmental-intelligence-design-decisions)
17. [Behavioral Intelligence Design Decisions](#17-behavioral-intelligence-design-decisions)
18. [Architectural Horizon (Locked Future)](#18-architectural-horizon-locked-future)
19. [Relationship to Sentinel AI](#19-relationship-to-sentinel-ai)
20. [Architecture Decision Records (ADR)](#20-architecture-decision-records-adr)
21. [Changelog](#21-changelog)

---

## 1. Project Name & Mission

**Project Name:** God's Eye

**Mission Statement:**  
God's Eye is a real-time situational awareness platform that transforms raw video streams into persistent spatial, temporal, and behavioral intelligence.

The system answers:

- Who is present?
- Where did they come from? Where did they go? What path did they follow?
- Which camera observed them?
- What events occurred over time?
- What is happening right now?
- **What is normal for this environment, and what deviates from it?**
- **What behavioral patterns exist, and are they being followed or broken?**
- **What is likely to happen next, based on learned history?**

The system evolves through six progressive layers:

```
Perception → Identity → Multi-Camera → Environmental Intelligence
→ Behavioral Pattern Detection → Temporal Memory → Situational Reasoning
→ Identity Behavioral Intelligence
```

### Architectural Identity

At its core, God's Eye constructs and maintains three interlocking graphs:

- **Identity Graph** — who exists, where they've been, how they relate
- **Spatial Graph** — how cameras connect, how zones relate physically
- **Temporal Graph** — what happened, when, in what order

The new intelligence layers add a fourth:

- **Behavioral Graph** — what patterns exist, what deviates, what predicts future state

Everything is built to serve these four graphs.

---

## 2. Core Philosophy & Commandments

### Design Rules

**Rule 1 — Benchmarks Over Assumptions**  
No optimization without measurement. No architecture without validation. Every performance and accuracy claim references a benchmark with hardware, dataset, and metric specified.

**Rule 2 — Simplicity First**  
The simplest working solution is preferred. Complexity must be justified by a benchmark or documented requirement.

**Rule 3 — Incremental Development**  
Build in strict phase order. Phase N does not begin until Phase N-1 passes all gate criteria. Never implement future-phase components early.

**Rule 4 — Modularity**  
Every subsystem is independently replaceable. Replaceable means: same typed interface in, same typed interface out. Interface changes require an ADR entry.

**Rule 5 — Observability**  
No black boxes. Every subsystem exposes its metrics in the format specified in §8. Silence from a subsystem is a failure signal.

**Rule 6 — Explicit Failure Modes**  
Every subsystem defines its failure behavior. Silent degradation is prohibited.

**Rule 7 — Learned Model Outputs Carry Uncertainty**  
Every output from a statistical or learned model must include a confidence score and confidence band. Low-confidence outputs are flagged, never silently actioned.

**Rule 8 — Explainability is Required**  
Every anomaly alert, behavioral prediction, and pattern match must include a human-readable explanation of contributing factors. "Score: 0.87" is not an acceptable output.

**Rule 9 — Operational Mode Governs Output Trust**  
The system has three operational modes (§6). Behavioral and anomaly outputs are only produced in Operational mode. The system must never produce actionable alerts from an unvalidated model.

**Rule 10 — Scene Before Identity**  
Scene-level intelligence is always built and validated before identity-level intelligence. You cannot define anomalous individual behavior without a stable model of normal scene behavior.

### Commandments

1. Benchmark first. Optimize second.
2. Never trust assumptions — measure them.
3. Never build future phases early.
4. Never overengineer. Complexity earns its place.
5. Keep every module replaceable.
6. Measure everything that moves.
7. Document every decision in the ADR log.
8. Prefer clarity over cleverness.
9. Build one layer at a time.
10. Failure modes are features. Define them explicitly.
11. No model output without a confidence score.
12. No alert without an explanation.
13. No behavioral output from an unvalidated model.

---

## 3. What God's Eye Is NOT

God's Eye is NOT:

- A face recognition project
- A dashboard project
- A React project
- A FastAPI project
- A database project
- A chatbot project
- A surveillance-first system (it is an intelligence-first system)
- A system that makes decisions about people autonomously

These may exist around the system. They are not the system itself.

God's Eye produces structured, explainable intelligence. Humans and downstream systems act on it.

---

## 4. Canonical Data Schemas

These are the inter-module contracts. All modules consume and produce these types exactly. Schema changes require an ADR entry.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PERCEPTION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    x1: float          # Top-left x (pixels, absolute)
    y1: float          # Top-left y (pixels, absolute)
    x2: float          # Bottom-right x (pixels, absolute)
    y2: float          # Bottom-right y (pixels, absolute)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class Detection:
    detection_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int                        # Unix nanoseconds — never float seconds
    bbox: BoundingBox
    confidence: float                        # [0.0, 1.0]
    class_label: str
    source_resolution: tuple[int, int]       # (width, height)


@dataclass
class Track:
    track_id: str                            # Tracker-local ID
    camera_id: str
    state: TrackState
    bbox: BoundingBox
    velocity: tuple[float, float]            # (dx, dy) pixels/frame — NEW
    first_frame_id: int
    last_frame_id: int
    lost_frame_count: int
    detection_history: list[str]             # detection_ids, capped at N


class TrackState(Enum):
    ACTIVE = "active"
    LOST = "lost"
    DEAD = "dead"


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTITY SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityStatus(Enum):
    ACTIVE = "active"
    LOST = "lost"
    PURGED = "purged"

@dataclass
class Identity:
    global_id: str
    status: IdentityStatus
    embedding: np.ndarray
    embedding_history: list[np.ndarray]      # Rolling window, max EMBEDDING_HISTORY_LEN
    first_seen_ns: int
    last_seen_ns: int
    last_camera_id: str
    camera_history: list[str]
    track_id_history: list[tuple[str, str]]  # [(camera_id, track_id)]
    confidence: float
    purge_at_ns: Optional[int]
    behavioral_profile_id: Optional[str]     # Links to BehavioralProfile (Phase 6)

# Identity State Transitions:
#   ACTIVE  → LOST    : last_seen_ns older than IDENTITY_LOST_TIMEOUT_S
#   LOST    → ACTIVE  : Re-ID match >= REID_MATCH_THRESHOLD
#   LOST    → PURGED  : current_time > purge_at_ns
#   PURGED  → (none)  : terminal — embeddings zeroed, behavioral profile flagged for audit


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class EventType(Enum):
    # Perception events
    PERSON_ENTERED_FRAME    = "person_entered_frame"
    PERSON_LEFT_FRAME       = "person_left_frame"
    # Identity events
    IDENTITY_CONFIRMED      = "identity_confirmed"
    IDENTITY_LOST           = "identity_lost"
    IDENTITY_PURGED         = "identity_purged"
    CROSS_CAMERA_TRANSITION = "cross_camera_transition"
    # Environmental events
    SCENE_LIGHTING_CHANGED  = "scene_lighting_changed"
    OCCUPANCY_THRESHOLD     = "occupancy_threshold"
    BACKGROUND_MODEL_UPDATED = "background_model_updated"
    ENVIRONMENTAL_ANOMALY   = "environmental_anomaly"
    # Behavioral events
    TRAJECTORY_ANOMALY      = "trajectory_anomaly"
    DWELL_ANOMALY           = "dwell_anomaly"
    CROWD_ANOMALY           = "crowd_anomaly"
    BEHAVIORAL_PATTERN_MATCH = "behavioral_pattern_match"
    BEHAVIORAL_PREDICTION   = "behavioral_prediction"
    # System events
    SYSTEM_MODE_CHANGED     = "system_mode_changed"
    MODEL_DRIFT_DETECTED    = "model_drift_detected"
    WARM_UP_COMPLETE        = "warm_up_complete"

@dataclass
class Event:
    event_id: str
    event_type: EventType
    global_id: Optional[str]
    camera_id: str
    zone_id: Optional[str]                   # Logical zone within camera FOV
    timestamp_ns: int
    frame_id: int
    confidence: float                        # REQUIRED — never omit
    explanation: str                         # REQUIRED — human-readable reason
    metadata: dict                           # Event-type-specific payload (flat, JSON-serializable)


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Zone:
    zone_id: str
    camera_id: str
    display_name: str
    polygon: list[tuple[float, float]]       # Normalized [0,1] coordinates
    zone_type: str                           # "entrance", "exit", "dwell", "transit", "restricted"
    expected_dwell_s: tuple[float, float]    # (min, max) — learned empirically, seeded manually


@dataclass
class CameraNode:
    camera_id: str
    display_name: str
    source_uri: str
    location_label: str
    resolution: tuple[int, int]
    fps_nominal: float
    zones: list[Zone]
    adjacent_cameras: list[str]
    transition_priors: dict[str, float]      # {target_camera_id: probability}
    overlap_regions: list[BoundingBox]


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENTAL INTELLIGENCE SCHEMAS  (Phase 3.5)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BackgroundModel:
    camera_id: str
    model_type: str                          # "gmm" | "running_average"
    version: int
    created_ns: int
    last_updated_ns: int
    warm_up_complete: bool
    frame_count: int                         # Frames used to build this model
    confidence: float                        # Model stability score [0.0, 1.0]
    # Serialized model state stored separately — this is the metadata record


@dataclass
class LightingCondition(Enum):
    DAY_BRIGHT   = "day_bright"
    DAY_NORMAL   = "day_normal"
    DAY_OVERCAST = "day_overcast"
    DUSK_DAWN    = "dusk_dawn"
    NIGHT_LIT    = "night_lit"
    NIGHT_DARK   = "night_dark"
    ARTIFICIAL   = "artificial"


@dataclass
class SceneState:
    camera_id: str
    timestamp_ns: int
    lighting_condition: LightingCondition
    occupancy_count: int                     # Current detected persons
    occupancy_density: float                 # persons / normalized area [0.0, 1.0]
    background_model_id: str
    foreground_ratio: float                  # Fraction of frame with detected motion
    is_anomalous: bool
    anomaly_confidence: float
    anomaly_explanation: str


@dataclass
class OccupancyBaseline:
    camera_id: str
    zone_id: Optional[str]
    time_bucket: str                         # "MON_09", "TUE_14" etc (day_hour)
    mean_occupancy: float
    std_occupancy: float
    sample_count: int
    last_updated_ns: int
    is_mature: bool                          # True when sample_count >= MIN_BASELINE_SAMPLES


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL INTELLIGENCE SCHEMAS  (Phase 4.5 + Phase 6)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    trajectory_id: str
    global_id: Optional[str]                # None for anonymous trajectories
    camera_id: str
    start_ns: int
    end_ns: int
    points: list[tuple[float, float, int]]   # [(norm_x, norm_y, timestamp_ns)]
    zones_visited: list[str]                 # Ordered zone_ids
    total_distance_px: float
    mean_speed_px_per_s: float
    dwell_times: dict[str, float]            # {zone_id: seconds}


@dataclass
class TrajectoryCluster:
    cluster_id: str
    camera_id: str
    centroid_path: list[tuple[float, float]] # Normalized (x,y) sequence
    member_count: int
    mean_duration_s: float
    std_duration_s: float
    last_observed_ns: int
    is_mature: bool                          # True when member_count >= MIN_CLUSTER_SAMPLES
    label: Optional[str]                     # Human-assigned label e.g. "north_to_exit"


@dataclass
class AnomalySignal:
    signal_id: str
    signal_type: str                         # "trajectory" | "dwell" | "crowd" | "behavioral"
    camera_id: str
    zone_id: Optional[str]
    global_id: Optional[str]
    timestamp_ns: int
    score: float                             # [0.0, 1.0] — higher = more anomalous
    confidence: float                        # [0.0, 1.0] — model confidence in score
    confidence_band: tuple[float, float]     # (lower, upper) 95% CI
    explanation: str                         # REQUIRED — human-readable
    contributing_factors: list[str]          # List of factor descriptions
    baseline_reference: str                  # What normal looks like (human-readable)
    suppressed: bool                         # True if confidence < ANOMALY_MIN_CONFIDENCE
    suppression_reason: Optional[str]


@dataclass
class BehavioralProfile:
    profile_id: str
    global_id: str
    created_ns: int
    last_updated_ns: int
    visit_count: int
    typical_entry_cameras: dict[str, float]  # {camera_id: frequency}
    typical_zones: dict[str, float]          # {zone_id: visit_frequency}
    typical_dwell_times: dict[str, float]    # {zone_id: mean_seconds}
    typical_trajectory_clusters: list[str]   # cluster_ids most frequently matched
    visit_time_distribution: dict[str, float] # {time_bucket: frequency}
    is_mature: bool                          # True when visit_count >= MIN_PROFILE_VISITS
    data_subject_purge_requested: bool       # GDPR erasure flag


@dataclass
class MovementPrediction:
    prediction_id: str
    global_id: str
    predicted_camera_id: str
    predicted_zone_id: Optional[str]
    predicted_arrival_ns: int
    confidence: float
    confidence_band: tuple[float, float]
    explanation: str
    basis: str                               # "trajectory_cluster" | "behavioral_profile" | "transition_prior"
```

---

## 5. Architecture Overview

```
Video Sources (RTSP / Webcam / File)
          │
          ▼
    [Ingestion Layer]                  ← Thread per source; frame queues with backpressure
          │
          ▼
    [Detection]                        ← YOLOv8 → List[Detection]
          │
          ▼
    [Tracking]                         ← ByteTrack → List[Track] (with velocity)
          │
          ├──────────────────────────────────────────────────┐
          ▼                                                  ▼
    [Identity Mapping Layer]           ← Phase 2      [Environmental Monitor]  ← Phase 3.5
    OSNet Re-ID → Identity                            Background Model
    Gallery → Events                                  Scene State → Events
          │                                                  │
          ▼                                                  ▼
    [Camera Graph]                     ← Phase 3      [Occupancy Tracker]      ← Phase 3.5
    Cross-camera identity                             Heatmaps
    Transition modeling                               Baseline builder
          │                                                  │
          └──────────────────────┬───────────────────────────┘
                                 ▼
                    [Behavioral Pattern Engine]          ← Phase 4.5
                    Trajectory extraction
                    Cluster modeling (DBSCAN)
                    Scene anomaly detection
                    Dwell analysis
                    Crowd dynamics
                                 │
                                 ▼
                    [Event Store + Temporal Memory]      ← Phase 4
                    Append-only event log
                    Movement history
                    Timeline reconstruction
                                 │
                                 ▼
                    [Identity Behavioral Profiler]       ← Phase 6
                    Per-identity profiles
                    Cross-visit analysis
                    Movement prediction
                                 │
                                 ▼
                    [Situational Reasoning Engine]       ← Phase 5
                    Structured query API
                    Anomaly aggregation
                    Prediction engine
                                 │
                                 ▼
                    [NLQ Interface]                      ← Phase 5
                    LLM + structured tool calls
```

**Data flow rule:** Each layer consumes schemas from §4 and produces schemas from §4. No private cross-boundary representations.

---

## 6. Operational Modes

The system operates in exactly one of three modes at any time. Mode governs what outputs are trusted and what alerts are produced.

### Mode Definitions

```
LEARNING_MODE
─────────────
When: New deployment, new camera added, model reset triggered, drift detected
What runs: All perception and tracking pipelines run normally
           Background models accumulate data
           Baseline occupancy samples accumulate
           Trajectory data collected but NOT clustered yet
What does NOT run: Anomaly alerts (suppressed)
                   Behavioral predictions (suppressed)
                   Identity behavioral profiling (suppressed)
Exit condition: ALL warm-up criteria met (see below)
Output: WARM_UP_COMPLETE event when transitioning to OPERATIONAL_MODE


OPERATIONAL_MODE
────────────────
When: All warm-up criteria passed, no drift detected
What runs: Full pipeline including anomaly detection, behavioral intelligence,
           predictions, and identity profiling (Phase 6+)
What alerts: All event types active
Exit condition: Drift detection trigger OR operator-forced reset
Output: Normal operation


DEGRADED_MODE
─────────────
When: Drift detected in any learned model, model validation failure,
      hardware resource constraint (GPU OOM, CPU threshold exceeded)
What runs: Perception, tracking, identity persistence only
           Anomaly and behavioral outputs: SUPPRESSED
What does NOT run: Environmental anomaly alerts
                   Behavioral predictions
                   Identity behavioral outputs
Exit condition: Operator-triggered revalidation + warm-up cycle
Output: MODEL_DRIFT_DETECTED event, SYSTEM_MODE_CHANGED event
```

### Warm-Up Criteria (Learning → Operational)

All of the following must be true before transitioning to OPERATIONAL_MODE:

| Criterion | Threshold | Config Key |
|-----------|-----------|------------|
| Background model frames | ≥ 10,000 frames per camera | `WARMUP_BG_FRAMES` |
| Occupancy baseline samples | ≥ 14 samples per time bucket (2 weeks) | `WARMUP_OCCUPANCY_SAMPLES` |
| Trajectory cluster minimum | ≥ 100 trajectories collected per camera | `WARMUP_TRAJECTORY_COUNT` |
| Background model stability | Confidence ≥ 0.85 | `WARMUP_BG_CONFIDENCE` |

Warm-up status is exposed via the metrics endpoint and the observability dashboard.

### Drift Detection Triggers (Operational → Degraded)

| Trigger | Condition |
|---------|-----------|
| Background model invalidation | Scene change detection: >40% of background model invalid over 10-minute window |
| Occupancy baseline break | Occupancy deviates >5σ from baseline for >30 minutes |
| Model confidence collapse | Background model confidence drops below 0.60 |
| Hardware resource critical | GPU memory >95% VRAM for >60 seconds |
| Operator manual trigger | Via config flag `FORCE_DEGRADED_MODE=1` |

---

## 7. Execution Environment

### Reference Hardware

| Component | Specification |
|-----------|---------------|
| CPU | 8-core x86-64, 3.0 GHz+ |
| RAM | 32 GB minimum (increased from v2 — behavioral models require additional memory) |
| GPU | NVIDIA GPU with 8 GB VRAM (CUDA 12.x) |
| Storage | SSD, 200 GB working space (behavioral model storage + trajectory history) |
| OS | Ubuntu 22.04 LTS |

### Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.11.x |
| CUDA | 12.1 |
| cuDNN | 8.9.x |
| PyTorch | 2.2.x |

### CPU-Only Degraded Mode

| Capability | GPU Mode | CPU-Only Mode |
|------------|----------|---------------|
| Detection FPS | ≥ 25 FPS @ 1080p | ≥ 6 FPS @ 720p |
| Re-ID | Full OSNet | Lightweight backbone |
| Background modeling | Real-time GMM | Running average only |
| Behavioral engine | Full | Disabled |
| Multi-stream | Up to N | 1 stream only |

CPU-only mode: `GODS_EYE_CPU_ONLY=1`. Behavioral intelligence requires GPU mode.

### Dependency Manifest

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "ultralytics>=8.0.0",
    "torchreid>=0.2.5",
    "scikit-learn>=1.4.0",         # DBSCAN, GMM, anomaly detection
    "scipy>=1.12.0",               # Statistical modeling
    "numpy>=1.26.0",
    "opencv-python>=4.9.0",
    "opencv-contrib-python>=4.9.0", # Background subtraction (MOG2, KNN)
    "prometheus-client>=0.20.0",
    "structlog>=24.0.0",
    "pydantic>=2.6.0",
    "networkx>=3.2.0",             # Camera graph, trajectory graph
    "hdbscan>=0.8.33",             # Hierarchical trajectory clustering
]
```

---

## 8. Observability Contract

### Required Metrics — All Subsystems

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

| Metric | Type |
|--------|------|
| `gods_eye_identities_active` | Gauge |
| `gods_eye_identities_lost` | Gauge |
| `gods_eye_identities_purged_total` | Counter |
| `gods_eye_reid_match_confidence` | Histogram |
| `gods_eye_gallery_size` | Gauge |

### Additional Metrics — Environmental Layer

| Metric | Type | Labels |
|--------|------|--------|
| `gods_eye_operational_mode` | Gauge (enum) | `camera_id` |
| `gods_eye_background_model_confidence` | Gauge | `camera_id` |
| `gods_eye_warmup_progress` | Gauge | `camera_id`, `criterion` |
| `gods_eye_scene_occupancy` | Gauge | `camera_id`, `zone_id` |
| `gods_eye_scene_anomaly_score` | Gauge | `camera_id` |
| `gods_eye_drift_detected_total` | Counter | `camera_id`, `trigger` |

### Additional Metrics — Behavioral Layer

| Metric | Type | Labels |
|--------|------|--------|
| `gods_eye_trajectory_clusters` | Gauge | `camera_id` |
| `gods_eye_anomaly_signals_total` | Counter | `camera_id`, `signal_type` |
| `gods_eye_anomaly_suppressed_total` | Counter | `camera_id`, `reason` |
| `gods_eye_behavioral_profiles` | Gauge | — |
| `gods_eye_prediction_accuracy` | Histogram | `basis` |

### Exposure Format

Prometheus `/metrics` endpoint on `localhost:9090` (configurable via `GODS_EYE_METRICS_PORT`).

Structured logs via `structlog` in JSON:

```json
{
  "timestamp": "<ISO8601>",
  "level": "info|warning|error",
  "subsystem": "<module_name>",
  "camera_id": "<id or null>",
  "operational_mode": "<mode>",
  "event": "<description>",
  "data": {}
}
```

---

## 9. Concurrency Model

### Pipeline Architecture

```
Camera Thread (1 per source)
    │  [FrameQueue — maxsize=FRAME_QUEUE_SIZE]
    ▼
Detection Worker Thread
    │  [DetectionQueue]
    ├──────────────────────────────┐
    ▼                             ▼
Tracking Worker Thread      Environmental Monitor Thread
    │  [TrackQueue]               │  [SceneStateQueue]
    ▼                             ▼
Identity Mapping Thread ←── Behavioral Pattern Thread
    │  [EventQueue]
    ▼
Event Writer Thread
```

### Shared State Rules

| Resource | Owner Thread | Access Rule |
|----------|--------------|-------------|
| Identity Gallery | Identity Mapping Thread | Single-writer via queue messages |
| Background Models | Environmental Monitor Thread | Single-writer; read via snapshot copy |
| Trajectory Store | Behavioral Pattern Thread | Single-writer via queue |
| Occupancy Baselines | Behavioral Pattern Thread | Single-writer via queue |
| Camera Graph | Main (init-time) | Immutable after init; read-only safe |
| Metrics registry | Prometheus client | Thread-safe direct write |
| Event log | Event Writer Thread | Append-only via EventQueue |

### Queue Overflow Policy

Drop oldest frame. Log every drop with `reason="queue_overflow"`. Frame drop rate > 5% over 60s triggers DEGRADED_MODE.

---

## 10. Security, Privacy & Compliance Architecture

### Data Minimization

- Raw video frames are **never persisted** by God's Eye.
- Embeddings are stored as opaque vectors — no names, faces, or PII at the God's Eye layer.
- Behavioral profiles store statistical patterns only — not raw trajectory data beyond the retention window.

### Identity & Model Retention

| Data Type | Default TTL | Config Key |
|-----------|-------------|------------|
| Identity (lost → purge) | 86,400s (24h) | `IDENTITY_TTL_S` |
| Raw trajectory data | 7 days | `TRAJECTORY_RETENTION_DAYS` |
| Trajectory clusters | 90 days | `CLUSTER_RETENTION_DAYS` |
| Occupancy baselines | 365 days | `BASELINE_RETENTION_DAYS` |
| Background model | Until replaced | versioned |
| Behavioral profiles | 90 days | `BEHAVIORAL_PROFILE_RETENTION_DAYS` |
| Audit log | 2 years | `AUDIT_RETENTION_DAYS` |

**Model artifact governance:** Learned models (background models, trajectory clusters, anomaly thresholds, behavioral profiles) are derived data products. They are versioned, timestamped, and governed by their own retention policies. When an identity is purged, any behavioral profile linked to that identity is flagged `data_subject_purge_requested=True` and scheduled for deletion within 72 hours.

### Access Control

- All API endpoints require token authentication (`GODS_EYE_API_TOKEN`)
- All access is logged in the audit trail
- Behavioral profile queries require elevated scope (`GODS_EYE_SCOPE=behavioral`)

### Audit Trail

Append-only `audit.jsonl`. Records:
- Every identity creation and purge
- Every behavioral profile creation, update, and deletion
- Every anomaly signal generated (including suppressed)
- Every cross-camera transition
- Every external query
- Every operational mode change

### Compliance Scope

Designed to support GDPR (Art. 5, 17, 22), CCPA, and BIPA compliance:

- **Art. 5 (data minimization):** Raw video not retained; embeddings are pseudonymous
- **Art. 17 (right to erasure):** `data_subject_purge_requested` flag triggers full profile deletion
- **Art. 22 (automated decision-making):** God's Eye produces signals, not decisions. No autonomous action taken by the system. Human review required for all anomaly alerts.
- **BIPA:** Biometric data (embeddings) subject to configurable retention limits

Deployers set policy via configuration. God's Eye provides the mechanisms.

---

## 11. Engineering Standards

All code must:

- Use type hints (`mypy --strict` enforced)
- Be modular — one class, one responsibility
- Be independently testable with mocked interfaces
- Be benchmarkable via `benchmarks/benchmark_<module>.py`
- Use structured logging via `structlog`
- Avoid hidden globals
- Prefer composition over inheritance

### File Structure

```
gods_eye/
├── schemas/              # §4 schemas — no imports from other gods_eye modules
├── ingestion/            # Camera threads, frame queues
├── detection/            # YOLOv8 wrapper
├── tracking/             # ByteTrack wrapper + velocity estimation
├── reid/                 # OSNet, embedding extraction
├── identity/             # IML, gallery, state machine
├── camera_graph/         # CameraNode topology (Phase 3)
├── environmental/        # Background models, scene state, occupancy (Phase 3.5)
│   ├── background/       # GMM, running average background modeling
│   ├── occupancy/        # Heatmaps, baseline builder
│   └── lighting/         # Lighting condition classifier
├── behavioral/           # Trajectory, clustering, anomaly detection (Phase 4.5)
│   ├── trajectory/       # Extraction, normalization
│   ├── clustering/       # DBSCAN / HDBSCAN trajectory clustering
│   ├── anomaly/          # Anomaly scoring, suppression, explainability
│   └── crowd/            # Crowd dynamics
├── identity_profiler/    # Behavioral profiles, predictions (Phase 6)
├── events/               # Event store, event writer (Phase 4)
├── memory/               # Temporal memory, timeline (Phase 4)
├── reasoning/            # Query engine, NLQ interface (Phase 5)
├── observability/        # Prometheus metrics, structured logger
├── config/               # Config loader, defaults, env var resolution
└── benchmarks/           # Per-module benchmark harnesses
```

---

## 12. Development Workflow Template

Every task uses this template before implementation begins. No exceptions.

```markdown
## Task: [Name]

### Goal
What outcome does this task achieve?

### Implementation Plan
Step-by-step description.

### Interface
- Input: [typed schema consumed]
- Output: [typed schema produced]
- Confidence/uncertainty output: [how model uncertainty is expressed]
- Explainability output: [what explanation is generated]
- Config keys introduced: [list]

### Operational Mode Impact
- Does this task produce outputs that should be suppressed in LEARNING_MODE? [yes/no]
- Does this task affect mode transitions? [yes/no]

### Risks
What could go wrong? What is uncertain?

### Tradeoffs
What was considered and rejected? Why this approach?

### Implementation
[Code / changes]

### Changes Made
Actual changes vs. plan (document delta).

### Testing
- Unit tests: [what]
- Integration tests: [what]
- Failure injection: [what failure modes are tested]
- Benchmark: [metric, hardware, expected result]
```

---

## 13. Testing Specification

### Test Categories

| Category | Scope | Tool |
|----------|-------|------|
| Unit | Single class/function, mocked interfaces | `pytest` |
| Integration | Adjacent modules via schemas | `pytest` |
| Pipeline | End-to-end frame → event | `pytest` + synthetic stream |
| Behavioral | Trajectory clustering, anomaly detection accuracy | `pytest` + labeled dataset |
| Benchmark | Performance on reference hardware | `benchmarks/` harness |

### Test Data Sources

| Source | Use |
|--------|-----|
| MOT17 dataset | Tracking accuracy (MOTA, IDF1) |
| Market-1501 | Re-ID accuracy (Rank-1, mAP) |
| VIRAT / UCF-Crime | Behavioral anomaly detection (precision/recall) |
| Synthetic RTSP streams | Pipeline stability, concurrency, failure injection |
| Recorded real-world clips | Regression, demo reproducibility |

### Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `schemas/` | 100% |
| `identity/` | 90% |
| `environmental/` | 85% |
| `behavioral/anomaly/` | 90% — anomaly suppression logic especially |
| `detection/` | 80% |
| `tracking/` | 80% |
| `reid/` | 80% |

### Failure Injection Tests

| Scenario | Expected Behavior |
|----------|-------------------|
| RTSP stream drops | Reconnect with backoff; log ERROR; continue |
| GPU OOM | Fallback CPU mode; DEGRADED_MODE; log CRITICAL |
| Gallery overflow | LRU eviction; log WARNING |
| Background model invalidation | DEGRADED_MODE; suppress anomaly alerts; log ERROR |
| Sudden scene change (lights off) | Drift detection; DEGRADED_MODE; LEARNING_MODE re-entry |
| Identity purge during active behavioral profile | Profile flagged for deletion within 72h; audit log entry |
| Anomaly signal below confidence threshold | Signal suppressed; suppression logged; not emitted |

---

## 14. Project Roadmap

### Phase 1 — Perception Layer

**Goal:** Stable, real-time, single-camera detection and tracking.

**Deliverables:** Video ingestion (webcam, RTSP, file), YOLOv8 detection, ByteTrack tracking with velocity, threaded pipeline with backpressure, Prometheus metrics, structured logging, CPU degraded mode, benchmark harness, failure injection suite, Docker Compose demo.

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| FPS — 1080p RTSP, GPU | ≥ 25 FPS sustained 10 min |
| FPS — 720p, CPU-only | ≥ 6 FPS sustained 5 min |
| Zero unhandled exceptions | 10-minute continuous run |
| MOTA on MOT17-04 | ≥ 0.60 |
| ID switches on MOT17-04 | ≤ 150 |
| Frame drop rate | < 2% over any 60s window |
| Metrics endpoint | All 8 required metrics present and live |
| Stream recovery | Reconnect within 5s of drop |

---

### Phase 2 — Identity Persistence Layer

**Goal:** Maintain identity through occlusion and reappearance.

**Architecture:** ByteTrack ID → IML → OSNet embedding → cosine similarity → gallery → EMA update → Global Identity UUID

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Re-ID Rank-1 on Market-1501 | ≥ 0.88 |
| Identity continuity through 2s occlusion | ≥ 85% re-link rate |
| Gallery operation latency | ≤ 5ms p99 |
| False merge rate | < 1% |
| FPS regression vs Phase 1 | < 10% |

---

### Phase 3 — Multi-Camera Intelligence

**Goal:** Track identities seamlessly across cameras.

**Components:** Camera graph, transition modeling, cross-camera Re-ID, temporal sync, predictive transition hints.

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Cross-camera identity linking accuracy | ≥ 80% |
| Transition detection latency | ≤ 2s from appearance to confirmed link |
| Phase 2 regression | All Phase 2 gates still pass |

---

### Phase 3.5 — Environmental Intelligence Layer

**Goal:** Build a persistent, adaptive model of each camera's environment. The system learns what "normal" looks like before it can identify what is not normal.

**This phase is scene-level only. No identity data is used.**

**Components:**

**Background Modeling**
- Algorithm: Gaussian Mixture Model (GMM) via OpenCV `BackgroundSubtractorMOG2`
- Fallback (CPU mode): Running average background subtraction
- Model versioning: every model update increments version; previous version retained for rollback
- Warm-up: LEARNING_MODE until `WARMUP_BG_FRAMES` accumulated

**Lighting Condition Classifier**
- Input: frame brightness histogram + time-of-day
- Output: `LightingCondition` enum
- Purpose: context for anomaly scoring (nighttime occupancy baseline differs from daytime)
- Algorithm: histogram-based rule classifier (no ML required) + time-bucket lookup

**Occupancy Heatmaps**
- Accumulated detection density per camera per zone
- Updated every frame in LEARNING_MODE; used for anomaly scoring in OPERATIONAL_MODE
- Stored as 2D float arrays, downsampled to 64×64 normalized grid

**Occupancy Baseline Builder**
- Time-bucketed (day + hour): "MON_09", "TUE_14" etc.
- Statistics: mean, std per bucket, per camera, per zone
- Maturity: `is_mature = True` when `sample_count >= MIN_BASELINE_SAMPLES` (default: 14 = 2 weeks)
- Anomaly threshold: occupancy deviates > `OCCUPANCY_ANOMALY_SIGMA` std from bucket mean

**Scene Anomaly Detection**
- Input: current `SceneState` + `OccupancyBaseline`
- Output: `AnomalySignal` (type=`environmental_anomaly`) with confidence, explanation, suppression flag
- Suppressed unless `operational_mode == OPERATIONAL_MODE`
- Suppressed if confidence < `ANOMALY_MIN_CONFIDENCE` (default: 0.70)

**Drift Detection**
- Monitors background model confidence and occupancy baseline deviation
- Triggers DEGRADED_MODE on thresholds defined in §6

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Background model warm-up | ≥ 10,000 frames per camera accumulated |
| Background model confidence | ≥ 0.85 at warm-up completion |
| Occupancy baseline maturity | ≥ 14 samples per time bucket |
| Environmental anomaly precision | ≥ 0.80 on VIRAT labeled test set |
| Environmental anomaly recall | ≥ 0.70 on VIRAT labeled test set |
| Scene state latency | ≤ 10ms per frame |
| Drift detection response | DEGRADED_MODE within 60s of trigger condition |
| Zero suppression failures | No anomaly signal emitted during LEARNING_MODE |

---

### Phase 4 — Temporal Memory

**Goal:** Persist and reconstruct the full history of events and movements.

**Storage:** Graph database (Neo4j or equivalent). **Storage technology finalized in ADR before Phase 4 begins.** NLQ tool call patterns (Phase 5) must be prototyped first to inform storage query requirements.

**Components:** Append-only Event log, movement history per identity, timeline reconstruction, replay capability.

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Event write throughput | ≥ 1,000 events/sec sustained |
| Timeline query latency (last 1h) | ≤ 200ms p95 |
| Replay accuracy | 100% event ordering preserved |
| Storage growth | ≤ 1 GB per camera per 24h (pre-TTL) |

---

### Phase 4.5 — Behavioral Pattern Detection

**Goal:** Learn what behavioral patterns exist in each scene, detect deviations, and produce explainable anomaly signals.

**This phase is scene-level only. Patterns belong to the scene, not to individual identities.**

**Components:**

**Trajectory Extraction**
- Input: `List[Track]` per frame
- Output: completed `Trajectory` objects when a track ends or exceeds `TRAJECTORY_MAX_DURATION_S`
- Normalization: all coordinates normalized to [0,1] within camera FOV
- Minimum length: trajectories < `TRAJECTORY_MIN_POINTS` points are discarded

**Trajectory Clustering (HDBSCAN)**
- Algorithm: HDBSCAN (preferred over DBSCAN — handles variable cluster density)
- Feature vector: resampled normalized path (fixed N waypoints) + mean speed + duration
- Output: `TrajectoryCluster` records — one per discovered pattern
- Warm-up: LEARNING_MODE until `WARMUP_TRAJECTORY_COUNT` trajectories collected
- Maturity: cluster `is_mature = True` when `member_count >= MIN_CLUSTER_SAMPLES` (default: 20)
- Refit schedule: clusters refit every `CLUSTER_REFIT_INTERVAL_H` hours (default: 24h)

**Trajectory Anomaly Detection**
- Method: distance from nearest cluster centroid, normalized by cluster std
- Output: `AnomalySignal` (type=`trajectory_anomaly`) with score, confidence band, explanation
- Explanation format: "Trajectory deviates from nearest known pattern (cluster: north_to_exit) by 2.3σ. Unusual path toward restricted zone."
- Suppressed in LEARNING_MODE and below confidence threshold

**Dwell Time Anomaly Detection**
- Baseline: zone dwell time distribution per time bucket (mean, std)
- Anomaly: dwell time > `DWELL_ANOMALY_SIGMA` std from baseline
- Output: `AnomalySignal` (type=`dwell_anomaly`)
- Explanation format: "Person has remained in Zone B (entrance lobby) for 14.2 minutes. Typical dwell time at this hour is 0.8 ± 0.4 minutes."

**Crowd Dynamics**
- Metrics: local density, flow direction coherence, speed distribution
- Anomaly: density or flow metrics deviate from time-bucketed baseline
- Output: `AnomalySignal` (type=`crowd_anomaly`)

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Trajectory clustering stability | Cluster assignments stable over 72h window (NMI ≥ 0.85) |
| Trajectory anomaly precision | ≥ 0.78 on VIRAT + UCF-Crime test set |
| Trajectory anomaly recall | ≥ 0.72 on VIRAT + UCF-Crime test set |
| Dwell anomaly false positive rate | < 5% per 24h operational window |
| Explanation completeness | 100% of emitted signals have non-empty explanation |
| Suppression correctness | 0 signals emitted in LEARNING_MODE |
| Latency | Anomaly scoring ≤ 50ms per trajectory |

---

### Phase 5 — Situational Reasoning

**Goal:** Transform memory into queryable intelligence.

**NLQ Architecture:** LLM (Claude claude-sonnet-4-20250514) with structured tool calls into graph and event store. Stateless per query.

**Tool Definitions:**

```python
tools = [
    "get_identity_timeline",           # Full movement history for a global_id
    "search_events_by_type",           # Filter event log by type, time, camera, zone
    "get_camera_path",                 # Reconstruct cross-camera path for identity
    "list_active_identities",          # Current identities in ACTIVE state
    "query_location_at_time",          # Where was X at timestamp T?
    "get_anomaly_signals",             # Recent anomaly signals with explanations
    "get_occupancy_baseline",          # What is normal occupancy for zone/time?
    "get_trajectory_clusters",         # What movement patterns exist in a scene?
    "get_behavioral_prediction",       # Where is identity X likely to appear next?
    "get_scene_state",                 # Current SceneState for a camera
]
```

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Tool call accuracy on 50-query benchmark | ≥ 85% correct tool selection |
| Query response latency | ≤ 3s p95 for 1h history |
| NLQ factual correctness | ≥ 90% on held-out QA set |

---

### Phase 6 — Identity Behavioral Intelligence

**Goal:** Build per-identity behavioral profiles, detect cross-visit anomalies, and generate movement predictions.

**Legal prerequisite:** A documented legal basis for per-identity behavioral profiling must exist before this phase begins. The legal basis and its implications must be recorded in an ADR.

**Components:**

**Behavioral Profile Builder**
- Activates only for identities with `visit_count >= MIN_PROFILE_VISITS` (default: 5)
- Tracks: typical entry cameras, typical zones, typical dwell times, typical trajectory clusters, visit time distribution
- Update schedule: after each completed visit
- Maturity: `is_mature = True` when `visit_count >= MIN_PROFILE_VISITS`

**Cross-Visit Anomaly Detection**
- Compares current visit behavior against `BehavioralProfile`
- Output: `AnomalySignal` (type=`behavioral_pattern_match`) with explanation
- Explanation format: "Identity X's current behavior differs from their typical pattern (5 prior visits). Normally enters via Camera 1 and proceeds to Zone B. Currently entered via Camera 4 (unusual) and has not visited Zone B."

**Movement Prediction Engine**
- Basis priority: behavioral profile (highest confidence) → trajectory cluster → transition prior
- Output: `MovementPrediction` with confidence band and explanation
- Minimum confidence to emit: `PREDICTION_MIN_CONFIDENCE` (default: 0.65)

**Phase Gate Criteria:**

| Criterion | Threshold |
|-----------|-----------|
| Prediction accuracy (arrival camera) | ≥ 70% on held-out visit sequences |
| Prediction confidence calibration | Predicted confidence within 0.10 of empirical accuracy |
| Cross-visit anomaly precision | ≥ 0.80 |
| Profile build latency | ≤ 200ms per visit update |
| Erasure compliance | Profile deletion within 72h of purge request |

---

## 15. Re-ID Design Decisions

| Decision | Choice | ADR |
|----------|--------|-----|
| Embedding model | OSNet (torchreid) | ADR-001 |
| Similarity metric | Cosine similarity | ADR-002 |
| Embedding update | EMA | ADR-003 |
| Identity-track separation | IML | ADR-004 |

### Configuration Keys

```python
REID_MATCH_THRESHOLD: float = 0.75
REID_EMA_ALPHA: float = 0.9
IDENTITY_LOST_TIMEOUT_S: int = 300
IDENTITY_TTL_S: int = 86400
EMBEDDING_HISTORY_LEN: int = 50
GALLERY_MAX_SIZE: int = 10000
```

---

## 16. Environmental Intelligence Design Decisions

| Decision | Choice | ADR |
|----------|--------|-----|
| Background modeling algorithm | GMM (MOG2) with running average fallback | ADR-005 |
| Occupancy baseline granularity | Day + hour time buckets | ADR-006 |
| Lighting classification | Histogram-based rule classifier | ADR-007 |

### Configuration Keys

```python
WARMUP_BG_FRAMES: int = 10000
WARMUP_BG_CONFIDENCE: float = 0.85
WARMUP_OCCUPANCY_SAMPLES: int = 14
OCCUPANCY_ANOMALY_SIGMA: float = 3.0
ANOMALY_MIN_CONFIDENCE: float = 0.70
BG_LEARNING_RATE: float = 0.01
CLUSTER_REFIT_INTERVAL_H: int = 24
```

---

## 17. Behavioral Intelligence Design Decisions

| Decision | Choice | ADR |
|----------|--------|-----|
| Trajectory clustering algorithm | HDBSCAN | ADR-008 |
| Trajectory feature vector | Resampled path + speed + duration | ADR-009 |
| Anomaly scoring method | Distance from cluster centroid, normalized by cluster std | ADR-010 |
| Prediction basis priority | Profile → cluster → transition prior | ADR-011 |

### Configuration Keys

```python
TRAJECTORY_MIN_POINTS: int = 10
TRAJECTORY_MAX_DURATION_S: int = 300
WARMUP_TRAJECTORY_COUNT: int = 100
MIN_CLUSTER_SAMPLES: int = 20
DWELL_ANOMALY_SIGMA: float = 3.0
PREDICTION_MIN_CONFIDENCE: float = 0.65
MIN_PROFILE_VISITS: int = 5
BEHAVIORAL_PROFILE_RETENTION_DAYS: int = 90
TRAJECTORY_RETENTION_DAYS: int = 7
```

---

## 18. Architectural Horizon (Locked Future)

Capabilities named here so present data structures do not foreclose them. Not in scope for current phases.

**Federated Learning Across Deployments:** Share behavioral pattern knowledge across God's Eye deployments without sharing raw data. Trajectory cluster centroids could be shared in privacy-preserving form. Requires federated learning infrastructure — Phase 7+.

**Active Learning for Model Improvement:** Operator review of flagged anomalies feeds back into cluster and anomaly model retraining. Requires a review UI and feedback schema — Phase 7+.

**Graph Neural Network for Behavioral Modeling:** Replace statistical clustering with GNN-based trajectory and behavioral modeling for higher accuracy. Trajectory and behavioral schemas are designed to support this transition.

**Real-Time Threat Scoring:** Aggregate anomaly signals across multiple cameras into a unified scene threat score. Requires multi-camera anomaly correlation — Phase 5.5 candidate.

---

## 19. Relationship to Sentinel AI

| System | Focus |
|--------|-------|
| Sentinel AI | Face recognition, alerting, dashboards, monitoring |
| God's Eye | Identity persistence, spatial/temporal/behavioral intelligence |

**Provisional Integration Contract:**

```
Sentinel AI → God's Eye:
  Pushes: Track events, face recognition results
  Format: God's Eye Event schema
  Transport: Message queue (ADR pending)

God's Eye → Sentinel AI:
  Exposes: Identity graph, timeline queries, anomaly signals, behavioral predictions
  Format: Structured JSON
  Transport: Authenticated REST API
  Scope: Anomaly signals require elevated scope
```

Integration begins only after both systems independently pass their phase gate criteria.

---

## 20. Architecture Decision Records (ADR)

---

**ADR-001 — Embedding Model: OSNet**
- **Status:** Accepted
- **Decision:** OSNet via torchreid
- **Rejected:** MobileNetV3 (below Rank-1 gate), EfficientNet Re-ID (3x inference cost)
- **Revisit:** If Phase 2 gate fails, or a model achieves ≥2% Rank-1 improvement at equal cost

**ADR-002 — Similarity Metric: Cosine Similarity**
- **Status:** Accepted
- **Decision:** Cosine similarity (magnitude-invariant, cheap, within 1% of learned metrics on normalized OSNet embeddings)
- **Rejected:** Euclidean (magnitude-sensitive), Siamese (requires training infrastructure)
- **Revisit:** If false merge rate > 1% in Phase 2 gate

**ADR-003 — Embedding Update: EMA**
- **Status:** Accepted
- **Decision:** EMA with α=0.9
- **Rejected:** Static template (brittle), simple mean (drowns recent signal), online learning (GPU overhead)
- **Revisit:** If identity drift under appearance change produces false negative > 15%

**ADR-004 — Identity-Track Separation: IML**
- **Status:** Accepted — permanent
- **Decision:** IML separates ByteTrack ephemeral IDs from persistent global UUIDs
- **Revisit:** Never

**ADR-005 — Background Modeling: GMM (MOG2)**
- **Status:** Accepted
- **Decision:** OpenCV MOG2 (GMM-based) as primary; running average as CPU fallback
- **Rejected:** Frame differencing (too noisy, no shadow handling), deep learning background subtraction (prohibitive GPU cost at Phase 3.5)
- **Revisit:** If background model confidence < 0.85 after warm-up on real deployment cameras

**ADR-006 — Occupancy Baseline: Day+Hour Buckets**
- **Status:** Accepted
- **Decision:** 168 buckets (7 days × 24 hours). Simple, interpretable, 2-week warm-up.
- **Rejected:** Continuous temporal model (requires significantly more data and computation for marginal gain at this stage)
- **Revisit:** If baseline fails to capture intra-day patterns sufficiently (detected via false positive rate > 5%)

**ADR-007 — Lighting Classification: Histogram Rules**
- **Status:** Accepted
- **Decision:** Histogram-based rule classifier + time-of-day lookup. No ML.
- **Rationale:** Rule-based is sufficient for 7 lighting conditions and is perfectly explainable. ML classifier adds complexity without benefit here.
- **Revisit:** If classification accuracy < 85% on real deployment cameras

**ADR-008 — Trajectory Clustering: HDBSCAN**
- **Status:** Accepted
- **Decision:** HDBSCAN over DBSCAN
- **Rationale:** Real-world trajectory clusters have variable density. HDBSCAN handles this natively; DBSCAN requires careful ε tuning per scene. HDBSCAN also produces cluster stability scores useful for maturity assessment.
- **Rejected:** K-means (requires k specification, assumes spherical clusters), DBSCAN (density parameter sensitivity)
- **Revisit:** If clustering fails Phase 4.5 gate criteria

**ADR-009 — Trajectory Features: Resampled Path + Speed + Duration**
- **Status:** Accepted
- **Decision:** Resample trajectory to fixed N waypoints (default: 20), append mean speed and total duration. Normalize all spatial coordinates.
- **Rationale:** Fixed-length feature vector required for clustering. Resampling preserves path shape regardless of tracking FPS variation. Speed and duration add discrimination without requiring complex feature engineering.
- **Revisit:** If cluster quality (NMI) < 0.85 on real deployment data

**ADR-010 — Anomaly Scoring: Distance from Cluster Centroid**
- **Status:** Accepted
- **Decision:** Anomaly score = distance from nearest cluster centroid, normalized by cluster standard deviation. Score expressed in σ units.
- **Rationale:** Interpretable (σ units map to natural language explanations), computationally cheap, well-understood.
- **Rejected:** Isolation Forest (less interpretable, harder to explain), Autoencoder reconstruction error (requires training, GPU overhead)
- **Revisit:** If precision/recall fails Phase 4.5 gate

**ADR-011 — Prediction Basis Priority: Profile → Cluster → Prior**
- **Status:** Accepted
- **Decision:** For movement prediction, prefer behavioral profile (highest specificity) over trajectory cluster (scene-level) over camera transition prior (weakest signal)
- **Rationale:** More specific = higher confidence when available. Graceful fallback ensures predictions always have a basis, with confidence calibrated to specificity.
- **Revisit:** If prediction accuracy fails Phase 6 gate

---

## 21. Changelog

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026-06-01 | Initial spec |
| 2.0 | 2026-06-10 | Added canonical schemas, execution environment, observability contract, concurrency model, security architecture, testing specification, numerical phase gate criteria, ADR log, architectural horizon, development workflow template |
| 3.0 | 2026-06-11 | Major capability expansion: Phase 3.5 (Environmental Intelligence), Phase 4.5 (Behavioral Pattern Detection), Phase 6 (Identity Behavioral Intelligence). Added operational modes (Learning/Operational/Degraded), warm-up criteria, drift detection, uncertainty quantification requirement, explainability requirement, model artifact governance, behavioral schemas (Trajectory, TrajectoryCluster, AnomalySignal, BehavioralProfile, MovementPrediction, SceneState, OccupancyBaseline), Zone schema, velocity on Track, new ADRs 005-011, updated file structure, updated metrics, updated dependency manifest, compliance additions (Art. 22, erasure). |
