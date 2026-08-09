# Phase 6 — Behavioral Intelligence & Trajectory Prediction: Architectural Review & Implementation Blueprint

**Project:** God's Eye (`katharv59-hub/gods_eye`)  
**Milestone:** Phase 6 — Behavioral Intelligence & Trajectory Prediction  
**Status:** **ARCHITECTURAL BLUEPRINT & FEASIBILITY REVIEW (NO CODE IMPLEMENTED)**  
**Authoritative Basis:** `GODS_EYE_MASTER_SPEC.md` (§5, §6, §14 Phase 6), `docs/audits/PHASE_5_FINAL_AUDIT.md`, `docs/phases/phase_5/PHASE_5_ARCHITECTURE_REVIEW.md`, `PHASE_4_FREEZE.md`  
**Date:** August 8, 2026  

---

## 1. Executive Summary

This document establishes the authoritative architectural blueprint and sub-phase implementation plan for **Phase 6 — Behavioral Intelligence & Trajectory Prediction**.

Phase 6 extends the situational awareness capabilities of God's Eye from **historical observation and natural language query reasoning (Phases 1–5)** to **predictive spatial-temporal intelligence (Phase 6)**. It incorporates deferred capabilities (`get_trajectory_clusters` and `get_behavioral_prediction`), building trajectory sequence models, Markov transition networks, and behavioral anomaly detectors over the frozen Phase 4 `GraphStore` and `EventStore`.

### Key Architectural Directives:
1. **Strict Deterministic/Probabilistic Isolation:** Prediction and clustering algorithms are isolated from persistence source-of-truth tables. Probabilistic prediction outputs (`PredictionResult`) explicitly include confidence scores and probability distributions over validated evidence.
2. **Phase 1–5 Freeze Compatibility:** Zero source code changes to Phase 1–4 ingestion/perception/persistence engines or Phase 5 reasoning contracts. Phase 6 operates as an additive worker and tool layer above existing contracts.
3. **Additive Tool Scope:** Exposes the two deferred tools as Tool 9 (`get_trajectory_clusters`) and Tool 10 (`get_behavioral_prediction`) within `ToolDispatcher`.

---

## 2. Current System Baseline (Phases 1–5)

| Phase | Delivered Component | Core Authority / Interface | Freeze Status |
|---|---|---|---|
| **Phase 1** | Ingestion & Frame Pipeline | `FramePacket`, `YOLOv8Detector`, `ByteTrack` | **FROZEN** |
| **Phase 2** | Re-ID & Identity Mapping | `OSNetExtractor`, `IdentityMapper`, `IdentityGallery` | **FROZEN** |
| **Phase 3** | Multi-Camera Graph & Transitions | `CameraGraph`, `SQLiteGraphStore` | **FROZEN** |
| **Phase 3.5** | Environmental Intelligence | `EnvironmentalWorker`, `OccupancyTracker`, `BackgroundModeler` | **FROZEN** |
| **Phase 4** | Temporal Persistence Foundation | `TemporalWorker`, `SQLiteEventStore`, `TimelineEngine`, `ReplayEngine` | **FROZEN** |
| **Phase 5** | Situational Reasoning Stack | `UserQuery`, `ToolDispatcher`, `NLQPlanner`, `ReasoningEngine` | **FROZEN (343 tests)** |

---

## 3. Master Spec Phase 6 Requirements Matrix

| Requirement (§ Master Spec) | Current Support | Missing Capability | Phase 6 Action |
|---|---|---|---|
| **§5 Spatial-Temporal Trajectory Clustering** | Phase 4 `GraphStore` records transition edges | Trajectory sequence feature extractor & clustering engine | Implement `TrajectoryClusterer` & `get_trajectory_clusters` |
| **§5 Next-Location Prediction** | Phase 4 reconstructs historical visit segments | Transition probability sequence model (Markov/N-Gram) | Implement `BehavioralPredictor` & `get_behavioral_prediction` |
| **§6 Behavioral Anomaly Detection** | Phase 3.5 detects environmental anomalies | Identity route deviation & abnormal dwell time detection | Implement `BehavioralAnomalyDetector` |
| **§14 Tool 9 (`get_trajectory_clusters`)** | Intercepted as deferred in Phase 5 | Cluster centroid & route frequency extraction | Expose Tool 9 in `ToolDispatcher` |
| **§14 Tool 10 (`get_behavioral_prediction`)** | Intercepted as deferred in Phase 5 | Probabilistic next-camera/zone horizon prediction | Expose Tool 10 in `ToolDispatcher` |

---

## 4. Analysis of Deferred Capabilities

### A. Tool 9: `get_trajectory_clusters`
- **Purpose:** Extracts recurring camera-to-camera movement routes and spatial trajectory patterns across active or historical identities.
- **Required Inputs:** `camera_id`, `zone_id`, `min_cluster_size` (default: 3), `time_window_s` (default: 86,400s).
- **Algorithm Strategy:** Sequence alignment similarity over transition edges (`edge_transitions` in `GraphStore`) combined with DBSCAN/Hierarchical clustering.
- **Output Contract:** List of `TrajectoryCluster` objects (`cluster_id`, `route_sequence`, `frequency`, `mean_duration_s`, `confidence`).

### B. Tool 10: `get_behavioral_prediction`
- **Purpose:** Predicts the most probable next camera location(s) and arrival time window for a tracked identity given recent spatial history.
- **Required Inputs:** `global_id`, `current_camera_id`, `prediction_horizon_s` (default: 300s).
- **Algorithm Strategy:** Higher-order Markov Transition Matrix combined with historical transition dwell distributions built over `GraphStore`.
- **Output Contract:** `PredictionResult` (`global_id`, `current_camera_id`, `predicted_destinations: list[tuple[camera_id, probability]]`, `expected_arrival_ns`, `confidence`, `confidence_band`).

---

## 5. Deterministic vs. Probabilistic Boundary

```
Deterministic Layer (Phase 4 / Phase 5.2 Stores)
  ├─ SQLiteEventStore (Raw logged events)
  └─ SQLiteGraphStore (node_identities, edge_transitions, edge_observations)
        │
        ▼ (Deterministic Evidence Extraction)
Phase 6 Feature Extractor (Camera transition sequences & dwell statistics)
        │
        ▼ (Probabilistic Inference Boundary)
Phase 6 Prediction & Clustering Engines
  ├─ TrajectoryClusterer (DBSCAN / Sequence Alignment)
  └─ BehavioralPredictor (1st/2nd Order Markov Transition Matrix)
        │
        ▼ (Explicit Confidence & Uncertainty Payload)
ReasoningEngine & ToolDispatcher (Exposes Evidence-Backed Tool 9 & 10 Results)
```

- **Rule of Truth:** Predictions MUST NOT alter persistent historical observations. A prediction of `cam_02` at $T+60\text{s}$ remains a probabilistic expectation (`probability=0.82`) until confirmed by a physical detector observation.

---

## 6. Proposed Phase 6 Data Contracts & Schemas

### `gods_eye/schemas/behavioral.py`

```python
@dataclass
class TrajectoryCluster:
    """Clustered spatial route sequence pattern (§5 Master Spec)."""

    cluster_id: str
    camera_sequence: list[str]
    occurrence_count: int
    mean_duration_s: float
    confidence: float
    sample_global_ids: list[str]


@dataclass
class PredictedDestination:
    """Candidate future location prediction (§5 Master Spec)."""

    camera_id: str
    probability: float  # [0.0, 1.0]
    expected_arrival_ns: int
    confidence_band: tuple[float, float]


@dataclass
class PredictionResult:
    """Probabilistic next-location prediction envelope (§5 Master Spec)."""

    global_id: str
    current_camera_id: str
    timestamp_ns: int
    predictions: list[PredictedDestination]
    overall_confidence: float
    evidence_count: int
    explanation: str
```

---

## 7. Real-Time Computation & Concurrency Model

- **Asynchronous Worker Execution:** Behavioral feature extraction and Markov transition matrix updates execute asynchronously inside `BehavioralWorker(threading.Thread)`.
- **Non-Blocking Ingestion Guarantee:** Perceptual frame processing (`DetectionWorker`, `TrackingWorker`, `IdentityWorker`) MUST NOT wait on behavioral prediction queries.
- **Query Latency Target:** Deterministic local transition matrix lookup and prediction horizon evaluation execute in $p95 \le 50\text{ ms}$.

---

## 8. Sub-Phase Implementation Roadmap

```
Phase 6.1: Behavioral Data Contracts & Schemas (gods_eye/schemas/behavioral.py)
    │
Phase 6.2: Trajectory Sequence & Feature Extractor (gods_eye/behavioral/extractor.py)
    │
Phase 6.3: Trajectory Clustering Engine (gods_eye/behavioral/clustering.py)
    │
Phase 6.4: Markov Behavioral Predictor (gods_eye/behavioral/predictor.py)
    │
Phase 6.5: ToolDispatcher & NLQPlanner Tool 9 & 10 Integration
    │
Phase 6.6: Integration Testing & Performance Benchmarks (tests/test_behavioral_*.py)
```

### Detailed Sub-Phase Breakdown:
1. **Sub-Phase 6.1 — Contracts & Schemas:** Create `gods_eye/schemas/behavioral.py` exporting `TrajectoryCluster`, `PredictedDestination`, `PredictionResult`.
2. **Sub-Phase 6.2 — Feature Extraction:** Create `gods_eye/behavioral/extractor.py` extracting transition sequences and temporal dwell distributions from `GraphStore`.
3. **Sub-Phase 6.3 — Trajectory Clustering:** Create `gods_eye/behavioral/clustering.py` implementing DBSCAN sequence clustering for route pattern discovery.
4. **Sub-Phase 6.4 — Behavioral Predictor:** Create `gods_eye/behavioral/predictor.py` implementing Markov transition modeling for next-camera prediction.
5. **Sub-Phase 6.5 — Tool Layer Integration:** Update `ToolDispatcher` (`gods_eye/reasoning/tools.py`) and `RuleBasedPlanner` (`gods_eye/reasoning/planner.py`) to activate Tools 9 & 10.
6. **Sub-Phase 6.6 — Benchmarks & Verification:** Create unit/integration tests and benchmark suite validating prediction accuracy ($\ge 80\%$) and zero regression across 343 baseline tests.

---

## 9. Phase 6 Engineering Performance Gates

| Gate Metric | Target Threshold | Measurement Method |
|---|---|---|
| **Prediction Accuracy ($k=1$)** | $\ge 80.0\%$ | Top-1 predicted next camera vs. actual ground truth transition |
| **Prediction Latency ($p95$)** | $\le 50.0\text{ ms}$ | Benchmark execution over 100 transition predictions |
| **Clustering Latency ($p95$)** | $\le 200.0\text{ ms}$ | Trajectory clustering over 1,000 transition edges |
| **Phase 1–5 Regression** | **0 regressions** | Full `pytest` suite execution (**343/343 passing**) |

---

## 10. Privacy & Institutional Security Boundary

- **No Student Profiling Data Persistence:** Behavioral models store camera IDs (`cam_01`, `cam_02`) and synthetic/hashed identity keys (`gid_...`). Zero student names, PII, or academic records are imported or persisted.
- **Access Control:** Behavioral prediction outputs inherit Phase 5 read-only RBAC controls.

---

## 11. Final Architectural Readiness Verdict

```
================================================================================
                       PHASE 6 ARCHITECTURAL VERDICT                            
================================================================================

            PHASE 6 (BEHAVIORAL INTELLIGENCE & PREDICTION):

                  APPROVED FOR IMPLEMENTATION PLANNING

- Baseline System Verification : Phase 1–5 Frozen & Verified (343 tests passing)
- Architectural Feasibility    : High (Extends Phase 4 GraphStore cleanly)
- Concurrency & Storage Impact : Zero impact on perception pipeline
- Phase 4 Freeze Compatibility : 100% Preserved (0 Phase 4 code changes required)

================================================================================
```
