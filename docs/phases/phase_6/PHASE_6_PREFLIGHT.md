# Phase 6 Pre-Flight — Architectural Contract Verification & Pre-Implementation Pre-Flight

**Project:** God's Eye (`katharv59-hub/gods_eye`)  
**Milestone:** Phase 6 Pre-Flight Contract & Architectural Verification  
**Status:** **PRE-FLIGHT AUDIT & VERIFICATION (NO CODE WRITTEN)**  
**Authoritative Basis:** `GODS_EYE_MASTER_SPEC.md` (§5, §6, §14 Phase 6), `docs/phases/phase_6/PHASE_6_ARCHITECTURE_REVIEW.md`, `PHASE_4_FREEZE.md`  
**Date:** August 8, 2026  

---

## 1. Master Spec Verification Matrix

Every requirement in `GODS_EYE_MASTER_SPEC.md` pertaining to Phase 6 was verified against the architecture blueprint:

| Requirement (§ Master Spec) | Master Spec Evidence | Explicit / Derived | Blueprint Interpretation | Verdict |
|---|---|---|---|---|
| **Trajectory Clustering (§5)** | *"Group recurring spatial movement routes across camera nodes into cluster centroids"* | Explicit | Sequence alignment + DBSCAN sequence clustering | **PASS** |
| **Next-Location Prediction (§5)** | *"Predict the candidate destination camera and arrival window for active identities"* | Explicit | 1st/2nd order Markov transition matrix over `edge_transitions` | **PASS** |
| **Behavioral Anomaly Detection (§6)** | *"Detect route deviations, loitering anomalies, and abnormal camera transition durations"* | Explicit | Statistical Z-score & frequency baselines | **PASS** |
| **Tool 9 (`get_trajectory_clusters`)** | *"Tool call returning spatial route clusters and frequency statistics"* | Explicit | Expose Tool 9 in `ToolDispatcher` | **PASS** |
| **Tool 10 (`get_behavioral_prediction`)** | *"Tool call returning next location prediction, transition probability, and confidence"* | Explicit | Expose Tool 10 in `ToolDispatcher` | **PASS** |
| **Prediction Accuracy $\ge 80\%$** | — | Derived (Internal Target) | Engineering benchmark gate | **DERIVED TARGET** |
| **Prediction Latency $\le 50\text{ ms}$** | — | Derived (Internal Target) | Engineering benchmark gate | **DERIVED TARGET** |
| **Clustering Latency $\le 200\text{ ms}$** | — | Derived (Internal Target) | Engineering benchmark gate | **DERIVED TARGET** |

> **CRITICAL CLARIFICATION:** Accuracy ($\ge 80\%$) and latency ($\le 50\text{ms}$) thresholds are **Internal Engineering Targets**, not explicit Master Spec gates. The Master Spec requires trajectory discovery, next-location prediction, and behavioral anomaly detection.

---

## 2. Scope Consistency & Anomaly Detection Integration

**Audit Finding:** The initial architecture review listed Behavioral Anomaly Detection in the requirements matrix but omitted a dedicated sub-phase in the roadmap.

**Resolution & Scope Harmonization:**
Behavioral Anomaly Detection (detecting route deviations and abnormal dwell durations) is an **Explicit Requirement** under §6 of the Master Spec. It is incorporated into the Sub-Phase Roadmap as **Sub-Phase 6.4 (Behavioral Anomaly Engine)**.

---

## 3. Fine-Grained Component Classification

| Component | Technical Nature | Source of Truth / Dependency | Deterministic vs Probabilistic |
|---|---|---|---|
| `GraphStore` Edge Retrieval | Data Fetch | SQLite DB (`edge_transitions`) | **DETERMINISTIC** |
| Sequence Feature Extraction | Feature Computation | `GraphStore` Transition Records | **DETERMINISTIC** |
| Dwell Distribution Calculation | Statistical Summary | Sample means / standard deviations | **STATISTICAL** |
| Route Sequence Clustering | Pattern Discovery | Levenshtein / Edit Distance + DBSCAN | **STATISTICAL / CLUSTERING** |
| Markov Transition Matrix | Transition Likelihood | N-Gram count matrices | **PROBABILISTIC** |
| Next-Location Prediction | Inference | Markov state transition probabilities | **PROBABILISTIC** |
| Anomaly Scoring | Anomaly Metric | Statistical Z-Score deviation from baseline | **STATISTICAL** |
| Confidence Band Calculation | Uncertainty Bound | Mean confidence $\pm 0.05$ | **DETERMINISTIC RULE** |
| Natural Language Explanation | Text Formatting | Deterministic template / LLM boundary | **DETERMINISTIC TEMPLATE** |

---

## 4. Minimum Viable Trajectory Clustering Algorithm

**Evaluated Algorithms:**
1. **DBSCAN on Raw Coordinates:** Rejected (Camera topology is discrete graph nodes, not continuous 2D Cartesian coordinates).
2. **K-Means on Route Strings:** Rejected (Requires fixed vector length; routes vary in length).
3. **Graph Route Edit Distance (Levenshtein) + Agglomerative / DBSCAN Clustering:** **SELECTED**.

**Recommended Strategy:**
- Treat trajectories as sequences of camera IDs (e.g. `["cam_01", "cam_02", "cam_05"]`).
- Compute pairwise Normalized Edit Distance between sequences.
- Apply DBSCAN with distance threshold $\epsilon = 0.3$ and `min_samples = 3`.
- **Deterministic Reproducibility:** Sort unique camera routes lexicographically prior to cluster assignment.

---

## 5. Markov Predictor Architecture & Cold-Start Strategy

**Selected Strategy:** 1st-Order Markov Chain with 2nd-Order Fallback and Unseen Route Handling.

- **State Representation:** Current camera node $C_t$ (1st Order) or tuple $(C_{t-1}, C_t)$ (2nd Order).
- **Probability Calculation:**
  $$P(C_{t+1} \mid C_t) = \frac{\text{Count}(C_t \to C_{t+1})}{\sum_{C'} \text{Count}(C_t \to C')}$$
- **Cold-Start / Sparse History Handling:**
  - If $C_t$ has $< 3$ historical transitions: Fall back to static camera graph adjacency or return `QueryStatus.INSUFFICIENT_EVIDENCE`.
  - **Zero Fabrication:** If an identity has zero recorded transitions, the predictor returns `QueryStatus.INSUFFICIENT_EVIDENCE` with zero probability.

---

## 6. Contract Review & Distinction: Probability vs. Confidence

- **Probability:** Mathematical likelihood of identity moving to camera $C_{t+1}$ given current location ($\sum P_i = 1.0$).
- **Confidence:** Provenance strength of the underlying historical evidence supporting the transition matrix (e.g. based on sample count and Re-ID observation confidences).

### Refined Contracts (`gods_eye/schemas/behavioral.py`)

```python
@dataclass
class PredictedDestination:
    """Candidate next location prediction with distinct probability and confidence."""

    camera_id: str
    probability: float  # [0.0, 1.0] Transition likelihood
    confidence: float   # [0.0, 1.0] Evidence provenance confidence
    confidence_band: tuple[float, float]
    expected_arrival_ns: int


@dataclass
class PredictionResult:
    """Probabilistic next-location prediction envelope (§5 Master Spec)."""

    global_id: str
    current_camera_id: str
    timestamp_ns: int
    predictions: list[PredictedDestination]
    overall_confidence: float
    evidence_count: int
    model_version: str
    explanation: str
```

---

## 7. Trajectory Cluster Privacy Minimization

**Privacy Directive:** To minimize unnecessary identity linkability, `TrajectoryCluster` will omit raw lists of individual `global_id` UUIDs.

```python
@dataclass
class TrajectoryCluster:
    """Clustered spatial route sequence pattern with identity privacy minimization."""

    cluster_id: str
    camera_sequence: list[str]
    occurrence_count: int
    unique_identity_count: int  # Aggregate count instead of raw list of UUIDs
    mean_duration_s: float
    confidence: float
```

---

## 8. Ephemeral vs. Derived Persistence Model

Phase 6 operates **100% read-only with respect to Phase 1–4 persistent source-of-truth tables**.

| State Item | Classification | Storage Location | Recomputable? |
|---|---|---|---|
| Transition Counts Matrix | **DERIVED** | In-Memory Cache (Rebuilt from `GraphStore`) | Yes |
| Trajectory Clusters | **DERIVED** | In-Memory Cache | Yes |
| Scene Occupancy Baselines | **DERIVED** | In-Memory `EnvironmentalWorker` | Yes |
| Prediction Results | **EPHEMERAL** | Memory-bound query return payload | Yes |

**Phase 4 Storage Conflict:** **NONE**. Zero Phase 4 table modifications required.

---

## 9. Non-Blocking Concurrency Boundary

- `BehavioralWorker` consumes temporal observations from an asynchronous, non-blocking queue (`PipelineQueue`).
- **Perception Non-Blocking Guarantee:** Perception worker threads (`DetectionWorker`, `TrackingWorker`, `IdentityWorker`) MUST NEVER wait on behavioral model updates or clustering computations.

---

## 10. Final Behavioral Queue Admission Policy

### Audit of `DROP_OLDEST` vs. Priority Admission
Naive `DROP_OLDEST` under queue backpressure can cause arbitrary loss of critical **cross-camera transition boundaries** (`CROSS_CAMERA_TRANSITION`) or identity state events (`IDENTITY_CONFIRMED`, `IDENTITY_LOST`), distorting transition probability matrices $P(C_{t+1} \mid C_t)$ and route sequences.

### Selected Policy: Priority-Based High-Watermark Admission
To preserve non-blocking execution while guaranteeing route topology integrity, `BehavioralWorker` enforces a **Priority-Based High-Watermark Admission Policy**:

1. **Observation Classification:**
   - **CRITICAL (Priority 1):** `CROSS_CAMERA_TRANSITION`, `IDENTITY_CONFIRMED`, `IDENTITY_LOST`, `ENVIRONMENTAL_ANOMALY`. Must NEVER be dropped under normal backpressure.
   - **PERIODIC (Priority 2):** Repeated frame-level presence observations of the same identity at the same camera node.
2. **Queue High Watermark (80% Capacity):**
   - When queue fill ratio $\le 80\%$: Enqueue all incoming observations.
   - When queue fill ratio $> 80\%$: Apply **Temporal Downsampling / Thinning** on Priority 2 (periodic) observations, preserving all Priority 1 (critical transition) items.
3. **Queue Overflow (100% Capacity):**
   - If queue reaches $100\%$ capacity despite thinning Priority 2 items, drop Priority 2 items (`DROP_PERIODIC`). Log warning metric `gods_eye_behavioral_queue_drops_total{priority="periodic"}`.
4. **Perception Non-Blocking Guarantee:** Queue enqueue operates via non-blocking `put_nowait()`. Perception pipeline threads NEVER block.
5. **Reproducibility Invariant:** In the absence of queue overflow, model construction from historical `GraphStore` edges is 100% deterministic and reproducible.

---

## 11. Corrected Sub-Phase Roadmap

```
Sub-Phase 6.1: Behavioral Schemas & Contracts (gods_eye/schemas/behavioral.py)
    │
Sub-Phase 6.2: Trajectory Sequence & Feature Extractor (gods_eye/behavioral/extractor.py)
    │
Sub-Phase 6.3: Spatial Trajectory Clustering Engine (gods_eye/behavioral/clustering.py)
    │
Sub-Phase 6.4: Behavioral Anomaly Detection Engine (gods_eye/behavioral/anomaly.py)
    │
Sub-Phase 6.5: Markov Behavioral Predictor (gods_eye/behavioral/predictor.py)
    │
Sub-Phase 6.6: Tool 9 & 10 Integration & Reasoning Dispatcher Activation
    │
Sub-Phase 6.7: Integration Testing, Hardening & Benchmarking (343 Baseline Tests + Phase 6 Tests)
```

---

## 12. Final Implementation Decision

```
================================================================================
                    PHASE 6.1 IMPLEMENTATION STATUS                             
================================================================================

                                  GO

- Master Spec Requirements Alignment : 100% Verified
- Queue Admission & Concurrency Policy: Priority High-Watermark (Non-Blocking)
- Probabilistic / Deterministic Boundary: Formally Defined
- Privacy & Identity Minimization     : Verified (Aggregate identity counts)
- Phase 4 Storage Freeze Compatibility: 100% Preserved (0 persistent DB changes)
- Implementation Roadmap Status       : Harmonized (Sub-Phases 6.1–6.7)

Approved First Task:
Sub-Phase 6.1 — Behavioral Schemas & Contracts (gods_eye/schemas/behavioral.py)

================================================================================
```
