# Phase 7 — Situational Intelligence & Probabilistic Reasoning: Architecture & Specification Review

## Executive Summary

- **Repository Baseline Commit**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Repository State**: Clean working tree, **429 / 429 passing tests**, 0 source code changes.
- **Review Mandate**: READ-ONLY architectural review and specification for Phase 7. Zero runtime code, zero database migrations, zero speculative feature implementations.
- **Final Verdict**: **CONDITIONALLY READY**

---

## 1. Current System Baseline (as of `v0.6.0-phase6-freeze`)

The God's Eye system currently implements a six-layer architecture spanning perception through behavioral prediction. Below is the verified source mapping for each layer:

```
[Ingestion Layer] (gods_eye/ingestion/stream.py)
       │
       ▼
[Detection Layer] (gods_eye/detection/yolo.py) → List[Detection]
       │
       ▼
[Tracking Layer] (gods_eye/tracking/bytetrack.py) → List[Track] (with velocity)
       │
       ├───────────────────────────────────────────────────────┐
       ▼                                                       ▼
[Identity Mapping Layer] (gods_eye/identity/mapper.py)    [Environmental Monitor] (gods_eye/environmental/)
  OSNet Re-ID (gods_eye/reid/osnet.py) → Gallery            Background Subtraction (MOG2)
  Global Identity UUID Assignment                           Occupancy Heatmaps & Baseline Builder
       │                                                       │
       ▼                                                       ▼
[Camera Graph] (gods_eye/camera_graph/)                  [Environmental Anomaly Detector]
  Cross-camera topology & transitions                     Scene State & Anomaly Signals
       │                                                       │
       └──────────────────────────┬────────────────────────────┘
                                  ▼
                     [Event Store & Persistence] (gods_eye/events/, gods_eye/memory/)
                       SQLiteEventStore (append-only event log)
                       SQLiteGraphStore (identity transition edges & spatial topology)
                                  │
                                  ▼
                     [Temporal Memory & Timeline Engine] (gods_eye/memory/timeline.py)
                       Identity, Camera, Zone Timelines
                       Deterministic Replay Engine (gods_eye/memory/replay.py)
                                  │
                                  ▼
                     [Behavioral Intelligence Layer] (gods_eye/behavioral/)
                       TrajectorySequenceExtractor (extractor.py)
                       TrajectoryClusteringEngine (clustering.py) — Levenshtein edit distance
                       BehavioralAnomalyDetector (anomaly.py)
                       MarkovBehavioralPredictor (predictor.py) — 1st-order Markov chain
                                  │
                                  ▼
                     [Reasoning & NLQ Interface Layer] (gods_eye/reasoning/)
                       ToolDispatcher (tools.py) — Tools 1 through 10
                       RuleBasedPlanner / NLQPlanner (planner.py)
                       ReasoningEngine (engine.py) — Structured tool orchestrator
```

### Verified File & Package Mapping

| Pipeline Layer | Primary Source Files | Primary Output Schemas |
|---|---|---|
| Ingestion & Perception | `gods_eye/ingestion/stream.py`, `gods_eye/detection/yolo.py` | `Detection`, `BoundingBox` |
| Tracking & Re-ID | `gods_eye/tracking/bytetrack.py`, `gods_eye/reid/osnet.py` | `Track` |
| Identity Mapping | `gods_eye/identity/mapper.py`, `gallery.py`, `state.py` | `Identity`, `IdentityState` |
| Persistence & Memory | `gods_eye/events/event_store.py`, `gods_eye/memory/graph_store.py` | `Event`, `TemporalObservation` |
| Timeline & Replay | `gods_eye/memory/timeline.py`, `replay.py` | `IdentityTimeline`, `CameraTimeline` |
| Behavioral Intelligence | `gods_eye/behavioral/extractor.py`, `clustering.py`, `anomaly.py`, `predictor.py` | `TrajectoryCluster`, `PredictedDestination`, `BehavioralAnomalyResult` |
| Reasoning Layer | `gods_eye/reasoning/tools.py`, `planner.py`, `engine.py` | `UserQuery`, `ToolCall`, `Evidence`, `ReasoningResult` |

---

## 2. Master Spec Phase 7 Requirements

An exhaustive audit of `GODS_EYE_MASTER_SPEC.md` (v3.0) yields the following explicit Phase 7+ requirements. Requirements not found in the Master Spec are explicitly classified as `NOT SPECIFIED BY MASTER SPEC`.

| Requirement | Master Spec Section | Existing Code Support | Missing Implementation | Dependencies | Verification Method |
|---|---|---|---|---|---|
| **Federated Pattern Sharing** | §18 (Architectural Horizon) | `TrajectoryCluster` schema | Cross-deployment cluster aggregation & privacy protocols | Phase 6 Clustering Engine | Synthetic multi-node cluster alignment tests |
| **Active Learning & Feedback** | §18 (Architectural Horizon) | `AnomalySignal.suppressed` | Operator review schema, feedback ingestion, retraining trigger | Phase 5/6 Anomaly Detectors | Precision/recall delta on feedback loops |
| **Sentinel AI Integration** | §19 | REST API endpoints, Event schema | External message queue transport & authentication handshake | Phase 5 Reasoning API | Mock REST API contract test suite |
| **GNN Behavioral Modeling** | §18 | `TrajectorySequence` | Graph Neural Network trajectory embeddings | GraphStore transition edges | NMI comparison vs edit-distance DBSCAN |
| **Cross-Camera Threat Scoring** | §18 (candidate §5.5) | `BehavioralAnomalyResult` | Threat aggregation engine across concurrent streams | Phase 3.5 & Phase 6.4 Anomaly Detectors | Multi-camera anomaly injection benchmark |

> [!IMPORTANT]
> The roadmap in `GODS_EYE_MASTER_SPEC.md` §14 explicitly defines Phases 1 through 6. Phase 7 represents the **Situational Intelligence & Deployment Scale Layer** defined in §18 (Architectural Horizon) and §19 (System Integrations).

---

## 3. Phase 6 → Phase 7 Dependency Audit

| Component | Interface / Contract | Stability | Status & Mismatch Notes |
|---|---|---|---|
| `TrajectorySequence` | Dataclass (`extractor.py`) | **Stable** | Ready for Phase 7 consumption. Collapses consecutive observations cleanly. |
| `TrajectoryCluster` | Schema (`schemas/behavioral.py`) | **Stable** | Aggregate identity privacy preserved (`unique_identity_count`). |
| `BehavioralAnomalyResult` | Schema (`schemas/behavioral.py`) | **Stable** | Contains score, threshold, nearest cluster ID, and explanation. |
| `PredictedDestination` | Schema (`schemas/behavioral.py`) | **Stable** | Stores candidate camera, transition probability, and expected arrival time. |
| `PredictionResult` | Schema (`schemas/behavioral.py`) | **Stable** | Aggregate prediction payload containing candidates & explanation. |
| Tool 9 (`get_trajectory_clusters`) | `ToolDispatcher` method | **Stable** | Dispatches to `TrajectoryClusteringEngine`. Read-only, privacy-preserved. |
| Tool 10 (`get_behavioral_prediction`) | `ToolDispatcher` method | **Stable** | Dispatches to `MarkovBehavioralPredictor`. Clears predictor state before fitting. |
| `ReasoningEngine` | Orchestrator (`engine.py`) | **Stable** | Dispatches queries to planner and tools, aggregates evidence into `ReasoningResult`. |
| `NLQPlanner` | Intent Router (`planner.py`) | **Stable** | Rule-based keyword matching routing queries to Tools 1–10. |
| `SQLiteGraphStore` | Persistence (`graph_store.py`) | **Stable** | Stores spatial nodes and directional transition edges with weights and timestamps. |

---

## 4. Critical Limitations from Phase 6

### A. Synthetic Prediction Validation
- **Current Result**: 80.0% Top-1 accuracy, 100.0% Top-3 accuracy on 500 held-out temporal transitions.
- **Crucial Limitation**: Evaluated strictly against a **Deterministic Synthetic Transition Graph**. It verifies the temporal transition math of `MarkovBehavioralPredictor`, but does NOT prove real-world CCTV performance.
- **Phase 7 Requirement**: Phase 7 must not assume 80% real-world accuracy. It must handle low-confidence predictions gracefully.

### B. Tool 9 Clustering Complexity
- **Current Latency**: 3.62 ms @ $N=100$, 150.11 ms @ $N=1,000$, 226.40 ms @ $N=10,000$.
- **Crucial Limitation**: Pairwise Levenshtein sequence edit distance scales quadratically ($O(N^2 \cdot L^2)$).
- **Phase 7 Requirement**: Phase 7 queries calling Tool 9 must enforce explicit time-window bounds (`start_ns`, `end_ns`) to constrain historical candidate sizes.

### C. Tool 10 Markov Architecture
- **Current Model**: First-order Markov chain ($P(C_{t+1} \mid C_t)$).
- **Crucial Limitation**: Remembering only the current camera ignores prior path history, velocity, and time-of-day context.
- **Phase 7 Requirement**: Phase 7 situational reasoning must combine 1st-order Markov priors with multi-hop temporal timeline context.

### D. Identity Uncertainty
- **Crucial Limitation**: Global identity UUIDs (`global_id`) are derived via OSNet embeddings and Cosine Similarity thresholds ($0.75$). Occlusion or extreme lighting can cause false merges or identity splits.
- **Phase 7 Requirement**: Phase 7 reasoning must treat `global_id` as a probabilistic hypothesis rather than absolute ground truth.

### E. Temporal Uncertainty & Gaps
- **Crucial Limitation**: Disjunct camera fields of view leave unmonitored spatial gaps. Open visits have undetermined exit timestamps.
- **Phase 7 Requirement**: Phase 7 must accommodate open-ended visits, missing observation intervals, and out-of-order event ingestion.

---

## 5. Phase 7 Architectural Options

We evaluate four structural options for Phase 7:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Option A: Deterministic Tool Extension (Rule-Based Only)               │
│ Option B: Fully Probabilistic Temporal Engine (Bayesian Network)         │
│ Option C: Pure Graph-Based Situational Reasoning (Graph DB + Cypher)     │
│ Option D: Hybrid Deterministic Core + Probabilistic Risk Evaluator      │
└─────────────────────────────────────────────────────────────────────────┘
```

| Evaluation Dimension | Option A (Deterministic) | Option B (Probabilistic) | Option C (Graph-Based) | **Option D (Hybrid - RECOMMENDED)** |
|---|---|---|---|---|
| **Architecture** | Extend RuleBasedPlanner with more static tools | Dynamic Bayesian Network across time steps | Persistent Graph DB traversal queries | Deterministic Tool Layer + Probabilistic Hypothesis Evaluator |
| **Advantages** | Zero latency overhead, 100% bit-exact determinism | Rigorous uncertainty modeling | Fast path/graph pattern matching | Strict determinism for facts; calibrated confidence for hypotheses |
| **Disadvantages** | Cannot evaluate complex multi-factor scene hypotheses | Extremely high computational complexity ($O(V^T)$) | Requires external database dependency (Neo4j) | Requires careful separation of facts vs hypotheses |
| **Complexity** | Low | Very High | High | Moderate |
| **Latency (p95)** | < 5 ms | > 500 ms | ~ 50 ms | **< 25 ms** |
| **Determinism** | 100% | Low | High | **100% for Evidence, Calibrated for Risk** |
| **Explainability** | High (static strings) | Low (black-box probabilities) | Moderate (Cypher trace) | **High (Structured contributing factors)** |
| **Implementation Risk**| Low | High | High | **Low-Moderate** |

---

## 6. Recommended Architecture: Option D (Hybrid Layer)

**Selected Architecture**: **Option D — Hybrid Deterministic Core + Probabilistic Situational Risk Evaluator**.

### Why Option D is Selected
1. **Master Spec Compliance**: Preserves Rule 1 (benchmarks over assumptions), Rule 7 (uncertainty scores required), Rule 8 (explainability required), and Rule 11 (no uncalibrated model outputs).
2. **Determinism Preservation**: Query execution, evidence retrieval, timeline reconstruction, and tool dispatches remain 100% deterministic and bit-exact.
3. **Uncertainty & Risk Handling**: Probabilistic evaluations (e.g., identity ambiguity, multi-camera trajectory anomaly risk) are encapsulated inside a dedicated `SituationalRiskEvaluator` that outputs explicit confidence bands `(lower, upper)`.
4. **Solo-Developer Feasibility**: Does not require heavy graph database infrastructure (Neo4j) or black-box neural networks. Integrates directly with existing `SQLiteEventStore` and `SQLiteGraphStore`.

---

## 7. Phase 7 Proposed Data Contracts

> [!NOTE]
> The following schemas represent architectural designs only. All proposed contracts are tagged `PROPOSED — NOT IMPLEMENTED`.

```python
# PROPOSED — NOT IMPLEMENTED
@dataclass(frozen=True)
class SituationalState:
    """Represents overall multi-camera scene state at a given timestamp."""
    timestamp_ns: int
    active_identity_count: int
    active_cameras: tuple[str, ...]
    system_mode: OperationalMode
    active_anomalies: tuple[BehavioralAnomalyResult, ...]
    overall_risk_score: float                # [0.0, 1.0]
    explanation: str

# PROPOSED — NOT IMPLEMENTED
@dataclass(frozen=True)
class Hypothesis:
    """A candidate situational hypothesis generated from evidence."""
    hypothesis_id: str
    hypothesis_type: str                     # "trajectory_deviation" | "unusual_presence" | "identity_swap"
    subject_global_id: Optional[str]
    confidence: float                        # [0.0, 1.0]
    confidence_band: tuple[float, float]     # 95% CI
    supporting_evidence_ids: tuple[str, ...]
    explanation: str

# PROPOSED — NOT IMPLEMENTED
@dataclass(frozen=True)
class RiskSignal:
    """Aggregated risk score combining environmental and behavioral anomalies."""
    signal_id: str
    timestamp_ns: int
    risk_level: str                          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    score: float                             # [0.0, 1.0]
    contributing_factors: tuple[str, ...]
    explanation: str
```

---

## 8. Deterministic / Probabilistic Boundary

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      DETERMINISTIC BOUNDARY (100% Exact)                 │
├───────────────────────────────────────────────────────────────────────────┤
│ • EventStore logs & event IDs                                             │
│ • GraphStore spatial topology & edge weights                              │
│ • Timeline reconstruction (visits, dwell times, entry/exit)              │
│ • Tool selection (Tools 1–10 + future Tools 11–12)                       │
│ • Evidence ID provenance and serialization                                │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      PROBABILISTIC BOUNDARY (Calibrated)                 │
├───────────────────────────────────────────────────────────────────────────┤
│ • Markov destination prediction probability P(C_{t+1} | C_t)              │
│ • Anomaly deviation score (distance in σ units normalized to [0,1])       │
│ • Identity Re-ID similarity match scores                                  │
│ • Multi-factor Situational Risk Evaluation                                │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        LLM BOUNDARY (Strict Scope)                        │
├───────────────────────────────────────────────────────────────────────────┤
│ • Natural Language Query intent extraction (NLQPlanner)                   │
│ • Natural Language Explanation formatting for human operators             │
│ • CANNOT: Synthesize facts, override evidence, or emit unevidenced claims  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Phase 7 Query & Situational Reasoning Flow

```
User Query ("Evaluate security risk in Zone A for person X")
       │
       ▼
[NLQPlanner / RuleBasedPlanner] ──(Intent: get_situational_risk)
       │
       ▼
[ToolDispatcher] ──(Dispatches parallel queries)
       │
       ├──────► Tool 1 (`get_identity_timeline`) ──► Fact Evidence
       ├──────► Tool 6 (`get_anomaly_signals`)    ──► Anomaly Evidence
       └──────► Tool 10 (`get_behavioral_prediction`) ──► Markov Evidence
       │
       ▼
[SituationalRiskEvaluator] ──(Combines evidence & evaluates risk score)
       │
       ▼
[EvidenceVerifier] ──(Validates non-contradictions & checks confidence threshold)
       │
       ▼
[ReasoningEngine] ──(Builds structured ReasoningResult with human explanation)
       │
       ▼
Output: ReasoningResult (Status=SUCCESS, Risk=MEDIUM, Score=0.68, Evidence=[...])
```

---

## 10. Failure & Uncertainty Model

| Failure Scenario | System Reaction | Emitted Status | Explanation Output |
|---|---|---|---|
| **Zero Trajectories / Cold Start** | Return empty predictions, fallback to spatial prior | `INSUFFICIENT_EVIDENCE` | "Insufficient historical observations for camera X." |
| **Contradictory Identity Timeline** | Flag timeline conflict, retain both evidence logs | `AMBIGUOUS` | "Conflicting observations detected across Camera 2 and 4." |
| **Low Prediction Confidence (< 0.65)** | Suppress candidate alert | `SUPPRESSED` | "Prediction confidence (0.42) below threshold (0.65)." |
| **Unknown Camera Parameter** | Reject prediction query | `INVALID_PARAMETER` | "Camera ID 'cam_99' not found in spatial graph." |
| **LLM Provider Failure / Timeout** | Fallback to `RuleBasedPlanner` regex matcher | `SUCCESS` (Fallback) | "Routed via rule-based keyword planner." |

---

## 11. Performance Architecture Targets

| Performance Dimension | Spec Category | Scale $N=100$ | Scale $N=1,000$ | Scale $N=10,000$ |
|---|---|---|---|---|
| **Tool 9 Clustering Latency** | Internal Engineering Target | < 5 ms | < 160 ms | < 250 ms |
| **Tool 10 Prediction Latency** | Internal Engineering Target | < 1 ms | < 10 ms | < 80 ms |
| **Planner Intent Routing** | Internal Engineering Target | < 0.05 ms | < 0.05 ms | < 0.05 ms |
| **Reasoning Engine E2E** | Master Spec Gate (§14 Phase 5) | < 50 ms | < 200 ms | < 500 ms |
| **Timeline Query Latency** | Master Spec Gate (§14 Phase 4) | < 10 ms | < 50 ms | < 200 ms (p95) |

---

## 12. Observability Specifications

Phase 7 requires the following Prometheus metrics added to `MetricsRegistry`:

- `gods_eye_situational_risk_evaluations_total` (Counter: `["risk_level", "status"]`)
- `gods_eye_situational_risk_score` (Histogram: `buckets=(0.1, 0.3, 0.5, 0.7, 0.85, 0.95)`)
- `gods_eye_reasoning_hypothesis_count` (Histogram: `buckets=(1, 2, 5, 10)`)
- `gods_eye_evidence_verification_failures_total` (Counter: `["reason"]`)

---

## 13. Security & Privacy Architecture

1. **Identity Privacy Minimization**: All Phase 7 situational risk and clustering APIs must expose aggregate counts (`unique_identity_count`). Raw `global_id` UUID lists must never be exposed over unauthenticated public boundaries.
2. **Prompt Injection Protection**: All string inputs to `NLQPlanner` must be sanitized against regex delimiter escapes and system prompt overrides.
3. **Audit Trail**: Every situational query execution and risk signal emission must write an entry to `audit.jsonl`.

---

## 14. Test Strategy for Phase 7

- **Unit Tests**: Test `SituationalRiskEvaluator` against synthetic evidence objects (100% coverage target).
- **Integration Tests**: Verify Tool 1–10 aggregation through `ReasoningEngine`.
- **Determinism Tests**: Verify bit-exact field output across repeated risk query dispatches.
- **Privacy Tests**: Audit serializations for zero `global_id` leakage in aggregate outputs.
- **Benchmarking**: Latency and scale benchmarks added to `benchmarks/benchmark_phase7_situational.py`.

---

## 15. Phase 7 Implementation Roadmap (Sub-Phases 7.1 – 7.5)

```
7.1 Situational Schemas & Contracts     (Schemas: SituationalState, RiskSignal, Hypothesis)
7.2 Situational Risk Evaluator         (Core: Multi-factor evidence aggregation)
7.3 Tool 11 & Tool 12 Integration      (Tools: get_situational_risk, get_hypothesis_tree)
7.4 NLQ Planner & Rule Integration     (Planner: Intent rules for situational queries)
7.5 E2E Validation & Benchmarking      (Validation: Held-out validation & latency benchmarking)
```

| Sub-Phase | Primary Objective | Affected Files | Stop Condition |
|---|---|---|---|
| **7.1** | Define frozen situational schemas | `gods_eye/schemas/situational.py` | Schema unit tests pass (100% coverage) |
| **7.2** | Build situational risk evaluator engine | `gods_eye/reasoning/situational.py` | Risk score evaluation tests pass |
| **7.3** | Register Tool 11 & Tool 12 in dispatcher | `gods_eye/reasoning/tools.py` | Tool 11 & 12 dispatches pass read-only checks |
| **7.4** | Add NLQPlanner intent routing rules | `gods_eye/reasoning/planner.py` | Intent routing tests pass |
| **7.5** | Full integration & latency benchmark | `tests/test_phase_7_validation.py` | All repository tests pass cleanly |

---

## 16. Go / No-Go Decision

### Decision: **CONDITIONALLY READY**

#### Required Prerequisites Before Sub-Phase 7.1 Implementation:
1. **Approval of Phase 7 Architecture Review**: Formal sign-off on `docs/phases/phase_7/PHASE_7_ARCHITECTURE_REVIEW.md`.
2. **Preservation of Phase 1–6 Baseline**: Zero regressions against existing 429 passing pytest tests.
3. **Synthetic Limitations Acknowledgment**: Real-world CCTV feed validation must remain clearly identified as pending live deployment.

---

## 17. Final Architectural Verdict

1. **What Phase 7 Should Build**:
   - Hybrid deterministic situational risk evaluator.
   - Tool 11 (`get_situational_risk`) and Tool 12 (`get_hypothesis_tree`).
   - Structured human-readable risk explanations.
2. **What Phase 7 Must NOT Build**:
   - Must NOT build black-box probabilistic neural net reasoners.
   - Must NOT add heavy graph database infrastructure (Neo4j).
   - Must NOT allow LLMs to fabricate unevidenced facts.
3. **What Phase 6 Already Provides**:
   - 1st-order Markov destination predictor (`MarkovBehavioralPredictor`).
   - Levenshtein sequence edit-distance clustering (`TrajectoryClusteringEngine`).
   - Behavioral anomaly detector (`BehavioralAnomalyDetector`).
   - Tools 9 and 10 integrated into `ToolDispatcher` and `ReasoningEngine`.
4. **Highest-Risk Component**:
   - Balancing multi-factor risk scoring across conflicting or partial evidence without producing false positive alerts.
5. **Recommended First Implementation Task**:
   - **Sub-Phase 7.1**: Create data schemas in `gods_eye/schemas/situational.py` and register exports in `gods_eye/schemas/__init__.py`.
