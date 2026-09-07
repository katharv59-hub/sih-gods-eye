# Phase 7 — Contract Hardening & Pre-Implementation Audit

## Executive Summary

- **Repository Baseline Commit**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Audit Target**: `docs/phases/phase_7/PHASE_7_ARCHITECTURE_REVIEW.md`
- **Review Scope**: READ-ONLY pre-implementation contract hardening audit. Zero runtime code modifications, zero schema changes, zero test modifications.
- **Final Verdict**: **CONDITIONALLY READY** (Sub-Phase 7.1 implementation requires adoption of the 7 contract hardening directives defined in Section 13).

---

## 1. Hypothesis Identity Model Audit

The Phase 7 Architecture Review proposed:
```python
subject_global_id: Optional[str]  # PROPOSED IN ARCHITECTURE REVIEW
```

### Critical Flaws Identified
1. **Privacy Boundary Violation**: Exposing raw `global_id` UUID strings in a public hypothesis contract violates Data Minimization (§10) and exposes persistent PII identifiers across external boundaries.
2. **False Identity Certainty**: Treating `global_id` as ground truth ignores OSNet Re-ID matching uncertainty ($0.75$ Cosine Similarity threshold), false merges, and track splits.

### Hardened Contract Specification
- **Public / External Contract**: Replace `subject_global_id` with `subject_ref: Optional[str]` (an opaque pseudonymous token, e.g., `"sub_8f2a9d"`).
- **Identity Confidence**: Include explicit `identity_confidence: float` representing Re-ID similarity confidence.
- **Competing Identity Hypotheses**: Support competing identity candidate links:
```python
# HARDENED SPECIFICATION — PROPOSED FOR SUB-PHASE 7.1
@dataclass(frozen=True)
class IdentityCandidate:
    subject_ref: str
    confidence: float

@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    hypothesis_type: str
    primary_subject_ref: Optional[str]
    identity_confidence: float
    competing_candidates: tuple[IdentityCandidate, ...]
    confidence: float
    evidence_ids: tuple[str, ...]
    explanation: str
```

---

## 2. Confidence vs Probability vs Statistical Confidence Interval

The Phase 7 Architecture Review proposed:
```python
confidence_band: tuple[float, float]  # 95% CI
```

### Critical Audit Findings
> [!CAUTION]
> The system **MUST NOT** claim outputs represent a "95% Confidence Interval (CI)".

1. **Lack of Parametric Distribution**: Cosine similarity scores, GMM background confidence, and 1st-order Markov counts are heuristic and frequency-ratio statistics. They do not possess closed-form Student's t or Gaussian parametric sample distributions.
2. **Misleading Terminology**: Presenting a heuristic lower/upper bound as a "95% CI" violates Rule 1 (benchmarks over assumptions) and Rule 7 (learned model outputs carry uncertainty).

### Semantic Definitions Matrix

| Concept | Mathematical Definition | God's Eye Implementation | Valid Contract Field |
|---|---|---|---|
| **Probability** | Normalized outcome likelihood $\in [0.0, 1.0]$ | Markov transition ratio $\frac{count(C_i \to C_j)}{\sum count(C_i \to \cdot)}$ | `probability: float` |
| **Confidence Score** | Heuristic model certainty $\in [0.0, 1.0]$ | Calibrated similarity or cluster distance score | `confidence: float` |
| **Evidence Strength** | Observation sample sufficiency | $\min\left(1.0, \frac{N_{samples}}{N_{threshold}}\right)$ | `evidence_strength: float` |
| **Calibrated Probability** | Empirical accuracy on held-out data | Top-1 accuracy alignment | `calibrated_confidence: float` |
| **Confidence Interval** | Parametric Gaussian/Student's t bound | **NOT SUPPORTED BY CURRENT BASELINE** | **PROHIBITED** |

### Hardened Contract Recommendation
Replace `confidence_band: tuple[float, float]` with `confidence: float` and `evidence_strength: float`.

---

## 3. Risk Score Formulation Audit

### Audit of Arbitrary Weighted Scoring
The architecture review discussed candidate formulas such as:
$$\text{risk} = 0.4 \cdot \text{anomaly} + 0.3 \cdot \text{prediction} + 0.3 \cdot \text{environmental}$$

> [!WARNING]
> Arbitrary weighted formulas are **NOT YET JUSTIFIED** by empirical multi-camera test datasets.

### Correct Phase 7 Risk Strategy
1. **Initial Implementation (Sub-Phase 7.2)**: Risk evaluation must be strictly **Rule-Based**:
   - `CRITICAL`: Trajectory anomaly $> 3.0\sigma$ AND Dwell time $> 5.0\sigma$ in restricted zone.
   - `HIGH`: Trajectory anomaly $> 2.5\sigma$ OR Environmental occupancy deviation $> 4.0\sigma$.
   - `MEDIUM`: Trajectory anomaly $> 1.5\sigma$ OR Unlikely transition probability $< 0.10$.
   - `LOW`: Baseline behavior.
2. **Deferred Calibration**: Continuous float risk weights ($[0.0, 1.0]$) must remain deferred until held-out labeled multi-camera risk datasets are collected.

---

## 4. Situational State Contract Audit

### Flaws in Proposed `SituationalState`
1. **Single Timestamp Insufficiency**: A situational state evaluates a temporal window, not an instantaneous point in time. Single `timestamp_ns` fails to represent open visits or windowed anomalies.
2. **Missing Window Boundaries**: Window boundaries `(window_start_ns, window_end_ns)` are strictly required.

### Hardened Dataclass Contract
```python
# HARDENED SPECIFICATION — PROPOSED FOR SUB-PHASE 7.1
@dataclass(frozen=True)
class SituationalState:
    state_id: str
    window_start_ns: int
    window_end_ns: int
    active_identity_count: int
    active_cameras: tuple[str, ...]
    system_mode: OperationalMode
    anomaly_ids: tuple[str, ...]
    overall_risk_level: str                  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    explanation: str
```

---

## 5. Risk Signal Contract Audit

### Hardened `RiskSignal` Schema
```python
# HARDENED SPECIFICATION — PROPOSED FOR SUB-PHASE 7.1
@dataclass(frozen=True)
class RiskSignal:
    signal_id: str
    window_start_ns: int
    window_end_ns: int
    risk_level: str                          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    risk_score: float                        # Heuristic rule-based score [0.0, 1.0]
    evidence_ids: tuple[str, ...]            # Traceable provenance IDs
    contributing_factors: tuple[str, ...]
    explanation: str                         # Human-readable explanation
    suppressed: bool                         # True if confidence < threshold or LEARNING_MODE
    suppression_reason: Optional[str]
    evaluator_version: str
```

---

## 6. Failure States & Canonical Reasoning Statuses

The Phase 7 contracts must reuse the existing canonical `QueryStatus` enum from `gods_eye/schemas/reasoning.py`:

```python
class QueryStatus(Enum):
    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS = "ambiguous"
    INVALID_PARAMETER = "invalid_parameter"
    ERROR = "error"
```

> [!IMPORTANT]
> Phase 7 **MUST NOT** introduce a separate or conflicting status enum. All tool dispatches and reasoning results use `QueryStatus`.

---

## 7. Temporal Semantics

1. **Observation vs Processing Time**:
   - `timestamp_ns`: Event occurrence Unix nanoseconds.
   - `ingestion_timestamp_ns`: Pipeline processing Unix nanoseconds.
2. **Temporal Windowing**: Situational queries specify `window_start_ns` and `window_end_ns`.
3. **Handling Gaps & Open Visits**: Open visits (no exit event) have `exit_timestamp_ns = None` and dwell duration computed dynamically against current `window_end_ns`.

---

## 8. Evidence Provenance Architecture

Every Phase 7 hypothesis and risk signal must strictly preserve evidence chain provenance:

```
[Hypothesis / RiskSignal]
       │
       ▼  (contains evidence_ids: tuple[str, ...])
[Evidence Schema] (gods_eye/schemas/reasoning.py)
       │
       ├──► source_event_id       ──► SQLiteEventStore (event log)
       ├──► source_edge_id        ──► SQLiteGraphStore (transition edge)
       └──► source_observation_id ──► TimelineEngine (observation)
```

**Rule**: Zero hypotheses or risk signals may be constructed without traceable evidence IDs.

---

## 9. Deterministic / Probabilistic / LLM Classification Matrix

| Component | Classification | Justification |
|---|---|---|
| **SQLiteEventStore / SQLiteGraphStore** | **DETERMINISTIC** | Immutable, append-only persistence |
| **TimelineReconstructionEngine** | **DETERMINISTIC** | Bit-exact event ordering and replay |
| **Tool Dispatcher (Tools 1–10)** | **DETERMINISTIC** | Exact schema querying and filtering |
| **Markov Predictor Matrix** | **PROBABILISTIC** | Frequency-ratio transition probabilities |
| **Anomaly Deviation Scoring** | **PROBABILISTIC** | Statistical distance in $\sigma$ units |
| **Situational Risk Evaluator** | **DETERMINISTIC RULES** | Rule-based decision tables over evidence |
| **NLQ Intent Planner** | **DETERMINISTIC / LLM** | Regex rule primary; LLM fallback |
| **Explanation Formatter** | **LLM-ASSISTED** | Read-only human formatting of evidence |

---

## 10. LLM Boundary Constraints

The LLM interface (`NLQPlanner` / Explanation Formatter) operates under strict architectural constraints:

1. **Read-Only Non-Authoritative Role**: The LLM is purely a natural language interface.
2. **Forbidden Actions**:
   - MUST NOT generate unevidenced facts.
   - MUST NOT alter prediction probabilities or anomaly scores.
   - MUST NOT invent identity UUIDs, camera IDs, or timestamps.
   - MUST NOT override system suppression flags.

---

## 11. Three-Tier Privacy Data Exposure Model

```
┌───────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: INTERNAL STORAGE (Full System Access)                            │
│ Exposes: Raw global_id UUIDs, trajectory point vectors, GraphStore edges  │
├───────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: OPERATOR INTERFACE (Authenticated Analyst View)                 │
│ Exposes: Opaque subject tokens (`sub_8f2a`), Camera IDs, Anomaly signals, │
│          Confidence scores, Evidence IDs, Explanations                    │
├───────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: PUBLIC / EXTERNAL API (Data Minimization Boundary)               │
│ Exposes: Aggregate identity counts (`active_identity_count`), Scene State,│
│          Risk Levels (`LOW`/`MEDIUM`/`HIGH`), Suppressed status          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Contract Minimalism Audit

The following proposed fields from the initial architecture review were audited and **REMOVED** as over-engineered:

1. `confidence_band: tuple[float, float]` — REMOVED (Unjustified statistical 95% CI claim).
2. `subject_global_id: str` — REMOVED from public contracts (Replaced with opaque `primary_subject_ref: Optional[str]`).
3. `overall_risk_score: float` — REMOVED from core contracts until empirical weight calibration is performed (Replaced with rule-based `risk_level: str`).

---

## 13. Phase 7.1 Implementation Gate Directives

The following 7 contract hardening directives are **MANDATORY** for Sub-Phase 7.1 implementation:

| # | Question | Decision / Mandatory Directive |
|---|---|---|
| **1** | Is `subject_global_id` acceptable? | **NO**. Replaced with pseudonymous `primary_subject_ref: Optional[str]`. |
| **2** | Is `95% CI` acceptable? | **NO**. "95% CI" terminology is prohibited. Replaced with `confidence: float`. |
| **3** | Is `overall_risk_score` justified? | **NO**. Float formula marked `NOT YET JUSTIFIED`. Initial implementation must be rule-based `risk_level`. |
| **4** | Are statuses compatible? | **YES**. Must use canonical `QueryStatus` enum from `gods_eye/schemas/reasoning.py`. |
| **5** | Is single timestamp sufficient? | **NO**. Situational states require `(window_start_ns, window_end_ns)` boundaries. |
| **6** | Is evidence provenance sufficient? | **YES**. Every hypothesis and risk signal must contain `evidence_ids: tuple[str, ...]`. |
| **7** | Is the LLM boundary constrained? | **YES**. Strictly read-only formatting; zero fact generation capability. |

---

## 14. Final Verdict

### Verdict: **GO FOR 7.1** (Subject to Hardened Contracts)

Sub-Phase 7.1 schema implementation is **APPROVED** to proceed using the hardened dataclasses and privacy boundaries defined in this document.

**Phase 7 runtime code has NOT been implemented.**
