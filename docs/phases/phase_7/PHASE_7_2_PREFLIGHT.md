# Sub-Phase 7.2 — Situational Risk Evaluator: Deep Preflight & Contract Audit

## Executive Summary

- **Repository Baseline Commit**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Preflight Target**: Sub-Phase 7.2 — `SituationalRiskEvaluator`
- **Audit Scope**: READ-ONLY preflight design and contract audit. Zero runtime code, zero schema changes, zero test modifications.
- **Final Verdict**: **GO FOR 7.2**

---

## 1. Master Spec Requirement Tracing

An audit of `GODS_EYE_MASTER_SPEC.md` (§5, §6, §14, §18) traces the exact status of Phase 7.2 capabilities:

| Proposed Capability | Master Spec Reference | Classification | Implementation Status |
|---|---|---|---|
| Multi-Signal Risk Aggregation | §5 Architecture Overview, §18 | **REQUIRED** | To be implemented in Sub-Phase 7.2 |
| Operational Mode Suppression | §6 Operational Modes | **REQUIRED** | Supported by `SystemMode` in `gods_eye/schemas/environment.py` |
| Evidence Provenance Preservation | §2 Rule 8, §4 Canonical Schemas | **REQUIRED** | Supported by `Evidence` schema in `gods_eye/schemas/reasoning.py` |
| Rule-Based Risk Severity Mapping | §18 Architectural Horizon | **SUPPORTED** | Defined in `gods_eye/schemas/situational.py` (`RiskSignal`) |
| Empirical Multi-Camera Risk Calibration | §2 Rule 1 | **DEFERRED** | Requires physical multi-camera deployment dataset |
| Arbitrary Float Risk Score Formulas | None | **NOT SPECIFIED** | Explicitly prohibited; rule-severity score used instead |

---

## 2. Input Evidence Audit

The `SituationalRiskEvaluator` consumes structured signals from Phase 3.5, 4, 5, and 6. Each input signal is audited below:

| Input Signal Schema | Source Layer | Semantics | Timestamp Field | Confidence Field | Classification | Risk Contribution |
|---|---|---|---|---|---|---|
| `BehavioralAnomalyResult` | Phase 6.4 (`anomaly.py`) | Trajectory deviation score in $\sigma$ units | `timestamp_ns` (extracted from trajectory) | `confidence: float` | **Probabilistic** | High score ($>2.5\sigma$) indicates anomalous spatial path |
| `PredictionResult` | Phase 6.5 (`predictor.py`) | Markov next-camera destination forecasts | `timestamp_ns` | `confidence: float` | **Probabilistic** | Low confidence or unpredicted destination path |
| `SceneState` | Phase 3.5 (`environment.py`)| Environmental occupancy & lighting state | `timestamp_ns` | N/A | **Deterministic** | Occupancy deviation against time-bucketed baseline |
| `Event` | Phase 4 (`event_store.py`) | Persistent log events | `timestamp_ns` | `confidence: float` | **Deterministic** | Specific critical event types (e.g. `ANOMALY_DETECTED`) |
| `Evidence` | Phase 5 (`reasoning.py`) | Unified tool/query evidence payload | `timestamp_ns` | `confidence: float` | **Hybrid** | Traceable evidence wrapper for all input signals |

> [!IMPORTANT]
> **Semantic Rule**: Probability is NOT risk ($P(C_{t+1} \mid C_t) \ne \text{Risk}$). Model confidence is NOT probability. Risk represents rule-evaluated potential operational danger.

---

## 3. Risk Level Determination Rules

The hardened contract review defines four categorical risk levels: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.

### Deterministic Risk Rule Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CRITICAL RISK                                                               │
│ Condition: Trajectory Anomaly > 3.5σ AND Zone Dwell > 5.0σ in Restricted Zone│
│ Risk Score: 1.00                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ HIGH RISK                                                                   │
│ Condition: Trajectory Anomaly > 2.5σ OR Occupancy Deviation > 4.0σ           │
│ Risk Score: 0.75                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ MEDIUM RISK                                                                 │
│ Condition: Trajectory Anomaly > 1.5σ OR Unlikely Transition (P < 0.10)      │
│ Risk Score: 0.50                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ LOW RISK                                                                    │
│ Condition: All signals within normal baseline thresholds                    │
│ Risk Score: 0.25                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!WARNING]
> Arbitrary uncalibrated floating point thresholds are **NOT EMPIRICALLY JUSTIFIED**. Thresholds used in Sub-Phase 7.2 represent deterministic rule boundaries that map discrete signal bounds directly to risk levels.

---

## 4. `risk_score` Semantics Definition

In `RiskSignal`, `risk_score: float` is defined under **Interpretation A: Normalized Rule-Severity Score**:

- `risk_score` is a discrete, step-normalized scalar $\in [0.0, 1.0]$ derived directly from rule severity:
  - `LOW` $\to 0.25$
  - `MEDIUM` $\to 0.50$
  - `HIGH` $\to 0.75$
  - `CRITICAL` $\to 1.00$
- **Rejection**: Interpretations B (calibrated probability), C (confidence score), D (evidence strength), and E (arbitrary weighted heuristic) are strictly rejected until physical CCTV deployment datasets are available.

---

## 5. Prohibited Arbitrary Risk Weights

Formulas such as:
$$\text{risk} = 0.4 \cdot \text{anomaly} + 0.3 \cdot \text{prediction} + 0.3 \cdot \text{environmental}$$
are **STRICTLY PROHIBITED** from becoming implementation truth.

Sub-Phase 7.2 will use **Deterministic Rule Precedence** over discrete risk levels.

---

## 6. Deterministic Rule Precedence

When multiple evidence signals fire simultaneously over an evaluation window, risk levels are aggregated using **Maximum Severity Precedence**:

1. **Precedence Hierarchy**: $\text{CRITICAL} > \text{HIGH} > \text{MEDIUM} > \text{LOW}$.
2. **Non-Downgrading Rule**: A lower-severity signal (e.g. `LOW`) can NEVER downgrade or diminish a higher-severity signal (e.g. `HIGH`).
3. **Additive Provenance**: All contributing factor strings and evidence IDs from all firing signals are combined into the output `contributing_factors` and `evidence_ids` tuples without dropping high-severity provenance.

---

## 7. Evidence Provenance Architecture

Every generated `RiskSignal` must retain full provenance traceability back to underlying persistence layers:

$$\text{RiskSignal} \xrightarrow{\text{evidence\_ids}} \text{Evidence} \xrightarrow{\text{source\_id}} \begin{cases} \text{SQLiteEventStore (event\_id)} \\ \text{SQLiteGraphStore (edge\_id)} \\ \text{TimelineEngine (observation\_id)} \end{cases}$$

**Rule**: If `evidence_items` is empty, `RiskSignal` MUST be emitted with `risk_level="LOW"`, `risk_score=0.25`, and marked `suppressed=True` with `suppression_reason="Insufficient evidence: zero evidence items provided"`.

---

## 8. Suppression Semantics & Precedence

A `RiskSignal` MUST be suppressed under any of the following deterministic conditions:

| Priority | Suppression Condition | `suppressed` | `suppression_reason` |
|---|---|---|---|
| **1 (Highest)** | `system_mode == SystemMode.LEARNING_MODE` | `True` | `"System in LEARNING_MODE: baseline warm-up incomplete"` |
| **2** | `system_mode == SystemMode.DEGRADED_MODE` | `True` | `"System in DEGRADED_MODE: model drift or hardware constraint"` |
| **3** | `len(evidence_items) == 0` | `True` | `"Insufficient evidence: zero evidence items provided"` |
| **4** | `evidence_confidence < 0.70` | `True` | `"Evidence confidence below threshold (0.70)"` |
| **5 (Lowest)** | Normal Operational Evaluation | `False` | `None` |

---

## 9. Temporal Windowing Rules

1. **Window Invariants**: `window_start_ns <= window_end_ns`.
2. **Maximum Window Duration**: $24\text{ hours} = 86,400,000,000,000\text{ ns}$.
3. **Stale Evidence Filtering**: Evidence items with `timestamp_ns < window_start_ns` or `timestamp_ns > window_end_ns` are filtered out prior to risk evaluation.
4. **Future Timestamp Protection**: If `window_start_ns` or `window_end_ns` exceeds current time by $> 1.0\text{s}$, raise `ValueError`.

---

## 10. Determinism Invariants

`SituationalRiskEvaluator` is a 100% pure, deterministic rule engine:
- Zero random number generation (`random`, `uuid4()` calls).
- Zero LLM interaction.
- Zero mutable global state.
- Given identical inputs `(evidence_items, system_mode, window_start_ns, window_end_ns)`, the evaluator produces bit-exact identical `RiskSignal` instances.

---

## 11. Probabilistic vs Deterministic Boundary

- **Evaluator Classification**: **DETERMINISTIC RULE ENGINE**.
- The evaluator consumes probabilistic scores produced by Phase 6 engines (`BehavioralAnomalyResult.score`, `PredictionResult.confidence`) and maps them using deterministic decision rules. It does not alter or recompute underlying probabilities.

---

## 12. Reasoning Layer Integration

`SituationalRiskEvaluator` sits **BELOW** the Reasoning Layer (`ToolDispatcher`, `NLQPlanner`, `ReasoningEngine`):

$$\text{Evidence Signals} \longrightarrow \text{SituationalRiskEvaluator} \longrightarrow \text{RiskSignal} \stackrel{\text{Phase 7.3}}{\longrightarrow} \text{Tool 11 (`get_situational_risk`)}$$

Sub-Phase 7.2 does NOT modify `ToolDispatcher`, `NLQPlanner`, or `ReasoningEngine`. Tool 11 integration is deferred to Sub-Phase 7.3.

---

## 13. Performance Targets

- **Evaluation Latency**: $< 5.0\text{ ms}$ for $N=100$ evidence items.
- **Execution Mode**: In-memory, synchronous evaluation. Zero thread blocking, zero GPU overhead.

---

## 14. Unit Test Plan (18 Required Scenarios)

The test suite `tests/test_situational_evaluator.py` must implement the following 18 unit tests:

1. `test_empty_evidence_suppressed_low_risk`
2. `test_learning_mode_suppression`
3. `test_degraded_mode_suppression`
4. `test_low_risk_baseline_evaluation`
5. `test_medium_risk_trajectory_anomaly`
6. `test_high_risk_trajectory_anomaly`
7. `test_critical_risk_combined_anomaly`
8. `test_max_severity_precedence`
9. `test_additive_contributing_factors_and_evidence_ids`
10. `test_low_confidence_evidence_suppression`
11. `test_stale_evidence_filtering`
12. `test_invalid_temporal_window_rejection`
13. `test_deterministic_repeated_execution`
14. `test_risk_score_step_bounds`
15. `test_suppression_reason_consistency`
16. `test_evaluator_version_string`
17. `test_read_only_invariants`
18. `test_phase1_to_phase6_regression`

---

## 15. Phase 6 Freeze Verification

Sub-Phase 7.2 requires **ZERO** modifications to:
- `gods_eye/behavioral/*`
- `gods_eye/memory/*`
- `gods_eye/events/*`
- `gods_eye/tracking/*`
- `gods_eye/reid/*`
- SQLite database schemas or migrations.

---

## 16. Final Implementation Contract

```python
# PROPOSED IMPLEMENTATION CONTRACT FOR SUB-PHASE 7.2
class SituationalRiskEvaluator:
    """Deterministic situational risk evaluator engine."""

    def __init__(self, evaluator_version: str = "v7.2.0") -> None:
        self._evaluator_version = evaluator_version

    def evaluate(
        self,
        evidence_items: tuple[Evidence, ...] | list[Evidence],
        system_mode: SystemMode,
        window_start_ns: int,
        window_end_ns: int,
    ) -> RiskSignal:
        """Evaluate input evidence and produce a deterministic RiskSignal."""
        ...
```

---

## 17. Final Verdict

### Verdict: **GO FOR 7.2**

Sub-Phase 7.2 is fully specified, deterministic, and safe for implementation.

**Phase 7.2 runtime code has NOT been implemented.**
