# Sub-Phase 7.3 — Tool 11 Integration & NLQ Router: Deep Preflight

## Executive Summary

- **Repository Baseline Commit**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Current Test Count**: **477 / 477 passing tests**
- **Preflight Target**: Sub-Phase 7.3 — Tool 11 Integration (`get_situational_risk`) & RuleBasedPlanner Router Integration
- **Preflight Mode**: STRICT PREFLIGHT-ONLY. Zero runtime code, zero test modifications, zero commits.
- **Tool-Level Verdicts**:
  - **Tool 11 (`get_situational_risk`)**: **GO**
  - **Tool 12 (`get_hypothesis_tree`)**: **BLOCKED (CONTRACT GAP)**
- **Overall Preflight Verdict**: **CONDITIONAL GO FOR 7.3**

---

## 1. Authoritative Baseline Audit

An audit across repository documentation (`GODS_EYE_MASTER_SPEC.md`, `PHASE_7_ARCHITECTURE_REVIEW.md`, `PHASE_7_CONTRACT_REVIEW.md`, `PHASE_7_2_DEEP_AUDIT.md`) establishes the baseline state:

- **Phase 6 Freeze**: Committed at `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`).
- **Phase 7.1 Schemas**: `IdentityCandidate`, `Hypothesis`, `SituationalState`, `RiskSignal` frozen in `gods_eye/schemas/situational.py`.
- **Phase 7.2 Engine**: `SituationalRiskEvaluator` implemented in `gods_eye/situational/evaluator.py`, deep-audited and conditionally approved.
- **Tool Registry Baseline**: Tools 1–10 currently registered in `gods_eye/reasoning/tools.py` (`ToolDispatcher`) and `gods_eye/reasoning/planner.py` (`RuleBasedPlanner`).

---

## 2. Scope Definition & Tool 12 Contract Gap

The Phase 7 Architecture Review originally grouped Tool 11 (`get_situational_risk`) and Tool 12 (`get_hypothesis_tree`) under Sub-Phase 7.3. A rigorous contract audit reveals:

1. **Tool 11 (`get_situational_risk`)**: Backed by the completed Phase 7.2 `SituationalRiskEvaluator` engine and frozen `RiskSignal` schema. Fully specified and ready for integration into `ToolDispatcher` and `RuleBasedPlanner`. Status: **GO**.
2. **Tool 12 (`get_hypothesis_tree`)**: **CONTRACT GAP**. While the `Hypothesis` schema was frozen in Sub-Phase 7.1, no hypothesis generation engine, ranking logic, or tree builder exists in the codebase or Master Spec. Implementing Tool 12 in Sub-Phase 7.3 would force inventing semantics from scratch without specification. Status: **BLOCKED pending hypothesis engine specification in Sub-Phase 7.4**.

---

## 3. Tool 11 Contract (`get_situational_risk`)

### Method Signature & Registration
```python
def get_situational_risk(
    self,
    window_start_ns: int,
    window_end_ns: int,
    system_mode: str = "operational_mode",
    evidence_ids: Optional[list[str]] = None,
) -> ToolResult:
    """Evaluate situational risk over a temporal window (Tool 11)."""
```

### Input Specifications
- `window_start_ns`: Integer, required, $\ge 0$, must satisfy `window_start_ns <= window_end_ns`.
- `window_end_ns`: Integer, required, $\ge 0$, duration `(window_end_ns - window_start_ns)` must not exceed 24 hours ($86,400,000,000,000\text{ ns}$).
- `system_mode`: String, optional, default `"operational_mode"`. Must map to valid `SystemMode` enum value (`"learning_mode"`, `"operational_mode"`, `"degraded_mode"`).
- `evidence_ids`: Optional list of strings to restrict evaluation evidence items.

### Output Payload
- Returns `ToolResult`:
  - `call_id`: Matching input call ID string.
  - `tool_name`: `"get_situational_risk"`.
  - `success`: `True` if execution succeeds.
  - `data`: Serialized dictionary representation of `RiskSignal` (`risk_signal.to_dict()`).
  - `evidence_list`: List of canonical `Evidence` items contributing to the risk signal.
  - `error_message`: `None` on success, error message string on failure.

---

## 4. Tool 12 Contract Gap Analysis

- **Deficiency**: `gods_eye/situational/` contains `SituationalRiskEvaluator` (which produces `RiskSignal`), but lacks a `HypothesisGenerator` component to instantiate or rank `Hypothesis` objects.
- **Contract Requirement**: Before Tool 12 (`get_hypothesis_tree`) can be implemented, Sub-Phase 7.4 must define:
  1. `HypothesisGenerator` engine architecture and ranking formulas.
  2. Input evidence requirements for hypothesis candidate generation.
  3. Recursion and branching factor limits for hypothesis tree construction.
- **Decision**: Tool 12 is **BLOCKED** and explicitly excluded from Sub-Phase 7.3 implementation scope.

---

## 5. Privacy Contract

- **Identity Privacy Minimization**: Tool 11 output payloads (`data` dictionary and `explanation`) MUST NOT contain raw `global_id` UUID strings, `subject_global_id`, ReID embeddings, or gallery images.
- **Evidence References**: Only evidence IDs (`evidence_ids`) and pseudonymous subject references (`subject_ref`) are exposed across the reasoning boundary.

---

## 6. Determinism Contract

- **Pure Evaluation**: Tool 11 execution is 100% deterministic, synchronous, in-memory, and read-only.
- **Deterministic Ordering**: `evidence_ids` and `contributing_factors` retain stable, sorted tuple ordering.
- **Timestamp Integrity**: Real-time `time.time_ns()` is used solely for sanity checking future timestamp bounds ($> \text{now} + 1.0\text{s}$) and does not mutate output payload content.

---

## 7. Read-Only Invariant

Tool 11 execution MUST NOT alter database or store state:
- Zero writes to `SQLiteEventStore` or `SQLiteGraphStore`.
- Zero state mutations in `SituationalRiskEvaluator` or Phase 6 engines.

---

## 8. Reasoning-Layer Architecture & Data Flow

```
UserQuery ("Is there a high situational risk?")
      │
      ▼
RuleBasedPlanner.plan()
      │ (matches "situational risk" intent -> ToolCall("get_situational_risk"))
      ▼
ToolDispatcher.dispatch()
      │
      ▼
get_situational_risk(window_start_ns, window_end_ns, system_mode)
      │
      ├─► Queries candidate Evidence from EventStore & GraphStore over window
      │
      ▼
SituationalRiskEvaluator.evaluate(evidence_items, system_mode, window_start_ns, window_end_ns)
      │
      ▼
RiskSignal -> ToolResult(data=risk_signal.to_dict(), evidence_list=evidence_items)
```

---

## 9. Evidence Flow into Tool 11

1. `get_situational_risk` queries existing `EventStore` and `GraphStore` records matching `[window_start_ns, window_end_ns]`.
2. Records are mapped to canonical `Evidence` dataclass objects, preserving payload attributes (`trajectory_anomaly_sigma`, `zone_dwell_sigma`, `is_restricted_zone`, `occupancy_deviation_sigma`, `unlikely_transition_probability`).
3. `SituationalRiskEvaluator.evaluate()` evaluates the evidence list and emits a `RiskSignal`.

---

## 10. NLQ Routing Contract

`RuleBasedPlanner` in `gods_eye/reasoning/planner.py` will route user queries to Tool 11 based on deterministic query patterns:

### Supported Query Intent Patterns for Tool 11
- `"situational risk"`
- `"security risk"`
- `"risk level"`
- `"is there any risk"`
- `"evaluate risk"`
- `"how risky"`

### Intent Disambiguation Precedence
1. `"prediction"`, `"destination"`, `"where will identity go"` $\to$ Tool 9 (`get_behavioral_prediction`)
2. `"cluster"`, `"group"`, `"trajectory pattern"` $\to$ Tool 10 (`get_trajectory_clusters`)
3. `"risk"`, `"threat level"`, `"situational hazard"` $\to$ Tool 11 (`get_situational_risk`)

---

## 11. QueryStatus Mapping Table

| Execution Condition | Resulting `QueryStatus` | `ToolResult.success` | Notes |
|---|---|---|---|
| Valid window & evidence evaluated | `QueryStatus.SUCCESS` | `True` | Standard successful risk evaluation |
| Valid window, evaluation suppressed | `QueryStatus.SUCCESS` | `True` | `RiskSignal.suppressed=True` with exact suppression reason |
| Zero evidence in temporal window | `QueryStatus.INSUFFICIENT_EVIDENCE` | `True` | Evaluated as `LOW` risk with `suppressed=True` |
| Invalid window (`start > end` or $> 24\text{h}$) | `QueryStatus.INVALID_QUERY` | `False` | `error_message` populated |
| Unsupported query intent | `QueryStatus.AMBIGUOUS` | `False` | Fallback planner status |

---

## 12. Evidence Provenance & Provenance Preservation

`ToolResult.evidence_list` contains the exact `Evidence` objects whose `evidence_id` values appear in `RiskSignal.evidence_ids`. Zero synthetic or manufactured evidence objects are generated.

---

## 13. Performance Targets

- **Latency Target**: $< 5.0\text{ ms}$ for $N=100$ evidence items.
- **Memory Allocation**: $O(N)$ temporary objects; zero memory retention post-dispatch.

---

## 14. Failure Safety

- **Missing / Invalid Arguments**: Trapped by `RuleBasedPlanner` or `ToolDispatcher`, returning `QueryStatus.INVALID_QUERY` with an explicit error string.
- **Store Query Failures**: If underlying store query fails, handle gracefully and return `ToolResult(success=False, error_message=...)`.

---

## 15. Phase 1–6 Freeze Integrity

Sub-Phase 7.3 requires **ZERO** modifications to:
- `gods_eye/behavioral/*`
- `gods_eye/memory/*`
- `gods_eye/events/*`
- `gods_eye/tracking/*`
- `gods_eye/reid/*`
- SQLite schemas and database files.

---

## 16. Phase 7.1 / 7.2 Freeze Integrity

Sub-Phase 7.3 requires **ZERO** modifications to:
- `gods_eye/schemas/situational.py`
- `gods_eye/situational/evaluator.py`

---

## 17. Master Spec Compliance Matrix

| Requirement | Source | 7.3 Contract | Status |
|---|---|---|---|
| Tool 11 Registration | §14 Phase 5/6 Reasoning | Register `get_situational_risk` in `ToolDispatcher` | **PASS** |
| NLQ Planner Intent Routing | §14 RuleBasedPlanner | Add `"risk"` intent patterns to `RuleBasedPlanner` | **PASS** |
| Approved Tools Set Update | §14 Approved Tools | Add `get_situational_risk` to `APPROVED_TOOLS` | **PASS** |
| Privacy Preservation | §2 Rule 6, §4 Schemas | Omit `global_id` in public payloads | **PASS** |
| Tool 12 Implementation | §18 Horizon | Defer Tool 12 until hypothesis engine spec is finalized | **CONTRACT GAP** |

---

## 18. Implementation File Boundaries

### Allowed File Modifications
1. `gods_eye/reasoning/tools.py`: Register `get_situational_risk` method and constructor dependency in `ToolDispatcher`.
2. `gods_eye/reasoning/planner.py`: Add `"get_situational_risk"` to `APPROVED_TOOLS` and implement intent matching rules in `RuleBasedPlanner`.

### Allowed File Creations
1. `tests/test_situational_tool11.py`: Unit test suite for Tool 11 integration and NLQ routing.
2. `docs/phases/phase_7/PHASE_7_3_COMPLETION.md`: Completion report.

### Prohibited Modifications
- All Phase 1–6 engines, stores, schemas, and Phase 7.1/7.2 files.

---

## 19. Unit Test Plan (15 Required Scenarios)

The test suite `tests/test_situational_tool11.py` must validate:

1. `test_tool11_dispatcher_registration`: Confirm `get_situational_risk` is registered in `ToolDispatcher`.
2. `test_tool11_successful_evaluation`: Evaluate valid evidence over window and verify `ToolResult`.
3. `test_tool11_low_risk_baseline`: Verify low-risk signal output structure.
4. `test_tool11_medium_risk_signal`: Verify medium-risk trajectory anomaly signal payload.
5. `test_tool11_high_risk_signal`: Verify high-risk signal payload.
6. `test_tool11_critical_risk_signal`: Verify critical-risk signal payload.
7. `test_tool11_learning_mode_suppression`: Verify suppression payload when `system_mode="learning_mode"`.
8. `test_tool11_degraded_mode_suppression`: Verify suppression payload when `system_mode="degraded_mode"`.
9. `test_tool11_empty_evidence_insufficient`: Verify empty evidence handling.
10. `test_tool11_invalid_temporal_window`: Verify `start > end` returns `success=False` with error message.
11. `test_tool11_read_only_invariant`: Verify zero DB store writes.
12. `test_tool11_privacy_invariant`: Confirm zero `global_id` strings in result dictionary.
13. `test_nlq_planner_routes_to_tool11`: Test natural language queries ("Is there a high situational risk?") resolve to Tool 11 call.
14. `test_nlq_planner_approved_tools`: Verify `"get_situational_risk"` is in `APPROVED_TOOLS`.
15. `test_phase1_to_phase6_regression`: Full test suite regression verification.

---

## 20. Tool-Level & Overall Verdicts

- **Tool 11 (`get_situational_risk`)**: **GO**
- **Tool 12 (`get_hypothesis_tree`)**: **BLOCKED (CONTRACT GAP)**
- **Overall Preflight Verdict**: **CONDITIONAL GO FOR 7.3**

---

## 21. Stop Condition & Final Status

Sub-Phase 7.3 Preflight is complete. Zero runtime code has been implemented.
