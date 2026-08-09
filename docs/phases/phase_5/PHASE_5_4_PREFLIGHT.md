# Phase 5.4 — Reasoning Engine Orchestration & Evidence Verification: Pre-Flight Review

**Project:** God's Eye (`katharv59-hub/gods_eye`)  
**Milestone:** Sub-Phase 5.4 — Reasoning Engine Orchestration & Evidence Verification  
**Status:** **PRE-FLIGHT REVIEW & ARCHITECTURAL AUDIT (NO CODE WRITTEN)**  
**Authoritative Basis:** `GODS_EYE_MASTER_SPEC.md` (§5, §8, §10, §14 Phase 5), `gods_eye/schemas/reasoning.py`, `gods_eye/reasoning/tools.py`, `gods_eye/reasoning/planner.py`, Phase 5.1/5.2/5.3 Completion Reports  
**Date:** August 8, 2026  

---

## 1. Current Phase 5 Pipeline Reconstruction

The actual, verified state of the Phase 5 pipeline across existing codebase modules is:

```
UserQuery (Natural Language Input)
    │
    ▼
NLQPlanner.create_plan(UserQuery)  [gods_eye/reasoning/planner.py]
    │  (Prompt Injection Protection + RuleBasedPlanner / LLMProvider)
    ▼
(QueryStatus, list[ToolCall], explanation)
    │
    ▼ [Phase 5.4 Orchestration Boundary]
ReasoningEngine.process_query(UserQuery)  [gods_eye/reasoning/engine.py]
    │
    ├── For each ToolCall:
    │     ToolDispatcher.dispatch(ToolCall)  [gods_eye/reasoning/tools.py]
    │       ▼
    │     ToolResult (success, data, evidence_list)
    │
    ▼
[Evidence Collector & Verification]
  - Collects all retrieved Evidence objects
  - Performs evidence sufficiency check (Rule 8 Explainability)
  - Calculates confidence & confidence_band (Rule 7 Uncertainty)
    │
    ▼
ReasoningResult (query_id, status, answer, confidence, confidence_band, evidence, tool_calls, execution_time_ms, explanation)
```

---

## 2. ReasoningEngine Responsibility

The `ReasoningEngine` (`gods_eye/reasoning/engine.py`) is responsible for orchestrating end-to-end query processing:

**Responsibilities:**
1. Receives raw `UserQuery`.
2. Delegates intent planning to `NLQPlanner.create_plan()`.
3. Sequentially executes generated `ToolCall` plans via `ToolDispatcher.dispatch()`.
4. Collects and deduplicates `Evidence` objects returned in `ToolResult.evidence_list`.
5. Verifies evidence sufficiency (returns `QueryStatus.INSUFFICIENT_EVIDENCE` if no evidence supports the query).
6. Constructs a factual summary string (`answer`) directly backed by evidence.
7. Aggregates tool confidences into an explicit `confidence` score and `confidence_band`.
8. Enforces security scope authentication and query timeouts ($\le 3\text{s}$).

**Non-Responsibilities:**
- Does NOT execute SQL or access SQLite database handles directly.
- Does NOT bypass `ToolDispatcher` or register new tools.
- Does NOT perform raw video capture, object detection, tracking, or identity mapping.
- Does NOT mutate Phase 1–4 persistent stores or environmental state.

---

## 3. Tool Orchestration Model

- **Execution Model:** Deterministic sequential tool execution. If a plan contains multiple tool calls (e.g. `list_active_identities` followed by `get_identity_timeline`), calls are executed in order.
- **Partial Failure Handling:** If a tool call in a multi-call plan fails (`ToolResult.success == False`), the engine captures the failure, logs a warning, and continues processing remaining calls or returns `QueryStatus.EXECUTION_ERROR` if critical evidence is missing.

---

## 4. Evidence Verification & Hallucination Boundary (Rule 7 & 8)

- **Rule 8 (Explainability / Provenance):** Every claim in `ReasoningResult.answer` MUST cite explicit `Evidence` records retrieved from `ToolResult.evidence_list`.
- **Zero Hallucination Rule:** If `ToolResult.data` returns empty results or `evidence_list` is empty, `ReasoningResult.status` MUST be set to `QueryStatus.INSUFFICIENT_EVIDENCE` with the explanation `"No matching historical records found in EventStore or GraphStore."`
- **Strict Evidence Deduplication:** `Evidence` items are deduplicated by `(source_store, record_type, record_id)` and deterministically sorted by `(timestamp_ns, record_id)`.

---

## 5. Confidence Aggregation Semantics (Rule 7)

- **Confidence Score:** Derived from retrieved `Evidence` confidence fields (`Event.confidence`, `Observation.confidence`).
  - Single evidence record: `confidence = evidence.confidence`.
  - Multiple evidence records: `confidence = mean(e.confidence for e in evidence)`.
  - Empty evidence list: `confidence = 0.0`.
- **Confidence Band:** `confidence_band = (max(0.0, confidence - 0.05), min(1.0, confidence + 0.05))`.
- **Open Decision:** Advanced Bayesian or multi-camera fusion confidence aggregation (deferred to future refinement; mean confidence model used for Phase 5.4).

---

## 6. Failure Model Mapping

| Pipeline Condition | Engine Action | Output `QueryStatus` | Evidence Attached |
|---|---|---|---|
| Planner returns `INVALID_QUERY` | Return early with planner explanation | `INVALID_QUERY` | `[]` |
| Planner returns `AMBIGUOUS` | Return early with ambiguity clarification prompt | `AMBIGUOUS` | `[]` |
| Tool execution returns `success=False` | Log warning, return execution error | `EXECUTION_ERROR` | `[]` |
| Tool returns `data=[]` / empty evidence | Mark insufficient evidence | `INSUFFICIENT_EVIDENCE` | `[]` |
| Successful tool execution with evidence | Format answer string & aggregate evidence | `SUCCESS` | Retained `Evidence` list |

---

## 7. Determinism & Read-Only Invariants

- **Determinism:** Given identical `UserQuery` input and static persistent database state, `ReasoningEngine.process_query()` emits 100% identical `ReasoningResult` payloads.
- **Read-Only Invariant:** Execution causes ZERO mutations in `SQLiteEventStore`, `SQLiteGraphStore`, `IdentityGallery`, `IdentityMapper`, or `EnvironmentalWorker`.

---

## 8. LLM Boundary & Answer Synthesis

- `ReasoningEngine` handles deterministic orchestration, tool call execution, evidence collection, and rule-based template answer synthesis.
- Natural language answer formatting follows clean deterministic templates (e.g. *"Identity {gid} was observed at camera {cam} starting at {time} for duration {dwell}s."*).
- Advanced generative LLM answer synthesis is encapsulated cleanly and does NOT replace deterministic evidence structures.

---

## 9. Observability Requirements (§8 Master Spec)

Phase 5.4 will register low-cardinality metrics in `gods_eye/observability/metrics.py`:
- `gods_eye_reasoning_engine_queries_total{status}` (Counter)
- `gods_eye_reasoning_engine_latency_ms` (Histogram)

---

## 10. Phase 5.4 Test Plan

1. `test_reasoning_engine_successful_single_tool_query`: Test end-to-end execution of a valid identity timeline query.
2. `test_reasoning_engine_multi_tool_execution`: Test sequential execution of multiple tool calls.
3. `test_reasoning_engine_insufficient_evidence_handling`: Verify `QueryStatus.INSUFFICIENT_EVIDENCE` when queries return empty results.
4. `test_reasoning_engine_ambiguous_query_handling`: Verify handling of ambiguous queries returned by `NLQPlanner`.
5. `test_reasoning_engine_prompt_injection_rejection`: Verify end-to-end rejection of prompt injection attempts.
6. `test_reasoning_engine_tool_failure_handling`: Test engine behavior when a backend tool raises a database error.
7. `test_reasoning_engine_confidence_calculation`: Verify confidence and `confidence_band` calculation logic.
8. `test_reasoning_engine_read_only_invariant`: Assert zero database mutations across `EventStore` and `GraphStore`.
9. `test_reasoning_engine_determinism`: Assert identical outputs for identical queries over static database states.

---

## 11. Implementation Scope for Sub-Phase 5.4

- **Files to Create:**
  1. `gods_eye/reasoning/engine.py` — `ReasoningEngine` class orchestrating `NLQPlanner` and `ToolDispatcher`.
  2. `tests/test_reasoning_engine.py` — 10 unit tests covering orchestration, evidence aggregation, confidence scoring, failure handling, and read-only invariants.
  3. `docs/phases/phase_5/PHASE_5_4_COMPLETION.md` — Sub-Phase 5.4 completion report.
- **Files to Modify:**
  1. `gods_eye/reasoning/__init__.py` — Export `ReasoningEngine`.
  2. `gods_eye/observability/metrics.py` — Register `reasoning_engine_queries_total` and `reasoning_engine_latency_ms`.

---

## 12. Phase 4 Freeze Compatibility

Sub-Phase 5.4 operates purely above `NLQPlanner` and `ToolDispatcher`. Zero modifications to Phase 1–4 source code or persistence engines will occur.

---

PHASE 5.4 STATUS:
SAFE TO IMPLEMENT
