# Phase 5 — Situational Reasoning & Natural Language Querying (NLQ): Architectural Review & Implementation Blueprint

**Project:** God's Eye (`katharv59-hub/gods_eye`)  
**Milestone:** Phase 5 — Situational Reasoning & Natural Language Querying (NLQ)  
**Status:** **ARCHITECTURE REVIEW & IMPLEMENTATION PLAN (NO CODE WRITTEN)**  
**Authoritative Basis:** `GODS_EYE_MASTER_SPEC.md` (v3.0, 2026-06-11), ADR-001 through ADR-012, Phase 4 Freeze Checkpoint (`2e51484`, `v0.4.0-phase4-freeze`), `docs/audits/REPOSITORY_AUDIT_PHASE4.md`  
**Date:** August 8, 2026  

---

## 1. Executive Summary

Phase 5 transforms God's Eye from a persistent spatiotemporal memory engine into a queryable, evidence-backed situational intelligence system. As mandated by `GODS_EYE_MASTER_SPEC.md` §14, Phase 5 provides natural language query interpretation and structured tool calling over the authoritative persistent stores established in Phase 4 (`SQLiteEventStore` and `SQLiteGraphStore`) and derived read-only query engines (`TimelineReconstructionEngine` and `DeterministicReplayEngine`).

This document provides a comprehensive architectural audit, canonical data contract definition, deterministic tool interface specification, concurrency boundary analysis, and sub-phase implementation plan. 

**Key Architectural Invariants:**
1. **Phase 4 Freeze Protection:** Phase 5 is 100% read-only relative to Phase 4 memory stores and perception pipelines. It introduces ZERO mutations to `EventStore`, `GraphStore`, `IdentityGallery`, or live pipeline state.
2. **Strict Deterministic/Probabilistic Boundary:** Probabilistic LLM logic is strictly constrained to query intent parsing, tool parameter extraction, and human-readable answer synthesis. Data extraction, temporal calculations, camera path tracing, and event retrieval execute 100% deterministically within Python tool handlers. Hallucinated facts are prevented by requiring every answer statement to cite verified `Evidence` records retrieved from Phase 4 stores.
3. **Non-Blocking Execution:** Reasoning queries run asynchronously on dedicated worker threads or via isolated query endpoints, ensuring user queries NEVER block perception, tracking, identity mapping, or temporal observation persistence.

---

## 2. Master Spec Phase 5 Requirements

The following requirements are extracted directly from `GODS_EYE_MASTER_SPEC.md` (§1, §2, §5, §8, §10, §14 Phase 5, §19):

| Requirement | Spec Location | Current Repository Interface / State | Status | Architectural Implication |
|---|---|---|---|---|
| **Transform Memory into Queryable Intelligence** | §14 Phase 5 | `TimelineReconstructionEngine`, `DeterministicReplayEngine`, `SQLiteEventStore`, `SQLiteGraphStore` | **READY (Phase 4)** | Phase 5 reads directly from persistent Phase 4 stores. |
| **Stateless LLM Tool-Calling Architecture** | §14 Phase 5, §5 | `gods_eye/reasoning/` (empty package placeholder) | **MISSING** | Stateless query execution using structured tool-calling. |
| **10 Standard Tool Definitions** | §14 Phase 5 | Individual methods exist in `TimelineReconstructionEngine`, `GraphStore`, `EventStore`, `EnvironmentalWorker` | **PARTIALLY IMPLEMENTED** | Standardized tool wrapper layer required to unify query calls into clean typed schemas. |
| **Explainability & Evidence Requirement** | §2 Rule 8, §14 Phase 5 | `Event.explanation`, `ReplayFrame`, `VisitSegment` | **PARTIALLY IMPLEMENTED** | Every answer must cite exact `Evidence` records (event IDs, observation IDs, frame IDs, timestamps). |
| **Confidence & Uncertainty Quantification** | §2 Rule 7, §14 Phase 5 | `Event.confidence`, `TemporalObservation.confidence` | **PARTIALLY IMPLEMENTED** | Output `ReasoningResult` must aggregate tool output confidences into an explicit confidence score and band. |
| **Elevated Scope Authentication** | §10, §14 Phase 5 | `Settings.api_token` | **PARTIALLY IMPLEMENTED** | API endpoint requires token authentication and scope validation (`GODS_EYE_SCOPE=behavioral`/`reasoning`). |
| **Tool Selection Accuracy Gate** | §14 Phase 5 | Benchmarks in `benchmarks/` | **MISSING** | Benchmark harness evaluating $\ge 85\%$ correct tool selection on 50-query dataset. |
| **Query Latency Gate** | §14 Phase 5 | Benchmark harnesses | **MISSING** | Query response latency $\le 3\text{s } p95$ for 1-hour temporal history window. |
| **Factual Correctness Gate** | §14 Phase 5 | Held-out QA test suite | **MISSING** | $\ge 90\%$ factual correctness on held-out QA dataset. |

---

## 3. Current Repository Capability

The repository audit (`docs/audits/REPOSITORY_AUDIT_PHASE4.md`) confirms that all underlying spatiotemporal memory stores required by Phase 5 are fully implemented, tested, and frozen:

```
[Phase 4 Persistence Foundation - FROZEN]
├── SQLiteEventStore (data/events.db)           -> Authoritative Event Log (58,823 events/s)
├── SQLiteGraphStore (data/graph.db)            -> Spatiotemporal Graph Projection (41,667 records/s)
├── TimelineReconstructionEngine               -> Identity/Camera/Zone Reconstructor (0.84 ms p95)
└── DeterministicReplayEngine                  -> Replay Stream Generator (0.84 ms p95)

[Phase 5 Reasoning Layer - UNIMPLEMENTED Placeholder]
└── gods_eye/reasoning/__init__.py             -> Empty module placeholder (5 lines)
```

### Verified Phase 4 Entry Points Available for Phase 5
1. `TimelineReconstructionEngine.reconstruct_identity_timeline(global_id, start_ns, end_ns)`
2. `TimelineReconstructionEngine.reconstruct_camera_timeline(camera_id, start_ns, end_ns)`
3. `TimelineReconstructionEngine.reconstruct_zone_timeline(zone_id, start_ns, end_ns)`
4. `SQLiteGraphStore.get_identity_trajectory(global_id, start_ns, end_ns)`
5. `SQLiteGraphStore.get_camera_transitions(from_camera_id, to_camera_id, start_ns, end_ns)`
6. `SQLiteGraphStore.get_zone_occupancy(zone_id, start_ns, end_ns)`
7. `SQLiteEventStore.query_events(event_type, camera_id, global_id, zone_id, start_ns, end_ns, limit)`
8. `DeterministicReplayEngine.stream_replay(start_ns, end_ns, camera_id, global_id)`

---

## 4. Phase 5 Architecture & Data Flow

```
                               ┌────────────────────────────────────────┐
                               │           User Natural Language        │
                               │                Query                   │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                               ┌────────────────────────────────────────┐
                               │        ReasoningEngine API Boundary    │
                               │  (Authentication & Scope Enforcement)  │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                               ┌────────────────────────────────────────┐
                               │       NLQ Planner / Intent Parser      │
                               │ (LLM / Function Calling Orchestrator)  │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼ [ToolCall Schemas]
                               ┌────────────────────────────────────────┐
                               │       Deterministic Tool Dispatcher     │
                               │       (gods_eye.reasoning.tools)       │
                               └──────┬───────────┬───────────┬─────────┘
                                      │           │           │
           ┌──────────────────────────┘           │           └──────────────────────────┐
           ▼                                      ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐                ┌─────────────────────┐
│ Timeline Engine Tool│                │  Event Query Tool   │                │  Graph Store Tool   │
│  (Read-Only Query)  │                │  (Read-Only Query)  │                │  (Read-Only Query)  │
└──────────┬──────────┘                └──────────┬──────────┘                └──────────┬──────────┘
           │                                      │                                      │
           ▼                                      ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐                ┌─────────────────────┐
│ TimelineReconstruct │                │  SQLiteEventStore   │                │  SQLiteGraphStore   │
│       Engine        │                │   (data/events.db)  │                │   (data/graph.db)   │
└──────────┬──────────┘                └──────────┬──────────┘                └──────────┬──────────┘
           │                                      │                                      │
           └──────────────────────────┬───────────┴──────────────────────────────────────┘
                                      │
                                      ▼ [ToolResult & Verified Evidence Records]
                               ┌────────────────────────────────────────┐
                               │      Answer Synthesis & Verification   │
                               │  (Attaches Provenance & Confidence)    │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                               ┌────────────────────────────────────────┐
                               │            ReasoningResult             │
                               │    (Answer + Evidence + Confidence)    │
                               └────────────────────────────────────────┘
```

---

## 5. Component Responsibilities

1. **`ReasoningEngine` (`gods_eye.reasoning.engine`)**: Entry point API. Receives `UserQuery`, enforces security scope (`GODS_EYE_SCOPE=reasoning`), orchestrates query execution, enforces timeouts ($\le 3\text{s}$), records metrics, and returns formatted `ReasoningResult`.
2. **`NLQPlanner` (`gods_eye.reasoning.planner`)**: Translates natural language strings into one or more structured `ToolCall` objects. Uses an LLM provider interface (`LLMProvider`) or rule-based fallback parser to extract target entities (`global_id`, `camera_id`, `zone_id`), time windows (`start_ns`, `end_ns`), and event types.
3. **`ToolDispatcher` (`gods_eye.reasoning.tools.dispatcher`)**: Registry of deterministic tool handlers. Validates `ToolCall` arguments against Pydantic/dataclass schemas, invokes the appropriate read-only Phase 4 store engine, catches errors gracefully, and returns structured `ToolResult` objects.
4. **`EvidenceVerifier` (`gods_eye.reasoning.verifier`)**: Extracts concrete spatiotemporal records (events, graph edges, visit segments) from `ToolResult` data and constructs immutable `Evidence` references. Verifies that synthesized answer statements cite explicit event IDs, observation IDs, or timestamps.

---

## 6. Canonical Data Contracts

The Phase 5 data contracts will be defined in `gods_eye/schemas/reasoning.py` without mutating existing Phase 1–4 schemas:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

class QueryStatus(Enum):
    SUCCESS = "success"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_QUERY = "invalid_query"
    EXECUTION_ERROR = "execution_error"

@dataclass(frozen=True)
class UserQuery:
    query_id: str
    text: str
    timestamp_ns: int
    context_camera_id: Optional[str] = None
    default_time_window_s: float = 3600.0  # Default 1h history

@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_store: str                      # "event_store" | "graph_store"
    record_type: str                       # "event" | "observation_edge" | "transition_edge" | "visit_segment"
    record_id: str                         # UUID of underlying record
    timestamp_ns: int
    camera_id: Optional[str]
    global_id: Optional[str]
    explanation: str
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    data: Any
    evidence_list: list[Evidence] = field(default_factory=list)
    error_message: Optional[str] = None

@dataclass(frozen=True)
class ReasoningResult:
    query_id: str
    status: QueryStatus
    answer: str
    confidence: float                      # Range [0.0, 1.0]
    confidence_band: tuple[float, float]   # (lower, upper)
    evidence: list[Evidence] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    execution_time_ms: float = 0.0
    explanation: str = ""
```

---

## 7. Deterministic Tool Contracts (§14 Phase 5 Specification)

Phase 5 exposes exactly **10 deterministic tools** backed by Phase 4 query engines:

| Tool Name | Master Spec Definition (§14) | Authoritative Backend Interface | Input Arguments | Output Description |
|---|---|---|---|---|
| `get_identity_timeline` | Movement history for a global_id | `TimelineReconstructionEngine.reconstruct_identity_timeline()` | `global_id: str`, `start_ns: int`, `end_ns: int` | `IdentityTimeline` with visit segments and dwell times |
| `search_events_by_type` | Filter event log by type, time, camera, zone | `SQLiteEventStore.query_events()` | `event_type: Optional[str]`, `camera_id: Optional[str]`, `global_id: Optional[str]`, `zone_id: Optional[str]`, `start_ns: Optional[int]`, `end_ns: Optional[int]`, `limit: int` | List of matching `Event` records |
| `get_camera_path` | Reconstruct cross-camera path for identity | `SQLiteGraphStore.get_camera_transitions()` | `global_id: str`, `start_ns: int`, `end_ns: int` | Ordered list of `TransitionEdge` records |
| `list_active_identities` | Current identities in ACTIVE state | `SQLiteGraphStore.get_active_identities()` | `camera_id: Optional[str]` | List of active `global_id` strings with last seen timestamps |
| `query_location_at_time` | Where was X at timestamp T? | `TimelineReconstructionEngine` & `SQLiteGraphStore` | `global_id: str`, `target_ns: int` | `CameraNode` and `Zone` location at time T (or `None`) |
| `get_anomaly_signals` | Recent anomaly signals with explanations | `SQLiteEventStore.query_events(event_type=ENVIRONMENTAL_ANOMALY)` | `camera_id: Optional[str]`, `start_ns: Optional[int]`, `end_ns: Optional[int]` | List of environmental anomaly `Event` records |
| `get_occupancy_baseline` | Normal occupancy for zone/time | `EnvironmentalWorker.get_baseline()` | `camera_id: str`, `zone_id: Optional[str]`, `time_bucket: Optional[str]` | `OccupancyBaseline` statistical mean and std |
| `get_trajectory_clusters` | Scene movement patterns | Stubs / GraphStore Occupancy | `camera_id: str` | Discovered scene trajectory clusters (Phase 4.5/6 stub) |
| `get_behavioral_prediction` | Next likely identity location | `CameraGraph.get_transition_priors()` | `global_id: str`, `current_camera_id: str` | Candidate next `camera_id` with transition probability |
| `get_scene_state` | Current SceneState for a camera | `EnvironmentalWorker.get_scene_state()` | `camera_id: str` | Latest `SceneState` record (lighting, density, motion) |

---

## 8. NLQ / Reasoning Boundary Rules

To enforce Rule 7 (Uncertainty), Rule 8 (Explainability), and prevent model hallucinations:

1. **Zero Groundless Claims:** Answers MUST be synthesized strictly from records returned in `ToolResult.evidence_list`. If no records match, `ReasoningResult.status` MUST be set to `QueryStatus.INSUFFICIENT_EVIDENCE` with the explanation `"No matching historical records found in EventStore or GraphStore."`
2. **Ambiguity Rejection:** If a natural language query specifies a vague subject (e.g., *"Where did the guy in the red jacket go?"*) without a resolved `global_id`, the system MUST return `QueryStatus.AMBIGUOUS` requesting identity disambiguation.
3. **Strict Time Window Resolution:** Relative time terms (e.g., *"in the last 10 minutes"*, *"yesterday"*) are deterministically converted into absolute Unix nanosecond ranges `(start_ns, end_ns)` using `UserQuery.timestamp_ns` before invoking tools.
4. **LLM Scope Limit:** The LLM provider is strictly prohibited from executing raw database queries or altering memory state. LLM invocation is encapsulated inside `NLQPlanner.parse_intent()` and `AnswerSynthesizer.format_response()`.

---

## 9. Concurrency & Isolation Architecture

- **Read Isolation:** Phase 5 queries issue `SELECT` statements on SQLite WAL mode databases (`events.db` and `graph.db`). SQLite WAL mode allows concurrent readers to query without blocking concurrent `TemporalWorker` batch writes.
- **Perception Pipeline Non-Blocking Guarantee:** The `ReasoningEngine` executes in a separate thread pool or async handler. Perception workers (`DetectionWorker`, `TrackingWorker`, `IdentityWorker`, `TemporalWorker`) operate on independent background threads with zero lock contention against Phase 5 queries.
- **Resource Bounding:** Query processing uses a strict timeout executor (`concurrent.futures.ThreadPoolExecutor(max_workers=4)`). Queries taking longer than 3.0 seconds are cancelled and return `QueryStatus.EXECUTION_ERROR` with timeout diagnostic details.

---

## 10. Observability Requirements (§8 Master Spec)

Phase 5 will extend `MetricsRegistry` in `gods_eye/observability/metrics.py` with low-cardinality metrics:

```python
# Phase 5 Reasoning Metrics
self.reasoning_queries_total = Counter(
    "gods_eye_reasoning_queries_total",
    "Total reasoning queries processed",
    ["status"],
    registry=reg,
)
self.reasoning_query_latency_ms = Histogram(
    "gods_eye_reasoning_query_latency_ms",
    "Reasoning query processing latency in milliseconds",
    buckets=(10, 50, 100, 250, 500, 1000, 2000, 3000, 5000),
    registry=reg,
)
self.reasoning_tool_calls_total = Counter(
    "gods_eye_reasoning_tool_calls_total",
    "Total deterministic tool invocations",
    ["tool_name", "status"],
    registry=reg,
)
```

---

## 11. Testing Strategy & Performance Gates

### Test Suite Structure
1. `tests/test_reasoning_schemas.py`: Validation of `UserQuery`, `ToolCall`, `Evidence`, `ToolResult`, `ReasoningResult`.
2. `tests/test_reasoning_tools.py`: Unit tests for all 10 deterministic tool implementations against mock/in-memory `SQLiteEventStore` and `SQLiteGraphStore`.
3. `tests/test_reasoning_planner.py`: Intent parsing tests validating translation of natural language queries to structured `ToolCall` objects.
4. `tests/test_reasoning_engine.py`: Integration tests verifying end-to-end query execution, evidence attachment, confidence scoring, ambiguity rejection, and timeout handling.
5. `tests/test_reasoning_benchmarks.py`: Performance benchmark evaluating 50-query benchmark suite against Master Spec performance gates.

### Master Spec Phase Gate Criteria (§14 Phase 5)

| Criterion | Target Threshold | Verification Method |
|---|---|---|
| **Tool Selection Accuracy** | $\ge 85\%$ correct selection | 50-query benchmark evaluation |
| **Query Latency ($p95$)** | $\le 3,000\text{ ms } p95$ | 1-hour temporal history window benchmark |
| **Factual Correctness** | $\ge 90\%$ correctness | Held-out QA test suite |
| **Evidence Provenance** | $100\%$ evidence citation | Assertion that every success answer cites non-empty `Evidence` list |
| **Phase 1–4 Regression** | **0 regressions (272/272 pass)** | Full pytest suite execution |

---

## 12. Sub-Phase Implementation Roadmap

Phase 5 will be implemented incrementally across **5 sub-phases**:

### Sub-Phase 5.1 — Reasoning Schemas & Canonical Contracts
- **Files to Create:** `gods_eye/schemas/reasoning.py`, `tests/test_reasoning_schemas.py`
- **Files to Modify:** `gods_eye/schemas/__init__.py`
- **Goal:** Establish `UserQuery`, `ToolCall`, `Evidence`, `ToolResult`, `ReasoningResult`, and `QueryStatus` contracts.

### Sub-Phase 5.2 — Deterministic Tool Layer
- **Files to Create:** `gods_eye/reasoning/tools.py`, `tests/test_reasoning_tools.py`
- **Files to Modify:** `gods_eye/reasoning/__init__.py`
- **Goal:** Implement and test all 10 deterministic tool handlers wrapping `SQLiteEventStore`, `SQLiteGraphStore`, `TimelineReconstructionEngine`, and `DeterministicReplayEngine`.

### Sub-Phase 5.3 — NLQ Intent Planner & Rule Engine
- **Files to Create:** `gods_eye/reasoning/planner.py`, `tests/test_reasoning_planner.py`
- **Goal:** Implement `NLQPlanner` for parsing natural language queries into `ToolCall` sequences with support for structured pattern matching and optional LLM provider interface.

### Sub-Phase 5.4 — Reasoning Engine Orchestrator & Evidence Synthesis
- **Files to Create:** `gods_eye/reasoning/engine.py`, `gods_eye/reasoning/verifier.py`, `tests/test_reasoning_engine.py`
- **Files to Modify:** `gods_eye/observability/metrics.py`
- **Goal:** Assemble full reasoning pipeline, attach verified `Evidence` records, calculate confidence scores, enforce scope security, and log metrics.

### Sub-Phase 5.5 — Query Benchmarks & End-to-End Hardening
- **Files to Create:** `benchmarks/benchmark_reasoning_query.py`, `tests/test_reasoning_benchmarks.py`
- **Goal:** Execute 50-query benchmark suite, validate tool accuracy ($\ge 85\%$), verify query latency ($\le 3\text{s } p95$), verify factual correctness ($\ge 90\%$), and confirm 0 regressions across full project test suite.

---

## 13. Phase 4 Freeze Compatibility Verification

- [x] `gods_eye/schemas/observation.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/adapter.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/admission_control.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/events/event_store.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/events/event_writer.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/graph_store.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/graph_writer.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/timeline_engine.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/replay_engine.py` — **UNTOUCHED / FROZEN**
- [x] `gods_eye/memory/temporal_worker.py` — **UNTOUCHED / FROZEN**

---

## 14. Architectural Risk Register

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| **LLM Tool Selection Ambiguity** | Medium | Medium | Use strict keyword pattern matching rules first; fallback to LLM tool calling only when pattern match is ambiguous. |
| **Query Latency Spikes on Large DBs** | Low | High | Enforce strict query time windows (`start_ns`, `end_ns`) and pagination limits (`limit=1000`). Utilizes existing composite indexes. |
| **LLM Provider Availability / Network Latency** | Medium | Low | Ensure deterministic tool dispatcher operates independently of LLM. Fallback response provided if LLM API times out. |

---

## 15. Final Architectural Decision

### Decision: **PHASE 5 READY FOR IMPLEMENTATION**

**Rationale:**
1. All Phase 4 spatiotemporal memory stores, event logs, timeline engines, and replay engines are 100% frozen, benchmarked, and operational (`2e51484`, `v0.4.0-phase4-freeze`).
2. The entire 272-test project suite passes with zero regressions.
3. The Phase 5 architectural design preserves strict read-only isolation over Phase 4 memory components and guarantees non-blocking pipeline operation.

---

## 16. Immediate Next Implementation Step

Upon approval to proceed to Phase 5 implementation:
- **Begin Sub-Phase 5.1 — Reasoning Schemas & Canonical Contracts** by creating `gods_eye/schemas/reasoning.py` and `tests/test_reasoning_schemas.py`.
