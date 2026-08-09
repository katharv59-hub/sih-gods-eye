# Phase 5.2 — Deterministic Tool Layer: Final Pre-Flight Review

**Project:** God's Eye (`katharv59-hub/gods_eye`)  
**Milestone:** Sub-Phase 5.2 — Deterministic Tool Layer  
**Status:** **PRE-FLIGHT REVIEW & CONTRACT AUDIT (NO CODE WRITTEN)**  
**Authoritative Basis:** `GODS_EYE_MASTER_SPEC.md` (v3.0), Existing ADRs, `gods_eye/schemas/reasoning.py` (Phase 5.1), `docs/audits/REPOSITORY_AUDIT_PHASE4.md`, `PHASE_4_FREEZE.md`  
**Date:** August 8, 2026  

---

## 1. Tool Contract & Backend Mapping Matrix

| Tool | Existing Backend | Existing API | Input Contract | Output Contract | Read-Only? | Deterministic? | Missing Dependency | Status |
|---|---|---|---|---|---|---|---|---|
| `get_identity_timeline` | `TimelineReconstructionEngine` | `reconstruct_identity_timeline()` | `global_id: str`, `start_ns: Optional[int]`, `end_ns: Optional[int]` | `IdentityTimeline` (visits list & dwell times) | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `search_events_by_type` | `SQLiteEventStore` | `query_events()` | `event_type: Optional[str]`, `camera_id: Optional[str]`, `global_id: Optional[str]`, `zone_id: Optional[str]`, `start_ns: Optional[int]`, `end_ns: Optional[int]`, `limit: int` | `list[Event]` | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `get_camera_path` | `SQLiteGraphStore` | `get_camera_transitions()` | `global_id: str`, `from_camera_id: Optional[str]`, `to_camera_id: Optional[str]`, `start_ns: Optional[int]`, `end_ns: Optional[int]` | `list[dict]` (transition edges) | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `list_active_identities` | `SQLiteGraphStore` | `get_active_identities()` | `camera_id: Optional[str]` | `list[dict]` (active identity nodes) | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `query_location_at_time` | `TimelineReconstructionEngine` & `SQLiteGraphStore` | `reconstruct_identity_timeline()` / `get_identity_trajectory()` | `global_id: str`, `target_ns: int` | `dict` (location snapshot: camera_id, zone_id, timestamp_ns) | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `get_anomaly_signals` | `SQLiteEventStore` | `query_events(event_type=ENVIRONMENTAL_ANOMALY)` | `camera_id: Optional[str]`, `start_ns: Optional[int]`, `end_ns: Optional[int]`, `limit: int` | `list[Event]` (filtered environmental anomalies) | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `get_occupancy_baseline` | `OccupancyBaselineBuilder` / `SQLiteGraphStore` | `get_baseline()` | `camera_id: str`, `zone_id: Optional[str]`, `time_bucket: Optional[str]` | `dict` / `OccupancyBaseline` (mean, std, samples) | Yes | Yes | None | **IMPLEMENTABLE NOW** |
| `get_scene_state` | `EnvironmentalWorker` / `SQLiteGraphStore` | `get_scene_state()` / `get_camera_node()` | `camera_id: str` | `dict` / `SceneState` (lighting, density, motion) | Yes | Yes | None | **IMPLEMENTABLE NOW** |

---

## 2. Phase 5.1 Contract Compatibility

The Phase 5.1 schemas defined in `gods_eye/schemas/reasoning.py` (`QueryStatus`, `UserQuery`, `ToolCall`, `Evidence`, `ToolResult`, `ReasoningResult`) were inspected against all 8 tool contracts:

- **`UserQuery`**: Successfully represents query context (`query_id`, `text`, `timestamp_ns`, `context_camera_id`, `default_time_window_s`).
- **`ToolCall`**: Accurately encapsulates `call_id`, `tool_name`, and `arguments: dict[str, Any]` for all 8 tool signatures.
- **`ToolResult`**: Wraps handler output (`success`, `data`, `evidence_list`, `error_message`) without requiring schema modifications.
- **`Evidence`**: Directly attaches `source_store` (`"event_store"` / `"graph_store"` / `"environmental"`), `record_type` (`"event"`, `"obs_edge"`, `"trans_edge"`, `"visit_segment"`), and `record_id` (UUIDs) extracted from Phase 4 query outputs.
- **`ReasoningResult`**: Consumes tool execution outputs and aggregated evidence lists cleanly.
- **`QueryStatus`**: Contains `SUCCESS`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, `INVALID_QUERY`, and `EXECUTION_ERROR` to represent all tool execution outcomes.

**Compatibility Verdict:** **100% COMPATIBLE. NO SCHEMA MODIFICATIONS REQUIRED.**

---

## 3. Read-Only Invariant & Phase 4 Freeze Verification

- **Read-Only Invariant:** All 8 tool handlers execute read-only queries (`SELECT` SQL queries, derived timeline reconstruction algorithms, and parameter inspection). Zero `INSERT`, `UPDATE`, `DELETE`, or queue mutations are performed.
- **Phase 4 Freeze Compatibility:** Phase 5.2 tool handlers in `gods_eye/reasoning/tools.py` will wrap existing Phase 4 public interfaces as consumers without modifying:
  - `gods_eye/events/event_store.py` (`SQLiteEventStore`, `BaseEventStore`)
  - `gods_eye/memory/graph_store.py` (`SQLiteGraphStore`, `BaseGraphStore`)
  - `gods_eye/memory/timeline_engine.py` (`TimelineReconstructionEngine`)
  - `gods_eye/memory/replay_engine.py` (`DeterministicReplayEngine`)
  - `gods_eye/memory/temporal_worker.py` (`TemporalWorker`)
  - All Phase 1–4 perception and schema modules.

---

## 4. Error & Edge-Case Audit Matrix

| Scenario / Edge Case | Tool Handler Behavior | QueryStatus / Exception Behavior | Evidence Attachment |
|---|---|---|---|
| **Unknown `global_id`** | Returns `ToolResult(success=True, data=[])` | `INSUFFICIENT_EVIDENCE` | Empty `evidence_list` |
| **Unknown `camera_id`** | Returns `ToolResult(success=True, data=[])` | `INSUFFICIENT_EVIDENCE` | Empty `evidence_list` |
| **Unknown `zone_id`** | Returns `ToolResult(success=True, data=[])` | `INSUFFICIENT_EVIDENCE` | Empty `evidence_list` |
| **Empty Result Set** | Returns `ToolResult(success=True, data=[])` | `INSUFFICIENT_EVIDENCE` | Empty `evidence_list` |
| **Invalid Time Range (`start_ns > end_ns`)** | Catches validation error, returns `ToolResult(success=False, error_message=...)` | `INVALID_QUERY` | Empty `evidence_list` |
| **Invalid Timestamp (`timestamp_ns <= 0`)** | Catches validation error, returns `ToolResult(success=False, error_message=...)` | `INVALID_QUERY` | Empty `evidence_list` |
| **Invalid `event_type` string** | Catches ValueError during Enum lookup, returns `ToolResult(success=False)` | `INVALID_QUERY` | Empty `evidence_list` |
| **Missing Required Argument** | Catches KeyError/TypeError during argument parsing, returns `ToolResult(success=False)` | `INVALID_QUERY` | Empty `evidence_list` |
| **SQLite Storage / Query Error** | Catches `sqlite3.Error`, logs warning, returns `ToolResult(success=False, error_message=...)` | `EXECUTION_ERROR` | Empty `evidence_list` |

---

## 5. Phase 5.2 Test Plan Requirements

1. **`test_tool_dispatcher_registration`**: Verify that all 8 tools are registered in `ToolDispatcher` with valid input schemas.
2. **`test_get_identity_timeline_tool`**: Test valid timeline query, missing identity, time range filtering, and evidence generation.
3. **`test_search_events_by_type_tool`**: Test filtering by event type, camera, global identity, time window, limit, and invalid event type.
4. **`test_get_camera_path_tool`**: Test multi-camera transition path retrieval and evidence edge extraction.
5. **`test_list_active_identities_tool`**: Test active identity query across cameras and camera-filtered active query.
6. **`test_query_location_at_time_tool`**: Test location snapshot resolution at exact timestamp T and handle missing location.
7. **`test_get_anomaly_signals_tool`**: Test environmental anomaly query filtering.
8. **`test_get_occupancy_baseline_tool`**: Test occupancy baseline parameter lookup.
9. **`test_get_scene_state_tool`**: Test camera scene state inspection.
10. **`test_read_only_guarantee`**: Assert zero database mutations across all 8 tool handler invocations.

---

## 6. Open Architectural Decisions & Specific Implementation Scope

- **Implementation Scope (Phase 5.2):**
  1. Create `gods_eye/reasoning/tools.py` implementing `ToolDispatcher` and 8 deterministic tool handlers (`get_identity_timeline`, `search_events_by_type`, `get_camera_path`, `list_active_identities`, `query_location_at_time`, `get_anomaly_signals`, `get_occupancy_baseline`, `get_scene_state`).
  2. Update `gods_eye/reasoning/__init__.py` to export `ToolDispatcher` and tool handler interfaces.
  3. Create `tests/test_reasoning_tools.py` implementing the 10 test groups in the test plan.
  4. Create `docs/phases/phase_5/PHASE_5_2_COMPLETION.md`.
- **Out of Scope (Deferred):** `get_trajectory_clusters` (Phase 4.5/6) and `get_behavioral_prediction` (Phase 6).

---

PHASE 5.2 STATUS:
SAFE TO IMPLEMENT
