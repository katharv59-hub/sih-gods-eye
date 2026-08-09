# Phase 5 — Tool Scope Verification Report

**Project:** God's Eye (`katharv59-hub/gods_eye`)  
**Milestone:** Phase 5 — Situational Reasoning & Natural Language Querying (NLQ)  
**Status:** **TOOL SCOPE VERIFICATION & CONTRACT AUDIT (NO CODE WRITTEN)**  
**Authoritative Basis:** `GODS_EYE_MASTER_SPEC.md` (v3.0, 2026-06-11), Existing ADRs, Source Code, `docs/audits/REPOSITORY_AUDIT_PHASE4.md`, `PHASE_4_FREEZE.md`  
**Date:** August 8, 2026  

---

## 1. Tool Scope Verification Matrix

| Tool | Explicitly Required by Master Spec? | Spec Section | Existing Implementation? | Existing Data Source? | Requires New Algorithm/Model? | Phase 5 Scope? | Decision |
|---|---|---|---|---|---|---|---|
| `get_identity_timeline` | Yes | §14 (Phase 4 & 5) | `TimelineReconstructionEngine.reconstruct_identity_timeline()` | `SQLiteGraphStore` / `SQLiteEventStore` | No | Yes | **IMPLEMENT IN PHASE 5** |
| `search_events_by_type` | Yes | §14 (Phase 4 & 5) | `SQLiteEventStore.query_events()` | `SQLiteEventStore` (`data/events.db`) | No | Yes | **IMPLEMENT IN PHASE 5** |
| `get_camera_path` | Yes | §14 (Phase 4 & 5) | `SQLiteGraphStore.get_camera_transitions()` | `SQLiteGraphStore` (`data/graph.db`) | No | Yes | **IMPLEMENT IN PHASE 5** |
| `list_active_identities` | Yes | §14 (Phase 4 & 5) | `SQLiteGraphStore.get_active_identities()` / `IdentityGallery` | `SQLiteGraphStore` / `IdentityGallery` | No | Yes | **IMPLEMENT IN PHASE 5** |
| `query_location_at_time` | Yes | §14 (Phase 4 & 5) | `TimelineReconstructionEngine` & `SQLiteGraphStore` | `SQLiteGraphStore` / `TimelineEngine` | No | Yes | **IMPLEMENT IN PHASE 5** |
| `get_anomaly_signals` | Yes | §14 (Phase 3.5 & 5) | `SQLiteEventStore.query_events(event_type=ENVIRONMENTAL_ANOMALY)` | `SQLiteEventStore` | No | Yes | **IMPLEMENT IN PHASE 5** |
| `get_occupancy_baseline` | Yes | §14 (Phase 3.5 & 5) | `OccupancyBaselineBuilder.get_baseline()` | `OccupancyBaselineBuilder` | No | Yes | **IMPLEMENT IN PHASE 5** |
| `get_trajectory_clusters` | Yes (Phase 4.5/6) | §4, §14.5, ADR-008 | No (`gods_eye/behavioral/` missing) | None | Yes (HDBSCAN) | Deferred | **DEFER TO LATER PHASE** |
| `get_behavioral_prediction` | Yes (Phase 6) | §4, §14 Phase 6, ADR-011 | Partial (`CameraGraph.get_transition_priors()`) | `CameraGraph` | Yes (Profile Model) | Fallback Only / Phase 6 | **DEFER TO LATER PHASE** |
| `get_scene_state` | Yes | §14 (Phase 3.5 & 5) | `EnvironmentalWorker.get_scene_state()` | `EnvironmentalWorker` / `GraphStore` | No | Yes | **IMPLEMENT IN PHASE 5** |

---

## 2. Special Case 1 Verification — `get_trajectory_clusters`

1. **Where does the Master Spec require it?** Master Spec §4 (`TrajectoryCluster` schema), §14.5 (Phase 4.5 Behavioral Pattern Detection), and §17 (ADR-008 HDBSCAN Trajectory Clustering).
2. **Which exact repository module currently performs trajectory clustering?** **NONE.** The directory `gods_eye/behavioral/` referenced in Master Spec §11 does not exist in the codebase.
3. **Which existing API exposes those clusters?** **NONE.**
4. **What algorithm currently exists?** **NONE.** (ADR-008 selects HDBSCAN, but no implementation exists in the codebase).
5. **If none exists, is clustering supposed to be implemented in Phase 5?** **NO.** Master Spec §14 Phase 5 defines Situational Reasoning as querying existing memory stores (`EventStore`, `GraphStore`, `TimelineEngine`, `ReplayEngine`). Trajectory clustering belongs to Phase 4.5 / Phase 6 (Behavioral Pattern Engine).
6. **If it belongs to another phase, which phase?** Phase 4.5 (Behavioral Pattern Detection) / Phase 6 (Identity Behavioral Intelligence).
7. **Final Decision:** **DEFER TO LATER PHASE**. In Phase 5, invoking `get_trajectory_clusters` will return an empty result list with status message `"Trajectory clustering deferred to Phase 4.5/6; no active trajectory clusters found."` rather than blocking Phase 5 tool registration.

---

## 3. Special Case 2 Verification — `get_behavioral_prediction`

1. **Where does the Master Spec require it?** Master Spec §4 (`MovementPrediction` schema), §14 Phase 6 (Identity Behavioral Intelligence), and §17 (ADR-011 Prediction Basis Priority: Profile $\rightarrow$ Cluster $\rightarrow$ Prior).
2. **Which exact repository module currently performs behavioral prediction?** `gods_eye/camera_graph/camera_graph.py` provides static camera transition probabilities (`CameraNode.transition_priors`). Per-identity behavioral profile prediction (`gods_eye/identity_profiler/`) does NOT exist.
3. **Which model currently produces predictions?** Static spatial transition priors stored on `CameraNode` topology (`transition_priors: dict[str, float]`).
4. **What existing data/API feeds it?** `CameraGraph.get_adjacent_cameras()` and transition priors in `gods_eye/camera_graph/camera_graph.py`.
5. **Is behavioral prediction explicitly Phase 5?** **NO.** Full behavioral prediction is explicitly Phase 6 (Identity Behavioral Intelligence, Spec §14 Phase 6).
6. **Is it deferred to a later phase?** **YES.** Per-identity behavioral prediction is deferred to Phase 6.
7. **Final Decision:** **DEFER TO LATER PHASE**. In Phase 5, basic movement prediction falls back strictly on Phase 3 `CameraGraph` transition priors (lowest basis per ADR-011), while per-identity profile predictions are deferred to Phase 6.

---

## 4. Phase 5.1 Schema Verification

Phase 5.1 is strictly scoped to creating `gods_eye/schemas/reasoning.py` and `tests/test_reasoning_schemas.py`.

### Schema Contract Analysis

1. **`UserQuery`**:
   - Explicitly required: Yes.
   - Represented elsewhere: No.
   - Fields: `query_id: str`, `text: str`, `timestamp_ns: int`, `context_camera_id: Optional[str]`, `default_time_window_s: float`.
   - Speculative fields: None. All fields map directly to query handling.

2. **`ToolCall`**:
   - Explicitly required: Yes (Spec §14 Phase 5 tool calling).
   - Represented elsewhere: No.
   - Fields: `call_id: str`, `tool_name: str`, `arguments: dict[str, Any]`.
   - Speculative fields: None.

3. **`Evidence`**:
   - Explicitly required: Yes (Spec §2 Rule 8 Explainability).
   - Represented elsewhere: No.
   - Fields: `evidence_id: str`, `source_store: str`, `record_type: str`, `record_id: str`, `timestamp_ns: int`, `camera_id: Optional[str]`, `global_id: Optional[str]`, `explanation: str`, `payload: dict[str, Any]`.
   - Speculative fields: None. Maps directly to underlying event and graph IDs.

4. **`ToolResult`**:
   - Explicitly required: Yes.
   - Represented elsewhere: No.
   - Fields: `call_id: str`, `tool_name: str`, `success: bool`, `data: Any`, `evidence_list: list[Evidence]`, `error_message: Optional[str]`.
   - Speculative fields: None.

5. **`ReasoningResult`**:
   - Explicitly required: Yes (Spec §2 Rule 7 Confidence & Rule 8 Explanation).
   - Represented elsewhere: No.
   - Fields: `query_id: str`, `status: QueryStatus`, `answer: str`, `confidence: float`, `confidence_band: tuple[float, float]`, `evidence: list[Evidence]`, `tool_calls: list[ToolCall]`, `execution_time_ms: float`, `explanation: str`.
   - Speculative fields: None.

6. **`QueryStatus`**:
   - Enum: `SUCCESS`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, `INVALID_QUERY`, `EXECUTION_ERROR`.

---

## 5. Verification Conclusion & Decision

- All 8 Phase 5 tools (`get_identity_timeline`, `search_events_by_type`, `get_camera_path`, `list_active_identities`, `query_location_at_time`, `get_anomaly_signals`, `get_occupancy_baseline`, `get_scene_state`) consume existing Phase 3–4 persistent engines and data sources.
- The 2 deferred tools (`get_trajectory_clusters` and `get_behavioral_prediction`) are safely scoped as Phase 4.5/6 deferrals, returning structured empty/fallback responses without blocking tool dispatch.
- Phase 5.1 is strictly isolated to schema contracts (`gods_eye/schemas/reasoning.py` and `tests/test_reasoning_schemas.py`) and introduces zero changes to Phase 1–4 source code or tests.

---

PHASE 5.1 STATUS:
SAFE TO IMPLEMENT

**First Implementation Task:** Create `gods_eye/schemas/reasoning.py` defining `UserQuery`, `ToolCall`, `Evidence`, `ToolResult`, `ReasoningResult`, and `QueryStatus`, update `gods_eye/schemas/__init__.py` exports, and add unit test suite in `tests/test_reasoning_schemas.py`.
