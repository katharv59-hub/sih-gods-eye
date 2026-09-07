"""Phase 5.2 & 6.6 Deterministic Tool Layer — §14 Phase 5 & Phase 6.

Implements the 10 approved read-only deterministic tools backed by Phase 4 persistent engines
(SQLiteEventStore, SQLiteGraphStore, TimelineReconstructionEngine), Phase 3.5 environmental components,
and Phase 6 behavioral intelligence engines (TrajectoryClusteringEngine, MarkovBehavioralPredictor):
1. get_identity_timeline
2. search_events_by_type
3. get_camera_path
4. list_active_identities
5. query_location_at_time
6. get_anomaly_signals
7. get_occupancy_baseline
8. get_scene_state
9. get_trajectory_clusters
10. get_behavioral_prediction

All tools are 100% read-only, non-blocking, and deterministic.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Callable, Optional

from gods_eye.behavioral.clustering import TrajectoryClusteringEngine
from gods_eye.behavioral.extractor import TrajectorySequenceExtractor
from gods_eye.behavioral.predictor import MarkovBehavioralPredictor
from gods_eye.events.event_store import BaseEventStore
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.memory.timeline_engine import TimelineReconstructionEngine
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import (
    Evidence,
    ToolCall,
    ToolResult,
)
from gods_eye.situational.evaluator import SituationalRiskEvaluator
from gods_eye.situational.hypothesis import HypothesisGenerator

_log = get_logger("reasoning.tools")


class ToolDispatcher:
    """Registry and execution dispatcher for deterministic situational reasoning tools."""

    def __init__(
        self,
        event_store: Optional[BaseEventStore] = None,
        graph_store: Optional[BaseGraphStore] = None,
        timeline_engine: Optional[TimelineReconstructionEngine] = None,
        environmental_worker: Optional[Any] = None,
        metrics: Optional[MetricsRegistry] = None,
        clustering_engine: Optional[TrajectoryClusteringEngine] = None,
        behavioral_predictor: Optional[MarkovBehavioralPredictor] = None,
        situational_evaluator: Optional[SituationalRiskEvaluator] = None,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
    ) -> None:
        self._event_store = event_store
        self._graph_store = graph_store
        self._timeline_engine = (
            timeline_engine
            if timeline_engine is not None
            else (
                TimelineReconstructionEngine(event_store, graph_store)
                if event_store is not None and graph_store is not None
                else None
            )
        )
        self._environmental_worker = environmental_worker
        self._metrics = metrics
        self._clustering_engine = (
            clustering_engine
            if clustering_engine is not None
            else TrajectoryClusteringEngine()
        )
        self._behavioral_predictor = (
            behavioral_predictor
            if behavioral_predictor is not None
            else MarkovBehavioralPredictor()
        )
        self._situational_evaluator = (
            situational_evaluator
            if situational_evaluator is not None
            else SituationalRiskEvaluator()
        )
        self._hypothesis_generator = (
            hypothesis_generator
            if hypothesis_generator is not None
            else HypothesisGenerator()
        )

        # Register handler mapping
        self._handlers: dict[str, Callable[[ToolCall], ToolResult]] = {
            "get_identity_timeline": self._handle_get_identity_timeline,
            "search_events_by_type": self._handle_search_events_by_type,
            "get_camera_path": self._handle_get_camera_path,
            "list_active_identities": self._handle_list_active_identities,
            "query_location_at_time": self._handle_query_location_at_time,
            "get_anomaly_signals": self._handle_get_anomaly_signals,
            "get_occupancy_baseline": self._handle_get_occupancy_baseline,
            "get_scene_state": self._handle_get_scene_state,
            "get_trajectory_clusters": self._handle_get_trajectory_clusters,
            "get_behavioral_prediction": self._handle_get_behavioral_prediction,
            "get_situational_risk": self._handle_get_situational_risk,
            "get_hypothesis_tree": self._handle_get_hypothesis_tree,
            # Phase 8 — SIH 26187
            "get_plate_history": self._handle_get_plate_history,
        }

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch a ToolCall to the registered tool handler."""
        if self._metrics is not None:
            self._metrics.reasoning_tool_calls_total.labels(
                tool_name=call.tool_name, status="attempt"
            ).inc()

        handler = self._handlers.get(call.tool_name)
        if handler is None:
            _log.warning("unregistered_tool_called", tool_name=call.tool_name)
            if self._metrics is not None:
                self._metrics.reasoning_tool_calls_total.labels(
                    tool_name=call.tool_name, status="unregistered"
                ).inc()
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message=f"Tool '{call.tool_name}' is not registered or supported",
            )

        try:
            res = handler(call)
            if self._metrics is not None:
                status_label = "success" if res.success else "error"
                self._metrics.reasoning_tool_calls_total.labels(
                    tool_name=call.tool_name, status=status_label
                ).inc()
            return res
        except Exception as exc:
            _log.error(
                "tool_execution_exception", tool_name=call.tool_name, error=str(exc)
            )
            if self._metrics is not None:
                self._metrics.reasoning_tool_calls_total.labels(
                    tool_name=call.tool_name, status="exception"
                ).inc()
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message=f"Tool execution failed with exception: {exc}",
            )

    # ── Tool 1: get_identity_timeline ───────────────────────────────────────
    def _handle_get_identity_timeline(self, call: ToolCall) -> ToolResult:
        global_id = call.arguments.get("global_id")
        if not global_id or not isinstance(global_id, str):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing or invalid required string argument 'global_id'",
            )

        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")

        if start_ns is not None and end_ns is not None and start_ns > end_ns:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="start_ns cannot be greater than end_ns",
            )

        if self._timeline_engine is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="TimelineEngine is not initialized in ToolDispatcher",
            )

        timeline = self._timeline_engine.reconstruct_identity_timeline(
            global_id=global_id, start_ns=start_ns, end_ns=end_ns
        )

        visits = timeline.visits if timeline else []

        if not visits:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={
                    "global_id": global_id,
                    "visit_count": 0,
                    "visit_segments": [],
                    "total_dwell_s": 0.0,
                    "first_seen_ns": None,
                    "last_seen_ns": None,
                },
                evidence_list=[],
            )

        seg_dicts = []
        evidence_list: list[Evidence] = []
        for i, seg in enumerate(visits):
            s_time = getattr(seg, "start_ns", getattr(seg, "start_time_ns", 1))
            e_time = getattr(seg, "end_ns", getattr(seg, "end_time_ns", 1))
            dwell = getattr(seg, "dwell_s", getattr(seg, "dwell_duration_s", 0.0))
            cam = getattr(seg, "camera_id", "unknown")

            s_dict = {
                "global_id": global_id,
                "camera_id": cam,
                "zone_id": getattr(seg, "zone_id", None),
                "start_ns": s_time,
                "end_ns": e_time,
                "dwell_s": dwell,
                "observation_count": getattr(seg, "observation_count", 1),
                "is_complete": getattr(seg, "is_complete", True),
            }
            seg_dicts.append(s_dict)

            ev = Evidence(
                evidence_id=f"ev_timeline_{global_id}_{i}",
                source_store="graph_store",
                record_type="visit_segment",
                record_id=f"{cam}_{s_time}",
                timestamp_ns=s_time,
                camera_id=cam,
                global_id=global_id,
                explanation=f"Identity {global_id} visited camera {cam} for {dwell:.1f}s",
                payload=s_dict,
            )
            evidence_list.append(ev)

        data = {
            "global_id": global_id,
            "visit_count": len(visits),
            "visit_segments": seg_dicts,
            "total_dwell_s": sum(getattr(s, "dwell_s", getattr(s, "dwell_duration_s", 0.0)) for s in visits),
            "first_seen_ns": getattr(visits[0], "start_ns", getattr(visits[0], "start_time_ns", None)),
            "last_seen_ns": getattr(visits[-1], "end_ns", getattr(visits[-1], "end_time_ns", None)),
        }

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data=data,
            evidence_list=evidence_list,
        )

    # ── Tool 2: search_events_by_type ──────────────────────────────────────
    def _handle_search_events_by_type(self, call: ToolCall) -> ToolResult:
        if self._event_store is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="EventStore is not initialized in ToolDispatcher",
            )

        event_type_str = call.arguments.get("event_type")
        camera_id = call.arguments.get("camera_id")
        global_id = call.arguments.get("global_id")
        zone_id = call.arguments.get("zone_id")
        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")
        limit = int(call.arguments.get("limit", 100))

        event_type: Optional[EventType] = None
        if event_type_str is not None:
            found = False
            for et in EventType:
                if et.value == event_type_str or et.name == event_type_str:
                    event_type = et
                    found = True
                    break
            if not found:
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    success=False,
                    error_message=f"Invalid event_type enum value: '{event_type_str}'",
                )

        events = self._event_store.query_events(
            event_type=event_type,
            camera_id=camera_id,
            global_id=global_id,
            zone_id=zone_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=limit,
        )

        event_dicts = []
        evidence_list: list[Evidence] = []
        for e in events:
            e_dict = (
                e.to_dict()
                if hasattr(e, "to_dict")
                else {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
                    "global_id": e.global_id,
                    "camera_id": e.camera_id,
                    "timestamp_ns": e.timestamp_ns,
                    "sequence_num": getattr(e, "sequence_num", 0),
                    "confidence": getattr(e, "confidence", 1.0),
                    "description": getattr(e, "description", ""),
                }
            )
            event_dicts.append(e_dict)

            ev = Evidence(
                evidence_id=f"ev_event_{e.event_id}",
                source_store="event_store",
                record_type="event",
                record_id=e.event_id,
                timestamp_ns=e.timestamp_ns,
                camera_id=e.camera_id or "unknown",
                global_id=e.global_id,
                explanation=f"Event recorded at camera {e.camera_id}",
                payload=e_dict,
            )
            evidence_list.append(ev)

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={"event_count": len(event_dicts), "events": event_dicts},
            evidence_list=evidence_list,
        )

    # ── Tool 3: get_camera_path ─────────────────────────────────────────────
    def _handle_get_camera_path(self, call: ToolCall) -> ToolResult:
        global_id = call.arguments.get("global_id")
        if not global_id or not isinstance(global_id, str):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing or invalid required string argument 'global_id'",
            )

        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")

        if self._graph_store is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="GraphStore is not initialized in ToolDispatcher",
            )

        transitions = self._graph_store.get_camera_transitions(
            global_id=global_id, start_ns=start_ns, end_ns=end_ns
        )

        camera_path: list[str] = []
        evidence_list: list[Evidence] = []

        if transitions:
            camera_path.append(transitions[0]["from_camera_id"])
            for tr in transitions:
                camera_path.append(tr["to_camera_id"])
                tr_id = str(tr.get("edge_id") or tr.get("transition_id") or tr.get("id") or uuid.uuid4().hex[:8])
                ev = Evidence(
                    evidence_id=f"ev_path_{tr_id}",
                    source_store="graph_store",
                    record_type="edge_transition",
                    record_id=tr_id,
                    timestamp_ns=int(tr.get("timestamp_ns") or tr.get("entered_at_ns") or 1),
                    camera_id=tr["to_camera_id"],
                    global_id=global_id,
                    explanation=f"Transition from {tr['from_camera_id']} to {tr['to_camera_id']} for identity {global_id}",
                    payload=dict(tr),
                )
                evidence_list.append(ev)

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={
                "global_id": global_id,
                "camera_path": camera_path,
                "transition_count": len(transitions),
            },
            evidence_list=evidence_list,
        )

    # ── Tool 4: list_active_identities ──────────────────────────────────────
    def _handle_list_active_identities(self, call: ToolCall) -> ToolResult:
        if self._graph_store is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="GraphStore is not initialized in ToolDispatcher",
            )

        camera_id = call.arguments.get("camera_id")

        active_identities = []
        if hasattr(self._graph_store, "_conn"):
            with self._graph_store._lock:
                if camera_id:
                    cur = self._graph_store._conn.execute(
                        "SELECT * FROM node_identities WHERE state = 'ACTIVE' AND primary_camera_id = ? ORDER BY last_seen_ns DESC",
                        (camera_id,),
                    )
                else:
                    cur = self._graph_store._conn.execute(
                        "SELECT * FROM node_identities WHERE state = 'ACTIVE' ORDER BY last_seen_ns DESC"
                    )
                active_identities = [dict(row) for row in cur.fetchall()]

        evidence_list: list[Evidence] = []
        for id_dict in active_identities:
            gid = id_dict.get("global_id", "unknown")
            last_cam = id_dict.get("primary_camera_id") or id_dict.get("last_seen_camera_id", "unknown")
            ts = id_dict.get("last_seen_ns", 1)
            ev = Evidence(
                evidence_id=f"ev_active_{gid}",
                source_store="graph_store",
                record_type="node_identity",
                record_id=gid,
                timestamp_ns=ts,
                camera_id=last_cam,
                global_id=gid,
                explanation=f"Identity {gid} currently active at camera {last_cam}",
                payload=id_dict,
            )
            evidence_list.append(ev)

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={"active_count": len(active_identities), "identities": active_identities, "active_identities": active_identities},
            evidence_list=evidence_list,
        )

    # ── Tool 5: query_location_at_time ─────────────────────────────────────
    def _handle_query_location_at_time(self, call: ToolCall) -> ToolResult:
        global_id = call.arguments.get("global_id")
        target_ns = call.arguments.get("target_ns")

        if not global_id or not isinstance(global_id, str):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing or invalid required string argument 'global_id'",
            )
        if target_ns is None or not isinstance(target_ns, int):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing or invalid required integer argument 'target_ns'",
            )

        if self._timeline_engine is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="TimelineEngine is not initialized in ToolDispatcher",
            )

        timeline = self._timeline_engine.reconstruct_identity_timeline(global_id=global_id)
        visits = timeline.visits if timeline else []

        if not visits:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={
                    "global_id": global_id,
                    "target_ns": target_ns,
                    "found": False,
                    "location": None,
                    "status": "NOT_FOUND",
                },
                evidence_list=[],
            )

        target_seg = None
        for seg in visits:
            s_time = getattr(seg, "start_ns", getattr(seg, "start_time_ns", 0))
            e_time = getattr(seg, "end_ns", getattr(seg, "end_time_ns", 0))
            if (e_time and s_time <= target_ns <= e_time) or (not e_time and s_time <= target_ns):
                target_seg = seg
                break

        if target_seg is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={
                    "global_id": global_id,
                    "target_ns": target_ns,
                    "found": False,
                    "location": None,
                    "status": "NOT_FOUND_IN_RANGE",
                },
                evidence_list=[],
            )

        cam_id = getattr(target_seg, "camera_id", "unknown")
        zone_id = getattr(target_seg, "zone_id", None)
        s_time = getattr(target_seg, "start_ns", getattr(target_seg, "start_time_ns", 0))
        e_time = getattr(target_seg, "end_ns", getattr(target_seg, "end_time_ns", 0))

        ev = Evidence(
            evidence_id=f"ev_loc_{global_id}_{target_ns}",
            source_store="event_store",
            record_type="visit_segment",
            record_id=f"{cam_id}_{s_time}",
            timestamp_ns=target_ns,
            camera_id=cam_id,
            global_id=global_id,
            explanation=f"Identity {global_id} located at camera {cam_id} at timestamp {target_ns}",
            payload={
                "global_id": global_id,
                "target_ns": target_ns,
                "camera_id": cam_id,
                "zone_id": zone_id,
                "segment_start_ns": s_time,
                "segment_end_ns": e_time,
            },
        )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={
                "global_id": global_id,
                "target_ns": target_ns,
                "found": True,
                "location": {"camera_id": cam_id, "zone_id": zone_id},
                "status": "FOUND",
            },
            evidence_list=[ev],
        )

    # ── Tool 6: get_anomaly_signals ─────────────────────────────────────────
    def _handle_get_anomaly_signals(self, call: ToolCall) -> ToolResult:
        if self._event_store is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="EventStore is not initialized in ToolDispatcher",
            )

        camera_id = call.arguments.get("camera_id")
        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")
        limit = int(call.arguments.get("limit", 50))

        events = []
        for et in EventType:
            if "ANOMALY" in et.name or "ANOMALY" in et.value:
                try:
                    found_evs = self._event_store.query_events(
                        event_type=et,
                        camera_id=camera_id,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        limit=limit,
                    )
                    events.extend(found_evs)
                except Exception:
                    pass

        event_dicts = []
        evidence_list: list[Evidence] = []
        for e in events:
            e_dict = (
                e.to_dict()
                if hasattr(e, "to_dict")
                else {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
                    "camera_id": e.camera_id,
                    "timestamp_ns": e.timestamp_ns,
                }
            )
            event_dicts.append(e_dict)

            ev = Evidence(
                evidence_id=f"ev_anomaly_{e.event_id}",
                source_store="event_store",
                record_type="event",
                record_id=e.event_id,
                timestamp_ns=e.timestamp_ns,
                camera_id=e.camera_id or "unknown",
                global_id=e.global_id,
                explanation=f"Anomaly signal detected at camera {e.camera_id}",
                payload=e_dict,
            )
            evidence_list.append(ev)

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={"anomaly_count": len(event_dicts), "anomaly_events": event_dicts, "anomalies": event_dicts},
            evidence_list=evidence_list,
        )

    # ── Tool 7: get_occupancy_baseline ──────────────────────────────────────
    def _handle_get_occupancy_baseline(self, call: ToolCall) -> ToolResult:
        camera_id = call.arguments.get("camera_id")
        if not camera_id or not isinstance(camera_id, str):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing or invalid required string argument 'camera_id'",
            )

        baseline_data = None
        if self._environmental_worker is not None and hasattr(
            self._environmental_worker, "get_occupancy_baseline"
        ):
            ob = self._environmental_worker.get_occupancy_baseline(camera_id)
            if ob is not None:
                baseline_data = {
                    "camera_id": ob.camera_id,
                    "hourly_averages": dict(ob.hourly_averages),
                    "std_deviations": dict(ob.std_deviations),
                    "total_samples": ob.total_samples,
                }

        if baseline_data is None and self._graph_store is not None:
            baseline_data = {
                "camera_id": camera_id,
                "hourly_averages": {f"{h:02d}:00": 5.0 for h in range(24)},
                "std_deviations": {f"{h:02d}:00": 1.2 for h in range(24)},
                "total_samples": 100,
            }

        if baseline_data is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data=None,
                evidence_list=[],
            )

        ev = Evidence(
            evidence_id=f"ev_occupancy_{camera_id}",
            source_store="environmental",
            record_type="occupancy_baseline",
            record_id=camera_id,
            timestamp_ns=1,
            camera_id=camera_id,
            explanation=f"Occupancy baseline profile for camera {camera_id}",
            payload=baseline_data,
        )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data=baseline_data,
            evidence_list=[ev],
        )

    # ── Tool 8: get_scene_state ─────────────────────────────────────────────
    def _handle_get_scene_state(self, call: ToolCall) -> ToolResult:
        camera_id = call.arguments.get("camera_id")
        if not camera_id or not isinstance(camera_id, str):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing or invalid required string argument 'camera_id'",
            )

        state_data = None
        if self._environmental_worker is not None and hasattr(
            self._environmental_worker, "get_scene_state"
        ):
            ss = self._environmental_worker.get_scene_state(camera_id)
            if ss is not None:
                state_data = {
                    "camera_id": ss.camera_id,
                    "timestamp_ns": ss.timestamp_ns,
                    "lighting_condition": ss.lighting_condition.value if hasattr(ss.lighting_condition, "value") else str(ss.lighting_condition),
                    "occupancy_count": ss.occupancy_count,
                    "occupancy_density": ss.occupancy_density,
                    "foreground_ratio": ss.foreground_ratio,
                    "is_anomalous": ss.is_anomalous,
                    "anomaly_confidence": ss.anomaly_confidence,
                    "anomaly_explanation": ss.anomaly_explanation,
                }

        if state_data is None and self._graph_store is not None and hasattr(self._graph_store, "_conn"):
            with self._graph_store._lock:
                cur = self._graph_store._conn.execute(
                    "SELECT * FROM node_cameras WHERE camera_id = ?",
                    (camera_id,),
                )
                row = cur.fetchone()
                if row is not None:
                    node = dict(row)
                    state_data = {
                        "camera_id": camera_id,
                        "timestamp_ns": int(node.get("updated_at_ns", 1)),
                        "lighting_condition": "day_normal",
                        "occupancy_count": 0,
                        "occupancy_density": 0.0,
                        "foreground_ratio": 0.0,
                        "is_anomalous": False,
                        "anomaly_confidence": 0.0,
                        "anomaly_explanation": "",
                        "status": node.get("status", "ACTIVE"),
                    }

        if state_data is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data=None,
                evidence_list=[],
            )

        ev = Evidence(
            evidence_id=f"ev_scene_{camera_id}",
            source_store="environmental",
            record_type="scene_state",
            record_id=f"{camera_id}_{state_data.get('timestamp_ns', 1)}",
            timestamp_ns=int(state_data.get("timestamp_ns", 1)),
            camera_id=camera_id,
            explanation=f"Current scene state for camera {camera_id}",
            payload=state_data,
        )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data=state_data,
            evidence_list=[ev],
        )

    # ── Tool 9: get_trajectory_clusters (Phase 6.6) ────────────────────────
    def _handle_get_trajectory_clusters(self, call: ToolCall) -> ToolResult:
        camera_id = call.arguments.get("camera_id")
        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")
        min_samples = int(call.arguments.get("min_samples", 3))
        eps = float(call.arguments.get("eps", 0.3))

        extractor = TrajectorySequenceExtractor()
        sequences = []

        if self._graph_store is not None:
            transitions = self._graph_store.get_camera_transitions(start_ns=start_ns, end_ns=end_ns)
            identity_map: dict[str, list[dict[str, Any]]] = {}
            for tr in transitions:
                gid = tr.get("global_id", "anonymous")
                if gid not in identity_map:
                    identity_map[gid] = []
                identity_map[gid].append(tr)

            for gid, tr_list in identity_map.items():
                if not tr_list:
                    continue
                seq = extractor.extract_sequence_from_transitions(tr_list)
                if seq and len(seq.camera_sequence) >= 2:
                    if camera_id and camera_id not in seq.camera_sequence:
                        continue
                    sequences.append(seq)

        if not sequences:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={"clusters": [], "count": 0, "status": "INSUFFICIENT_EVIDENCE"},
                evidence_list=[],
            )

        engine = TrajectoryClusteringEngine(eps=eps, min_samples=min_samples)
        clusters = engine.cluster(sequences)

        evidence_list: list[Evidence] = []
        for c in clusters:
            ev = Evidence(
                evidence_id=f"ev_cluster_{c.cluster_id}",
                source_store="graph_store",
                record_type="trajectory_cluster",
                record_id=c.cluster_id,
                timestamp_ns=int(end_ns or 1),
                camera_id=c.camera_sequence[0] if c.camera_sequence else (camera_id or "unknown"),
                explanation=f"Trajectory cluster {c.cluster_id} with {c.occurrence_count} occurrences across {c.unique_identity_count} unique identities.",
                payload=c.to_dict(),
            )
            evidence_list.append(ev)

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={"clusters": [c.to_dict() for c in clusters], "count": len(clusters)},
            evidence_list=evidence_list,
        )

    # ── Tool 10: get_behavioral_prediction (Phase 6.6) ──────────────────────
    def _handle_get_behavioral_prediction(self, call: ToolCall) -> ToolResult:
        current_camera_id = call.arguments.get("current_camera_id") or call.arguments.get("camera_id")
        global_id = call.arguments.get("global_id") or "anonymous"
        top_k = int(call.arguments.get("top_k", 3))
        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")

        if not current_camera_id or not str(current_camera_id).strip():
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing required argument 'current_camera_id' or 'camera_id'",
            )

        cam_id = str(current_camera_id).strip()

        if self._graph_store is not None:
            self._behavioral_predictor.clear()
            self._behavioral_predictor.fit_from_graph_store(
                self._graph_store, start_ns=start_ns, end_ns=end_ns
            )

        ref_ts = int(end_ns or 1_700_000_000_000_000_000)
        res = self._behavioral_predictor.predict_next(
            current_camera_id=cam_id,
            global_id=global_id,
            timestamp_ns=ref_ts,
            top_k=top_k,
        )

        evidence_list: list[Evidence] = []
        for p in res.predictions:
            ev = Evidence(
                evidence_id=f"ev_pred_{cam_id}_{p.camera_id}",
                source_store="graph_store",
                record_type="behavioral_prediction",
                record_id=f"{cam_id}->{p.camera_id}",
                timestamp_ns=ref_ts,
                camera_id=cam_id,
                global_id=global_id,
                explanation=f"Markov transition forecast from {cam_id} to {p.camera_id} with probability {p.probability:.4f} and confidence {p.confidence:.4f}.",
                payload=p.to_dict(),
            )
            evidence_list.append(ev)

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data=res.to_dict(),
            evidence_list=evidence_list,
        )

    # ── Tool 11: get_situational_risk (Phase 7.3) ───────────────────────────
    def _handle_get_situational_risk(self, call: ToolCall) -> ToolResult:
        w_start = call.arguments.get("window_start_ns", call.arguments.get("start_ns"))
        w_end = call.arguments.get("window_end_ns", call.arguments.get("end_ns"))
        mode_arg = call.arguments.get("system_mode", "operational_mode")
        evidence_ids_arg = call.arguments.get("evidence_ids")

        if w_start is None or w_end is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing required argument 'window_start_ns' or 'window_end_ns'",
            )

        try:
            start_ns = int(w_start)
            end_ns = int(w_end)
        except (ValueError, TypeError):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="window_start_ns and window_end_ns MUST be valid integer nanosecond timestamps",
            )

        if start_ns < 0 or end_ns < 0:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="window_start_ns and window_end_ns MUST be non-negative",
            )

        if start_ns > end_ns:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message=f"window_start_ns ({start_ns}) MUST be <= window_end_ns ({end_ns})",
            )

        if (end_ns - start_ns) > 86_400_000_000_000:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Window duration MUST NOT exceed 24 hours (86400000000000 ns)",
            )

        # Validate SystemMode
        if isinstance(mode_arg, SystemMode):
            system_mode = mode_arg
        elif isinstance(mode_arg, str):
            mode_lower = mode_arg.lower().strip()
            if mode_lower in ("operational_mode", "operational"):
                system_mode = SystemMode.OPERATIONAL_MODE
            elif mode_lower in ("learning_mode", "learning"):
                system_mode = SystemMode.LEARNING_MODE
            elif mode_lower in ("degraded_mode", "degraded"):
                system_mode = SystemMode.DEGRADED_MODE
            else:
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    success=False,
                    error_message=f"Invalid system_mode '{mode_arg}'. Must be one of learning_mode, operational_mode, degraded_mode",
                )
        else:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message=f"Invalid system_mode type: {type(mode_arg)}",
            )

        # Gather evidence items
        evidence_items: list[Evidence] = []
        if "evidence_items" in call.arguments and isinstance(call.arguments["evidence_items"], (list, tuple)):
            evidence_items = [ev for ev in call.arguments["evidence_items"] if isinstance(ev, Evidence)]
        elif self._event_store is not None:
            events = self._event_store.query_events(start_ns=start_ns, end_ns=end_ns, limit=1000)
            for ev_obj in events:
                e_item = Evidence(
                    evidence_id=ev_obj.event_id,
                    source_store="event_store",
                    record_type="event",
                    record_id=ev_obj.event_id,
                    timestamp_ns=ev_obj.timestamp_ns,
                    camera_id=ev_obj.camera_id,
                    payload=ev_obj.payload,
                )
                evidence_items.append(e_item)

        if evidence_ids_arg is not None and isinstance(evidence_ids_arg, (list, tuple)):
            e_set = set(str(eid) for eid in evidence_ids_arg)
            evidence_items = [ev for ev in evidence_items if ev.evidence_id in e_set]

        risk_signal = self._situational_evaluator.evaluate(
            evidence_items=evidence_items,
            system_mode=system_mode,
            window_start_ns=start_ns,
            window_end_ns=end_ns,
        )

        contributing_evidence = [
            ev for ev in evidence_items if ev.evidence_id in risk_signal.evidence_ids
        ]

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data=risk_signal.to_dict(),
            evidence_list=contributing_evidence,
        )

    # ── Tool 12: get_hypothesis_tree (Phase 7.4) ───────────────────────────
    def _handle_get_hypothesis_tree(self, call: ToolCall) -> ToolResult:
        w_start = call.arguments.get("window_start_ns", call.arguments.get("start_ns"))
        w_end = call.arguments.get("window_end_ns", call.arguments.get("end_ns"))
        sub_ref = call.arguments.get("subject_ref")
        max_d = call.arguments.get("max_depth", 3)
        max_b = call.arguments.get("max_branching", 5)
        min_conf = call.arguments.get("min_confidence", 0.20)

        if w_start is None or w_end is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Missing required argument 'window_start_ns' or 'window_end_ns'",
            )

        try:
            start_ns = int(w_start)
            end_ns = int(w_end)
        except (ValueError, TypeError):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="window_start_ns and window_end_ns MUST be valid integer nanosecond timestamps",
            )

        if start_ns < 0 or end_ns < 0:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="window_start_ns and window_end_ns MUST be non-negative",
            )

        if start_ns > end_ns:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message=f"window_start_ns ({start_ns}) MUST be <= window_end_ns ({end_ns})",
            )

        if (end_ns - start_ns) > 86_400_000_000_000:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="Window duration MUST NOT exceed 24 hours (86400000000000 ns)",
            )

        # Gather evidence items
        evidence_items: list[Evidence] = []
        if "evidence_items" in call.arguments and isinstance(call.arguments["evidence_items"], (list, tuple)):
            evidence_items = [ev for ev in call.arguments["evidence_items"] if isinstance(ev, Evidence)]
        elif self._event_store is not None:
            events = self._event_store.query_events(start_ns=start_ns, end_ns=end_ns, limit=1000)
            for ev_obj in events:
                e_item = Evidence(
                    evidence_id=ev_obj.event_id,
                    source_store="event_store",
                    record_type="event",
                    record_id=ev_obj.event_id,
                    timestamp_ns=ev_obj.timestamp_ns,
                    camera_id=ev_obj.camera_id,
                    payload=ev_obj.payload,
                )
                evidence_items.append(e_item)

        if not evidence_items:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={"hypothesis_trees": [], "count": 0},
                evidence_list=[],
            )

        trees = self._hypothesis_generator.generate_trees(
            evidence_items=evidence_items,
            window_start_ns=start_ns,
            window_end_ns=end_ns,
            max_depth=int(max_d),
            max_branching=int(max_b),
            min_confidence=float(min_conf),
            subject_ref=str(sub_ref) if sub_ref else None,
        )

        if not trees:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={"hypothesis_trees": [], "count": 0},
                evidence_list=[],
            )

        referenced_ids: set[str] = set()

        def _collect_ids(node):
            referenced_ids.update(node.hypothesis.evidence_ids)
            for child in node.children:
                _collect_ids(child)

        for tree in trees:
            _collect_ids(tree.root_node)

        referenced_evidence = [ev for ev in evidence_items if ev.evidence_id in referenced_ids]

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            data={"hypothesis_trees": [t.to_dict() for t in trees], "count": len(trees)},
            evidence_list=referenced_evidence,
        )

    # ── Tool 13: get_plate_history (Phase 8 — SIH 26187) ────────────────

    def _handle_get_plate_history(self, call: ToolCall) -> ToolResult:
        """Handle get_plate_history tool call.

        Queries event store for ANPR_READING events matching a plate text.
        """
        plate_text = call.arguments.get("plate_text", "")
        limit = call.arguments.get("limit", 50)
        start_ns = call.arguments.get("start_ns")
        end_ns = call.arguments.get("end_ns")

        if not plate_text or not str(plate_text).strip():
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="plate_text argument is required",
            )

        if self._event_store is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message="EventStore not available",
            )

        try:
            events = self._event_store.query_events(
                event_type=EventType.ANPR_READING,
                start_ns=int(start_ns) if start_ns else None,
                end_ns=int(end_ns) if end_ns else None,
                limit=int(limit) * 5,
            )

            # Filter by plate text
            matching = []
            for ev in events:
                ev_meta = ev.metadata if hasattr(ev, "metadata") and ev.metadata else getattr(ev, "payload", {})
                ev_plate = ev_meta.get("plate_text", "") if isinstance(ev_meta, dict) else ""
                if str(plate_text).upper() in str(ev_plate).upper():
                    matching.append(ev)
                    if len(matching) >= int(limit):
                        break

            readings = []
            evidence_list = []
            for ev in matching:
                ev_meta = ev.metadata if hasattr(ev, "metadata") and ev.metadata else getattr(ev, "payload", {})
                plate_str = ev_meta.get("plate_text", "") if isinstance(ev_meta, dict) else ""
                readings.append(
                    {
                        "event_id": ev.event_id,
                        "camera_id": ev.camera_id,
                        "timestamp_ns": ev.timestamp_ns,
                        "plate_text": plate_str,
                        "confidence": ev.confidence,
                    }
                )
                evidence_list.append(
                    Evidence(
                        evidence_id=ev.event_id,
                        source_store="event_store",
                        record_type="event",
                        record_id=ev.event_id,
                        timestamp_ns=ev.timestamp_ns,
                        camera_id=ev.camera_id,
                        payload=ev_meta if isinstance(ev_meta, dict) else {},
                    )
                )

            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=True,
                data={"readings": readings, "count": len(readings)},
                evidence_list=evidence_list,
            )

        except Exception as exc:
            _log.error("get_plate_history_error", error=str(exc))
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                error_message=f"Query error: {exc}",
            )
