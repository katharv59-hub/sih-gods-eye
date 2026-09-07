"""Phase 5.3 & 6.6 NLQ Intent Planner & Security Boundary — §14 Phase 5 & Phase 6.

Converts natural language user queries (UserQuery) into structured, validated ToolCall plans.
Operates 100% read-only and memory-bound. Does NOT execute tools or modify database state.
Provides an abstract LLMProvider interface and a deterministic offline RuleBasedPlanner fallback.
Now includes Phase 6.6 behavioral intelligence tools (Tool 9 & Tool 10).
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.event import EventType
from gods_eye.schemas.reasoning import (
    QueryStatus,
    ToolCall,
    UserQuery,
)

_log = get_logger("reasoning.planner")

APPROVED_TOOLS: set[str] = {
    "get_identity_timeline",
    "search_events_by_type",
    "get_camera_path",
    "list_active_identities",
    "query_location_at_time",
    "get_anomaly_signals",
    "get_occupancy_baseline",
    "get_scene_state",
    "get_trajectory_clusters",
    "get_behavioral_prediction",
    "get_situational_risk",
    "get_hypothesis_tree",
    # Phase 8 — SIH 26187
    "get_plate_history",
}

DEFERRED_TOOLS: set[str] = set()


class LLMProvider(ABC):
    """Abstract interface for LLM intent parsing and function calling."""

    @abstractmethod
    def parse_query(self, query: UserQuery) -> dict[str, Any]:
        """Parse natural language user query into tool call dictionary structure."""
        ...


class RuleBasedPlanner:
    """Deterministic rule-based NLQ intent and entity extractor.

    Translates supported natural language query patterns into validated ToolCall objects.
    Enforces deterministic offline baseline execution.
    """

    @staticmethod
    def plan(query: UserQuery) -> tuple[QueryStatus, list[ToolCall], str]:
        """Generate a validated plan from a UserQuery using deterministic rules."""
        text_lower = query.text.lower().strip()

        if not text_lower:
            return (
                QueryStatus.INVALID_QUERY,
                [],
                "Query text cannot be empty",
            )

        # Entity extraction patterns
        global_id_match = re.search(
            r"\b(gid_[a-zA-Z0-9_-]+|[0-9a-fA-F-]{36}|person_\w+|identity_\w+)\b",
            query.text,
            re.IGNORECASE,
        )
        global_id = global_id_match.group(1) if global_id_match else None

        camera_matches = re.findall(
            r"\b(cam_[a-zA-Z0-9_-]+|camera_\w+)\b", query.text, re.IGNORECASE
        )
        camera_id = camera_matches[0] if camera_matches else None
        from_camera_id = camera_matches[0] if len(camera_matches) >= 1 else None
        to_camera_id = camera_matches[1] if len(camera_matches) >= 2 else None

        zone_match = re.search(
            r"\b(zone_[a-zA-Z0-9_-]+|zone_\w+)\b", query.text, re.IGNORECASE
        )
        zone_id = zone_match.group(1) if zone_match else None

        # Timestamp extraction
        ts_match = re.search(
            r"\b(timestamp|ns|at)\s*[:=]?\s*(\d{10,19})\b", query.text, re.IGNORECASE
        )
        target_ns = int(ts_match.group(2)) if ts_match else None

        # Relative time window parsing (e.g. "last 15 minutes", "last 1 hour")
        start_ns: Optional[int] = None
        end_ns: Optional[int] = query.timestamp_ns
        rel_time_match = re.search(
            r"\blast\s+(\d+)\s+(minute|min|hour|hr|second|sec)s?\b", text_lower
        )
        if rel_time_match:
            amount = int(rel_time_match.group(1))
            unit = rel_time_match.group(2)
            seconds = amount
            if unit in ("minute", "min"):
                seconds = amount * 60
            elif unit in ("hour", "hr"):
                seconds = amount * 3600
            start_ns = query.timestamp_ns - int(seconds * 1e9)

        # Plate extraction (e.g. "plate DL01AB1234", "plate: MH12DE1433", or plate regex)
        plate_match = re.search(
            r"\bplate\s*[:=]?\s*([a-zA-Z0-9_-]+)\b", query.text, re.IGNORECASE
        )
        plate_text = plate_match.group(1) if plate_match else None
        if not plate_text:
            gen_plate = re.search(r"\b([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4})\b", query.text)
            if gen_plate:
                plate_text = gen_plate.group(1)

        # Rule 13: License plate history query (Phase 8 — SIH 26187)
        if any(
            w in text_lower
            for w in [
                "plate history",
                "license plate",
                "number plate",
                "plate reading",
                "where was plate",
                "track plate",
                "find plate",
                "plate sightings",
            ]
        ) or (plate_text and any(w in text_lower for w in ["plate", "vehicle", "seen", "sightings", "history", "where"])):
            args_plate: dict[str, Any] = {}
            if plate_text:
                args_plate["plate_text"] = plate_text
            if camera_id:
                args_plate["camera_id"] = camera_id
            if start_ns:
                args_plate["start_ns"] = start_ns
                args_plate["end_ns"] = end_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_plate_history",
                arguments=args_plate,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_plate_history"

        # Rule 9: Trajectory clusters query (Phase 6.6)
        if any(
            w in text_lower
            for w in [
                "trajectory cluster",
                "cluster trajectory",
                "trajectory pattern",
                "common routes",
                "movement patterns",
                "common paths",
                "route clusters",
            ]
        ):
            args: dict[str, Any] = {}
            if camera_id:
                args["camera_id"] = camera_id
            if start_ns:
                args["start_ns"] = start_ns
                args["end_ns"] = end_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_trajectory_clusters",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_trajectory_clusters"

        # Rule 10: Behavioral prediction query (Phase 6.6)
        if any(
            w in text_lower
            for w in [
                "predict movement",
                "behavioral prediction",
                "predict identity",
                "likely next",
                "next camera",
                "likely to go",
                "usually go",
                "where will",
                "where does someone go",
                "next destinations",
                "next destination",
                "likely destinations",
            ]
        ):
            if not camera_id:
                return (
                    QueryStatus.AMBIGUOUS,
                    [],
                    "Behavioral prediction query requires camera_id or current_camera_id",
                )
            args = {"current_camera_id": camera_id}
            if global_id:
                args["global_id"] = global_id
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_behavioral_prediction",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_behavioral_prediction"

        # Rule 11: Situational risk query (Phase 7.3)
        if any(
            w in text_lower
            for w in [
                "situational risk",
                "security risk",
                "risk level",
                "is there any risk",
                "evaluate risk",
                "how risky",
                "threat level",
                "situational hazard",
            ]
        ):
            w_start = start_ns if start_ns is not None else max(0, query.timestamp_ns - int(query.default_time_window_s * 1e9))
            w_end = end_ns if end_ns is not None else query.timestamp_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_situational_risk",
                arguments={
                    "window_start_ns": w_start,
                    "window_end_ns": w_end,
                    "system_mode": "operational_mode",
                },
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_situational_risk"

        # Rule 12: Hypothesis tree query (Phase 7.4)
        if any(
            w in text_lower
            for w in [
                "hypothesis tree",
                "competing hypotheses",
                "active hypotheses",
                "why is this situation",
                "hypotheses for",
                "show hypothesis",
                "explain hypotheses",
            ]
        ):
            w_start = start_ns if start_ns is not None else max(0, query.timestamp_ns - int(query.default_time_window_s * 1e9))
            w_end = end_ns if end_ns is not None else query.timestamp_ns
            args: dict[str, Any] = {
                "window_start_ns": w_start,
                "window_end_ns": w_end,
                "max_depth": 3,
                "min_confidence": 0.20,
            }
            if global_id:
                args["subject_ref"] = global_id
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_hypothesis_tree",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_hypothesis_tree"

        # Priority Rule 1: Explicit Event Search Query (e.g. "search event", "show events", or matching explicit EventType)
        matched_event_type: Optional[str] = None
        for et in EventType:
            if et.value.lower() in text_lower:
                matched_event_type = et.value
                break

        if any(w in text_lower for w in ["search event", "show event", "list event", "events"]) or matched_event_type:
            args = {}
            if matched_event_type:
                args["event_type"] = matched_event_type
            if camera_id:
                args["camera_id"] = camera_id
            if global_id:
                args["global_id"] = global_id
            if zone_id:
                args["zone_id"] = zone_id
            if start_ns:
                args["start_ns"] = start_ns
                args["end_ns"] = end_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="search_events_by_type",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to search_events_by_type"

        # Rule 2: Point-in-time location query
        if target_ns or (any(w in text_lower for w in ["where was", "location at"]) and target_ns):
            if not global_id:
                return (
                    QueryStatus.AMBIGUOUS,
                    [],
                    "Location query requires identity global_id",
                )
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="query_location_at_time",
                arguments={"global_id": global_id, "target_ns": target_ns},
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to query_location_at_time"

        # Rule 3: Camera transition path query
        if any(
            w in text_lower for w in ["camera path", "moved between", "transition path", "camera transition"]
        ):
            if not global_id:
                return (
                    QueryStatus.AMBIGUOUS,
                    [],
                    "Camera path query requires identity global_id",
                )
            args = {"global_id": global_id}
            if from_camera_id and to_camera_id:
                args["from_camera_id"] = from_camera_id
                args["to_camera_id"] = to_camera_id
            if start_ns:
                args["start_ns"] = start_ns
                args["end_ns"] = end_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_camera_path",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_camera_path"

        # Rule 4: Identity timeline query
        if any(
            w in text_lower for w in ["timeline", "history", "where did", "where has", "where is person"]
        ):
            if not global_id:
                return (
                    QueryStatus.AMBIGUOUS,
                    [],
                    "Timeline query requires identity global_id",
                )
            args = {"global_id": global_id}
            if start_ns:
                args["start_ns"] = start_ns
                args["end_ns"] = end_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_identity_timeline",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_identity_timeline"

        # Rule 5: Active identities query
        if any(
            w in text_lower
            for w in ["active identities", "active right now", "who is active", "currently active"]
        ):
            args = {}
            if camera_id:
                args["camera_id"] = camera_id
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="list_active_identities",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to list_active_identities"

        # Rule 6: Environmental anomaly query
        if any(w in text_lower for w in ["anomaly", "anomalies", "environmental anomaly"]):
            args = {}
            if camera_id:
                args["camera_id"] = camera_id
            if start_ns:
                args["start_ns"] = start_ns
                args["end_ns"] = end_ns
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_anomaly_signals",
                arguments=args,
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_anomaly_signals"

        # Rule 7: Occupancy baseline query
        if any(w in text_lower for w in ["occupancy baseline", "normal occupancy"]):
            if not camera_id:
                return (
                    QueryStatus.AMBIGUOUS,
                    [],
                    "Occupancy baseline query requires camera_id",
                )
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_occupancy_baseline",
                arguments={"camera_id": camera_id},
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_occupancy_baseline"

        # Rule 8: Scene state query
        if any(w in text_lower for w in ["scene state", "current state", "lighting"]):
            if not camera_id:
                return (
                    QueryStatus.AMBIGUOUS,
                    [],
                    "Scene state query requires camera_id",
                )
            tc = ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:8]}",
                tool_name="get_scene_state",
                arguments={"camera_id": camera_id},
            )
            return QueryStatus.SUCCESS, [tc], "Mapped to get_scene_state"

        return (
            QueryStatus.INVALID_QUERY,
            [],
            "Query intent could not be mapped to any supported Phase 5 tool",
        )


class NLQPlanner:
    """Natural Language Query Intent Planner and Security Boundary.

    Parses UserQuery objects into validated ToolCall sequences using an LLMProvider or RuleBasedPlanner.
    Enforces strict security allowlists, parameter validation, and fail-closed security.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._metrics = metrics

    def create_plan(self, query: UserQuery) -> tuple[QueryStatus, list[ToolCall], str]:
        """Create a validated ToolCall plan from a UserQuery."""
        if self._metrics is not None:
            self._metrics.reasoning_planner_queries_total.labels(status="attempt").inc()

        # Input text safety sanitization (Prompt Injection Protection)
        text_lower = query.text.lower()
        if any(
            bad in text_lower
            for bad in ["ignore previous", "drop table", "select * from", "system_call", "raw_exec"]
        ):
            _log.warning("prompt_injection_attempt_detected", query_id=query.query_id)
            if self._metrics is not None:
                self._metrics.reasoning_planner_queries_total.labels(status="injection_rejected").inc()
            return (
                QueryStatus.INVALID_QUERY,
                [],
                "Query rejected due to prompt injection attempt or unsafe content",
            )

        status: QueryStatus
        tool_calls: list[ToolCall] = []
        explanation: str = ""

        if self._llm_provider is not None:
            try:
                raw_plan = self._llm_provider.parse_query(query)
                status, tool_calls, explanation = self._validate_llm_plan(raw_plan)
            except Exception as exc:
                _log.error("llm_provider_failed_falling_back_to_rules", error=str(exc))
                status, tool_calls, explanation = RuleBasedPlanner.plan(query)
        else:
            status, tool_calls, explanation = RuleBasedPlanner.plan(query)

        if self._metrics is not None:
            self._metrics.reasoning_planner_queries_total.labels(status=status.value).inc()

        return status, tool_calls, explanation

    def _validate_llm_plan(
        self, raw_plan: dict[str, Any]
    ) -> tuple[QueryStatus, list[ToolCall], str]:
        """Validate LLM provider outputs against strict allowlist and argument contracts."""
        raw_status = raw_plan.get("status", "success")
        try:
            status = QueryStatus(raw_status)
        except ValueError:
            status = QueryStatus.INVALID_QUERY

        explanation = str(raw_plan.get("explanation", "Parsed by LLMProvider"))
        raw_calls = raw_plan.get("tool_calls", [])

        if not isinstance(raw_calls, list):
            return QueryStatus.INVALID_QUERY, [], "LLM tool_calls output MUST be a list"

        validated_calls: list[ToolCall] = []

        for call_data in raw_calls:
            if not isinstance(call_data, dict):
                continue
            tool_name = str(call_data.get("tool_name", ""))
            call_id = str(call_data.get("call_id", f"call_{uuid.uuid4().hex[:8]}"))
            arguments = call_data.get("arguments", {})

            # Strict Tool Allowlist Enforcement
            if tool_name not in APPROVED_TOOLS:
                _log.warning("unauthorized_tool_call_attempted_by_llm", tool_name=tool_name)
                return (
                    QueryStatus.INVALID_QUERY,
                    [],
                    f"LLM proposed unauthorized or deferred tool '{tool_name}'",
                )

            if not isinstance(arguments, dict):
                return QueryStatus.INVALID_QUERY, [], "ToolCall arguments MUST be a dictionary"

            tc = ToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments)
            validated_calls.append(tc)

        return status, validated_calls, explanation
