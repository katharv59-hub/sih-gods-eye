"""Phase 5.4 Reasoning Engine Orchestration & Evidence Verification — §14 Phase 5.

Orchestrates the end-to-end situational reasoning pipeline:
UserQuery -> NLQPlanner -> ToolCall -> ReasoningEngine -> ToolDispatcher -> ToolResult -> Evidence Verification -> ReasoningResult.

Operates 100% read-only and memory-bound. Enforces Rule 7 (Confidence Band) and Rule 8 (Explainability/Evidence).
Does NOT execute SQL directly or mutate Phase 1-4 state.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.reasoning.planner import NLQPlanner
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.reasoning import (
    Evidence,
    QueryStatus,
    ReasoningResult,
    ToolCall,
    ToolResult,
    UserQuery,
)

_log = get_logger("reasoning.engine")


class ReasoningEngine:
    """Orchestrator for Situational Reasoning queries and evidence verification."""

    def __init__(
        self,
        planner: Optional[NLQPlanner] = None,
        dispatcher: Optional[ToolDispatcher] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self._planner = planner if planner is not None else NLQPlanner()
        self._dispatcher = dispatcher if dispatcher is not None else ToolDispatcher()
        self._metrics = metrics

    def process_query(self, query: UserQuery) -> ReasoningResult:
        """Process a natural language UserQuery and produce a canonical ReasoningResult."""
        t_start = time.perf_counter()

        if self._metrics is not None:
            self._metrics.reasoning_engine_queries_total.labels(status="attempt").inc()

        # Step 1: Delegate planning to NLQPlanner
        plan_status, tool_calls, plan_explanation = self._planner.create_plan(query)

        if plan_status != QueryStatus.SUCCESS or not tool_calls:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            if self._metrics is not None:
                self._metrics.reasoning_engine_queries_total.labels(
                    status=plan_status.value
                ).inc()
                self._metrics.reasoning_engine_latency_ms.observe(t_ms)

            return ReasoningResult(
                query_id=query.query_id,
                status=plan_status,
                answer="",
                confidence=0.0,
                confidence_band=(0.0, 0.0),
                evidence=[],
                tool_calls=[],
                execution_time_ms=t_ms,
                explanation=plan_explanation,
            )

        # Step 2: Execute tool calls through ToolDispatcher
        executed_calls: list[ToolCall] = []
        collected_evidence: list[Evidence] = []
        tool_results: list[ToolResult] = []

        for call in tool_calls:
            executed_calls.append(call)
            t_res = self._dispatcher.dispatch(call)
            tool_results.append(t_res)

            if not t_res.success:
                _log.warning(
                    "tool_execution_failed_in_engine",
                    call_id=call.call_id,
                    error=t_res.error_message,
                )
                t_ms = (time.perf_counter() - t_start) * 1000.0
                if self._metrics is not None:
                    self._metrics.reasoning_engine_queries_total.labels(
                        status="execution_error"
                    ).inc()
                    self._metrics.reasoning_engine_latency_ms.observe(t_ms)

                return ReasoningResult(
                    query_id=query.query_id,
                    status=QueryStatus.EXECUTION_ERROR,
                    answer="",
                    confidence=0.0,
                    confidence_band=(0.0, 0.0),
                    evidence=[],
                    tool_calls=executed_calls,
                    execution_time_ms=t_ms,
                    explanation=f"Tool execution failed for '{call.tool_name}': {t_res.error_message}",
                )

            collected_evidence.extend(t_res.evidence_list)

        # Step 3: Evidence Deduplication & Deterministic Ordering
        unique_evidence: dict[tuple[str, str, str], Evidence] = {}
        for ev in collected_evidence:
            key = (ev.source_store, ev.record_type, ev.record_id)
            if key not in unique_evidence:
                unique_evidence[key] = ev

        evidence_list = list(unique_evidence.values())
        evidence_list.sort(key=lambda e: (e.timestamp_ns, e.record_id))

        # Step 4: Evidence Sufficiency Verification (Rule 8)
        if not evidence_list:
            t_ms = (time.perf_counter() - t_start) * 1000.0
            if self._metrics is not None:
                self._metrics.reasoning_engine_queries_total.labels(
                    status="insufficient_evidence"
                ).inc()
                self._metrics.reasoning_engine_latency_ms.observe(t_ms)

            return ReasoningResult(
                query_id=query.query_id,
                status=QueryStatus.INSUFFICIENT_EVIDENCE,
                answer="No matching historical records or evidence found in EventStore or GraphStore.",
                confidence=0.0,
                confidence_band=(0.0, 0.0),
                evidence=[],
                tool_calls=executed_calls,
                execution_time_ms=t_ms,
                explanation="Execution completed successfully, but zero matching evidence records were retrieved.",
            )

        # Step 5: Confidence Aggregation & Band Calculation (Rule 7)
        conf_scores = []
        for ev in evidence_list:
            conf_val = float(ev.payload.get("confidence", 1.0))
            if 0.0 <= conf_val <= 1.0:
                conf_scores.append(conf_val)

        mean_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 1.0
        lower_band = max(0.0, round(mean_confidence - 0.05, 4))
        upper_band = min(1.0, round(mean_confidence + 0.05, 4))

        # Step 6: Fact-Backed Answer Formatting (Zero Hallucination)
        answer = self._format_fact_summary(tool_results, evidence_list)

        t_ms = (time.perf_counter() - t_start) * 1000.0
        if self._metrics is not None:
            self._metrics.reasoning_engine_queries_total.labels(status="success").inc()
            self._metrics.reasoning_engine_latency_ms.observe(t_ms)

        return ReasoningResult(
            query_id=query.query_id,
            status=QueryStatus.SUCCESS,
            answer=answer,
            confidence=round(mean_confidence, 4),
            confidence_band=(lower_band, upper_band),
            evidence=evidence_list,
            tool_calls=executed_calls,
            execution_time_ms=t_ms,
            explanation=f"Successfully verified {len(evidence_list)} evidence record(s) across {len(executed_calls)} tool invocation(s).",
        )

    def _format_fact_summary(
        self, tool_results: list[ToolResult], evidence_list: list[Evidence]
    ) -> str:
        """Format a deterministic, fact-backed summary from tool results and evidence."""
        summaries: list[str] = []

        for tr in tool_results:
            if not tr.data:
                continue

            if tr.tool_name == "get_identity_timeline":
                gid = tr.data.get("global_id", "")
                v_count = tr.data.get("visit_count", 0)
                visits = tr.data.get("visits", [])
                cams = ", ".join(dict.fromkeys(v["camera_id"] for v in visits))
                summaries.append(
                    f"Identity {gid} has {v_count} visit segment(s) across camera(s): {cams}."
                )

            elif tr.tool_name == "search_events_by_type":
                e_count = tr.data.get("event_count", 0)
                summaries.append(f"Found {e_count} matching event(s) in EventStore.")

            elif tr.tool_name == "get_camera_path":
                gid = tr.data.get("global_id", "")
                t_count = tr.data.get("transition_count", 0)
                summaries.append(f"Identity {gid} has {t_count} camera transition(s).")

            elif tr.tool_name == "list_active_identities":
                a_count = tr.data.get("active_count", 0)
                summaries.append(f"Found {a_count} active identity/identities.")

            elif tr.tool_name == "query_location_at_time":
                gid = tr.data.get("global_id", "")
                found = tr.data.get("found", False)
                loc = tr.data.get("location")
                if found and loc:
                    summaries.append(
                        f"Identity {gid} was located at camera {loc.get('camera_id')} at timestamp {tr.data.get('target_ns')}."
                    )

            elif tr.tool_name == "get_anomaly_signals":
                a_count = tr.data.get("anomaly_count", 0)
                summaries.append(f"Found {a_count} environmental anomaly signal(s).")

            elif tr.tool_name == "get_occupancy_baseline":
                cam = tr.data.get("camera_id")
                mean_occ = tr.data.get("mean_occupancy", 0.0)
                summaries.append(f"Camera {cam} has baseline occupancy mean of {mean_occ:.2f}.")

            elif tr.tool_name == "get_scene_state":
                cam = tr.data.get("camera_id")
                lighting = tr.data.get("lighting_condition")
                summaries.append(f"Camera {cam} scene state: lighting={lighting}.")

        if not summaries:
            return f"Verified {len(evidence_list)} evidence record(s)."

        return " ".join(summaries)
