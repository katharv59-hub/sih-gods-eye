"""Unit test suite for Sub-Phase 7.3 Tool 11 (get_situational_risk) and NLQ Router Integration."""

from __future__ import annotations

import time
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.reasoning.planner import APPROVED_TOOLS, RuleBasedPlanner
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.reasoning import (
    Evidence,
    QueryStatus,
    ToolCall,
    UserQuery,
)
from gods_eye.situational.evaluator import SituationalRiskEvaluator


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    evaluator = SituationalRiskEvaluator(evaluator_version="v7.3.0")
    return ToolDispatcher(situational_evaluator=evaluator)


def make_evidence(
    evidence_id: str = "ev_001",
    timestamp_ns: int = 1000,
    payload: dict | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_store="event_store",
        record_type="event",
        record_id="rec_001",
        timestamp_ns=timestamp_ns,
        explanation="Test evidence",
        payload=payload or {},
    )


class TestTool11Integration:
    # 1. test_tool11_dispatcher_registration
    def test_tool11_dispatcher_registration(self, dispatcher: ToolDispatcher) -> None:
        assert "get_situational_risk" in dispatcher._handlers

    # 2. test_tool11_successful_evaluation
    def test_tool11_successful_evaluation(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 1.8})
        call = ToolCall(
            call_id="c1",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["risk_level"] == "MEDIUM"
        assert res.data["risk_score"] == 0.50
        assert len(res.evidence_list) == 1
        assert res.evidence_list[0].evidence_id == "ev_1"

    # 3. test_tool11_low_risk_baseline
    def test_tool11_low_risk_baseline(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 0.5})
        call = ToolCall(
            call_id="c2",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["risk_level"] == "LOW"
        assert res.data["risk_score"] == 0.25
        assert res.data["suppressed"] is False

    # 4. test_tool11_medium_risk_signal
    def test_tool11_medium_risk_signal(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 1.8})
        call = ToolCall(
            call_id="c3",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["risk_level"] == "MEDIUM"

    # 5. test_tool11_high_risk_signal
    def test_tool11_high_risk_signal(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 2.8})
        call = ToolCall(
            call_id="c4",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["risk_level"] == "HIGH"
        assert res.data["risk_score"] == 0.75

    # 6. test_tool11_critical_risk_signal
    def test_tool11_critical_risk_signal(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence(
            "ev_1",
            1500,
            {
                "trajectory_anomaly_sigma": 3.8,
                "zone_dwell_sigma": 5.2,
                "is_restricted_zone": True,
            },
        )
        call = ToolCall(
            call_id="c5",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["risk_level"] == "CRITICAL"
        assert res.data["risk_score"] == 1.00

    # 7. test_tool11_learning_mode_suppression
    def test_tool11_learning_mode_suppression(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 3.0})
        call = ToolCall(
            call_id="c6",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "learning_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["suppressed"] is True
        assert "LEARNING_MODE" in res.data["suppression_reason"]

    # 8. test_tool11_degraded_mode_suppression
    def test_tool11_degraded_mode_suppression(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 3.0})
        call = ToolCall(
            call_id="c7",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "degraded_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["suppressed"] is True
        assert "DEGRADED_MODE" in res.data["suppression_reason"]

    # 9. test_tool11_empty_evidence
    def test_tool11_empty_evidence(self, dispatcher: ToolDispatcher) -> None:
        call = ToolCall(
            call_id="c8",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["risk_level"] == "LOW"
        assert res.data["suppressed"] is True
        assert "zero evidence" in res.data["suppression_reason"]

    # 10. test_tool11_invalid_temporal_window
    def test_tool11_invalid_temporal_window(self, dispatcher: ToolDispatcher) -> None:
        call = ToolCall(
            call_id="c9",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 3000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is False
        assert "MUST be <= window_end_ns" in res.error_message

    # 11. test_tool11_read_only_invariant
    def test_tool11_read_only_invariant(self, tmp_path) -> None:
        db_path = str(tmp_path / "test_store.db")
        es = SQLiteEventStore(db_path)
        gs = SQLiteGraphStore(db_path)
        disp = ToolDispatcher(event_store=es, graph_store=gs)

        call = ToolCall(
            call_id="c10",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
            },
        )
        res = disp.dispatch(call)
        assert res.success is True
        assert len(es.query_events()) == 0
        assert len(gs.get_camera_transitions()) == 0
        es.close()

    # 12. test_tool11_privacy_invariant
    def test_tool11_privacy_invariant(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 2.8})
        call = ToolCall(
            call_id="c11",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "system_mode": "operational_mode",
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        d_keys = list(res.data.keys())
        assert "global_id" not in d_keys
        assert "subject_global_id" not in d_keys
        assert "global_id" not in res.data["explanation"]

    # 13. test_nlq_planner_routes_to_tool11
    def test_nlq_planner_routes_to_tool11(self) -> None:
        q = UserQuery(
            query_id="q1",
            text="What is the current situational risk level?",
            timestamp_ns=2_000_000_000,
        )
        status, calls, desc = RuleBasedPlanner.plan(q)
        assert status == QueryStatus.SUCCESS
        assert len(calls) == 1
        assert calls[0].tool_name == "get_situational_risk"

    # 14. test_nlq_planner_approved_tools
    def test_nlq_planner_approved_tools(self) -> None:
        assert "get_situational_risk" in APPROVED_TOOLS

    # 15. test_phase1_to_phase6_regression
    def test_phase1_to_phase6_regression(self, dispatcher: ToolDispatcher) -> None:
        assert "get_identity_timeline" in dispatcher._handlers
        assert "get_behavioral_prediction" in dispatcher._handlers
