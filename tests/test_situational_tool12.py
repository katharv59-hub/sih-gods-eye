"""Unit test suite for Sub-Phase 7.4 Tool 12 (get_hypothesis_tree) and NLQ Router Integration."""

from __future__ import annotations

import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.reasoning.planner import APPROVED_TOOLS, RuleBasedPlanner
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.reasoning import (
    Evidence,
    QueryStatus,
    ToolCall,
    UserQuery,
)
from gods_eye.situational.hypothesis import HypothesisGenerator


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    gen = HypothesisGenerator(generator_version="v7.4.0")
    return ToolDispatcher(hypothesis_generator=gen)


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
        camera_id="cam_1",
        global_id="sub_alpha",
        explanation="Test evidence",
        payload=payload or {},
    )


class TestTool12Integration:
    # 1. test_tool12_dispatcher_registration
    def test_tool12_dispatcher_registration(self, dispatcher: ToolDispatcher) -> None:
        assert "get_hypothesis_tree" in dispatcher._handlers

    # 2. test_tool12_successful_dispatch
    def test_tool12_successful_dispatch(self, dispatcher: ToolDispatcher) -> None:
        ev1 = make_evidence("ev_1", 1200, {"trajectory_anomaly_sigma": 2.5, "confidence": 0.80})
        call = ToolCall(
            call_id="c12_1",
            tool_name="get_hypothesis_tree",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "evidence_items": [ev1],
                "max_depth": 3,
                "min_confidence": 0.20,
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert "hypothesis_trees" in res.data
        assert res.data["count"] == len(res.data["hypothesis_trees"])
        assert len(res.data["hypothesis_trees"]) == 1

    # 3. test_tool12_insufficient_evidence
    def test_tool12_insufficient_evidence(self, dispatcher: ToolDispatcher) -> None:
        call = ToolCall(
            call_id="c12_2",
            tool_name="get_hypothesis_tree",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "evidence_items": [],
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert res.data["hypothesis_trees"] == []
        assert res.data["count"] == 0
        assert res.evidence_list == []

    # 4. test_tool12_invalid_temporal_window
    def test_tool12_invalid_temporal_window(self, dispatcher: ToolDispatcher) -> None:
        call = ToolCall(
            call_id="c12_3",
            tool_name="get_hypothesis_tree",
            arguments={
                "window_start_ns": 3000,
                "window_end_ns": 2000,
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is False
        assert "MUST be <= window_end_ns" in res.error_message

    # 5. test_tool12_read_only_db_invariant
    def test_tool12_read_only_db_invariant(self, tmp_path) -> None:
        db_path = str(tmp_path / "test_store_12.db")
        es = SQLiteEventStore(db_path)
        gs = SQLiteGraphStore(db_path)
        disp = ToolDispatcher(event_store=es, graph_store=gs)

        call = ToolCall(
            call_id="c12_4",
            tool_name="get_hypothesis_tree",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
            },
        )
        res = disp.dispatch(call)
        assert res.success is True
        assert len(es.query_events()) == 0
        assert len(gs.get_camera_transitions()) == 0
        es.close()

    # 6. test_tool12_privacy_invariant
    def test_tool12_privacy_invariant(self, dispatcher: ToolDispatcher) -> None:
        ev = make_evidence("ev_1", 1500, {"confidence": 0.80})
        call = ToolCall(
            call_id="c12_5",
            tool_name="get_hypothesis_tree",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
                "evidence_items": [ev],
            },
        )
        res = dispatcher.dispatch(call)
        res_str = str(res.data)
        assert "global_id" not in res_str
        assert "subject_global_id" not in res_str

    # 7. test_tool12_nlq_planner_routing
    def test_tool12_nlq_planner_routing(self) -> None:
        q = UserQuery(
            query_id="q12_1",
            text="Show the active hypotheses for the current situation",
            timestamp_ns=2_000_000_000,
        )
        status, calls, desc = RuleBasedPlanner.plan(q)
        assert status == QueryStatus.SUCCESS
        assert len(calls) == 1
        assert calls[0].tool_name == "get_hypothesis_tree"

    # 8. test_tool12_approved_tools_registration
    def test_tool12_approved_tools_registration(self) -> None:
        assert "get_hypothesis_tree" in APPROVED_TOOLS

    # 9. test_tool11_remains_frozen_and_passing
    def test_tool11_remains_frozen_and_passing(self, dispatcher: ToolDispatcher) -> None:
        call = ToolCall(
            call_id="c11_test",
            tool_name="get_situational_risk",
            arguments={
                "window_start_ns": 1000,
                "window_end_ns": 2000,
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert "risk_level" in res.data

    # 10. test_phase6_predictor_remains_frozen_and_passing
    def test_phase6_predictor_remains_frozen_and_passing(self, dispatcher: ToolDispatcher) -> None:
        call = ToolCall(
            call_id="c10_test",
            tool_name="get_behavioral_prediction",
            arguments={
                "current_camera_id": "cam_1",
            },
        )
        res = dispatcher.dispatch(call)
        assert res.success is True
        assert "predictions" in res.data
