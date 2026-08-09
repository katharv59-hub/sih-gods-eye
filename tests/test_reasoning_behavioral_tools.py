"""Integration & Unit tests for Phase 6.6 Tool 9 & Tool 10 Activation."""

from __future__ import annotations

import pytest

from gods_eye.behavioral.clustering import TrajectoryClusteringEngine
from gods_eye.behavioral.predictor import MarkovBehavioralPredictor
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.reasoning.engine import ReasoningEngine
from gods_eye.reasoning.planner import NLQPlanner, RuleBasedPlanner
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.reasoning import QueryStatus, ToolCall, UserQuery


class TestBehavioralToolsIntegration:
    def setup_method(self):
        self.graph_store = SQLiteGraphStore(":memory:")
        self.event_store = SQLiteEventStore(":memory:")
        self.dispatcher = ToolDispatcher(
            event_store=self.event_store,
            graph_store=self.graph_store,
        )
        self.planner = NLQPlanner()
        self.engine = ReasoningEngine(planner=self.planner, dispatcher=self.dispatcher)

    def teardown_method(self):
        self.graph_store.close()
        self.event_store.close()

    # ── TOOL 9 TESTS ─────────────────────────────────────────────────────────
    def test_tool9_empty_history_insufficient_evidence(self):
        call = ToolCall("c1", "get_trajectory_clusters", {})
        res = self.dispatcher.dispatch(call)

        assert res.success is True
        assert res.data["status"] == "INSUFFICIENT_EVIDENCE"
        assert res.data["clusters"] == []
        assert res.evidence_list == []

    def test_tool9_successful_clustering_and_privacy(self):
        # Insert camera transitions
        self.graph_store.upsert_camera_node("cam_01")
        self.graph_store.upsert_camera_node("cam_02")
        self.graph_store.upsert_camera_node("cam_03")

        # 3 identical transitions for gid_1, gid_2, gid_3
        for i in range(1, 4):
            gid = f"gid_{i}"
            self.graph_store.add_transition_edge(f"t_{i}_1", gid, "cam_01", "cam_02", 1_000 + i)
            self.graph_store.add_transition_edge(f"t_{i}_2", gid, "cam_02", "cam_03", 2_000 + i)

        call = ToolCall("c2", "get_trajectory_clusters", {"min_samples": 2, "eps": 0.3})
        res = self.dispatcher.dispatch(call)

        assert res.success is True
        clusters = res.data["clusters"]
        assert len(clusters) >= 1
        assert res.evidence_list != []
        assert "global_id" not in res.data
        assert "global_ids" not in res.data

    def test_tool9_read_only_invariants(self):
        nodes_before = self.graph_store.count_nodes()
        edges_before = self.graph_store.count_edges()

        call = ToolCall("c3", "get_trajectory_clusters", {})
        self.dispatcher.dispatch(call)

        assert self.graph_store.count_nodes() == nodes_before
        assert self.graph_store.count_edges() == edges_before

    # ── TOOL 10 TESTS ────────────────────────────────────────────────────────
    def test_tool10_missing_camera_id_error(self):
        call = ToolCall("c4", "get_behavioral_prediction", {})
        res = self.dispatcher.dispatch(call)

        assert res.success is False
        assert "Missing required argument" in res.error_message

    def test_tool10_insufficient_evidence_unknown_camera(self):
        call = ToolCall("c5", "get_behavioral_prediction", {"current_camera_id": "cam_unknown"})
        res = self.dispatcher.dispatch(call)

        assert res.success is True
        assert res.data["predictions"] == []
        assert res.data["overall_confidence"] == 0.0
        assert res.evidence_list == []

    def test_tool10_successful_prediction_probability_and_confidence(self):
        self.graph_store.upsert_camera_node("cam_01")
        self.graph_store.upsert_camera_node("cam_02")
        self.graph_store.add_transition_edge("t1", "gid_1", "cam_01", "cam_02", 1_000)

        call = ToolCall("c6", "get_behavioral_prediction", {"current_camera_id": "cam_01", "top_k": 3})
        res = self.dispatcher.dispatch(call)

        assert res.success is True
        preds = res.data["predictions"]
        assert len(preds) == 1
        assert preds[0]["camera_id"] == "cam_02"
        assert preds[0]["probability"] == 1.0
        assert res.evidence_list != []

    def test_tool10_read_only_invariants(self):
        edges_before = self.graph_store.count_edges()
        call = ToolCall("c7", "get_behavioral_prediction", {"current_camera_id": "cam_01"})
        self.dispatcher.dispatch(call)

        assert self.graph_store.count_edges() == edges_before

    # ── PLANNER ROUTING TESTS ───────────────────────────────────────────────
    def test_planner_tool9_intent_routing(self):
        q = UserQuery("q1", "What are the common trajectory clusters in the building?", 1_700_000_000_000_000_000)
        status, calls, exp = RuleBasedPlanner.plan(q)

        assert status == QueryStatus.SUCCESS
        assert len(calls) == 1
        assert calls[0].tool_name == "get_trajectory_clusters"

    def test_planner_tool10_intent_routing(self):
        q = UserQuery("q2", "Where is a person likely to go from cam_01?", 1_700_000_000_000_000_000)
        status, calls, exp = RuleBasedPlanner.plan(q)

        assert status == QueryStatus.SUCCESS
        assert len(calls) == 1
        assert calls[0].tool_name == "get_behavioral_prediction"
        assert calls[0].arguments["current_camera_id"] == "cam_01"

    def test_planner_tool10_ambiguous_without_camera(self):
        q = UserQuery("q3", "Where will people go next?", 1_700_000_000_000_000_000)
        status, calls, exp = RuleBasedPlanner.plan(q)

        assert status == QueryStatus.AMBIGUOUS
        assert calls == []

    # ── END-TO-END REASONING ENGINE TESTS ───────────────────────────────────
    def test_e2e_tool9_reasoning_flow_empty_history(self):
        q = UserQuery("q4", "Show common routes for cam_01", 1_700_000_000_000_000_000)
        res = self.engine.process_query(q)

        assert res.query_id == "q4"
        assert res.status == QueryStatus.INSUFFICIENT_EVIDENCE
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].tool_name == "get_trajectory_clusters"

    def test_e2e_tool9_reasoning_flow_with_clusters(self):
        self.graph_store.upsert_camera_node("cam_01")
        self.graph_store.upsert_camera_node("cam_02")
        self.graph_store.upsert_camera_node("cam_03")

        for i in range(1, 4):
            gid = f"gid_{i}"
            self.graph_store.add_transition_edge(f"t_{i}_1", gid, "cam_01", "cam_02", 1_000 + i)
            self.graph_store.add_transition_edge(f"t_{i}_2", gid, "cam_02", "cam_03", 2_000 + i)

        q = UserQuery("q4_succ", "Show common routes for cam_01", 1_700_000_000_000_000_000)
        res = self.engine.process_query(q)

        assert res.query_id == "q4_succ"
        assert res.status == QueryStatus.SUCCESS
        assert len(res.evidence) >= 1

    def test_e2e_tool10_reasoning_flow(self):
        self.graph_store.upsert_camera_node("cam_01")
        self.graph_store.upsert_camera_node("cam_02")
        self.graph_store.add_transition_edge("t1", "gid_1", "cam_01", "cam_02", 1_000)

        q = UserQuery("q5", "What camera is likely next after cam_01?", 1_700_000_000_000_000_000)
        res = self.engine.process_query(q)

        assert res.query_id == "q5"
        assert res.status == QueryStatus.SUCCESS
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].tool_name == "get_behavioral_prediction"
        assert len(res.evidence) >= 1
