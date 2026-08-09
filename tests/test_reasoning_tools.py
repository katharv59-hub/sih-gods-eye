"""Unit tests for Phase 5.2 Deterministic Tool Layer."""

from __future__ import annotations

import time
import uuid
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.memory.timeline_engine import TimelineReconstructionEngine
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import ToolCall


@pytest.fixture
def memory_stores():
    es = SQLiteEventStore(":memory:")
    gs = SQLiteGraphStore(":memory:")
    timeline_engine = TimelineReconstructionEngine(es, gs)
    yield es, gs, timeline_engine
    es.close()
    gs.close()


class TestToolDispatcherRegistration:
    def test_unregistered_tool_returns_failure(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)
        tc = ToolCall(call_id="c1", tool_name="unknown_tool")
        res = dispatcher.dispatch(tc)

        assert res.success is False
        assert "not registered" in res.error_message

    def test_registered_tools_exist(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        for name in [
            "get_identity_timeline",
            "search_events_by_type",
            "get_camera_path",
            "list_active_identities",
            "query_location_at_time",
            "get_anomaly_signals",
            "get_occupancy_baseline",
            "get_scene_state",
        ]:
            assert name in dispatcher._handlers


class TestGetIdentityTimelineTool:
    def test_known_identity_timeline(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        # Seed data
        gs.upsert_identity_node("gid_1", 1000, 2000, "cam_01")
        gs.add_observation_edge("obs_1", "gid_1", "cam_01", 1000)
        gs.add_observation_edge("obs_2", "gid_1", "cam_01", 2000)

        tc = ToolCall(
            call_id="c_tl",
            tool_name="get_identity_timeline",
            arguments={"global_id": "gid_1"},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["global_id"] == "gid_1"
        assert res.data["visit_count"] == 1
        assert len(res.evidence_list) == 1
        assert res.evidence_list[0].source_store == "graph_store"

    def test_unknown_identity_timeline(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        tc = ToolCall(
            call_id="c_tl_missing",
            tool_name="get_identity_timeline",
            arguments={"global_id": "nonexistent_gid"},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["visit_count"] == 0
        assert len(res.evidence_list) == 0

    def test_invalid_arguments(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        tc = ToolCall(call_id="c_err", tool_name="get_identity_timeline", arguments={"global_id": ""})
        res = dispatcher.dispatch(tc)
        assert res.success is False

        tc_range = ToolCall(
            call_id="c_err_2",
            tool_name="get_identity_timeline",
            arguments={"global_id": "gid_1", "start_ns": 200, "end_ns": 100},
        )
        res_range = dispatcher.dispatch(tc_range)
        assert res_range.success is False


class TestSearchEventsByTypeTool:
    def test_event_search_and_filtering(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        # Seed events
        evt1 = Event("e1", EventType.CROSS_CAMERA_TRANSITION, "gid_1", "cam_02", 1000, 10, 0.9, "Trans")
        evt2 = Event("e2", EventType.ENVIRONMENTAL_ANOMALY, None, "cam_01", 2000, 20, 0.8, "Anomaly")
        es.append(evt1)
        es.append(evt2)

        tc = ToolCall(
            call_id="c_search",
            tool_name="search_events_by_type",
            arguments={"event_type": "cross_camera_transition"},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["event_count"] == 1
        assert res.data["events"][0]["event_id"] == "e1"
        assert len(res.evidence_list) == 1
        assert res.evidence_list[0].source_store == "event_store"

    def test_invalid_event_type(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        tc = ToolCall(
            call_id="c_bad_evt",
            tool_name="search_events_by_type",
            arguments={"event_type": "invalid_event_category"},
        )
        res = dispatcher.dispatch(tc)
        assert res.success is False


class TestGetCameraPathTool:
    def test_camera_path_query(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        gs.add_transition_edge("t1", "gid_1", "cam_01", "cam_02", 1000, 1.5)

        tc = ToolCall(
            call_id="c_path",
            tool_name="get_camera_path",
            arguments={"global_id": "gid_1"},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["transition_count"] == 1
        assert len(res.evidence_list) == 1


class TestListActiveIdentitiesTool:
    def test_active_identities_query(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        gs.upsert_identity_node("gid_active", 100, 200, "cam_01", state="ACTIVE")

        tc = ToolCall(call_id="c_active", tool_name="list_active_identities")
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["active_count"] == 1
        assert res.data["identities"][0]["global_id"] == "gid_active"


class TestQueryLocationAtTimeTool:
    def test_location_at_target_timestamp(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        gs.upsert_identity_node("gid_1", 1000, 5000, "cam_01")
        gs.add_observation_edge("obs_1", "gid_1", "cam_01", 1000)
        gs.add_observation_edge("obs_2", "gid_1", "cam_01", 5000)

        tc = ToolCall(
            call_id="c_loc",
            tool_name="query_location_at_time",
            arguments={"global_id": "gid_1", "target_ns": 3000},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["found"] is True
        assert res.data["location"]["camera_id"] == "cam_01"

    def test_location_not_found(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        tc = ToolCall(
            call_id="c_loc_missing",
            tool_name="query_location_at_time",
            arguments={"global_id": "gid_missing", "target_ns": 3000},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["found"] is False


class TestGetAnomalySignalsTool:
    def test_anomaly_signal_retrieval(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        evt_anom = Event("a1", EventType.ENVIRONMENTAL_ANOMALY, None, "cam_01", 1500, 15, 0.85, "Motion anomaly")
        es.append(evt_anom)

        tc = ToolCall(call_id="c_anom", tool_name="get_anomaly_signals")
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["anomaly_count"] == 1


class TestGetOccupancyBaselineAndSceneStateTools:
    def test_occupancy_baseline_query(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        tc = ToolCall(
            call_id="c_base",
            tool_name="get_occupancy_baseline",
            arguments={"camera_id": "cam_01", "time_bucket": "MON_09"},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["camera_id"] == "cam_01"

    def test_scene_state_query(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        gs.upsert_camera_node("cam_01", status="ACTIVE")

        tc = ToolCall(
            call_id="c_scene",
            tool_name="get_scene_state",
            arguments={"camera_id": "cam_01"},
        )
        res = dispatcher.dispatch(tc)

        assert res.success is True
        assert res.data["camera_id"] == "cam_01"


class TestReadOnlyAndDeterminism:
    def test_read_only_invariant(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        # Seed data
        es.append(Event("e1", EventType.IDENTITY_CONFIRMED, "g1", "c1", 100, 1, 1.0, "Conf"))
        gs.upsert_identity_node("g1", 100, 100, "c1")

        cnt_events_before = es.count()
        cnt_nodes_before = gs.count_nodes()["identities"]

        # Run tool calls
        dispatcher.dispatch(ToolCall("c1", "search_events_by_type"))
        dispatcher.dispatch(ToolCall("c2", "get_identity_timeline", arguments={"global_id": "g1"}))
        dispatcher.dispatch(ToolCall("c3", "list_active_identities"))

        assert es.count() == cnt_events_before
        assert gs.count_nodes()["identities"] == cnt_nodes_before

    def test_determinism_invariant(self, memory_stores) -> None:
        es, gs, te = memory_stores
        dispatcher = ToolDispatcher(es, gs, te)

        es.append(Event("e1", EventType.IDENTITY_CONFIRMED, "g1", "c1", 100, 1, 1.0, "Conf"))
        es.append(Event("e2", EventType.CROSS_CAMERA_TRANSITION, "g1", "c2", 200, 2, 1.0, "Trans"))

        tc = ToolCall("c_det", "search_events_by_type")
        res1 = dispatcher.dispatch(tc)
        res2 = dispatcher.dispatch(tc)

        assert res1.to_dict() == res2.to_dict()
