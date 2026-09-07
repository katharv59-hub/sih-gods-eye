"""Unit tests for Tool 13 (get_plate_history) and Planner Plate Routing (Phase 8 — SIH 26187)."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.reasoning.planner import NLQPlanner, RuleBasedPlanner, APPROVED_TOOLS
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import ToolCall, UserQuery, QueryStatus


class TestTool13RegistrationAndRouting:
    def test_approved_tools_contains_tool13(self) -> None:
        assert "get_plate_history" in APPROVED_TOOLS

    def test_planner_routes_plate_query(self) -> None:
        planner = NLQPlanner()

        # Query with explicit plate
        q1 = UserQuery(
            query_id="q1",
            text="Where was license plate DL01AB1234 seen in the last hour?",
            timestamp_ns=3600_000_000_000,
        )
        status, calls, explanation = planner.create_plan(q1)
        assert status == QueryStatus.SUCCESS
        assert len(calls) == 1
        assert calls[0].tool_name == "get_plate_history"
        assert calls[0].arguments.get("plate_text") == "DL01AB1234"

        # Query with plate keyword
        q2 = UserQuery(
            query_id="q2",
            text="Find plate history for vehicle seen at cam_01",
            timestamp_ns=1_000_000_000,
        )
        status2, calls2, _ = planner.create_plan(q2)
        assert status2 == QueryStatus.SUCCESS
        assert len(calls2) == 1
        assert calls2[0].tool_name == "get_plate_history"
        assert calls2[0].arguments.get("camera_id") == "cam_01"


class TestTool13DispatcherExecution:
    def test_tool13_missing_plate_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")
            store = SQLiteEventStore(db_path=db_path)
            try:
                dispatcher = ToolDispatcher(event_store=store)
                call = ToolCall(
                    call_id="call_1",
                    tool_name="get_plate_history",
                    arguments={},
                )
                result = dispatcher.dispatch(call)
                assert result.success is False
                assert "plate_text argument is required" in result.error_message
            finally:
                store.close()

    def test_tool13_successful_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")
            store = SQLiteEventStore(db_path=db_path)
            try:
                # Seed an ANPR reading event
                ev = Event(
                    event_id="ev_anpr_1",
                    event_type=EventType.ANPR_READING,
                    global_id=None,
                    camera_id="cam-01",
                    timestamp_ns=1_000_000_000,
                    frame_id=42,
                    confidence=0.95,
                    explanation="Plate DL01AB1234 detected",
                    metadata={"plate_text": "DL01AB1234", "confidence": 0.95},
                )
                store.append(ev)

                dispatcher = ToolDispatcher(event_store=store)
                call = ToolCall(
                    call_id="call_2",
                    tool_name="get_plate_history",
                    arguments={"plate_text": "DL01AB1234"},
                )
                result = dispatcher.dispatch(call)
                assert result.success is True
                assert result.data["count"] == 1
                assert result.data["readings"][0]["plate_text"] == "DL01AB1234"
                assert len(result.evidence_list) == 1
                assert result.evidence_list[0].evidence_id == "ev_anpr_1"
            finally:
                store.close()
