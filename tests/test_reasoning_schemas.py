"""Unit tests for Phase 5.1 Canonical Reasoning Schemas."""

from __future__ import annotations

import pytest

from gods_eye.schemas.reasoning import (
    Evidence,
    QueryStatus,
    ReasoningResult,
    ToolCall,
    ToolResult,
    UserQuery,
)


class TestUserQuery:
    def test_valid_user_query(self) -> None:
        q = UserQuery(
            query_id="q_123",
            text="Where did person X go?",
            timestamp_ns=1_700_000_000_000_000_000,
            context_camera_id="cam_01",
            default_time_window_s=1800.0,
        )
        assert q.query_id == "q_123"
        assert q.text == "Where did person X go?"
        assert q.timestamp_ns == 1_700_000_000_000_000_000
        assert q.context_camera_id == "cam_01"
        assert q.default_time_window_s == 1800.0

        d = q.to_dict()
        assert d["query_id"] == "q_123"
        assert d["text"] == "Where did person X go?"
        assert d["default_time_window_s"] == 1800.0

    def test_invalid_user_query_empty_id(self) -> None:
        with pytest.raises(ValueError, match="query_id MUST be a non-empty string"):
            UserQuery(query_id="", text="Where is X?", timestamp_ns=1000)

    def test_invalid_user_query_empty_text(self) -> None:
        with pytest.raises(ValueError, match="text MUST be a non-empty string"):
            UserQuery(query_id="q_1", text="   ", timestamp_ns=1000)

    def test_invalid_user_query_negative_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp_ns MUST be a positive Unix nanosecond timestamp"):
            UserQuery(query_id="q_1", text="Search", timestamp_ns=0)

    def test_invalid_user_query_negative_window(self) -> None:
        with pytest.raises(ValueError, match="default_time_window_s MUST be positive"):
            UserQuery(query_id="q_1", text="Search", timestamp_ns=1000, default_time_window_s=-10.0)


class TestToolCall:
    def test_valid_tool_call(self) -> None:
        tc = ToolCall(
            call_id="call_99",
            tool_name="get_identity_timeline",
            arguments={"global_id": "gid_42", "start_ns": 100, "end_ns": 200},
        )
        assert tc.call_id == "call_99"
        assert tc.tool_name == "get_identity_timeline"
        assert tc.arguments["global_id"] == "gid_42"

        d = tc.to_dict()
        assert d["call_id"] == "call_99"
        assert d["tool_name"] == "get_identity_timeline"

    def test_invalid_tool_call_empty_id(self) -> None:
        with pytest.raises(ValueError, match="call_id MUST be a non-empty string"):
            ToolCall(call_id="", tool_name="search_events")

    def test_invalid_tool_call_empty_name(self) -> None:
        with pytest.raises(ValueError, match="tool_name MUST be a non-empty string"):
            ToolCall(call_id="c_1", tool_name="  ")


class TestEvidence:
    def test_valid_evidence(self) -> None:
        ev = Evidence(
            evidence_id="ev_001",
            source_store="event_store",
            record_type="event",
            record_id="rec_abc",
            timestamp_ns=1_000_000,
            camera_id="cam_01",
            global_id="gid_01",
            explanation="Person entered camera FOV",
            payload={"frame_id": 42},
        )
        assert ev.evidence_id == "ev_001"
        assert ev.source_store == "event_store"
        assert ev.record_type == "event"
        assert ev.record_id == "rec_abc"

        d = ev.to_dict()
        assert d["evidence_id"] == "ev_001"
        assert d["payload"] == {"frame_id": 42}

    def test_invalid_evidence_empty_fields(self) -> None:
        with pytest.raises(ValueError, match="evidence_id MUST be a non-empty string"):
            Evidence(evidence_id="", source_store="event_store", record_type="event", record_id="r1", timestamp_ns=10)

        with pytest.raises(ValueError, match="source_store MUST be a non-empty string"):
            Evidence(evidence_id="e1", source_store="", record_type="event", record_id="r1", timestamp_ns=10)

        with pytest.raises(ValueError, match="record_type MUST be a non-empty string"):
            Evidence(evidence_id="e1", source_store="store", record_type="", record_id="r1", timestamp_ns=10)

        with pytest.raises(ValueError, match="record_id MUST be a non-empty string"):
            Evidence(evidence_id="e1", source_store="store", record_type="event", record_id=" ", timestamp_ns=10)

    def test_invalid_evidence_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp_ns MUST be a positive Unix nanosecond timestamp"):
            Evidence(evidence_id="e1", source_store="store", record_type="event", record_id="r1", timestamp_ns=-5)


class TestToolResult:
    def test_valid_tool_result(self) -> None:
        ev = Evidence("ev1", "event_store", "event", "r1", 100)
        tr = ToolResult(
            call_id="c_1",
            tool_name="search_events_by_type",
            success=True,
            data=[{"event_id": "r1"}],
            evidence_list=[ev],
        )
        assert tr.call_id == "c_1"
        assert tr.success is True
        assert len(tr.evidence_list) == 1

        d = tr.to_dict()
        assert d["call_id"] == "c_1"
        assert len(d["evidence_list"]) == 1

    def test_invalid_tool_result(self) -> None:
        with pytest.raises(ValueError, match="call_id MUST be a non-empty string"):
            ToolResult(call_id="", tool_name="search", success=False)


class TestReasoningResultAndStatus:
    def test_query_status_enum(self) -> None:
        assert QueryStatus.SUCCESS.value == "success"
        assert QueryStatus.AMBIGUOUS.value == "ambiguous"
        assert QueryStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert QueryStatus.INVALID_QUERY.value == "invalid_query"
        assert QueryStatus.EXECUTION_ERROR.value == "execution_error"

    def test_valid_reasoning_result(self) -> None:
        ev = Evidence("ev1", "event_store", "event", "r1", 100)
        tc = ToolCall("c1", "search_events")
        rr = ReasoningResult(
            query_id="q_1",
            status=QueryStatus.SUCCESS,
            answer="Identity X was last seen at Cam 1.",
            confidence=0.95,
            confidence_band=(0.90, 0.98),
            evidence=[ev],
            tool_calls=[tc],
            execution_time_ms=12.5,
            explanation="Verified via EventStore record r1",
        )
        assert rr.query_id == "q_1"
        assert rr.status == QueryStatus.SUCCESS
        assert rr.confidence == 0.95
        assert rr.confidence_band == (0.90, 0.98)

        d = rr.to_dict()
        assert d["query_id"] == "q_1"
        assert d["status"] == "success"
        assert d["confidence"] == 0.95
        assert d["confidence_band"] == [0.90, 0.98]

    def test_invalid_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="confidence MUST be between 0.0 and 1.0"):
            ReasoningResult(query_id="q1", status=QueryStatus.SUCCESS, answer="A", confidence=1.5)

    def test_invalid_confidence_band(self) -> None:
        with pytest.raises(ValueError, match="confidence_band MUST satisfy"):
            ReasoningResult(query_id="q1", status=QueryStatus.SUCCESS, answer="A", confidence_band=(0.9, 0.7))

    def test_invalid_execution_time(self) -> None:
        with pytest.raises(ValueError, match="execution_time_ms MUST be non-negative"):
            ReasoningResult(query_id="q1", status=QueryStatus.SUCCESS, answer="A", execution_time_ms=-1.0)
