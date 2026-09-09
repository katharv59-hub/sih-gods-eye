"""Canonical Situational Reasoning Data Contracts — Phase 5.1 (§4 Canonical Data Schemas).

Defines canonical data schemas for Phase 5 Reasoning and NLQ interfaces:
- QueryStatus (Enum)
- UserQuery (Dataclass)
- ToolCall (Dataclass)
- Evidence (Dataclass)
- ToolResult (Dataclass)
- ReasoningResult (Dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QueryStatus(Enum):
    """Execution status of a reasoning query."""

    SUCCESS = "success"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_QUERY = "invalid_query"
    EXECUTION_ERROR = "execution_error"


@dataclass
class UserQuery:
    """Canonical user natural language query payload (§4)."""

    query_id: str
    text: str
    timestamp_ns: int
    context_camera_id: Optional[str] = None
    default_time_window_s: float = 3600.0

    def __post_init__(self) -> None:
        """Validate UserQuery invariants."""
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id MUST be a non-empty string")
        if not self.text or not self.text.strip():
            raise ValueError("text MUST be a non-empty string")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")
        if self.default_time_window_s <= 0.0:
            raise ValueError("default_time_window_s MUST be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "timestamp_ns": self.timestamp_ns,
            "context_camera_id": self.context_camera_id,
            "default_time_window_s": self.default_time_window_s,
        }


@dataclass
class ToolCall:
    """Canonical invocation payload for deterministic tools (§14 Phase 5)."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ToolCall invariants."""
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id MUST be a non-empty string")
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name MUST be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }


@dataclass
class Evidence:
    """Canonical evidence item linking reasoning outputs to Phase 4 memory records (Rule 8)."""

    evidence_id: str
    source_store: str  # "event_store" | "graph_store" | "environmental"
    record_type: str   # "event" | "obs_edge" | "trans_edge" | "zone_edge" | "visit_segment"
    record_id: str     # UUID of underlying record
    timestamp_ns: int
    camera_id: Optional[str] = None
    global_id: Optional[str] = None
    explanation: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate Evidence invariants."""
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id MUST be a non-empty string")
        if not self.source_store or not self.source_store.strip():
            raise ValueError("source_store MUST be a non-empty string")
        if not self.record_type or not self.record_type.strip():
            raise ValueError("record_type MUST be a non-empty string")
        if not self.record_id or not self.record_id.strip():
            raise ValueError("record_id MUST be a non-empty string")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "evidence_id": self.evidence_id,
            "source_store": self.source_store,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "timestamp_ns": self.timestamp_ns,
            "camera_id": self.camera_id,
            "global_id": self.global_id,
            "explanation": self.explanation,
            "payload": self.payload,
        }

    @classmethod
    def from_event(
        cls,
        event: Any,
        evidence_id: Optional[str] = None,
        source_store: str = "event_store",
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> Evidence:
        """Construct a canonical Evidence record directly from a canonical Event."""
        payload = dict(event.metadata) if isinstance(getattr(event, "metadata", None), dict) else {}
        if extra_payload:
            payload.update(extra_payload)

        event_type_str = (
            event.event_type.value
            if hasattr(event.event_type, "value")
            else str(event.event_type)
        )
        if "event_type" not in payload:
            payload["event_type"] = event_type_str
        if "subject_ref" not in payload and getattr(event, "global_id", None):
            payload["subject_ref"] = event.global_id

        return cls(
            evidence_id=evidence_id or f"ev_doc_{event.event_id}",
            source_store=source_store,
            record_type=event_type_str.upper(),
            record_id=event.event_id,
            timestamp_ns=event.timestamp_ns,
            camera_id=event.camera_id,
            global_id=event.global_id,
            explanation=event.explanation,
            payload=payload,
        )



@dataclass
class ToolResult:
    """Canonical output from a deterministic tool handler."""

    call_id: str
    tool_name: str
    success: bool
    data: Any = None
    evidence_list: list[Evidence] = field(default_factory=list)
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate ToolResult invariants."""
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id MUST be a non-empty string")
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name MUST be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "error_message": self.error_message,
        }


@dataclass
class ReasoningResult:
    """Canonical result emitted by the Situational Reasoning Engine (Rule 7, Rule 8)."""

    query_id: str
    status: QueryStatus
    answer: str
    confidence: float = 1.0
    confidence_band: tuple[float, float] = (1.0, 1.0)
    evidence: list[Evidence] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    execution_time_ms: float = 0.0
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate ReasoningResult invariants."""
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id MUST be a non-empty string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")
        lower, upper = self.confidence_band
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError(
                f"confidence_band MUST satisfy 0.0 <= lower <= upper <= 1.0, got {self.confidence_band}"
            )
        if self.execution_time_ms < 0.0:
            raise ValueError("execution_time_ms MUST be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "query_id": self.query_id,
            "status": self.status.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "confidence_band": list(self.confidence_band),
            "evidence": [e.to_dict() for e in self.evidence],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "execution_time_ms": self.execution_time_ms,
            "explanation": self.explanation,
        }
