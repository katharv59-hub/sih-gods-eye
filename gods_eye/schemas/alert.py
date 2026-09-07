"""Alert Schemas — Phase 10.1 (SIH 26187).

Defines canonical data contracts for the Alert Engine:
- AlertStatus: lifecycle enum (DETECTED → CONFIRMED → ACTIVE → RESOLVED)
- Alert: alert record with severity sourced from RiskSignal.risk_level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AlertStatus(Enum):
    """Alert lifecycle states."""

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    RESOLVED = "resolved"


ALLOWED_SEVERITIES: set[str] = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


@dataclass
class Alert:
    """An actionable alert produced by the Alert Engine.

    Severity is sourced directly from SituationalRiskEvaluator output
    (RiskSignal.risk_level). No second severity taxonomy.

    Attributes:
        alert_id: Unique alert identifier.
        severity: Risk level from RiskSignal (LOW/MEDIUM/HIGH/CRITICAL).
        status: Lifecycle state.
        subject_ref: Pseudonymous subject reference.
        zone_id: Zone where the alert originated.
        camera_id: Camera where the alert originated.
        timestamp_ns: Alert creation time in Unix nanoseconds.
        risk_signal_id: ID of the RiskSignal that triggered this alert.
        hypothesis_ids: IDs of associated hypotheses.
        evidence_ids: IDs of associated evidence items.
        explanation: Human-readable alert explanation.
        metadata: Additional alert-specific payload.
        resolved_ns: Timestamp when alert was resolved (None if unresolved).
        resolution_note: Human-readable resolution explanation.
    """

    alert_id: str
    severity: str
    status: AlertStatus
    subject_ref: Optional[str]
    zone_id: Optional[str]
    camera_id: str
    timestamp_ns: int
    risk_signal_id: str
    hypothesis_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_ns: Optional[int] = None
    resolution_note: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate Alert invariants."""
        if not self.alert_id or not self.alert_id.strip():
            raise ValueError("alert_id MUST be a non-empty string")
        if self.severity not in ALLOWED_SEVERITIES:
            raise ValueError(
                f"severity MUST be one of {sorted(ALLOWED_SEVERITIES)}, got '{self.severity}'"
            )
        if not isinstance(self.status, AlertStatus):
            raise ValueError(f"status MUST be an AlertStatus instance, got {type(self.status)}")
        if not self.camera_id or not self.camera_id.strip():
            raise ValueError("camera_id MUST be a non-empty string")
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")
        if not self.risk_signal_id or not self.risk_signal_id.strip():
            raise ValueError("risk_signal_id MUST be a non-empty string")

        # Coerce lists to tuples for immutability
        if isinstance(self.hypothesis_ids, list):
            object.__setattr__(self, "hypothesis_ids", tuple(self.hypothesis_ids))
        if isinstance(self.evidence_ids, list):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "status": self.status.value,
            "subject_ref": self.subject_ref,
            "zone_id": self.zone_id,
            "camera_id": self.camera_id,
            "timestamp_ns": self.timestamp_ns,
            "risk_signal_id": self.risk_signal_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "evidence_ids": list(self.evidence_ids),
            "explanation": self.explanation,
            "metadata": self.metadata,
            "resolved_ns": self.resolved_ns,
            "resolution_note": self.resolution_note,
        }
