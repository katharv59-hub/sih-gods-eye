"""Alert Store Adapter — Phase 10.1 (SIH 26187).

Persists alerts through the canonical SQLiteEventStore rather than
maintaining an independent database. Alert records are stored as
Event objects with event_type=ALERT_CREATED and all alert-specific
fields carried in Event.metadata.

This eliminates the duplicate persistence architecture violation.
"""

from __future__ import annotations

import json
from typing import Optional

from gods_eye.events.event_store import BaseEventStore
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.alert import Alert, AlertStatus
from gods_eye.schemas.event import Event, EventType

_log = get_logger("alerts.alert_store")


def _alert_to_event(alert: Alert) -> Event:
    """Convert an Alert to a canonical Event for persistence."""
    metadata = dict(alert.metadata) if alert.metadata else {}
    metadata.update({
        "alert_severity": alert.severity,
        "alert_status": alert.status.value,
        "alert_subject_ref": alert.subject_ref,
        "alert_risk_signal_id": alert.risk_signal_id,
        "alert_hypothesis_ids": json.dumps(list(alert.hypothesis_ids)),
        "alert_evidence_ids": json.dumps(list(alert.evidence_ids)),
        "alert_resolved_ns": alert.resolved_ns,
        "alert_resolution_note": alert.resolution_note,
    })

    return Event(
        event_id=alert.alert_id,
        event_type=EventType.ALERT_CREATED,
        global_id=alert.subject_ref,
        camera_id=alert.camera_id,
        zone_id=alert.zone_id,
        timestamp_ns=alert.timestamp_ns,
        frame_id=0,
        confidence=0.0,
        explanation=alert.explanation,
        metadata=metadata,
    )


def _event_to_alert(event: Event) -> Alert:
    """Reconstruct an Alert from a persisted Event."""
    md = event.metadata or {}

    # Extract alert-specific fields from metadata
    severity = md.get("alert_severity", "MEDIUM")
    status_str = md.get("alert_status", "detected")
    subject_ref = md.get("alert_subject_ref")
    risk_signal_id = md.get("alert_risk_signal_id", "")
    hypothesis_ids_raw = md.get("alert_hypothesis_ids", "[]")
    evidence_ids_raw = md.get("alert_evidence_ids", "[]")
    resolved_ns = md.get("alert_resolved_ns")
    resolution_note = md.get("alert_resolution_note")

    # Parse JSON list fields
    if isinstance(hypothesis_ids_raw, str):
        hypothesis_ids = tuple(json.loads(hypothesis_ids_raw))
    else:
        hypothesis_ids = tuple(hypothesis_ids_raw) if hypothesis_ids_raw else ()

    if isinstance(evidence_ids_raw, str):
        evidence_ids = tuple(json.loads(evidence_ids_raw))
    else:
        evidence_ids = tuple(evidence_ids_raw) if evidence_ids_raw else ()

    # Reconstruct user metadata (strip alert-internal keys)
    alert_meta_keys = {
        "alert_severity", "alert_status", "alert_subject_ref",
        "alert_risk_signal_id", "alert_hypothesis_ids", "alert_evidence_ids",
        "alert_resolved_ns", "alert_resolution_note",
    }
    user_metadata = {k: v for k, v in md.items() if k not in alert_meta_keys}

    return Alert(
        alert_id=event.event_id,
        severity=severity,
        status=AlertStatus(status_str),
        subject_ref=subject_ref,
        zone_id=event.zone_id,
        camera_id=event.camera_id,
        timestamp_ns=event.timestamp_ns,
        risk_signal_id=risk_signal_id,
        hypothesis_ids=hypothesis_ids,
        evidence_ids=evidence_ids,
        explanation=event.explanation,
        metadata=user_metadata,
        resolved_ns=resolved_ns,
        resolution_note=resolution_note,
    )


class AlertStore:
    """Alert persistence adapter wrapping the canonical EventStore.

    Eliminates the duplicate SQLiteAlertStore / alerts.db architecture.
    All alert records are persisted as Event objects through the single
    canonical event persistence layer.
    """

    def __init__(self, event_store: BaseEventStore) -> None:
        self._event_store = event_store
        _log.info("alert_store_init", backend="EventStore_adapter")

    @property
    def event_store(self) -> BaseEventStore:
        """Return the underlying canonical event store."""
        return self._event_store

    def append_alert(self, alert: Alert) -> None:
        """Persist a new alert through the canonical event store."""
        event = _alert_to_event(alert)
        self._event_store.upsert(event)

    def append_batch(self, alerts: list[Alert]) -> None:
        """Persist a batch of alerts."""
        for alert in alerts:
            self.append_alert(alert)

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrieve a single alert by ID."""
        event = self._event_store.get_by_id(alert_id)
        if event is None:
            return None
        if event.event_type != EventType.ALERT_CREATED:
            return None
        return _event_to_alert(event)

    def get_alerts_by_severity(
        self,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Retrieve alerts filtered by severity."""
        events = self._event_store.query_events(
            event_type=EventType.ALERT_CREATED,
            limit=limit * 5,  # Over-fetch to compensate for post-filter
        )
        alerts = [_event_to_alert(e) for e in events]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by timestamp descending and limit
        alerts.sort(key=lambda a: a.timestamp_ns, reverse=True)
        return alerts[:limit]

    def get_active_alerts(self, limit: int = 100) -> list[Alert]:
        """Retrieve alerts that are not resolved."""
        events = self._event_store.query_events(
            event_type=EventType.ALERT_CREATED,
            limit=limit * 5,
        )
        alerts = [_event_to_alert(e) for e in events]
        active = [a for a in alerts if a.status != AlertStatus.RESOLVED]
        active.sort(key=lambda a: a.timestamp_ns, reverse=True)
        return active[:limit]

    def update_status(
        self,
        alert_id: str,
        status: AlertStatus,
        resolved_ns: Optional[int] = None,
        resolution_note: Optional[str] = None,
    ) -> None:
        """Update alert lifecycle status through the canonical event store."""
        event = self._event_store.get_by_id(alert_id)
        if event is None:
            _log.warning("alert_update_not_found", alert_id=alert_id)
            return

        # Update metadata with new status
        metadata = dict(event.metadata) if event.metadata else {}
        metadata["alert_status"] = status.value
        if resolved_ns is not None:
            metadata["alert_resolved_ns"] = resolved_ns
        if resolution_note is not None:
            metadata["alert_resolution_note"] = resolution_note

        updated_event = Event(
            event_id=event.event_id,
            event_type=event.event_type,
            global_id=event.global_id,
            camera_id=event.camera_id,
            zone_id=event.zone_id,
            timestamp_ns=event.timestamp_ns,
            frame_id=event.frame_id,
            confidence=event.confidence,
            explanation=event.explanation,
            metadata=metadata,
        )
        self._event_store.upsert(updated_event)

    def count(self) -> int:
        """Return total alert count."""
        events = self._event_store.query_events(
            event_type=EventType.ALERT_CREATED,
            limit=100_000,
        )
        return len(events)

    def close(self) -> None:
        """No-op — lifecycle managed by the canonical event store."""
        pass


# Backward-compatibility alias (deprecated)
SQLiteAlertStore = AlertStore
