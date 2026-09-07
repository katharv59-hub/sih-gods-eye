"""Alert Engine — Phase 10.1 (SIH 26187).

Creates alerts from RiskSignal/Hypothesis outputs with:
- Severity sourced from RiskSignal.risk_level (no second taxonomy)
- Configurable action threshold (ALERT_MIN_SEVERITY)
- Dedup window (INCIDENT_MERGE_WINDOW_S)
- Lifecycle: DETECTED → CONFIRMED → ACTIVE → RESOLVED
"""

from __future__ import annotations

import uuid
import time
from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.alert import Alert, AlertStatus, ALLOWED_SEVERITIES
from gods_eye.schemas.situational import RiskSignal, HypothesisTree

_log = get_logger("alerts.alert_engine")

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
DEFAULT_ALERT_MIN_SEVERITY = "MEDIUM"
DEFAULT_INCIDENT_MERGE_WINDOW_S = 30.0


class AlertEngine:
    """Alert creation and lifecycle management engine.

    Severity is sourced directly from SituationalRiskEvaluator output
    (RiskSignal.risk_level). Deduplication merges alerts for the same
    subject_ref + zone within INCIDENT_MERGE_WINDOW_S.
    """

    def __init__(
        self,
        alert_min_severity: str = DEFAULT_ALERT_MIN_SEVERITY,
        incident_merge_window_s: float = DEFAULT_INCIDENT_MERGE_WINDOW_S,
    ) -> None:
        if alert_min_severity not in ALLOWED_SEVERITIES:
            raise ValueError(
                f"alert_min_severity MUST be one of {sorted(ALLOWED_SEVERITIES)}, "
                f"got '{alert_min_severity}'"
            )
        self._alert_min_severity = alert_min_severity
        self._incident_merge_window_s = incident_merge_window_s
        # Dedup tracking: {(subject_ref, zone_id): (alert_id, last_timestamp_ns)}
        self._recent_alerts: dict[tuple[str, str], tuple[str, int]] = {}

    def evaluate_risk_signal(
        self,
        risk_signal: RiskSignal,
        camera_id: str,
        subject_ref: Optional[str] = None,
        zone_id: Optional[str] = None,
        hypothesis_ids: tuple[str, ...] = (),
    ) -> Optional[Alert]:
        """Evaluate a RiskSignal and create an alert if threshold met.

        Args:
            risk_signal: RiskSignal from SituationalRiskEvaluator.
            camera_id: Source camera.
            subject_ref: Pseudonymous subject reference.
            zone_id: Zone where the risk was assessed.
            hypothesis_ids: Associated hypothesis IDs.

        Returns:
            Alert if severity threshold met and not deduplicated, None otherwise.
        """
        # Check suppression
        if risk_signal.suppressed:
            _log.debug(
                "alert_suppressed_risk_signal",
                signal_id=risk_signal.signal_id,
                reason=risk_signal.suppression_reason,
            )
            return None

        # Check severity threshold
        signal_severity_order = SEVERITY_ORDER.get(risk_signal.risk_level, 0)
        min_severity_order = SEVERITY_ORDER.get(self._alert_min_severity, 1)

        if signal_severity_order < min_severity_order:
            return None

        # Deduplication check
        dedup_key = (subject_ref or "unknown", zone_id or "global")
        now_ns = risk_signal.window_end_ns

        if dedup_key in self._recent_alerts:
            prev_alert_id, prev_ts = self._recent_alerts[dedup_key]
            elapsed_s = (now_ns - prev_ts) / 1_000_000_000.0
            if elapsed_s < self._incident_merge_window_s:
                _log.debug(
                    "alert_deduplicated",
                    subject_ref=subject_ref,
                    zone_id=zone_id,
                    existing_alert=prev_alert_id,
                    elapsed_s=elapsed_s,
                )
                return None

        # Create new alert
        alert_id = f"alert_{uuid.uuid4().hex[:12]}"

        alert = Alert(
            alert_id=alert_id,
            severity=risk_signal.risk_level,
            status=AlertStatus.DETECTED,
            subject_ref=subject_ref,
            zone_id=zone_id,
            camera_id=camera_id,
            timestamp_ns=now_ns,
            risk_signal_id=risk_signal.signal_id,
            hypothesis_ids=hypothesis_ids,
            evidence_ids=risk_signal.evidence_ids,
            explanation=risk_signal.explanation,
        )

        # Track for dedup
        self._recent_alerts[dedup_key] = (alert_id, now_ns)

        _log.info(
            "alert_created",
            alert_id=alert_id,
            severity=risk_signal.risk_level,
            subject_ref=subject_ref,
            zone_id=zone_id,
        )

        return alert

    def confirm_alert(self, alert: Alert) -> Alert:
        """Transition alert from DETECTED to CONFIRMED."""
        return Alert(
            alert_id=alert.alert_id,
            severity=alert.severity,
            status=AlertStatus.CONFIRMED,
            subject_ref=alert.subject_ref,
            zone_id=alert.zone_id,
            camera_id=alert.camera_id,
            timestamp_ns=alert.timestamp_ns,
            risk_signal_id=alert.risk_signal_id,
            hypothesis_ids=alert.hypothesis_ids,
            evidence_ids=alert.evidence_ids,
            explanation=alert.explanation,
            metadata=alert.metadata,
        )

    def activate_alert(self, alert: Alert) -> Alert:
        """Transition alert from CONFIRMED to ACTIVE."""
        return Alert(
            alert_id=alert.alert_id,
            severity=alert.severity,
            status=AlertStatus.ACTIVE,
            subject_ref=alert.subject_ref,
            zone_id=alert.zone_id,
            camera_id=alert.camera_id,
            timestamp_ns=alert.timestamp_ns,
            risk_signal_id=alert.risk_signal_id,
            hypothesis_ids=alert.hypothesis_ids,
            evidence_ids=alert.evidence_ids,
            explanation=alert.explanation,
            metadata=alert.metadata,
        )

    def resolve_alert(self, alert: Alert, resolution_note: str = "Resolved") -> Alert:
        """Transition alert to RESOLVED."""
        return Alert(
            alert_id=alert.alert_id,
            severity=alert.severity,
            status=AlertStatus.RESOLVED,
            subject_ref=alert.subject_ref,
            zone_id=alert.zone_id,
            camera_id=alert.camera_id,
            timestamp_ns=alert.timestamp_ns,
            risk_signal_id=alert.risk_signal_id,
            hypothesis_ids=alert.hypothesis_ids,
            evidence_ids=alert.evidence_ids,
            explanation=alert.explanation,
            metadata=alert.metadata,
            resolved_ns=time.time_ns(),
            resolution_note=resolution_note,
        )

    def cleanup_stale_dedup_entries(self, current_ns: int) -> int:
        """Remove old dedup entries beyond the merge window."""
        stale_keys = []
        for key, (_, ts) in self._recent_alerts.items():
            if (current_ns - ts) / 1_000_000_000.0 > self._incident_merge_window_s * 2:
                stale_keys.append(key)

        for key in stale_keys:
            del self._recent_alerts[key]

        return len(stale_keys)
