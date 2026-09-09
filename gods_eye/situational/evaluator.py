"""Situational Risk Evaluator — Sub-Phase 7.2 (§5, §6 & §18 Master Spec).

Provides deterministic, synchronous, in-memory situational risk evaluation.
"""

from __future__ import annotations

import time
from typing import Sequence

from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.reasoning import Evidence
from gods_eye.schemas.situational import RiskSignal

MAX_WINDOW_DURATION_NS: int = 86_400_000_000_000  # 24 hours in nanoseconds
FUTURE_TIMESTAMP_TOLERANCE_NS: int = 1_000_000_000  # 1 second in nanoseconds
MIN_EVIDENCE_CONFIDENCE_THRESHOLD: float = 0.70


class SituationalRiskEvaluator:
    """Deterministic situational risk evaluator engine."""

    def __init__(self, evaluator_version: str = "v7.2.0") -> None:
        if not evaluator_version or not evaluator_version.strip():
            raise ValueError("evaluator_version MUST be a non-empty string")
        self._evaluator_version = evaluator_version

    @property
    def evaluator_version(self) -> str:
        """Return evaluator version string."""
        return self._evaluator_version

    def evaluate(
        self,
        evidence_items: tuple[Evidence, ...] | list[Evidence],
        system_mode: SystemMode,
        window_start_ns: int,
        window_end_ns: int,
    ) -> RiskSignal:
        """Evaluate input evidence items and return a deterministic RiskSignal.

        Args:
            evidence_items: Collection of input Evidence objects.
            system_mode: Current system operational mode (LEARNING, OPERATIONAL, DEGRADED).
            window_start_ns: Evaluation window start (nanoseconds).
            window_end_ns: Evaluation window end (nanoseconds).

        Returns:
            RiskSignal dataclass instance.
        """
        # 1. Temporal Window Validation
        if not isinstance(window_start_ns, int) or window_start_ns < 0:
            raise ValueError(f"window_start_ns MUST be a non-negative integer, got {window_start_ns}")
        if not isinstance(window_end_ns, int) or window_end_ns < 0:
            raise ValueError(f"window_end_ns MUST be a non-negative integer, got {window_end_ns}")
        if window_start_ns > window_end_ns:
            raise ValueError(
                f"window_start_ns ({window_start_ns}) MUST be <= window_end_ns ({window_end_ns})"
            )
        if (window_end_ns - window_start_ns) > MAX_WINDOW_DURATION_NS:
            raise ValueError(
                f"Window duration MUST NOT exceed 24 hours (86400000000000 ns), got {window_end_ns - window_start_ns}"
            )

        now_ns = time.time_ns()
        if (window_start_ns - now_ns) > FUTURE_TIMESTAMP_TOLERANCE_NS or (
            window_end_ns - now_ns
        ) > FUTURE_TIMESTAMP_TOLERANCE_NS:
            raise ValueError("Window timestamp MUST NOT exceed current time by more than 1 second")

        if not isinstance(system_mode, SystemMode):
            raise ValueError(f"system_mode MUST be a SystemMode instance, got {system_mode}")

        # 2. Filter evidence items inside [window_start_ns, window_end_ns]
        if isinstance(evidence_items, list):
            evidence_sequence: Sequence[Evidence] = tuple(evidence_items)
        elif isinstance(evidence_items, tuple):
            evidence_sequence = evidence_items
        else:
            raise ValueError("evidence_items MUST be a tuple or list of Evidence objects")

        valid_evidence: list[Evidence] = [
            ev
            for ev in evidence_sequence
            if isinstance(ev, Evidence) and window_start_ns <= ev.timestamp_ns <= window_end_ns
        ]

        # 3. Deterministic Suppression Precedence
        suppressed = False
        suppression_reason: str | None = None

        if system_mode == SystemMode.LEARNING_MODE:
            suppressed = True
            suppression_reason = "System in LEARNING_MODE: baseline warm-up incomplete"
        elif system_mode == SystemMode.DEGRADED_MODE:
            suppressed = True
            suppression_reason = "System in DEGRADED_MODE: model drift or hardware constraint"
        elif len(valid_evidence) == 0:
            suppressed = True
            suppression_reason = "Insufficient evidence: zero evidence items provided"
            return RiskSignal(
                signal_id=f"risk_sig_{window_start_ns}_{window_end_ns}",
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
                risk_level="LOW",
                risk_score=0.25,
                evidence_ids=(),
                contributing_factors=(),
                explanation="Risk evaluation SUPPRESSED (Insufficient evidence: zero evidence items provided). Assessed risk level: LOW (score: 0.25). Contributing factors: none.",
                suppressed=True,
                suppression_reason=suppression_reason,
                evaluator_version=self._evaluator_version,
            )
        else:
            # Check aggregate evidence confidence
            confidences = [
                float(ev.payload.get("confidence", 1.0))
                if isinstance(ev.payload, dict) and "confidence" in ev.payload
                else 1.0
                for ev in valid_evidence
            ]
            mean_conf = sum(confidences) / len(confidences) if confidences else 1.0
            if mean_conf < MIN_EVIDENCE_CONFIDENCE_THRESHOLD:
                suppressed = True
                suppression_reason = "Evidence confidence below threshold (0.70)"

        # 4. Evaluate Heuristic Rules Across Valid Evidence Items
        # Rule boundaries (STRICT > comparisons per preflight & prompt):
        # CRITICAL: trajectory_anomaly_sigma > 3.5 AND zone_dwell_sigma > 5.0 AND is_restricted_zone
        # HIGH: trajectory_anomaly_sigma > 2.5 OR occupancy_deviation_sigma > 4.0
        # MEDIUM: trajectory_anomaly_sigma > 1.5 OR 0.0 <= unlikely_transition_probability < 0.10
        # LOW: baseline

        firing_rules: set[str] = set()
        firing_evidence_map: dict[str, set[str]] = {
            "CRITICAL": set(),
            "HIGH": set(),
            "MEDIUM": set(),
        }

        for ev in valid_evidence:
            payload = ev.payload if isinstance(ev.payload, dict) else {}

            # Extract signals with payload key flexibility
            traj_sigma = payload.get(
                "trajectory_anomaly_sigma",
                payload.get("trajectory_sigma", payload.get("sigma", None)),
            )
            dwell_sigma = payload.get("zone_dwell_sigma", payload.get("dwell_sigma", None))
            is_restricted = payload.get(
                "is_restricted_zone",
                payload.get("restricted_zone", payload.get("is_restricted", False)),
            )
            occ_sigma = payload.get(
                "occupancy_deviation_sigma", payload.get("occupancy_sigma", None)
            )
            unlikely_prob = payload.get(
                "unlikely_transition_probability",
                payload.get("transition_probability", payload.get("probability", None)),
            )

            # Check CRITICAL rule
            if (
                traj_sigma is not None
                and float(traj_sigma) > 3.5
                and dwell_sigma is not None
                and float(dwell_sigma) > 5.0
                and bool(is_restricted) is True
            ):
                firing_rules.add("CRITICAL")
                firing_rules.add("trajectory_anomaly_gt_3.5sigma")
                firing_rules.add("zone_dwell_gt_5.0sigma")
                firing_rules.add("restricted_zone_condition")
                firing_evidence_map["CRITICAL"].add(ev.evidence_id)

            # Check HIGH rule
            high_fired = False
            if traj_sigma is not None and float(traj_sigma) > 2.5:
                firing_rules.add("HIGH")
                firing_rules.add("trajectory_anomaly_gt_2.5sigma")
                high_fired = True
            if occ_sigma is not None and float(occ_sigma) > 4.0:
                firing_rules.add("HIGH")
                firing_rules.add("occupancy_deviation_gt_4.0sigma")
                high_fired = True
            if high_fired:
                firing_evidence_map["HIGH"].add(ev.evidence_id)

            # Check MEDIUM rule
            medium_fired = False
            if traj_sigma is not None and float(traj_sigma) > 1.5:
                firing_rules.add("MEDIUM")
                firing_rules.add("trajectory_anomaly_gt_1.5sigma")
                medium_fired = True
            if unlikely_prob is not None and 0.0 <= float(unlikely_prob) < 0.10:
                firing_rules.add("MEDIUM")
                firing_rules.add("unlikely_transition_lt_0.10")
                medium_fired = True
            if medium_fired:
                firing_evidence_map["MEDIUM"].add(ev.evidence_id)

        # Cross-evidence co-occurrence per subject for multi-factor rules:
        # If CRITICAL hasn't fired yet on a single item, check if independent
        # evidence items for the same subject jointly satisfy the CRITICAL rule
        # (trajectory_anomaly > 3.5σ AND zone_dwell > 5.0σ AND is_restricted_zone).
        if "CRITICAL" not in firing_rules:
            subject_evidence: dict[str, list[Evidence]] = {}
            for ev in valid_evidence:
                payload = ev.payload if isinstance(ev.payload, dict) else {}
                sub = str(payload.get("subject_ref") or ev.global_id or "")
                if sub:
                    if sub not in subject_evidence:
                        subject_evidence[sub] = []
                    subject_evidence[sub].append(ev)

            for sub, sub_evs in subject_evidence.items():
                if len(sub_evs) < 2:
                    continue

                traj_items: list[tuple[float, str]] = []
                dwell_items: list[tuple[float, str]] = []
                restricted_eids: list[str] = []

                for ev in sub_evs:
                    payload = ev.payload if isinstance(ev.payload, dict) else {}
                    ts = payload.get(
                        "trajectory_anomaly_sigma",
                        payload.get("trajectory_sigma", payload.get("sigma", None)),
                    )
                    if ts is not None:
                        traj_items.append((float(ts), ev.evidence_id))

                    ds = payload.get("zone_dwell_sigma", payload.get("dwell_sigma", None))
                    if ds is not None:
                        dwell_items.append((float(ds), ev.evidence_id))

                    rz = payload.get(
                        "is_restricted_zone",
                        payload.get("restricted_zone", payload.get("is_restricted", False)),
                    )
                    if bool(rz) is True:
                        restricted_eids.append(ev.evidence_id)

                max_traj = max((t[0] for t in traj_items), default=None)
                max_dwell = max((d[0] for d in dwell_items), default=None)

                if (
                    max_traj is not None
                    and max_traj > 3.5
                    and max_dwell is not None
                    and max_dwell > 5.0
                    and len(restricted_eids) > 0
                ):
                    firing_rules.add("CRITICAL")
                    firing_rules.add("trajectory_anomaly_gt_3.5sigma")
                    firing_rules.add("zone_dwell_gt_5.0sigma")
                    firing_rules.add("restricted_zone_condition")
                    for val, eid in traj_items:
                        if val > 3.5:
                            firing_evidence_map["CRITICAL"].add(eid)
                    for val, eid in dwell_items:
                        if val > 5.0:
                            firing_evidence_map["CRITICAL"].add(eid)
                    for eid in restricted_eids:
                        firing_evidence_map["CRITICAL"].add(eid)

        # 5. Maximum Severity Precedence
        if "CRITICAL" in firing_rules:
            risk_level = "CRITICAL"
            risk_score = 1.00
        elif "HIGH" in firing_rules:
            risk_level = "HIGH"
            risk_score = 0.75
        elif "MEDIUM" in firing_rules:
            risk_level = "MEDIUM"
            risk_score = 0.50
        else:
            risk_level = "LOW"
            risk_score = 0.25

        # 6. Preserved Contributing Factors & Evidence Provenance
        possible_factors = (
            "trajectory_anomaly_gt_3.5sigma",
            "zone_dwell_gt_5.0sigma",
            "restricted_zone_condition",
            "trajectory_anomaly_gt_2.5sigma",
            "occupancy_deviation_gt_4.0sigma",
            "trajectory_anomaly_gt_1.5sigma",
            "unlikely_transition_lt_0.10",
        )
        contributing_factors = tuple(
            factor for factor in possible_factors if factor in firing_rules
        )

        if risk_level != "LOW":
            firing_ids: set[str] = set()
            for r_level, ids in firing_evidence_map.items():
                if r_level in firing_rules:
                    firing_ids.update(ids)
            evidence_ids = tuple(sorted(list(firing_ids)))
        else:
            evidence_ids = tuple(sorted(list({ev.evidence_id for ev in valid_evidence})))

        # 7. Deterministic Factual Explanation Construction
        factors_str = ", ".join(contributing_factors) if contributing_factors else "none"
        if suppressed:
            explanation = (
                f"Risk evaluation SUPPRESSED ({suppression_reason}). "
                f"Assessed risk level: {risk_level} (score: {risk_score:.2f}). "
                f"Contributing factors: {factors_str}."
            )
        else:
            explanation = (
                f"Assessed risk level: {risk_level} (score: {risk_score:.2f}). "
                f"Contributing factors: {factors_str}."
            )

        signal_id = f"risk_sig_{window_start_ns}_{window_end_ns}"

        return RiskSignal(
            signal_id=signal_id,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
            risk_level=risk_level,
            risk_score=risk_score,
            evidence_ids=evidence_ids,
            contributing_factors=contributing_factors,
            explanation=explanation,
            suppressed=suppressed,
            suppression_reason=suppression_reason,
            evaluator_version=self._evaluator_version,
        )
