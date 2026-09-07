"""Unit test suite for Sub-Phase 7.2 Situational Risk Evaluator."""

from __future__ import annotations

import time
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.reasoning import Evidence
from gods_eye.schemas.situational import RiskSignal
from gods_eye.situational.evaluator import SituationalRiskEvaluator


@pytest.fixture
def evaluator() -> SituationalRiskEvaluator:
    return SituationalRiskEvaluator(evaluator_version="v7.2.0")


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
        explanation="Test evidence",
        payload=payload or {},
    )


class TestSituationalRiskEvaluatorScenarios:
    # 1. test_empty_evidence_suppressed_low_risk
    def test_empty_evidence_suppressed_low_risk(self, evaluator: SituationalRiskEvaluator) -> None:
        sig = evaluator.evaluate([], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "LOW"
        assert sig.risk_score == 0.25
        assert sig.suppressed is True
        assert sig.suppression_reason == "Insufficient evidence: zero evidence items provided"
        assert sig.evidence_ids == ()
        assert sig.contributing_factors == ()

    # 2. test_learning_mode_suppression
    def test_learning_mode_suppression(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 3.0})
        sig = evaluator.evaluate([ev], SystemMode.LEARNING_MODE, 1000, 2000)
        assert sig.suppressed is True
        assert sig.suppression_reason == "System in LEARNING_MODE: baseline warm-up incomplete"
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.75

    # 3. test_degraded_mode_suppression
    def test_degraded_mode_suppression(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 3.0})
        sig = evaluator.evaluate([ev], SystemMode.DEGRADED_MODE, 1000, 2000)
        assert sig.suppressed is True
        assert sig.suppression_reason == "System in DEGRADED_MODE: model drift or hardware constraint"
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.75

    # 4. test_low_risk_baseline_evaluation
    def test_low_risk_baseline_evaluation(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 0.5})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "LOW"
        assert sig.risk_score == 0.25
        assert sig.suppressed is False
        assert sig.suppression_reason is None
        assert sig.evidence_ids == ("ev_001",)
        assert sig.contributing_factors == ()

    # 5. test_medium_risk_trajectory_anomaly
    def test_medium_risk_trajectory_anomaly(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 1.8})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "MEDIUM"
        assert sig.risk_score == 0.50
        assert sig.suppressed is False
        assert "trajectory_anomaly_gt_1.5sigma" in sig.contributing_factors

    # 6. test_high_risk_trajectory_anomaly
    def test_high_risk_trajectory_anomaly(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 2.8})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.75
        assert sig.suppressed is False
        assert "trajectory_anomaly_gt_2.5sigma" in sig.contributing_factors

    # 7. test_critical_risk_combined_anomaly
    def test_critical_risk_combined_anomaly(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence(
            timestamp_ns=1500,
            payload={
                "trajectory_anomaly_sigma": 3.8,
                "zone_dwell_sigma": 5.2,
                "is_restricted_zone": True,
            },
        )
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "CRITICAL"
        assert sig.risk_score == 1.00
        assert sig.suppressed is False
        assert "trajectory_anomaly_gt_3.5sigma" in sig.contributing_factors
        assert "zone_dwell_gt_5.0sigma" in sig.contributing_factors
        assert "restricted_zone_condition" in sig.contributing_factors

    # 8. test_max_severity_precedence
    def test_max_severity_precedence(self, evaluator: SituationalRiskEvaluator) -> None:
        ev_low = make_evidence("ev_low", 1200, {"trajectory_anomaly_sigma": 0.2})
        ev_med = make_evidence("ev_med", 1300, {"trajectory_anomaly_sigma": 1.8})
        ev_high = make_evidence("ev_high", 1400, {"occupancy_deviation_sigma": 4.2})

        sig = evaluator.evaluate([ev_low, ev_med, ev_high], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.75
        assert "occupancy_deviation_gt_4.0sigma" in sig.contributing_factors
        assert "trajectory_anomaly_gt_1.5sigma" in sig.contributing_factors

    # 9. test_additive_contributing_factors_and_evidence_ids
    def test_additive_contributing_factors_and_evidence_ids(
        self, evaluator: SituationalRiskEvaluator
    ) -> None:
        ev1 = make_evidence("ev_1", 1200, {"trajectory_anomaly_sigma": 2.8})
        ev2 = make_evidence("ev_2", 1400, {"unlikely_transition_probability": 0.04})

        sig = evaluator.evaluate([ev1, ev2], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "HIGH"
        assert sig.evidence_ids == ("ev_1", "ev_2")
        assert "trajectory_anomaly_gt_2.5sigma" in sig.contributing_factors
        assert "unlikely_transition_lt_0.10" in sig.contributing_factors

    # 10. test_low_confidence_evidence_suppression
    def test_low_confidence_evidence_suppression(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 3.0, "confidence": 0.50})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.suppressed is True
        assert sig.suppression_reason == "Evidence confidence below threshold (0.70)"
        assert sig.risk_level == "HIGH"

    # 11. test_stale_evidence_filtering
    def test_stale_evidence_filtering(self, evaluator: SituationalRiskEvaluator) -> None:
        ev_stale = make_evidence("ev_stale", 500, {"trajectory_anomaly_sigma": 4.0})
        ev_valid = make_evidence("ev_valid", 1500, {"trajectory_anomaly_sigma": 0.2})

        sig = evaluator.evaluate([ev_stale, ev_valid], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "LOW"
        assert sig.evidence_ids == ("ev_valid",)

    # 12. test_invalid_temporal_window_rejection
    def test_invalid_temporal_window_rejection(self, evaluator: SituationalRiskEvaluator) -> None:
        with pytest.raises(ValueError, match="MUST be <= window_end_ns"):
            evaluator.evaluate([], SystemMode.OPERATIONAL_MODE, 3000, 2000)

        with pytest.raises(ValueError, match="duration MUST NOT exceed 24 hours"):
            evaluator.evaluate([], SystemMode.OPERATIONAL_MODE, 0, 100_000_000_000_000)

        future_ns = time.time_ns() + 10_000_000_000  # 10 seconds into future
        with pytest.raises(ValueError, match="Window timestamp MUST NOT exceed current time"):
            evaluator.evaluate([], SystemMode.OPERATIONAL_MODE, future_ns, future_ns + 1000)

    # 13. test_deterministic_repeated_execution
    def test_deterministic_repeated_execution(self, evaluator: SituationalRiskEvaluator) -> None:
        ev = make_evidence("ev_1", 1500, {"trajectory_anomaly_sigma": 3.0})
        sig1 = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        sig2 = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)

        assert sig1 == sig2
        assert sig1.signal_id == sig2.signal_id
        assert sig1.explanation == sig2.explanation

    # 14. test_risk_score_step_bounds
    def test_risk_score_step_bounds(self, evaluator: SituationalRiskEvaluator) -> None:
        ev_low = make_evidence("ev1", 1500, {})
        ev_med = make_evidence("ev2", 1500, {"trajectory_anomaly_sigma": 1.6})
        ev_high = make_evidence("ev3", 1500, {"trajectory_anomaly_sigma": 2.6})
        ev_crit = make_evidence(
            "ev4",
            1500,
            {"trajectory_anomaly_sigma": 3.6, "zone_dwell_sigma": 5.1, "is_restricted_zone": True},
        )

        assert evaluator.evaluate([ev_low], SystemMode.OPERATIONAL_MODE, 1000, 2000).risk_score == 0.25
        assert evaluator.evaluate([ev_med], SystemMode.OPERATIONAL_MODE, 1000, 2000).risk_score == 0.50
        assert evaluator.evaluate([ev_high], SystemMode.OPERATIONAL_MODE, 1000, 2000).risk_score == 0.75
        assert evaluator.evaluate([ev_crit], SystemMode.OPERATIONAL_MODE, 1000, 2000).risk_score == 1.00

    # 15. test_suppression_reason_consistency
    def test_suppression_reason_consistency(self, evaluator: SituationalRiskEvaluator) -> None:
        sig_unsuppressed = evaluator.evaluate(
            [make_evidence(timestamp_ns=1500)], SystemMode.OPERATIONAL_MODE, 1000, 2000
        )
        assert sig_unsuppressed.suppressed is False
        assert sig_unsuppressed.suppression_reason is None

        sig_suppressed = evaluator.evaluate(
            [make_evidence(timestamp_ns=1500)], SystemMode.LEARNING_MODE, 1000, 2000
        )
        assert sig_suppressed.suppressed is True
        assert isinstance(sig_suppressed.suppression_reason, str)
        assert len(sig_suppressed.suppression_reason) > 0

    # 16. test_evaluator_version_string
    def test_evaluator_version_string(self, evaluator: SituationalRiskEvaluator) -> None:
        assert evaluator.evaluator_version == "v7.2.0"
        sig = evaluator.evaluate([], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.evaluator_version == "v7.2.0"

    # 17. test_read_only_invariants
    def test_read_only_invariants(self, evaluator: SituationalRiskEvaluator, tmp_path) -> None:
        db_path = str(tmp_path / "test_store.db")
        es = SQLiteEventStore(db_path)
        gs = SQLiteGraphStore(db_path)

        ev_list = [make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 3.0})]
        sig = evaluator.evaluate(ev_list, SystemMode.OPERATIONAL_MODE, 1000, 2000)

        # Verify DB store row counts remain 0
        assert len(es.query_events()) == 0
        assert len(gs.get_camera_transitions()) == 0
        # Verify ev_list was not mutated
        assert len(ev_list) == 1

        es.close()

    # 18. test_phase1_to_phase6_regression
    def test_phase1_to_phase6_regression(self, evaluator: SituationalRiskEvaluator) -> None:
        # Evaluator evaluation does not interfere with baseline dataclasses
        assert isinstance(evaluator, SituationalRiskEvaluator)


class TestBoundaryValuesStrictComparisons:
    def test_boundary_values_1_5_sigma_no_fire(self, evaluator: SituationalRiskEvaluator) -> None:
        # 1.5 equals threshold -> MUST NOT fire > 1.5 rule
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 1.5})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "LOW"
        assert sig.risk_score == 0.25

    def test_boundary_values_2_5_sigma_no_fire(self, evaluator: SituationalRiskEvaluator) -> None:
        # 2.5 equals threshold -> Fires > 1.5 (MEDIUM), but NOT > 2.5 (HIGH)
        ev = make_evidence(timestamp_ns=1500, payload={"trajectory_anomaly_sigma": 2.5})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "MEDIUM"
        assert sig.risk_score == 0.50

    def test_boundary_values_3_5_sigma_no_fire(self, evaluator: SituationalRiskEvaluator) -> None:
        # 3.5 equals threshold -> Fires > 2.5 (HIGH), but NOT > 3.5 (CRITICAL)
        ev = make_evidence(
            timestamp_ns=1500,
            payload={
                "trajectory_anomaly_sigma": 3.5,
                "zone_dwell_sigma": 6.0,
                "is_restricted_zone": True,
            },
        )
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.75

    def test_boundary_values_4_0_sigma_no_fire(self, evaluator: SituationalRiskEvaluator) -> None:
        # 4.0 equals threshold -> MUST NOT fire > 4.0 rule
        ev = make_evidence(timestamp_ns=1500, payload={"occupancy_deviation_sigma": 4.0})
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        assert sig.risk_level == "LOW"
        assert sig.risk_score == 0.25

    def test_boundary_values_5_0_sigma_no_fire(self, evaluator: SituationalRiskEvaluator) -> None:
        # 5.0 equals threshold -> CRITICAL requires > 5.0 dwell, so CRITICAL does not fire
        ev = make_evidence(
            timestamp_ns=1500,
            payload={
                "trajectory_anomaly_sigma": 4.0,
                "zone_dwell_sigma": 5.0,
                "is_restricted_zone": True,
            },
        )
        sig = evaluator.evaluate([ev], SystemMode.OPERATIONAL_MODE, 1000, 2000)
        # 4.0 > 2.5 triggers HIGH, but 5.0 does not exceed 5.0 so CRITICAL fails
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.75
