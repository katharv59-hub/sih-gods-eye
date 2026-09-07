"""Unit test suite for Phase 7.1 Situational Schemas."""

from __future__ import annotations

import dataclasses
import pytest

from gods_eye.schemas.environment import SystemMode as OperationalMode
from gods_eye.schemas.reasoning import QueryStatus
from gods_eye.schemas.situational import (
    Hypothesis,
    IdentityCandidate,
    RiskSignal,
    SituationalState,
)


class TestIdentityCandidate:
    def test_valid_identity_candidate(self) -> None:
        cand = IdentityCandidate(subject_ref="sub_123", confidence=0.85)
        assert cand.subject_ref == "sub_123"
        assert cand.confidence == 0.85
        assert cand.to_dict() == {"subject_ref": "sub_123", "confidence": 0.85}

    def test_invalid_confidence_rejection(self) -> None:
        with pytest.raises(ValueError, match="confidence MUST be between 0.0 and 1.0"):
            IdentityCandidate(subject_ref="sub_123", confidence=1.5)

        with pytest.raises(ValueError, match="confidence MUST be between 0.0 and 1.0"):
            IdentityCandidate(subject_ref="sub_123", confidence=-0.1)

    def test_empty_subject_ref_rejection(self) -> None:
        with pytest.raises(ValueError, match="subject_ref MUST be a non-empty string"):
            IdentityCandidate(subject_ref="", confidence=0.5)

        with pytest.raises(ValueError, match="subject_ref MUST be a non-empty string"):
            IdentityCandidate(subject_ref="   ", confidence=0.5)

    def test_frozen_mutation_rejection(self) -> None:
        cand = IdentityCandidate(subject_ref="sub_123", confidence=0.85)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cand.confidence = 0.90  # type: ignore[misc]


class TestHypothesis:
    def test_valid_hypothesis(self) -> None:
        c1 = IdentityCandidate(subject_ref="sub_1", confidence=0.9)
        c2 = IdentityCandidate(subject_ref="sub_2", confidence=0.1)
        hyp = Hypothesis(
            hypothesis_id="hyp_001",
            hypothesis_type="trajectory_deviation",
            primary_subject_ref="sub_1",
            identity_confidence=0.9,
            competing_candidates=(c1, c2),
            confidence=0.85,
            evidence_ids=("ev_1", "ev_2"),
            explanation="Subject trajectory deviates from north corridor cluster.",
        )
        assert hyp.hypothesis_id == "hyp_001"
        assert hyp.primary_subject_ref == "sub_1"
        assert len(hyp.competing_candidates) == 2
        assert hyp.evidence_ids == ("ev_1", "ev_2")

        d = hyp.to_dict()
        assert d["hypothesis_id"] == "hyp_001"
        assert d["competing_candidates"][0]["subject_ref"] == "sub_1"
        assert d["evidence_ids"] == ["ev_1", "ev_2"]

    def test_hypothesis_confidence_bounds(self) -> None:
        with pytest.raises(ValueError, match="confidence MUST be between 0.0 and 1.0"):
            Hypothesis(
                hypothesis_id="hyp_001",
                hypothesis_type="deviation",
                primary_subject_ref="sub_1",
                identity_confidence=0.9,
                competing_candidates=(),
                confidence=1.1,
                evidence_ids=("ev_1",),
                explanation="Valid explanation",
            )

    def test_hypothesis_identity_confidence_bounds(self) -> None:
        with pytest.raises(ValueError, match="identity_confidence MUST be between 0.0 and 1.0"):
            Hypothesis(
                hypothesis_id="hyp_001",
                hypothesis_type="deviation",
                primary_subject_ref="sub_1",
                identity_confidence=-0.2,
                competing_candidates=(),
                confidence=0.8,
                evidence_ids=("ev_1",),
                explanation="Valid explanation",
            )

    def test_competing_candidates_validation(self) -> None:
        with pytest.raises(ValueError, match="competing_candidates elements MUST be IdentityCandidate"):
            Hypothesis(
                hypothesis_id="hyp_001",
                hypothesis_type="deviation",
                primary_subject_ref="sub_1",
                identity_confidence=0.9,
                competing_candidates=("not_a_candidate",),  # type: ignore[arg-type]
                confidence=0.8,
                evidence_ids=("ev_1",),
                explanation="Valid explanation",
            )

    def test_immutable_collections_conversion(self) -> None:
        c1 = IdentityCandidate(subject_ref="sub_1", confidence=0.9)
        hyp = Hypothesis(
            hypothesis_id="hyp_001",
            hypothesis_type="deviation",
            primary_subject_ref="sub_1",
            identity_confidence=0.9,
            competing_candidates=[c1],  # type: ignore[arg-type]
            confidence=0.8,
            evidence_ids=["ev_1", "ev_2"],  # type: ignore[arg-type]
            explanation="Valid explanation",
        )
        assert isinstance(hyp.competing_candidates, tuple)
        assert isinstance(hyp.evidence_ids, tuple)

    def test_absence_of_raw_global_id_fields(self) -> None:
        hyp_fields = [f.name for f in dataclasses.fields(Hypothesis)]
        assert "global_id" not in hyp_fields
        assert "subject_global_id" not in hyp_fields
        assert "confidence_band" not in hyp_fields


class TestSituationalState:
    def test_valid_situational_state(self) -> None:
        state = SituationalState(
            state_id="state_100",
            window_start_ns=1000,
            window_end_ns=2000,
            active_identity_count=5,
            active_cameras=("cam_1", "cam_2"),
            system_mode=OperationalMode.OPERATIONAL_MODE,
            anomaly_ids=("anom_1",),
            overall_risk_level="MEDIUM",
            explanation="Multi-camera activity within normal ranges; one minor dwell anomaly.",
        )
        assert state.state_id == "state_100"
        assert state.active_identity_count == 5
        assert state.overall_risk_level == "MEDIUM"
        assert state.to_dict()["system_mode"] == "operational_mode"

    def test_invalid_temporal_window(self) -> None:
        with pytest.raises(ValueError, match="MUST be <= window_end_ns"):
            SituationalState(
                state_id="state_100",
                window_start_ns=3000,
                window_end_ns=2000,
                active_identity_count=5,
                active_cameras=("cam_1",),
                system_mode=OperationalMode.OPERATIONAL_MODE,
                anomaly_ids=(),
                overall_risk_level="LOW",
                explanation="Explanation",
            )

    def test_negative_active_identity_count(self) -> None:
        with pytest.raises(ValueError, match="active_identity_count MUST be a non-negative integer"):
            SituationalState(
                state_id="state_100",
                window_start_ns=1000,
                window_end_ns=2000,
                active_identity_count=-1,
                active_cameras=("cam_1",),
                system_mode=OperationalMode.OPERATIONAL_MODE,
                anomaly_ids=(),
                overall_risk_level="LOW",
                explanation="Explanation",
            )

    def test_invalid_risk_level(self) -> None:
        with pytest.raises(ValueError, match="overall_risk_level MUST be one of"):
            SituationalState(
                state_id="state_100",
                window_start_ns=1000,
                window_end_ns=2000,
                active_identity_count=5,
                active_cameras=("cam_1",),
                system_mode=OperationalMode.OPERATIONAL_MODE,
                anomaly_ids=(),
                overall_risk_level="EXTREME",
                explanation="Explanation",
            )

    def test_absence_of_overall_risk_score(self) -> None:
        state_fields = [f.name for f in dataclasses.fields(SituationalState)]
        assert "overall_risk_score" not in state_fields


class TestRiskSignal:
    def test_valid_risk_signal_unsuppressed(self) -> None:
        sig = RiskSignal(
            signal_id="sig_001",
            window_start_ns=1000,
            window_end_ns=2000,
            risk_level="HIGH",
            risk_score=0.88,
            evidence_ids=("ev_1", "ev_2"),
            contributing_factors=("trajectory_anomaly_3.2sigma", "restricted_zone_entry"),
            explanation="Unusual movement trajectory into high-security perimeter.",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.1.0",
        )
        assert sig.signal_id == "sig_001"
        assert sig.risk_level == "HIGH"
        assert sig.risk_score == 0.88
        assert not sig.suppressed
        assert sig.suppression_reason is None

    def test_valid_risk_signal_suppressed(self) -> None:
        sig = RiskSignal(
            signal_id="sig_002",
            window_start_ns=1000,
            window_end_ns=2000,
            risk_level="MEDIUM",
            risk_score=0.55,
            evidence_ids=("ev_1",),
            contributing_factors=("dwell_anomaly",),
            explanation="Dwell time exceeded threshold.",
            suppressed=True,
            suppression_reason="System in LEARNING_MODE",
            evaluator_version="v7.1.0",
        )
        assert sig.suppressed
        assert sig.suppression_reason == "System in LEARNING_MODE"

    def test_risk_score_bounds(self) -> None:
        with pytest.raises(ValueError, match="risk_score MUST be between 0.0 and 1.0"):
            RiskSignal(
                signal_id="sig_001",
                window_start_ns=1000,
                window_end_ns=2000,
                risk_level="HIGH",
                risk_score=1.5,
                evidence_ids=("ev_1",),
                contributing_factors=("factor",),
                explanation="Explanation",
                suppressed=False,
                suppression_reason=None,
                evaluator_version="v7.1.0",
            )

    def test_suppression_reason_required_when_suppressed(self) -> None:
        with pytest.raises(ValueError, match="suppression_reason MUST be a non-empty string when suppressed is True"):
            RiskSignal(
                signal_id="sig_001",
                window_start_ns=1000,
                window_end_ns=2000,
                risk_level="HIGH",
                risk_score=0.8,
                evidence_ids=("ev_1",),
                contributing_factors=("factor",),
                explanation="Explanation",
                suppressed=True,
                suppression_reason=None,
                evaluator_version="v7.1.0",
            )

    def test_unsuppressed_signal_rejects_suppression_reason(self) -> None:
        with pytest.raises(ValueError, match="suppression_reason MUST be None when suppressed is False"):
            RiskSignal(
                signal_id="sig_001",
                window_start_ns=1000,
                window_end_ns=2000,
                risk_level="HIGH",
                risk_score=0.8,
                evidence_ids=("ev_1",),
                contributing_factors=("factor",),
                explanation="Explanation",
                suppressed=False,
                suppression_reason="Invalid reason provided",
                evaluator_version="v7.1.0",
            )

    def test_immutable_collections(self) -> None:
        sig = RiskSignal(
            signal_id="sig_001",
            window_start_ns=1000,
            window_end_ns=2000,
            risk_level="HIGH",
            risk_score=0.8,
            evidence_ids=["ev_1"],  # type: ignore[arg-type]
            contributing_factors=["factor_1"],  # type: ignore[arg-type]
            explanation="Explanation",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.1.0",
        )
        assert isinstance(sig.evidence_ids, tuple)
        assert isinstance(sig.contributing_factors, tuple)

    def test_evaluator_version_required(self) -> None:
        with pytest.raises(ValueError, match="evaluator_version MUST be a non-empty string"):
            RiskSignal(
                signal_id="sig_001",
                window_start_ns=1000,
                window_end_ns=2000,
                risk_level="HIGH",
                risk_score=0.8,
                evidence_ids=("ev_1",),
                contributing_factors=("factor_1",),
                explanation="Explanation",
                suppressed=False,
                suppression_reason=None,
                evaluator_version="",
            )


class TestCrossContractInvariants:
    def test_deterministic_equality(self) -> None:
        c1 = IdentityCandidate(subject_ref="sub_1", confidence=0.9)
        c2 = IdentityCandidate(subject_ref="sub_1", confidence=0.9)
        assert c1 == c2

        hyp1 = Hypothesis(
            hypothesis_id="h1",
            hypothesis_type="type_a",
            primary_subject_ref="sub_1",
            identity_confidence=0.9,
            competing_candidates=(c1,),
            confidence=0.8,
            evidence_ids=("e1",),
            explanation="Exp",
        )
        hyp2 = Hypothesis(
            hypothesis_id="h1",
            hypothesis_type="type_a",
            primary_subject_ref="sub_1",
            identity_confidence=0.9,
            competing_candidates=(c2,),
            confidence=0.8,
            evidence_ids=("e1",),
            explanation="Exp",
        )
        assert hyp1 == hyp2

    def test_query_status_reuse(self) -> None:
        assert QueryStatus.SUCCESS.value == "success"
        assert QueryStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert QueryStatus.AMBIGUOUS.value == "ambiguous"

    def test_no_forbidden_privacy_or_confidence_fields(self) -> None:
        for cls in (IdentityCandidate, Hypothesis, SituationalState, RiskSignal):
            field_names = [f.name for f in dataclasses.fields(cls)]
            assert "global_id" not in field_names, f"{cls.__name__} contains global_id"
            assert "subject_global_id" not in field_names, f"{cls.__name__} contains subject_global_id"
            assert "confidence_band" not in field_names, f"{cls.__name__} contains confidence_band"
