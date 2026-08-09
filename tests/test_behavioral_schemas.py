"""Unit tests for Phase 6.1 Canonical Behavioral Data Schemas."""

from __future__ import annotations

import pytest

from gods_eye.schemas.behavioral import (
    PredictedDestination,
    PredictionResult,
    TrajectoryCluster,
)


class TestTrajectoryClusterSchema:
    def test_valid_construction_and_dict(self):
        cluster = TrajectoryCluster(
            cluster_id="cluster_01",
            camera_sequence=["cam_01", "cam_02", "cam_03"],
            occurrence_count=15,
            unique_identity_count=5,
            mean_duration_s=120.5,
            confidence=0.92,
        )
        assert cluster.cluster_id == "cluster_01"
        assert cluster.camera_sequence == ["cam_01", "cam_02", "cam_03"]
        assert cluster.occurrence_count == 15
        assert cluster.unique_identity_count == 5
        assert cluster.mean_duration_s == 120.5
        assert cluster.confidence == 0.92

        d = cluster.to_dict()
        assert d["cluster_id"] == "cluster_01"
        assert d["camera_sequence"] == ["cam_01", "cam_02", "cam_03"]
        assert d["occurrence_count"] == 15
        assert d["unique_identity_count"] == 5
        assert d["mean_duration_s"] == 120.5
        assert d["confidence"] == 0.92

    def test_privacy_contract_no_raw_global_id_list(self):
        cluster = TrajectoryCluster(
            cluster_id="cluster_02",
            camera_sequence=["cam_01", "cam_02"],
            occurrence_count=3,
            unique_identity_count=2,
            mean_duration_s=45.0,
            confidence=0.8,
        )
        # Verify privacy invariant: aggregate counts, no global_ids field
        assert hasattr(cluster, "unique_identity_count")
        assert not hasattr(cluster, "global_ids")
        assert not hasattr(cluster, "sample_global_ids")

    def test_empty_cluster_id_rejection(self):
        with pytest.raises(ValueError, match="cluster_id MUST be a non-empty string"):
            TrajectoryCluster("", ["cam_01"], 1, 1, 10.0, 0.5)

    def test_empty_camera_sequence_rejection(self):
        with pytest.raises(ValueError, match="camera_sequence MUST be a non-empty list"):
            TrajectoryCluster("cluster_01", [], 1, 1, 10.0, 0.5)

    def test_invalid_camera_element_rejection(self):
        with pytest.raises(ValueError, match="camera_sequence elements MUST be non-empty strings"):
            TrajectoryCluster("cluster_01", ["cam_01", ""], 1, 1, 10.0, 0.5)

    def test_negative_occurrence_count_rejection(self):
        with pytest.raises(ValueError, match="occurrence_count MUST be non-negative"):
            TrajectoryCluster("cluster_01", ["cam_01"], -1, 1, 10.0, 0.5)

    def test_negative_unique_identity_count_rejection(self):
        with pytest.raises(ValueError, match="unique_identity_count MUST be non-negative"):
            TrajectoryCluster("cluster_01", ["cam_01"], 1, -1, 10.0, 0.5)

    def test_negative_mean_duration_rejection(self):
        with pytest.raises(ValueError, match="mean_duration_s MUST be non-negative"):
            TrajectoryCluster("cluster_01", ["cam_01"], 1, 1, -5.0, 0.5)

    def test_invalid_confidence_rejection(self):
        with pytest.raises(ValueError, match="confidence MUST be between 0.0 and 1.0"):
            TrajectoryCluster("cluster_01", ["cam_01"], 1, 1, 10.0, 1.5)


class TestPredictedDestinationSchema:
    def test_valid_construction_and_dict(self):
        dest = PredictedDestination(
            camera_id="cam_02",
            probability=0.75,
            confidence=0.88,
            confidence_band=(0.80, 0.95),
            expected_arrival_ns=1_700_000_000_000_000_000,
        )
        assert dest.camera_id == "cam_02"
        assert dest.probability == 0.75
        assert dest.confidence == 0.88
        assert dest.confidence_band == (0.80, 0.95)
        assert dest.expected_arrival_ns == 1_700_000_000_000_000_000

        d = dest.to_dict()
        assert d["camera_id"] == "cam_02"
        assert d["probability"] == 0.75
        assert d["confidence"] == 0.88
        assert d["confidence_band"] == [0.80, 0.95]
        assert d["expected_arrival_ns"] == 1_700_000_000_000_000_000

    def test_empty_camera_id_rejection(self):
        with pytest.raises(ValueError, match="camera_id MUST be a non-empty string"):
            PredictedDestination("", 0.5, 0.5, (0.4, 0.6), 1_700_000_000_000_000_000)

    def test_probability_boundary_rejection(self):
        with pytest.raises(ValueError, match="probability MUST be between 0.0 and 1.0"):
            PredictedDestination("cam_01", 1.2, 0.5, (0.4, 0.6), 1_700_000_000_000_000_000)

    def test_confidence_boundary_rejection(self):
        with pytest.raises(ValueError, match="confidence MUST be between 0.0 and 1.0"):
            PredictedDestination("cam_01", 0.5, -0.1, (0.4, 0.6), 1_700_000_000_000_000_000)

    def test_invalid_confidence_band_ordering(self):
        with pytest.raises(ValueError, match="confidence_band MUST satisfy"):
            PredictedDestination("cam_01", 0.5, 0.5, (0.8, 0.4), 1_700_000_000_000_000_000)

    def test_invalid_timestamp_rejection(self):
        with pytest.raises(ValueError, match="expected_arrival_ns MUST be a positive Unix nanosecond timestamp"):
            PredictedDestination("cam_01", 0.5, 0.5, (0.4, 0.6), 0)


class TestPredictionResultSchema:
    def test_valid_construction_and_dict(self):
        dest = PredictedDestination("cam_02", 0.8, 0.9, (0.85, 0.95), 1_700_000_000_000_000_000)
        res = PredictionResult(
            global_id="gid_123",
            current_camera_id="cam_01",
            timestamp_ns=1_700_000_000_000_000_000,
            predictions=[dest],
            overall_confidence=0.9,
            evidence_count=4,
            model_version="v1.0",
            explanation="Predicted transition to cam_02 based on historical Markov transition matrix.",
        )
        assert res.global_id == "gid_123"
        assert res.current_camera_id == "cam_01"
        assert len(res.predictions) == 1
        assert res.predictions[0].camera_id == "cam_02"
        assert res.overall_confidence == 0.9
        assert res.evidence_count == 4

        d = res.to_dict()
        assert d["global_id"] == "gid_123"
        assert d["current_camera_id"] == "cam_01"
        assert len(d["predictions"]) == 1
        assert d["predictions"][0]["camera_id"] == "cam_02"

    def test_invalid_global_id_rejection(self):
        with pytest.raises(ValueError, match="global_id MUST be a non-empty string"):
            PredictionResult("", "cam_01", 1_700_000_000_000_000_000)

    def test_invalid_current_camera_id_rejection(self):
        with pytest.raises(ValueError, match="current_camera_id MUST be a non-empty string"):
            PredictionResult("gid_123", "", 1_700_000_000_000_000_000)

    def test_invalid_timestamp_rejection(self):
        with pytest.raises(ValueError, match="timestamp_ns MUST be a positive Unix nanosecond timestamp"):
            PredictionResult("gid_123", "cam_01", -100)

    def test_invalid_overall_confidence_rejection(self):
        with pytest.raises(ValueError, match="overall_confidence MUST be between 0.0 and 1.0"):
            PredictionResult("gid_123", "cam_01", 1_700_000_000_000_000_000, overall_confidence=1.5)

    def test_negative_evidence_count_rejection(self):
        with pytest.raises(ValueError, match="evidence_count MUST be non-negative"):
            PredictionResult("gid_123", "cam_01", 1_700_000_000_000_000_000, evidence_count=-1)

    def test_invalid_model_version_rejection(self):
        with pytest.raises(ValueError, match="model_version MUST be a non-empty string"):
            PredictionResult("gid_123", "cam_01", 1_700_000_000_000_000_000, model_version="")


class TestCrossContractInvariants:
    def test_probability_and_confidence_are_distinct(self):
        # Probability and confidence can have completely different values
        dest = PredictedDestination("cam_03", probability=0.35, confidence=0.95, confidence_band=(0.90, 0.99), expected_arrival_ns=1_700_000_000_000_000_000)
        assert dest.probability != dest.confidence

    def test_no_forced_probability_normalization_in_schema(self):
        # Schema allows individual destination predictions without requiring full array sum to 1.0 at construction time
        d1 = PredictedDestination("cam_02", 0.4, 0.9, (0.8, 0.95), 1_700_000_000_000_000_000)
        d2 = PredictedDestination("cam_03", 0.4, 0.9, (0.8, 0.95), 1_700_000_000_000_000_000)
        res = PredictionResult("gid_123", "cam_01", 1_700_000_000_000_000_000, predictions=[d1, d2])
        assert len(res.predictions) == 2
