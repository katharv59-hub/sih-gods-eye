"""Unit tests for Sub-Phase 6.5 Markov Behavioral Predictor."""

from __future__ import annotations

import pytest

from gods_eye.behavioral.extractor import TrajectorySequence
from gods_eye.behavioral.predictor import MarkovBehavioralPredictor
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.schemas.behavioral import PredictedDestination, PredictionResult


class TestMarkovBehavioralPredictor:
    def setup_method(self):
        self.predictor = MarkovBehavioralPredictor(model_version="markov_test_v1")

    def test_empty_history_returns_insufficient_evidence(self):
        res = self.predictor.predict_next("cam_01")
        assert isinstance(res, PredictionResult)
        assert res.predictions == []
        assert res.overall_confidence == 0.0
        assert res.evidence_count == 0
        assert "Insufficient transition evidence" in res.explanation

    def test_invalid_parameters_rejection(self):
        with pytest.raises(ValueError, match="top_k MUST be positive"):
            self.predictor.predict_next("cam_01", top_k=0)
        with pytest.raises(ValueError, match="timestamp_ns MUST be a positive Unix nanosecond timestamp"):
            self.predictor.predict_next("cam_01", timestamp_ns=0)

    def test_basic_single_transition_prediction(self):
        seq = TrajectorySequence(["cam_01", "cam_02"], [1_000, 2_000], 1.0, 1, 2)
        self.predictor.fit_from_sequences([seq])

        res = self.predictor.predict_next("cam_01", timestamp_ns=100_000_000_000)
        assert len(res.predictions) == 1
        p = res.predictions[0]
        assert p.camera_id == "cam_02"
        assert p.probability == 1.0
        assert p.expected_arrival_ns == 100_000_000_000 + 1_000_000_000

    def test_multiple_destinations_probability_normalization(self):
        # 3 transitions from cam_01 -> cam_02, 1 transition from cam_01 -> cam_03
        transitions = [
            {"from_camera_id": "cam_01", "to_camera_id": "cam_02", "transition_duration_s": 2.0},
            {"from_camera_id": "cam_01", "to_camera_id": "cam_02", "transition_duration_s": 2.0},
            {"from_camera_id": "cam_01", "to_camera_id": "cam_02", "transition_duration_s": 2.0},
            {"from_camera_id": "cam_01", "to_camera_id": "cam_03", "transition_duration_s": 5.0},
        ]
        self.predictor.fit_from_transitions(transitions)

        matrix = self.predictor.get_transition_matrix()
        assert matrix["cam_01"]["cam_02"] == 0.75
        assert matrix["cam_01"]["cam_03"] == 0.25
        assert sum(matrix["cam_01"].values()) == pytest.approx(1.0)

        res = self.predictor.predict_next("cam_01", top_k=5)
        assert len(res.predictions) == 2
        assert res.predictions[0].camera_id == "cam_02"
        assert res.predictions[0].probability == 0.75
        assert res.predictions[1].camera_id == "cam_03"
        assert res.predictions[1].probability == 0.25

    def test_deterministic_candidate_ordering_and_tie_breaking(self):
        # Tied probabilities: cam_01 -> cam_03 (count 2), cam_01 -> cam_02 (count 2)
        transitions = [
            {"from_camera_id": "cam_01", "to_camera_id": "cam_03"},
            {"from_camera_id": "cam_01", "to_camera_id": "cam_03"},
            {"from_camera_id": "cam_01", "to_camera_id": "cam_02"},
            {"from_camera_id": "cam_01", "to_camera_id": "cam_02"},
        ]
        self.predictor.fit_from_transitions(transitions)

        res1 = self.predictor.predict_next("cam_01", top_k=2)
        res2 = self.predictor.predict_next("cam_01", top_k=2)

        # Tie-breaker sorts lexicographically by camera_id: cam_02 before cam_03
        assert [p.camera_id for p in res1.predictions] == ["cam_02", "cam_03"]
        assert res1 == res2

    def test_no_self_transition_double_counting(self):
        seq = TrajectorySequence(["cam_01", "cam_01", "cam_02"], [1_000, 2_000, 3_000], 2.0, 1, 2)
        self.predictor.fit_from_sequences([seq])

        matrix = self.predictor.get_transition_matrix()
        assert "cam_01" in matrix
        assert "cam_01" not in matrix["cam_01"]  # No self transition count
        assert matrix["cam_01"]["cam_02"] == 1.0

    def test_fit_from_graph_store_read_only(self):
        gs = SQLiteGraphStore(":memory:")
        gs.upsert_camera_node("cam_01")
        gs.upsert_camera_node("cam_02")
        gs.add_transition_edge("t1", "gid_1", "cam_01", "cam_02", 1_000, transition_duration_s=4.0)

        nodes_before = gs.count_nodes()
        edges_before = gs.count_edges()

        self.predictor.fit_from_graph_store(gs)
        res = self.predictor.predict_next("cam_01")

        assert len(res.predictions) == 1
        assert res.predictions[0].camera_id == "cam_02"

        # Read-only verification
        assert gs.count_nodes() == nodes_before
        assert gs.count_edges() == edges_before
        gs.close()

    def test_probability_and_confidence_semantic_separation(self):
        # High probability (1.0) but low confidence (only 1 historical transition sample)
        self.predictor.fit_from_transitions([
            {"from_camera_id": "cam_01", "to_camera_id": "cam_02"}
        ])

        res = self.predictor.predict_next("cam_01")
        p = res.predictions[0]
        assert p.probability == 1.0
        assert p.confidence == 0.2  # Scaled by sample count 1
        assert p.probability != p.confidence

    def test_read_only_event_store_and_graph_store_invariants(self):
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")

        events_before = len(es.query_events())
        edges_before = gs.count_edges()

        self.predictor.predict_next("cam_01")

        assert len(es.query_events()) == events_before
        assert gs.count_edges() == edges_before

        es.close()
        gs.close()
