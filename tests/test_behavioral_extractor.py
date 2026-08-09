"""Unit tests for Phase 6.2 Trajectory Sequence & Feature Extractor."""

from __future__ import annotations

import pytest

from gods_eye.behavioral.extractor import (
    TrajectorySequence,
    TrajectorySequenceExtractor,
    normalized_sequence_distance,
)
from gods_eye.memory.graph_store import SQLiteGraphStore


class TestNormalizedSequenceDistance:
    def test_identical_sequences_return_zero(self):
        seq_a = ["cam_01", "cam_02", "cam_03"]
        seq_b = ["cam_01", "cam_02", "cam_03"]
        assert normalized_sequence_distance(seq_a, seq_b) == 0.0

    def test_both_empty_sequences_return_zero(self):
        assert normalized_sequence_distance([], []) == 0.0

    def test_one_empty_sequence_returns_one(self):
        assert normalized_sequence_distance(["cam_01"], []) == 1.0
        assert normalized_sequence_distance([], ["cam_01", "cam_02"]) == 1.0

    def test_symmetry(self):
        seq_a = ["cam_01", "cam_02", "cam_03"]
        seq_b = ["cam_01", "cam_04", "cam_03"]
        d1 = normalized_sequence_distance(seq_a, seq_b)
        d2 = normalized_sequence_distance(seq_b, seq_a)
        assert d1 == d2

    def test_bounded_result_range(self):
        seq_a = ["cam_01", "cam_02"]
        seq_b = ["cam_03", "cam_04", "cam_05"]
        dist = normalized_sequence_distance(seq_a, seq_b)
        assert 0.0 <= dist <= 1.0

    def test_input_mutation_prevention(self):
        seq_a = ["cam_01", "cam_02"]
        seq_b = ["cam_03", "cam_04"]
        seq_a_copy = list(seq_a)
        seq_b_copy = list(seq_b)
        normalized_sequence_distance(seq_a, seq_b)
        assert seq_a == seq_a_copy
        assert seq_b == seq_b_copy


class TestTrajectorySequenceExtractor:
    def setup_method(self):
        self.extractor = TrajectorySequenceExtractor()

    def test_empty_observations(self):
        seq = self.extractor.extract_sequence_from_observations([])
        assert seq.camera_sequence == []
        assert seq.timestamps_ns == []
        assert seq.total_duration_s == 0.0
        assert seq.transition_count == 0
        assert seq.unique_camera_count == 0
        assert seq.dwell_durations_s == {}
        assert seq.confidence == 1.0

    def test_single_observation(self):
        obs = [
            {"edge_id": "e1", "camera_id": "cam_01", "timestamp_ns": 1_000_000_000, "confidence": 0.95}
        ]
        seq = self.extractor.extract_sequence_from_observations(obs)
        assert seq.camera_sequence == ["cam_01"]
        assert seq.timestamps_ns == [1_000_000_000]
        assert seq.total_duration_s == 0.0
        assert seq.transition_count == 0
        assert seq.unique_camera_count == 1

    def test_consecutive_same_camera_observations_collapsing(self):
        # Repeated frame-level presence observations at cam_01 must collapse into single transition node
        obs = [
            {"edge_id": "e1", "camera_id": "cam_01", "timestamp_ns": 1_000_000_000, "confidence": 1.0},
            {"edge_id": "e2", "camera_id": "cam_01", "timestamp_ns": 2_000_000_000, "confidence": 1.0},
            {"edge_id": "e3", "camera_id": "cam_01", "timestamp_ns": 3_000_000_000, "confidence": 1.0},
            {"edge_id": "e4", "camera_id": "cam_02", "timestamp_ns": 5_000_000_000, "confidence": 0.9},
        ]
        seq = self.extractor.extract_sequence_from_observations(obs)
        assert seq.camera_sequence == ["cam_01", "cam_02"]
        assert seq.timestamps_ns == [1_000_000_000, 5_000_000_000]
        assert seq.transition_count == 1
        assert seq.unique_camera_count == 2
        assert seq.total_duration_s == 4.0

    def test_deterministic_ordering_with_identical_timestamps(self):
        # Tie-breaking by (timestamp_ns, sequence_num, edge_id)
        obs = [
            {"edge_id": "e_z", "camera_id": "cam_02", "timestamp_ns": 1_000, "sequence_num": 2},
            {"edge_id": "e_a", "camera_id": "cam_01", "timestamp_ns": 1_000, "sequence_num": 1},
        ]
        seq1 = self.extractor.extract_sequence_from_observations(obs)
        seq2 = self.extractor.extract_sequence_from_observations(reversed(obs))
        assert seq1.camera_sequence == ["cam_01", "cam_02"]
        assert seq1 == seq2

    def test_extract_sequence_from_transitions(self):
        trans = [
            {"edge_id": "t1", "from_camera_id": "cam_01", "to_camera_id": "cam_02", "timestamp_ns": 1_000, "transition_duration_s": 2.5, "confidence": 0.95},
            {"edge_id": "t2", "from_camera_id": "cam_02", "to_camera_id": "cam_03", "timestamp_ns": 2_000, "transition_duration_s": 3.0, "confidence": 0.90},
        ]
        seq = self.extractor.extract_sequence_from_transitions(trans)
        assert seq.camera_sequence == ["cam_01", "cam_02", "cam_03"]
        assert seq.transition_count == 2
        assert seq.unique_camera_count == 3
        assert seq.total_duration_s == 5.5
        assert seq.confidence == 0.925

    def test_read_only_invariant_against_graph_store(self):
        gs = SQLiteGraphStore(":memory:")
        gs.upsert_identity_node("gid_01", 1_000, 5_000, "cam_01")
        gs.add_observation_edge("obs_1", "gid_01", "cam_01", 1_000)
        gs.add_observation_edge("obs_2", "gid_01", "cam_02", 3_000)

        nodes_before = gs.count_nodes()
        edges_before = gs.count_edges()

        seq = self.extractor.extract_identity_trajectory(gs, "gid_01")
        assert seq.camera_sequence == ["cam_01", "cam_02"]

        # Read-only invariant check
        assert gs.count_nodes() == nodes_before
        assert gs.count_edges() == edges_before
        gs.close()

    def test_privacy_no_public_global_id_list(self):
        seq = self.extractor.extract_sequence_from_observations([
            {"edge_id": "e1", "camera_id": "cam_01", "timestamp_ns": 1_000}
        ])
        assert not hasattr(seq, "global_id")
        assert not hasattr(seq, "global_ids")
        d = seq.to_dict()
        assert "global_id" not in d
        assert "global_ids" not in d
