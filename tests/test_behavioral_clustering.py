"""Unit tests for Sub-Phase 6.3 Spatial Trajectory Clustering Engine."""

from __future__ import annotations

import pytest

from gods_eye.behavioral.clustering import TrajectoryClusteringEngine
from gods_eye.behavioral.extractor import TrajectorySequence
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.schemas.behavioral import TrajectoryCluster


class TestTrajectoryClusteringEngine:
    def test_invalid_parameters_rejection(self):
        with pytest.raises(ValueError, match="eps MUST be between 0.0 and 1.0"):
            TrajectoryClusteringEngine(eps=1.5, min_samples=3)
        with pytest.raises(ValueError, match="min_samples MUST be at least 1"):
            TrajectoryClusteringEngine(eps=0.3, min_samples=0)

    def test_empty_input(self):
        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=3)
        assert engine.cluster([]) == []

    def test_empty_trajectories_only(self):
        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=1)
        empty_seq = TrajectorySequence([], [], 0.0, 0, 0, {})
        assert engine.cluster([empty_seq]) == []

    def test_identical_trajectories_cluster(self):
        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=3)
        seq = TrajectorySequence(["cam_01", "cam_02", "cam_03"], [100, 200, 300], 10.0, 2, 3, {}, 0.95)
        inputs = [("gid_1", seq), ("gid_2", seq), ("gid_3", seq)]

        clusters = engine.cluster(inputs)
        assert len(clusters) == 1
        c = clusters[0]
        assert isinstance(c, TrajectoryCluster)
        assert c.camera_sequence == ["cam_01", "cam_02", "cam_03"]
        assert c.occurrence_count == 3
        assert c.unique_identity_count == 3
        assert c.mean_duration_s == 10.0
        assert c.confidence == 0.95

    def test_similar_trajectories_cluster_together(self):
        engine = TrajectoryClusteringEngine(eps=0.4, min_samples=3)
        # 3 trajectories with minor edit distance
        s1 = TrajectorySequence(["cam_01", "cam_02", "cam_03", "cam_04"], [1, 2, 3, 4], 20.0, 3, 4)
        s2 = TrajectorySequence(["cam_01", "cam_02", "cam_03", "cam_04"], [1, 2, 3, 4], 22.0, 3, 4)
        s3 = TrajectorySequence(["cam_01", "cam_02", "cam_05", "cam_04"], [1, 2, 3, 4], 18.0, 3, 4)

        clusters = engine.cluster([("id1", s1), ("id2", s2), ("id3", s3)])
        assert len(clusters) == 1
        c = clusters[0]
        assert c.occurrence_count == 3
        assert c.mean_duration_s == 20.0

    def test_different_trajectories_separate(self):
        engine = TrajectoryClusteringEngine(eps=0.2, min_samples=2)
        # Group A
        sa1 = TrajectorySequence(["cam_01", "cam_02"], [1, 2], 10.0, 1, 2)
        sa2 = TrajectorySequence(["cam_01", "cam_02"], [1, 2], 10.0, 1, 2)
        # Group B
        sb1 = TrajectorySequence(["cam_08", "cam_09", "cam_10"], [1, 2, 3], 30.0, 2, 3)
        sb2 = TrajectorySequence(["cam_08", "cam_09", "cam_10"], [1, 2, 3], 30.0, 2, 3)

        clusters = engine.cluster([("a1", sa1), ("a2", sa2), ("b1", sb1), ("b2", sb2)])
        assert len(clusters) == 2
        seqs = [c.camera_sequence for c in clusters]
        assert ["cam_01", "cam_02"] in seqs
        assert ["cam_08", "cam_09", "cam_10"] in seqs

    def test_noise_rejection(self):
        engine = TrajectoryClusteringEngine(eps=0.2, min_samples=3)
        # Only 2 trajectories, min_samples=3 => classified as noise
        s1 = TrajectorySequence(["cam_01", "cam_02"], [1, 2], 10.0, 1, 2)
        s2 = TrajectorySequence(["cam_01", "cam_02"], [1, 2], 10.0, 1, 2)

        clusters = engine.cluster([s1, s2])
        assert len(clusters) == 0

    def test_deterministic_repeated_execution(self):
        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=2)
        s1 = TrajectorySequence(["cam_01", "cam_02"], [1, 2], 10.0, 1, 2)
        s2 = TrajectorySequence(["cam_01", "cam_02"], [1, 2], 12.0, 1, 2)
        s3 = TrajectorySequence(["cam_05", "cam_06"], [1, 2], 20.0, 1, 2)
        s4 = TrajectorySequence(["cam_05", "cam_06"], [1, 2], 22.0, 1, 2)

        data = [("a", s1), ("b", s2), ("c", s3), ("d", s4)]
        res1 = engine.cluster(data)
        res2 = engine.cluster(reversed(data))

        assert len(res1) == len(res2)
        assert [c.cluster_id for c in res1] == [c.cluster_id for c in res2]
        assert [c.camera_sequence for c in res1] == [c.camera_sequence for c in res2]

    def test_medoid_representative_sequence_selection(self):
        engine = TrajectoryClusteringEngine(eps=0.4, min_samples=3)
        # medoid should be the sequence with minimum sum of distances to others
        s1 = TrajectorySequence(["cam_01", "cam_02", "cam_03"], [1, 2, 3], 10.0, 2, 3)
        s2 = TrajectorySequence(["cam_01", "cam_02", "cam_03"], [1, 2, 3], 10.0, 2, 3)
        s3 = TrajectorySequence(["cam_01", "cam_02", "cam_99"], [1, 2, 3], 10.0, 2, 3)

        clusters = engine.cluster([s1, s2, s3])
        assert len(clusters) == 1
        assert clusters[0].camera_sequence == ["cam_01", "cam_02", "cam_03"]

    def test_privacy_no_raw_global_ids(self):
        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=1)
        seq = TrajectorySequence(["cam_01"], [100], 5.0, 0, 1)
        clusters = engine.cluster([("secret_gid_123", seq)])

        assert len(clusters) == 1
        c = clusters[0]
        assert not hasattr(c, "global_ids")
        assert not hasattr(c, "secret_gid_123")
        assert "secret_gid_123" not in str(c.to_dict())

    def test_read_only_invariant(self):
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")

        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=1)
        seq = TrajectorySequence(["cam_01"], [100], 5.0, 0, 1)

        nodes_before = gs.count_nodes()
        edges_before = gs.count_edges()
        events_before = len(es.query_events())

        engine.cluster([seq])

        assert gs.count_nodes() == nodes_before
        assert gs.count_edges() == edges_before
        assert len(es.query_events()) == events_before

        es.close()
        gs.close()
