"""Unit tests for Sub-Phase 6.4 Behavioral Anomaly Detector."""

from __future__ import annotations

import pytest

from gods_eye.behavioral.anomaly import BehavioralAnomalyDetector
from gods_eye.behavioral.extractor import TrajectorySequence
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.schemas.behavioral import BehavioralAnomalyResult, TrajectoryCluster
from gods_eye.schemas.reasoning import QueryStatus


class TestBehavioralAnomalyDetector:
    def setup_method(self):
        self.detector = BehavioralAnomalyDetector(threshold=0.5)

    def test_invalid_threshold_rejection(self):
        with pytest.raises(ValueError, match="threshold MUST be between 0.0 and 1.0"):
            BehavioralAnomalyDetector(threshold=1.5)
        with pytest.raises(ValueError, match="threshold MUST be between 0.0 and 1.0"):
            BehavioralAnomalyDetector(threshold=-0.1)

    def test_insufficient_evidence_empty_clusters(self):
        traj = TrajectorySequence(["cam_01", "cam_02"], [100, 200], 10.0, 1, 2)
        res = self.detector.detect(traj, [])

        assert res.status == QueryStatus.INSUFFICIENT_EVIDENCE
        assert res.is_anomalous is False
        assert res.anomaly_score == 0.0
        assert res.nearest_cluster_id is None
        assert res.confidence == 0.0

    def test_insufficient_evidence_empty_trajectory(self):
        traj = TrajectorySequence([], [], 0.0, 0, 0)
        cluster = TrajectoryCluster("c1", ["cam_01"], 5, 2, 10.0, 0.9)
        res = self.detector.detect(traj, [cluster])

        assert res.status == QueryStatus.INSUFFICIENT_EVIDENCE
        assert res.is_anomalous is False

    def test_clearly_normal_trajectory(self):
        cluster = TrajectoryCluster("c1", ["cam_01", "cam_02", "cam_03"], 10, 3, 15.0, 0.95)
        traj = TrajectorySequence(["cam_01", "cam_02", "cam_03"], [100, 200, 300], 15.0, 2, 3, {}, 0.95)

        res = self.detector.detect(traj, [cluster])

        assert res.status == QueryStatus.SUCCESS
        assert res.is_anomalous is False
        assert res.anomaly_score == 0.0
        assert res.nearest_cluster_id == "c1"
        assert res.confidence == 0.95

    def test_clearly_divergent_trajectory(self):
        cluster = TrajectoryCluster("c1", ["cam_01", "cam_02", "cam_03"], 10, 3, 15.0, 0.9)
        traj = TrajectorySequence(["cam_99", "cam_98", "cam_97"], [100, 200, 300], 15.0, 2, 3)

        res = self.detector.detect(traj, [cluster])

        assert res.status == QueryStatus.SUCCESS
        assert res.is_anomalous is True
        assert res.anomaly_score == 1.0  # Completely disjoint edit distance
        assert res.nearest_cluster_id == "c1"

    def test_threshold_override(self):
        cluster = TrajectoryCluster("c1", ["cam_01", "cam_02", "cam_03", "cam_04"], 10, 3, 15.0, 0.9)
        traj = TrajectorySequence(["cam_01", "cam_02", "cam_03", "cam_99"], [100, 200, 300, 400], 15.0, 3, 4)
        # Edit distance = 1/4 = 0.25

        res_strict = self.detector.detect(traj, [cluster], threshold_override=0.20)
        assert res_strict.is_anomalous is True

        res_lenient = self.detector.detect(traj, [cluster], threshold_override=0.30)
        assert res_lenient.is_anomalous is False

    def test_deterministic_repeated_execution(self):
        c1 = TrajectoryCluster("c1", ["cam_01", "cam_02"], 5, 2, 10.0, 0.9)
        c2 = TrajectoryCluster("c2", ["cam_05", "cam_06"], 5, 2, 10.0, 0.9)
        traj = TrajectorySequence(["cam_01", "cam_02"], [100, 200], 10.0, 1, 2)

        res1 = self.detector.detect(traj, [c1, c2])
        res2 = self.detector.detect(traj, [c2, c1])

        assert res1.anomaly_score == res2.anomaly_score
        assert res1.nearest_cluster_id == res2.nearest_cluster_id
        assert res1.is_anomalous == res2.is_anomalous

    def test_privacy_no_raw_global_ids(self):
        cluster = TrajectoryCluster("c1", ["cam_01"], 5, 2, 10.0, 0.9)
        traj = TrajectorySequence(["cam_01"], [100], 5.0, 0, 1)

        res = self.detector.detect(traj, [cluster])
        d = res.to_dict()

        assert "global_id" not in d
        assert "global_ids" not in d

    def test_read_only_invariants(self):
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")

        cluster = TrajectoryCluster("c1", ["cam_01"], 5, 2, 10.0, 0.9)
        traj = TrajectorySequence(["cam_01"], [100], 5.0, 0, 1)

        nodes_before = gs.count_nodes()
        edges_before = gs.count_edges()
        events_before = len(es.query_events())

        self.detector.detect(traj, [cluster])

        assert gs.count_nodes() == nodes_before
        assert gs.count_edges() == edges_before
        assert len(es.query_events()) == events_before

        es.close()
        gs.close()
