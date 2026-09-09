"""Focused Unit & Integration Tests for Trajectory Anomaly & Learning Provenance — Fix #9B.

Tests all 21 required areas (§4.5 & §17 Master Spec):
1. Trajectory feature construction (TrajectoryFeature, to_vector)
2. Coordinate normalization
3. Fixed waypoint resampling
4. Mean speed calculation
5. Duration calculation
6. Learning sample collection
7. HDBSCAN fitting
8. Cluster member counts
9. Empirical centroid generation
10. Empirical std generation
11. Maturity threshold
12. Learning mode before maturity
13. No baseline -> no anomaly
14. Nearest mature cluster selection
15. Anomaly sigma calculation
16. Degenerate std handling
17. Canonical TRAJECTORY_ANOMALY Event
18. Evidence conversion via Evidence.from_event (zero extra_payload)
19. No hardcoded default cluster in production
20. No hardcoded std in production
21. Real video trajectory provenance (MOT17-04)
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from gods_eye.behavioral.anomaly import BehavioralAnomalyDetector
from gods_eye.behavioral.clustering import (
    TrajectoryClusteringEngine,
    compute_trajectory_feature_distance,
    fit_spatial_trajectory_clusters,
    hdbscan_cluster,
)
from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.schemas.behavioral import (
    SpatialAnomalyResult,
    SpatialPoint,
    SpatialTrajectory,
    SpatialTrajectoryCluster,
    TrajectoryFeature,
)
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import Evidence
from gods_eye.situational.evaluator import SituationalRiskEvaluator
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker
from gods_eye.zones.trajectory_engine import (
    TRAJECTORY_MIN_POINTS,
    TrajectoryAnomalyEngine,
)


def _make_bbox(cx: float, cy: float, w: float = 40.0, h: float = 80.0) -> BoundingBox:
    return BoundingBox(
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
    )


def _make_line_trajectory(
    track_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    n_pts: int = 15,
    duration_s: float = 1.5,
    camera_id: str = "cam_01",
) -> SpatialTrajectory:
    """Construct a synthetic mathematical test trajectory fixture for controlled geometry tests."""
    points: list[SpatialPoint] = []
    interval_ns = int(duration_s * 1e9 / max(1, n_pts - 1))
    for i in range(n_pts):
        alpha = float(i) / float(n_pts - 1)
        x = start[0] + alpha * (end[0] - start[0])
        y = start[1] + alpha * (end[1] - start[1])
        points.append(SpatialPoint(x=round(x, 4), y=round(y, 4), timestamp_ns=i * interval_ns))
    return SpatialTrajectory(track_id=track_id, camera_id=camera_id, points=points)


# ---------------------------------------------------------------------------
# 1. Trajectory Feature Construction
# ---------------------------------------------------------------------------
def test_trajectory_feature_construction() -> None:
    """Verify TrajectoryFeature stores waypoints, speed, duration, and vectorizes correctly."""
    waypoints = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
    feat = TrajectoryFeature(waypoints=waypoints, mean_speed=0.15, duration_s=2.0)

    assert feat.waypoints == waypoints
    assert feat.mean_speed == 0.15
    assert feat.duration_s == 2.0

    vec = feat.to_vector()
    # 3 waypoints * 2 coordinates + speed + duration = 8 dimensions
    assert len(vec) == 8
    assert vec[:6] == [0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
    assert vec[6] == 0.15
    assert vec[7] == 2.0


# ---------------------------------------------------------------------------
# 2. Coordinate Normalization
# ---------------------------------------------------------------------------
def test_trajectory_coordinate_normalization() -> None:
    """Verify pixel coordinates map accurately to [0.0, 1.0] and clamp out-of-bounds values."""
    engine = TrajectoryAnomalyEngine(min_points=5)
    resolution = (1000, 500)

    # Center point (500, 250) -> (0.5, 0.5)
    engine.record_track_point(
        track_id="t1",
        bbox=_make_bbox(cx=500.0, cy=250.0, w=20.0, h=40.0),
        camera_id="cam_01",
        timestamp_ns=100,
        source_resolution=resolution,
    )
    pt = engine._track_histories["t1"][0]
    assert pytest.approx(pt.x, rel=1e-3) == 0.5
    assert pytest.approx(pt.y, rel=1e-3) == 0.5

    # Negative coordinates clamped to 0.0
    engine.record_track_point(
        track_id="t2",
        bbox=BoundingBox(x1=-50.0, y1=-100.0, x2=-10.0, y2=-20.0),
        camera_id="cam_01",
        timestamp_ns=200,
        source_resolution=resolution,
    )
    pt_clamped = engine._track_histories["t2"][0]
    assert pt_clamped.x == 0.0
    assert pt_clamped.y == 0.0


# ---------------------------------------------------------------------------
# 3. Fixed Waypoint Resampling
# ---------------------------------------------------------------------------
def test_fixed_waypoint_resampling() -> None:
    """Verify trajectory resamples to exact N waypoints preserving cumulative distance."""
    traj = _make_line_trajectory("t_resample", start=(0.1, 0.1), end=(0.9, 0.9), n_pts=25)
    resampled_10 = traj.resample(n_points=10)
    assert len(resampled_10) == 10
    # First point matches start, last point matches end
    assert pytest.approx(resampled_10[0][0], abs=1e-3) == 0.1
    assert pytest.approx(resampled_10[0][1], abs=1e-3) == 0.1
    assert pytest.approx(resampled_10[-1][0], abs=1e-3) == 0.9
    assert pytest.approx(resampled_10[-1][1], abs=1e-3) == 0.9


# ---------------------------------------------------------------------------
# 4. Mean Speed Calculation
# ---------------------------------------------------------------------------
def test_trajectory_mean_speed() -> None:
    """Verify trajectory mean speed computes normalized path length divided by duration."""
    # Line from (0.0, 0.0) to (0.3, 0.4) length = 0.5, duration = 2.0s -> speed = 0.25
    traj = _make_line_trajectory("t_speed", start=(0.0, 0.0), end=(0.3, 0.4), n_pts=11, duration_s=2.0)
    speed = traj.compute_mean_speed()
    assert pytest.approx(speed, abs=1e-3) == 0.25


# ---------------------------------------------------------------------------
# 5. Duration Calculation
# ---------------------------------------------------------------------------
def test_trajectory_duration() -> None:
    """Verify trajectory duration computes exact timestamp delta in seconds."""
    pts = [
        SpatialPoint(x=0.1, y=0.1, timestamp_ns=1_000_000_000),
        SpatialPoint(x=0.2, y=0.2, timestamp_ns=2_500_000_000),
    ]
    traj = SpatialTrajectory(track_id="t_dur", camera_id="cam_01", points=pts)
    assert pytest.approx(traj.compute_duration_s(), abs=1e-3) == 1.5


# ---------------------------------------------------------------------------
# 6. Learning Sample Collection
# ---------------------------------------------------------------------------
def test_learning_sample_collection() -> None:
    """Verify eligible trajectories (>= min_points) are collected for learning."""
    engine = TrajectoryAnomalyEngine(min_points=10)

    # Ineligible track with 5 points
    short_traj = _make_line_trajectory("short", (0.1, 0.1), (0.2, 0.2), n_pts=5)
    assert engine.add_learning_sample(short_traj) is False

    # Eligible track with 12 points
    valid_traj = _make_line_trajectory("valid", (0.1, 0.1), (0.5, 0.5), n_pts=12)
    assert engine.add_learning_sample(valid_traj) is True
    assert len(engine.get_learning_samples()) == 1


# ---------------------------------------------------------------------------
# 7. HDBSCAN Fitting
# ---------------------------------------------------------------------------
def test_hdbscan_fitting() -> None:
    """Verify HDBSCAN discovers dense clusters and separates outliers as noise (-1)."""
    # 3 points close in cluster A, 3 points close in cluster B, 1 distant outlier
    dist_matrix = np.zeros((7, 7))
    for i in range(3):
        for j in range(3):
            dist_matrix[i, j] = 0.04 if i != j else 0.0
            dist_matrix[i + 3, j + 3] = 0.04 if i != j else 0.0
    for i in range(3):
        for j in range(3, 6):
            dist_matrix[i, j] = 4.0
            dist_matrix[j, i] = 4.0
    for i in range(6):
        dist_matrix[i, 6] = 25.0
        dist_matrix[6, i] = 25.0

    labels = hdbscan_cluster(dist_matrix, min_cluster_size=3, min_samples=2)
    assert len(labels) == 7
    # Points 0, 1, 2 have same non-negative label
    assert labels[0] == labels[1] == labels[2] and labels[0] >= 0
    # Points 3, 4, 5 have same non-negative label
    assert labels[3] == labels[4] == labels[5] and labels[3] >= 0
    # Clusters are distinct
    assert labels[0] != labels[3]
    # Outlier is noise (-1)
    assert labels[6] == -1


# ---------------------------------------------------------------------------
# 8. Cluster Member Counts
# ---------------------------------------------------------------------------
def test_cluster_member_counts() -> None:
    """Verify learned cluster member_count derives exactly from actual member trajectories."""
    # 4 parallel trajectories along corridor A
    trajs = [
        _make_line_trajectory(f"t_a_{i}", (0.20 + 0.01 * i, 0.1), (0.20 + 0.01 * i, 0.9), n_pts=10)
        for i in range(4)
    ]
    clusters = fit_spatial_trajectory_clusters(trajs, min_cluster_size=3, min_cluster_samples=10)
    assert len(clusters) == 1
    assert clusters[0].sample_count == 4


# ---------------------------------------------------------------------------
# 9. Empirical Centroid Generation
# ---------------------------------------------------------------------------
def test_empirical_centroid_generation() -> None:
    """Verify cluster centroid waypoints are the exact arithmetic mean of member waypoints."""
    t1 = _make_line_trajectory("t1", (0.2, 0.0), (0.2, 1.0), n_pts=10)
    t2 = _make_line_trajectory("t2", (0.4, 0.0), (0.4, 1.0), n_pts=10)
    t3 = _make_line_trajectory("t3", (0.3, 0.0), (0.3, 1.0), n_pts=10)

    clusters = fit_spatial_trajectory_clusters([t1, t2, t3], min_cluster_size=3, min_cluster_samples=3)
    assert len(clusters) == 1
    # Expected centroid x is (0.2 + 0.4 + 0.3) / 3 = 0.3
    for pt in clusters[0].centroid_waypoints:
        assert pytest.approx(pt[0], abs=1e-3) == 0.3


# ---------------------------------------------------------------------------
# 10. Empirical Std Generation
# ---------------------------------------------------------------------------
def test_empirical_std_generation() -> None:
    """Verify cluster std_dev is computed from member distance dispersion around centroid."""
    t1 = _make_line_trajectory("t1", (0.2, 0.0), (0.2, 1.0), n_pts=10)
    t2 = _make_line_trajectory("t2", (0.4, 0.0), (0.4, 1.0), n_pts=10)
    t3 = _make_line_trajectory("t3", (0.3, 0.0), (0.3, 1.0), n_pts=10)

    clusters = fit_spatial_trajectory_clusters([t1, t2, t3], min_cluster_size=3, min_cluster_samples=3)
    assert len(clusters) == 1
    std = clusters[0].std_dev
    assert std > 0.0
    # All tracks are at x=0.2, 0.3, 0.4 with centroid at 0.3; distance to centroid is 0.1, 0.0, 0.1
    # Standard deviation of [0.1, 0.0, 0.1] has mean 0.0667, sample variance is non-zero
    assert pytest.approx(std, abs=0.03) == 0.0577


# ---------------------------------------------------------------------------
# 11. Maturity Threshold
# ---------------------------------------------------------------------------
def test_maturity_threshold() -> None:
    """Verify cluster maturity is strictly determined by member_count >= min_cluster_samples."""
    trajs = [
        _make_line_trajectory(f"t_{i}", (0.5 + 0.005 * i, 0.1), (0.5 + 0.005 * i, 0.9), n_pts=10)
        for i in range(4)
    ]
    # min_cluster_samples = 5 (sample_count = 4 < 5 -> immature)
    immature_clusters = fit_spatial_trajectory_clusters(trajs, min_cluster_size=3, min_cluster_samples=5)
    assert len(immature_clusters) == 1
    assert immature_clusters[0].is_mature is False

    # min_cluster_samples = 4 (sample_count = 4 >= 4 -> mature)
    mature_clusters = fit_spatial_trajectory_clusters(trajs, min_cluster_size=3, min_cluster_samples=4)
    assert len(mature_clusters) == 1
    assert mature_clusters[0].is_mature is True


# ---------------------------------------------------------------------------
# 12. Learning Mode Before Maturity
# ---------------------------------------------------------------------------
def test_learning_mode_before_maturity() -> None:
    """Verify engine in LEARNING_MODE suppresses anomalies and emits sigma=0.0."""
    engine = TrajectoryAnomalyEngine(min_points=10, learning_mode=True)
    # Record 10 points for track
    for i in range(10):
        engine.record_track_point(
            track_id="t_learn",
            bbox=_make_bbox(cx=100.0, cy=100.0),
            camera_id="cam_01",
            timestamp_ns=1000 + i * 100,
            source_resolution=(1000, 1000),
        )

    res = engine.evaluate_track("t_learn")
    assert res is not None
    assert res.is_anomalous is False
    assert res.anomaly_sigma == 0.0
    assert "Learning mode active" in res.explanation


# ---------------------------------------------------------------------------
# 13. No Baseline -> No Anomaly
# ---------------------------------------------------------------------------
def test_no_baseline_no_anomaly() -> None:
    """Verify when no clusters exist, detector returns is_anomalous=False and sigma=0.0."""
    detector = BehavioralAnomalyDetector()
    traj = _make_line_trajectory("t_nobaseline", (0.1, 0.1), (0.9, 0.9), n_pts=10)
    res = detector.detect_spatial_anomaly(trajectory=traj, clusters=[], min_points=10)
    assert res.is_anomalous is False
    assert res.anomaly_sigma == 0.0
    assert res.nearest_cluster_id is None


# ---------------------------------------------------------------------------
# 14. Nearest Mature Cluster Selection
# ---------------------------------------------------------------------------
def test_nearest_mature_cluster_selection() -> None:
    """Verify trajectory matches nearest MATURE cluster, ignoring immature clusters."""
    # Immature cluster closer at (0.2, 0.2)
    immature = SpatialTrajectoryCluster(
        cluster_id="c_immature",
        centroid_waypoints=[(0.2, 0.2)] * 10,
        std_dev=0.05,
        sample_count=2,
        is_mature=False,
    )
    # Mature cluster farther at (0.5, 0.5)
    mature = SpatialTrajectoryCluster(
        cluster_id="c_mature",
        centroid_waypoints=[(0.5, 0.5)] * 10,
        std_dev=0.08,
        sample_count=25,
        is_mature=True,
    )

    detector = BehavioralAnomalyDetector()
    # Trajectory near (0.25, 0.25)
    traj = SpatialTrajectory(
        track_id="t_match",
        camera_id="cam_01",
        points=[SpatialPoint(x=0.25, y=0.25, timestamp_ns=i * 100) for i in range(10)],
    )

    res = detector.detect_spatial_anomaly(
        trajectory=traj,
        clusters=[immature, mature],
        min_points=10,
        allow_immature=False,
    )
    assert res.nearest_cluster_id == "c_mature"


# ---------------------------------------------------------------------------
# 15. Anomaly Sigma Calculation
# ---------------------------------------------------------------------------
def test_anomaly_sigma_calculation() -> None:
    """Verify exact formula: distance / cluster_std -> trajectory_anomaly_sigma."""
    cluster = SpatialTrajectoryCluster(
        cluster_id="c_fixed",
        centroid_waypoints=[(0.5, 0.5)] * 10,
        std_dev=0.10,
        sample_count=20,
        is_mature=True,
    )
    detector = BehavioralAnomalyDetector()

    # Trajectory at distance 0.35 from centroid
    traj = SpatialTrajectory(
        track_id="t_eval",
        camera_id="cam_01",
        points=[SpatialPoint(x=0.85, y=0.5, timestamp_ns=i * 100) for i in range(10)],
    )

    res = detector.detect_spatial_anomaly(
        trajectory=traj,
        clusters=[cluster],
        min_points=10,
        sigma_threshold=3.0,
    )
    # distance = 0.35, std = 0.10 -> sigma = 3.50
    assert pytest.approx(res.min_distance, abs=1e-3) == 0.35
    assert pytest.approx(res.anomaly_sigma, abs=1e-2) == 3.50
    assert res.is_anomalous is True


# ---------------------------------------------------------------------------
# 16. Degenerate Std Handling
# ---------------------------------------------------------------------------
def test_degenerate_std_handling() -> None:
    """Verify zero std does not raise ZeroDivisionError and clamps safely."""
    cluster = SpatialTrajectoryCluster(
        cluster_id="c_zero_std",
        centroid_waypoints=[(0.5, 0.5)] * 10,
        std_dev=0.0,
        sample_count=20,
        is_mature=True,
    )
    detector = BehavioralAnomalyDetector()
    traj = SpatialTrajectory(
        track_id="t_degen",
        camera_id="cam_01",
        points=[SpatialPoint(x=0.6, y=0.5, timestamp_ns=i * 100) for i in range(10)],
    )
    res = detector.detect_spatial_anomaly(trajectory=traj, clusters=[cluster], min_points=10)
    assert res is not None
    assert not math.isnan(res.anomaly_sigma)
    assert not math.isinf(res.anomaly_sigma)
    assert res.anomaly_sigma > 0.0


# ---------------------------------------------------------------------------
# 17. Canonical TRAJECTORY_ANOMALY Event
# ---------------------------------------------------------------------------
def test_canonical_trajectory_anomaly_event() -> None:
    """Verify TrajectoryAnomalyEngine emits canonical EventType.TRAJECTORY_ANOMALY."""
    engine = TrajectoryAnomalyEngine(min_points=10, learning_mode=False)
    cluster = SpatialTrajectoryCluster(
        cluster_id="learned_corridor_001",
        centroid_waypoints=[(0.5, 0.5)] * 10,
        std_dev=0.05,
        sample_count=25,
        is_mature=True,
    )
    engine.set_clusters([cluster])

    for i in range(12):
        engine.record_track_point(
            track_id="intruder_42",
            bbox=_make_bbox(cx=100.0, cy=100.0),
            camera_id="cam-perimeter",
            timestamp_ns=1_000_000_000 + i * 33_333_333,
            source_resolution=(1000, 1000),
        )

    res = engine.evaluate_track("intruder_42")
    assert res is not None
    event = engine.to_event(res, frame_id=12)

    assert event.event_type == EventType.TRAJECTORY_ANOMALY
    assert event.global_id == "intruder_42"
    assert event.camera_id == "cam-perimeter"
    assert event.frame_id == 12
    assert "trajectory_anomaly_sigma" in event.metadata
    assert event.metadata["trajectory_anomaly_sigma"] == res.anomaly_sigma


# ---------------------------------------------------------------------------
# 18. Evidence Conversion via Evidence.from_event
# ---------------------------------------------------------------------------
def test_evidence_conversion_without_extra_payload() -> None:
    """Verify Evidence.from_event consumes canonical trajectory event with zero extra_payload."""
    event = Event(
        event_id="ev_traj_test_99",
        event_type=EventType.TRAJECTORY_ANOMALY,
        global_id="subj_42",
        camera_id="cam-01",
        timestamp_ns=2_000_000_000,
        frame_id=30,
        confidence=0.88,
        explanation="Real trajectory anomaly",
        metadata={
            "trajectory_anomaly_sigma": 3.85,
            "min_cluster_distance": 0.385,
            "nearest_cluster_id": "learned_corridor_001",
            "is_anomalous": True,
            "confidence": 0.88,
        },
    )

    evidence = Evidence.from_event(event)
    assert evidence.record_type == "TRAJECTORY_ANOMALY"
    assert evidence.global_id == "subj_42"
    assert evidence.payload["trajectory_anomaly_sigma"] == 3.85
    assert evidence.payload["nearest_cluster_id"] == "learned_corridor_001"
    assert evidence.payload["is_anomalous"] is True


# ---------------------------------------------------------------------------
# 19. No Hardcoded Default Cluster in Production
# ---------------------------------------------------------------------------
def test_no_hardcoded_cluster_in_production() -> None:
    """Verify TrajectoryAnomalyEngine initializes with zero synthetic default clusters."""
    engine = TrajectoryAnomalyEngine()
    assert len(engine._clusters) == 0
    assert engine.learning_mode is True
    # Confirm no reference to 'normal_pedestrian_walkway'
    for c in engine._clusters:
        assert c.cluster_id != "normal_pedestrian_walkway"


# ---------------------------------------------------------------------------
# 20. No Hardcoded Std in Production
# ---------------------------------------------------------------------------
def test_no_hardcoded_std_in_production() -> None:
    """Verify learned cluster std_dev is dynamically derived from data, not static 0.08."""
    # Data with wider dispersion: 0.1 vs 0.3 -> larger std than tight data
    t1 = _make_line_trajectory("t1", (0.1, 0.0), (0.1, 1.0), n_pts=10)
    t2 = _make_line_trajectory("t2", (0.5, 0.0), (0.5, 1.0), n_pts=10)
    t3 = _make_line_trajectory("t3", (0.3, 0.0), (0.3, 1.0), n_pts=10)

    clusters = fit_spatial_trajectory_clusters([t1, t2, t3], min_cluster_size=3, min_cluster_samples=3)
    assert len(clusters) == 1
    # Empirical std of [0.1, 0.5, 0.3] around centroid 0.3 is ~0.1155, NOT hardcoded 0.08
    assert clusters[0].std_dev != 0.08
    assert pytest.approx(clusters[0].std_dev, abs=0.03) == 0.1155


# ---------------------------------------------------------------------------
# 21. Real Video Trajectory Provenance (MOT17-04)
# ---------------------------------------------------------------------------
def test_real_video_trajectory_provenance_mot17() -> None:
    """Verify real MOT17-04 video frames feed into TrajectoryAnomalyEngine, extract real track points,
    and demonstrate warm-up learning mode and unsupervised cluster fitting without synthetic baseline."""
    video_path = Path("tests/data/demo_videos/mot17_04_medium_density.mp4")
    if not video_path.exists():
        pytest.skip(f"Video file not present: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det = YOLODetector(Settings(detection_confidence_threshold=0.30))
    trk = ByteTrackTracker()
    engine = TrajectoryAnomalyEngine(min_points=10, min_cluster_samples=20)

    for f_idx in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        packet = FramePacket("cam-cctv-01", f_idx + 1, f_idx * 33_333_333, frame, (width, height))
        dets = det.detect(packet)
        tracks = trk.update(dets, f_idx + 1, "cam-cctv-01")
        for t in tracks:
            engine.record_track_point(
                track_id=t.track_id,
                bbox=t.bbox,
                camera_id="cam-cctv-01",
                timestamp_ns=f_idx * 33_333_333,
                source_resolution=(width, height),
            )
    cap.release()

    # In 30 frames (1 sec), 23 tracks are observed (below WARMUP_TRAJECTORY_COUNT=100 & MIN_CLUSTER_SAMPLES=20)
    # Verify the engine truthfully remains in LEARNING_MODE
    assert engine.learning_mode is True
    res_learn = engine.evaluate_track("1")
    assert res_learn is not None
    assert res_learn.is_anomalous is False
    assert res_learn.anomaly_sigma == 0.0
    assert "Learning mode active" in res_learn.explanation

    # Now fit unsupervised HDBSCAN clusters from the 23 real observed tracks
    learned_clusters = engine.fit_clusters(min_cluster_size=3, min_cluster_samples=3)
    assert len(learned_clusters) >= 1
    # Verify clusters have non-zero empirical member counts and valid centroid coordinates
    for c in learned_clusters:
        assert c.sample_count >= 3
        assert len(c.centroid_waypoints) == 10
        assert c.std_dev >= 0.0
        assert c.is_mature is True  # with min_cluster_samples=3

    # With mature learned clusters, evaluate Track 1
    engine.set_clusters(learned_clusters)
    res_evaluated = engine.evaluate_track("1")
    assert res_evaluated is not None
    assert res_evaluated.trajectory_points_count == 30
    assert res_evaluated.nearest_cluster_id is not None
    # Track 1 is a normal pedestrian walking in corridor x~0.22, so it matches its learned cluster
    assert res_evaluated.anomaly_sigma < 3.5
    assert res_evaluated.is_anomalous is False


# ---------------------------------------------------------------------------
# 22. End-to-End Situational Risk Chain with Zero Extra Payload
# ---------------------------------------------------------------------------
def test_e2e_situational_risk_chain_zero_extra_payload() -> None:
    """Verify that an end-to-end event chain with zero extra_payload fires CRITICAL risk rule."""
    ev_fence = Event(
        event_id="ev_fence_01",
        event_type=EventType.RESTRICTED_ZONE_INTRUSION,
        global_id="intruder_x",
        camera_id="cam-01",
        timestamp_ns=1_000_000_000,
        frame_id=1,
        confidence=0.90,
        explanation="Restricted perimeter entered",
        zone_id="zone_restricted",
        metadata={"is_restricted_zone": True, "zone_id": "zone_restricted", "confidence": 0.90},
    )
    ev_dwell = Event(
        event_id="ev_dwell_01",
        event_type=EventType.LOITERING_DETECTED,
        global_id="intruder_x",
        camera_id="cam-01",
        timestamp_ns=2_000_000_000,
        frame_id=30,
        confidence=0.90,
        explanation="Loitering detected in zone",
        zone_id="zone_restricted",
        metadata={"zone_dwell_sigma": 5.5, "dwell_seconds": 1.2, "confidence": 0.90},
    )
    ev_traj = Event(
        event_id="ev_traj_01",
        event_type=EventType.TRAJECTORY_ANOMALY,
        global_id="intruder_x",
        camera_id="cam-01",
        timestamp_ns=2_000_000_000,
        frame_id=30,
        confidence=0.90,
        explanation="Trajectory anomaly detected",
        metadata={"trajectory_anomaly_sigma": 4.1, "confidence": 0.90, "is_anomalous": True},
    )

    doc_fence = Evidence.from_event(ev_fence)
    doc_dwell = Evidence.from_event(ev_dwell)
    doc_traj = Evidence.from_event(ev_traj)

    evaluator = SituationalRiskEvaluator()
    risk_signal = evaluator.evaluate(
        evidence_items=[doc_fence, doc_dwell, doc_traj],
        system_mode=SystemMode.OPERATIONAL_MODE,
        window_start_ns=0,
        window_end_ns=3_000_000_000,
    )

    assert risk_signal.risk_level == "CRITICAL"
    assert "trajectory_anomaly_gt_3.5sigma" in risk_signal.contributing_factors
    assert "zone_dwell_gt_5.0sigma" in risk_signal.contributing_factors
    assert "restricted_zone_condition" in risk_signal.contributing_factors


# ---------------------------------------------------------------------------
# 22. HDBSCAN Algorithmic Compliance: Density-Based Cluster Discovery
# ---------------------------------------------------------------------------
def test_hdbscan_density_based_cluster_discovery() -> None:
    """Verify HDBSCAN discovers clusters of differing densities without global epsilon."""
    # Cluster A: dense (points 0, 1, 2 with distance 0.02)
    # Cluster B: sparse (points 3, 4, 5 with distance 0.5)
    # Point 6: isolated outlier
    pts = [
        [0.0, 0.0], [0.02, 0.0], [0.0, 0.02],
        [10.0, 10.0], [10.5, 10.0], [10.0, 10.5],
        [80.0, 80.0],
    ]
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    labels = hdbscan_cluster(dm, min_cluster_size=3)
    # Both dense cluster A and sparse cluster B must be discovered
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert labels[0] >= 0 and labels[3] >= 0
    assert labels[6] == -1  # Outlier is noise


# ---------------------------------------------------------------------------
# 23. HDBSCAN Algorithmic Compliance: Noise and Outlier Handling
# ---------------------------------------------------------------------------
def test_hdbscan_noise_outlier_handling() -> None:
    """Verify points failing core reachability or dropping before selection are labeled -1."""
    # 4 scattered points with mutual distance >= 20.0 (all noise for min_cluster_size=3)
    pts = [[0.0, 0.0], [25.0, 0.0], [0.0, 25.0], [50.0, 50.0]]
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    labels = hdbscan_cluster(dm, min_cluster_size=3, allow_single_cluster=False)
    assert all(label == -1 for label in labels)


# ---------------------------------------------------------------------------
# 24. HDBSCAN Algorithmic Compliance: Min Cluster Size Behavior
# ---------------------------------------------------------------------------
def test_hdbscan_min_cluster_size_behavior() -> None:
    """Verify subclusters smaller than min_cluster_size fall out as dropouts/noise."""
    # Cluster A (4 points), small group B (2 points < min_cluster_size=3)
    pts = [
        [0.0, 0.0], [0.02, 0.0], [0.0, 0.02], [0.02, 0.02],
        [50.0, 50.0], [50.02, 50.0],
    ]
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    labels = hdbscan_cluster(dm, min_cluster_size=3, allow_single_cluster=True)
    # Cluster A formed
    assert labels[0] == labels[1] == labels[2] == labels[3] == 0
    # Small group of 2 points falls out as noise because 2 < min_cluster_size=3
    assert labels[4] == -1 and labels[5] == -1


# ---------------------------------------------------------------------------
# 25. HDBSCAN Algorithmic Compliance: Deterministic Clustering
# ---------------------------------------------------------------------------
def test_hdbscan_deterministic_clustering() -> None:
    """Verify identical input distance matrices yield identical cluster labels across runs."""
    rng = np.random.default_rng(42)
    pts = np.vstack([
        rng.normal(loc=[0.2, 0.2], scale=0.02, size=(5, 2)),
        rng.normal(loc=[0.7, 0.7], scale=0.03, size=(5, 2)),
        rng.uniform(low=0.0, high=1.0, size=(3, 2)),
    ])
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    run1 = hdbscan_cluster(dm, min_cluster_size=4)
    run2 = hdbscan_cluster(dm, min_cluster_size=4)
    run3 = hdbscan_cluster(dm, min_cluster_size=4)
    assert run1 == run2 == run3


# ---------------------------------------------------------------------------
# 26. HDBSCAN Algorithmic Compliance: Cluster Stability and EOM Selection
# ---------------------------------------------------------------------------
def test_hdbscan_cluster_stability_eom_selection() -> None:
    """Verify Excess of Mass (EOM) selects parent cluster when its stability exceeds children."""
    pts = [
        [0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
        [0.25, 0.0], [0.35, 0.0], [0.45, 0.0],
        [50.0, 50.0],
    ]
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    labels = hdbscan_cluster(dm, min_cluster_size=3, allow_single_cluster=True)
    # Canonical HDBSCAN: points 1..4 dissolve at death_lambda=10.0; points 0, 5 drop out at 5.0, point 6 at 0.014
    assert labels[1] == labels[2] == labels[3] == labels[4] == 0
    assert labels[0] == -1
    assert labels[5] == -1
    assert labels[6] == -1


# ---------------------------------------------------------------------------
# 27. HDBSCAN Algorithmic Compliance: Disagreement with Naive Single-Linkage
# ---------------------------------------------------------------------------
def test_hdbscan_vs_naive_single_linkage_disagreement() -> None:
    """Verify HDBSCAN succeeds where any fixed-threshold single linkage pruning fails."""
    pts = [
        [0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.05, 0.05],
        [10.0, 10.0], [10.4, 10.0], [10.0, 10.4], [10.4, 10.4],
        [30.0, 30.0],
    ]
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    # HDBSCAN extracts both clusters
    hdb_labels = hdbscan_cluster(dm, min_cluster_size=3)
    assert hdb_labels[0] == hdb_labels[1] == hdb_labels[2] == hdb_labels[3] == 0
    assert hdb_labels[4] == hdb_labels[5] == hdb_labels[6] == hdb_labels[7] == 1
    assert hdb_labels[8] == -1

    # Naive single-linkage threshold pruning at t=0.2 shatters Cluster B (distances 0.40)
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    z_naive = sch.linkage(squareform(dm), method="single")
    flat_small = sch.fcluster(z_naive, t=0.2, criterion="distance")
    assert flat_small[4] != flat_small[5] or flat_small[5] != flat_small[6]


# ---------------------------------------------------------------------------
# 28. HDBSCAN Algorithmic Compliance: Canonical min_samples=3 Core Distance
# ---------------------------------------------------------------------------
def test_hdbscan_min_samples_explicit_third_nearest_including_self() -> None:
    """Verify explicit min_samples=3 uses the 3rd nearest point including self (index 2)."""
    # 6-point pairwise distance matrix:
    # Points 0, 1, 2 form a tight cluster with mutual distance 0.2.
    # Points 3, 4 are a separate pair at distance 0.8 from cluster {0, 1, 2}.
    # Point 5 is an outlier at distance 2.0.
    #
    # For point 0: sorted distances are [0.0 (self), 0.2 (pt1), 0.2 (pt2), 0.8 (pt3), 0.8 (pt4), 2.0 (pt5)].
    # - 1st nearest (index 0, self): 0.0
    # - 2nd nearest (index 1): 0.2
    # - 3rd nearest (index 2, canonical min_samples=3): 0.2
    # - 4th nearest (index 3, old buggy index k=min_samples=3): 0.8
    dm = np.array([
        [0.0, 0.2, 0.2, 0.8, 0.8, 2.0],
        [0.2, 0.0, 0.2, 0.8, 0.8, 2.0],
        [0.2, 0.2, 0.0, 0.8, 0.8, 2.0],
        [0.8, 0.8, 0.8, 0.0, 0.2, 2.0],
        [0.8, 0.8, 0.8, 0.2, 0.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
    ])

    sorted_dists = np.sort(dm, axis=1)
    k_canonical = 3 - 1  # 2
    k_buggy = 3

    assert pytest.approx(sorted_dists[0, k_canonical]) == 0.2
    assert pytest.approx(sorted_dists[0, k_buggy]) == 0.8

    # With canonical min_samples=3, cluster {0, 1, 2} forms at core distance 0.2.
    # Points 3, 4 (size 2 < min_cluster_size=3) and point 5 are labeled as noise (-1).
    labels = hdbscan_cluster(dm, min_cluster_size=3, min_samples=3, allow_single_cluster=True)
    assert labels == [0, 0, 0, -1, -1, -1]


# ---------------------------------------------------------------------------
# 29. HDBSCAN Algorithmic Compliance: Canonical min_samples=1 Semantics
# ---------------------------------------------------------------------------
def test_hdbscan_min_samples_one_behaves_correctly() -> None:
    """Verify explicit min_samples=1 sets core distances to 0.0 (index 0, self)."""
    dm = np.array([
        [0.0, 0.1, 0.1, 5.0, 5.0, 5.0],
        [0.1, 0.0, 0.1, 5.0, 5.0, 5.0],
        [0.1, 0.1, 0.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 0.0, 0.1, 0.1],
        [5.0, 5.0, 5.0, 0.1, 0.0, 0.1],
        [5.0, 5.0, 5.0, 0.1, 0.1, 0.0],
    ])
    # For min_samples=1, index is 1 - 1 = 0.
    # sorted_dists[:, 0] is distance to self (0.0).
    # Mutual reachability reduces to raw Euclidean distance max(0, 0, d) = d.
    labels = hdbscan_cluster(dm, min_cluster_size=3, min_samples=1)
    assert labels[0] == labels[1] == labels[2] == 0
    assert labels[3] == labels[4] == labels[5] == 1


# ---------------------------------------------------------------------------
# 30. HDBSCAN Algorithmic Compliance: min_samples Differs from min_cluster_size
# ---------------------------------------------------------------------------
def test_hdbscan_min_samples_differs_from_min_cluster_size() -> None:
    """Verify explicit min_samples != min_cluster_size controls core density independently."""
    # 4 points in a line at x = 0, 1, 2, 3 and one at x = 10
    pts = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    dm = np.abs(pts - pts.T)

    # min_cluster_size=4 with min_samples=2:
    # m=2 uses 2nd nearest (index 1), core distances are <= 1.0
    labels_dense = hdbscan_cluster(dm, min_cluster_size=4, min_samples=2, allow_single_cluster=True)
    assert labels_dense[0] == labels_dense[1] == labels_dense[2] == labels_dense[3] == 0
    assert labels_dense[4] == -1


# ---------------------------------------------------------------------------
# 31. HDBSCAN Algorithmic Compliance: Default min_samples Matches Canonical
# ---------------------------------------------------------------------------
def test_hdbscan_min_samples_default_preserves_canonical_behavior() -> None:
    """Verify min_samples=None produces identical results to min_samples=min_cluster_size."""
    pts = [
        [0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.05, 0.05],
        [10.0, 10.0], [10.05, 10.0], [10.0, 10.05], [10.05, 10.05],
        [50.0, 50.0],
    ]
    n = len(pts)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    labels_default = hdbscan_cluster(dm, min_cluster_size=4, min_samples=None)
    labels_explicit = hdbscan_cluster(dm, min_cluster_size=4, min_samples=4)
    assert labels_default == labels_explicit
    assert labels_default[0] == labels_default[1] == labels_default[2] == labels_default[3] == 0
    assert labels_default[4] == labels_default[5] == labels_default[6] == labels_default[7] == 1
    assert labels_default[8] == -1


# ---------------------------------------------------------------------------
# 32. HDBSCAN Algorithmic Compliance: Invalid min_samples Rejection
# ---------------------------------------------------------------------------
def test_hdbscan_min_samples_invalid_values_rejection() -> None:
    """Verify min_samples < 1 raises ValueError according to the API contract."""
    dm = np.zeros((5, 5))
    with pytest.raises(ValueError, match="min_samples must be >= 1"):
        hdbscan_cluster(dm, min_cluster_size=3, min_samples=0)

    with pytest.raises(ValueError, match="min_samples must be >= 1"):
        hdbscan_cluster(dm, min_cluster_size=3, min_samples=-3)


# ---------------------------------------------------------------------------
# 33. HDBSCAN Algorithmic Compliance: Canonical Root EOM Competition
# ---------------------------------------------------------------------------
def test_hdbscan_root_cluster_eom_competition_when_allowed() -> None:
    """Verify root cluster (cid 0) competes in EOM when allow_single_cluster=True.

    Dataset structure:
      - 6 points total with min_cluster_size=3, min_samples=1.
      - At lambda = 1.0 (delta = 1.0), root cluster splits into child 1 {0, 1, 2} and child 2 {3, 4, 5}.
      - Each child quickly dissolves at lambda = 1.25 (delta = 0.8).
      - Root stability: S(0) = 6 * (1.0 - 0.0) = 6.0.
      - Child 1 stability: S(1) = 3 * (1.25 - 1.0) = 0.75.
      - Child 2 stability: S(2) = 3 * (1.25 - 1.0) = 0.75.
      - Combined child stability: S(1) + S(2) = 1.50.
      - Comparison: S(0) = 6.0 > 1.50 = sum(S(children)).

    Canonical EOM behavior:
      - allow_single_cluster=True: root wins EOM, is selected, and deselects children -> [0, 0, 0, 0, 0, 0].
      - allow_single_cluster=False: single-cluster root disallowed -> children retained -> [0, 0, 0, 1, 1, 1].
    """
    dm = np.array([
        [0.0, 0.8, 0.8, 1.0, 1.0, 1.0],
        [0.8, 0.0, 0.8, 1.0, 1.0, 1.0],
        [0.8, 0.8, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.8, 0.8],
        [1.0, 1.0, 1.0, 0.8, 0.0, 0.8],
        [1.0, 1.0, 1.0, 0.8, 0.8, 0.0],
    ])

    # When allow_single_cluster=True, root stability (6.0) beats children (1.50)
    labels_single = hdbscan_cluster(dm, min_cluster_size=3, min_samples=1, allow_single_cluster=True)
    assert labels_single == [0, 0, 0, 0, 0, 0]

    # When allow_single_cluster=False, root cannot override children
    labels_multi = hdbscan_cluster(dm, min_cluster_size=3, min_samples=1, allow_single_cluster=False)
    assert labels_multi == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# 34. HDBSCAN Algorithmic Compliance: Root Loses to High-Stability Children
# ---------------------------------------------------------------------------
def test_hdbscan_root_cluster_loses_to_stable_children() -> None:
    """Verify root cluster loses EOM competition when children have higher stability.

    Dataset structure:
      - 6 points total with min_cluster_size=3, min_samples=1.
      - At lambda = 1.0, root splits into child 1 {0, 1, 2} and child 2 {3, 4, 5}.
      - Each child is very dense internally (delta = 0.1, lambda = 10.0).
      - Root stability: S(0) = 6 * (1.0 - 0.0) = 6.0.
      - Child 1 stability: S(1) = 3 * (10.0 - 1.0) = 27.0.
      - Child 2 stability: S(2) = 3 * (10.0 - 1.0) = 27.0.
      - Combined child stability: S(1) + S(2) = 54.0.
      - Comparison: S(0) = 6.0 < 54.0 = sum(S(children)).

    Canonical EOM behavior:
      - Even with allow_single_cluster=True, children win -> [0, 0, 0, 1, 1, 1].
    """
    dm = np.array([
        [0.0, 0.1, 0.1, 1.0, 1.0, 1.0],
        [0.1, 0.0, 0.1, 1.0, 1.0, 1.0],
        [0.1, 0.1, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.1, 0.1],
        [1.0, 1.0, 1.0, 0.1, 0.0, 0.1],
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.0],
    ])

    labels = hdbscan_cluster(dm, min_cluster_size=3, min_samples=1, allow_single_cluster=True)
    assert labels == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# 35. HDBSCAN Algorithmic Compliance: Canonical Dropout & Single-Cluster Labeling
# ---------------------------------------------------------------------------
def test_hdbscan_canonical_dropout_labeling_semantics() -> None:
    """Verify canonical HDBSCAN dropout labeling replaces arbitrary 0.5 * death_lambda cutoff.

    Specifically proves:
      1. A point in a selected subcluster whose dropout lambda falls below 0.5 * death_lambda
         is correctly assigned to the cluster (NOT rejected as noise).
      2. A point that dropped out of the root cluster at low density remains noise under
         allow_single_cluster=True.
      3. Under allow_single_cluster=False, the same unimodal dataset yields all noise.
      4. Label assignment is completely deterministic across repeated executions.
    """
    # 1. Subcluster member with dropout lambda < 0.5 * death_lambda
    # Cluster 0 core: [0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01] (dissolves at delta=0.01, lambda=100)
    # Point 4: [0.08, 0.0] (drops out at delta=0.08, lambda=12.5; 12.5 < 0.5 * 100 = 50.0)
    # Cluster 1: [10.0, 10.0], [10.01, 10.0], [10.0, 10.01], [10.01, 10.01]
    # Noise: [50.0, 50.0]
    pts_multi = [
        [0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01],
        [0.08, 0.0],
        [10.0, 10.0], [10.01, 10.0], [10.0, 10.01], [10.01, 10.01],
        [50.0, 50.0],
    ]
    n_multi = len(pts_multi)
    dm_multi = np.zeros((n_multi, n_multi))
    for i in range(n_multi):
        for j in range(n_multi):
            dm_multi[i, j] = math.hypot(pts_multi[i][0] - pts_multi[j][0], pts_multi[i][1] - pts_multi[j][1])

    labels_multi = hdbscan_cluster(dm_multi, min_cluster_size=4)
    # Points 0..4 belong to Cluster 0 (including point 4 with lam < 0.5 * death_lambda)
    assert labels_multi[0] == labels_multi[1] == labels_multi[2] == labels_multi[3] == 0
    assert labels_multi[4] == 0, f"Point 4 should belong to Cluster 0, got {labels_multi[4]}"
    # Points 5..8 belong to Cluster 1
    assert labels_multi[5] == labels_multi[6] == labels_multi[7] == labels_multi[8] == 1
    # Point 9 is gross outlier
    assert labels_multi[9] == -1

    # 2. Single-cluster root dropout: allow_single_cluster=True
    # Tight core: [0.0, 0.0], [0.01, 0.0], [0.0, 0.01] (dissolves at lambda=100)
    # Outlier: [10.0, 10.0] (drops out at lambda=0.0707 < 100)
    pts_single = [
        [0.0, 0.0], [0.01, 0.0], [0.0, 0.01],
        [10.0, 10.0],
    ]
    n_single = len(pts_single)
    dm_single = np.zeros((n_single, n_single))
    for i in range(n_single):
        for j in range(n_single):
            dm_single[i, j] = math.hypot(pts_single[i][0] - pts_single[j][0], pts_single[i][1] - pts_single[j][1])

    labels_single_true = hdbscan_cluster(dm_single, min_cluster_size=3, min_samples=1, allow_single_cluster=True)
    assert labels_single_true == [0, 0, 0, -1]

    # 3. Same unimodal dataset under allow_single_cluster=False yields all noise
    labels_single_false = hdbscan_cluster(dm_single, min_cluster_size=3, min_samples=1, allow_single_cluster=False)
    assert labels_single_false == [-1, -1, -1, -1]

    # 4. Determinism
    for _ in range(5):
        assert hdbscan_cluster(dm_multi, min_cluster_size=4) == labels_multi
        assert hdbscan_cluster(dm_single, min_cluster_size=3, min_samples=1, allow_single_cluster=True) == [0, 0, 0, -1]


# ---------------------------------------------------------------------------
# 36. HDBSCAN Algorithmic Compliance: Canonical Parent-on-Tie EOM Selection
# ---------------------------------------------------------------------------
def test_hdbscan_eom_tie_breaking_canonical_semantics() -> None:
    """Verify canonical HDBSCAN Excess of Mass (EOM) tie-breaking semantics.

    In canonical HDBSCAN (_hdbscan_tree.pyx / Campello et al., 2013), when parent
    cluster stability equals the sum of selected child cluster stabilities:
        S(parent) == sum(S(children))
    the parent wins the EOM competition (parent-on-tie).

    This test constructs mathematically exact ties and verifies:
      A. Non-root parent exact tie: parent wins over children under allow_single_cluster=False.
      B. Root parent exact tie:
         - allow_single_cluster=True: root wins the tie and selects all members.
         - allow_single_cluster=False: root remains excluded by contract, children win.
      C. Parent strictly greater than children: parent wins.
      D. Children strictly greater than parent: children win.
      E. Determinism: repeated executions produce identical label assignments.
    """
    # -----------------------------------------------------------------------
    # Part 1: Non-root cluster exact tie (P with 6 points ties with C1 and C2)
    # -----------------------------------------------------------------------
    # 9 points total:
    # Q = {6, 7, 8} (3 points, tight at d_q = 0.5, dissolves at lambda=2.0)
    # P = {0..5} (6 points):
    #   Root splits into Q and P at d_split1 = 2.0 (lambda_birth(P) = 0.5)
    #   P splits into C1 {0, 1, 2} and C2 {3, 4, 5} at d_split2 = 1.0 (lambda_split = 1.0)
    #   C1 and C2 each dissolve at d_death = 2/3 (lambda_death = 1.5)
    #
    # Stabilities:
    #   S(P)  = 6 * (lambda_split - lambda_birth) = 6 * (1.0 - 0.5) = 3.0
    #   S(C1) = 3 * (lambda_death - lambda_split) = 3 * (1.5 - 1.0) = 1.5
    #   S(C2) = 3 * (lambda_death - lambda_split) = 3 * (1.5 - 1.0) = 1.5
    #   sum(S(children)) = 1.5 + 1.5 = 3.0
    #
    # Exact tie: S(P) == sum(S(children)) == 3.0 (exact dyadic floating point math).
    # Canonical parent-on-tie: P wins and deselects C1, C2.
    # Resulting clusters: P {0..5} (label 0), Q {6..8} (label 1).
    d_death_tie = 2.0 / 3.0
    d_split2 = 1.0
    d_q = 0.5
    d_split1 = 2.0

    def _build_9pt_dm(d_death: float) -> np.ndarray:
        matrix = np.zeros((9, 9), dtype=float)
        for i in range(9):
            for j in range(9):
                if i == j:
                    matrix[i, j] = 0.0
                elif (i in [0, 1, 2] and j in [0, 1, 2]) or (i in [3, 4, 5] and j in [3, 4, 5]):
                    matrix[i, j] = d_death
                elif i in range(6) and j in range(6):
                    matrix[i, j] = d_split2
                elif i in [6, 7, 8] and j in [6, 7, 8]:
                    matrix[i, j] = d_q
                else:
                    matrix[i, j] = d_split1
        return matrix

    dm_tie = _build_9pt_dm(d_death_tie)

    # A. Exact tie: parent P wins under canonical >=
    labels_tie = hdbscan_cluster(dm_tie, min_cluster_size=3, min_samples=1, allow_single_cluster=False)
    assert labels_tie == [0, 0, 0, 0, 0, 0, 1, 1, 1], f"Expected parent P to win on tie, got {labels_tie}"

    # C. Parent strictly greater: d_death = 0.70 -> lambda_death = 1.4286 -> S(children) = 2.571 < S(P) = 3.0
    dm_parent_wins = _build_9pt_dm(0.70)
    labels_parent_wins = hdbscan_cluster(dm_parent_wins, min_cluster_size=3, min_samples=1, allow_single_cluster=False)
    assert labels_parent_wins == [0, 0, 0, 0, 0, 0, 1, 1, 1]

    # D. Children strictly greater: d_death = 0.65 -> lambda_death = 1.5385 -> S(children) = 3.231 > S(P) = 3.0
    dm_children_win = _build_9pt_dm(0.65)
    labels_children_win = hdbscan_cluster(dm_children_win, min_cluster_size=3, min_samples=1, allow_single_cluster=False)
    assert labels_children_win == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # -----------------------------------------------------------------------
    # Part 2: Root cluster exact tie (S(root) = 6.0 == S(C1) + S(C2) = 6.0)
    # -----------------------------------------------------------------------
    # 6 points:
    # Root splits at delta = 1.0 (lambda = 1.0) into C1 {0, 1, 2} and C2 {3, 4, 5}.
    # Each child dissolves at delta = 0.5 (lambda = 2.0).
    # S(root) = 6 * (1.0 - 0.0) = 6.0.
    # S(C1) = 3 * (2.0 - 1.0) = 3.0, S(C2) = 3 * (2.0 - 1.0) = 3.0.
    # sum(S(children)) = 3.0 + 3.0 = 6.0.
    dm_root_tie = np.array([
        [0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
        [0.5, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.5, 0.5],
        [1.0, 1.0, 1.0, 0.5, 0.0, 0.5],
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.0],
    ], dtype=float)

    # B1. allow_single_cluster=True: root wins tie against children
    labels_root_tie_true = hdbscan_cluster(dm_root_tie, min_cluster_size=3, min_samples=1, allow_single_cluster=True)
    assert labels_root_tie_true == [0, 0, 0, 0, 0, 0], f"Expected root to win on tie when allowed, got {labels_root_tie_true}"

    # B2. allow_single_cluster=False: root excluded by contract, children win
    labels_root_tie_false = hdbscan_cluster(dm_root_tie, min_cluster_size=3, min_samples=1, allow_single_cluster=False)
    assert labels_root_tie_false == [0, 0, 0, 1, 1, 1], f"Expected children when root disallowed, got {labels_root_tie_false}"

    # E. Determinism across 5 repeated runs
    for _ in range(5):
        assert hdbscan_cluster(dm_tie, min_cluster_size=3, min_samples=1, allow_single_cluster=False) == [0, 0, 0, 0, 0, 0, 1, 1, 1]
        assert hdbscan_cluster(dm_root_tie, min_cluster_size=3, min_samples=1, allow_single_cluster=True) == [0, 0, 0, 0, 0, 0]
        assert hdbscan_cluster(dm_root_tie, min_cluster_size=3, min_samples=1, allow_single_cluster=False) == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# 37. Speed Distance Sensitivity
# ---------------------------------------------------------------------------
def test_trajectory_speed_distance_sensitivity() -> None:
    """Verify that changing mean speed while holding spatial geometry constant changes distance."""
    # Two trajectories with identical horizontal path (0.1, 0.5) -> (0.9, 0.5)
    # t_fast: duration = 1.0s -> speed = 0.8 / 1.0 = 0.8 FOV/s
    # t_slow: duration = 2.0s -> speed = 0.8 / 2.0 = 0.4 FOV/s
    t_fast = _make_line_trajectory("fast", (0.1, 0.5), (0.9, 0.5), n_pts=15, duration_s=1.0)
    t_slow = _make_line_trajectory("slow", (0.1, 0.5), (0.9, 0.5), n_pts=15, duration_s=2.0)

    feat_fast = t_fast.extract_features(10)
    feat_slow = t_slow.extract_features(10)

    # Spatial waypoints are identical
    for p1, p2 in zip(feat_fast.waypoints, feat_slow.waypoints):
        assert pytest.approx(p1[0], abs=1e-4) == p2[0]
        assert pytest.approx(p1[1], abs=1e-4) == p2[1]

    # Speeds are materially different
    assert feat_fast.mean_speed > feat_slow.mean_speed

    # Production distance must be strictly positive (> 0.0)
    dist = compute_trajectory_feature_distance(feat_fast, feat_slow)
    assert dist > 0.0, f"Expected distance > 0 for differing speeds, got {dist}"

    # Self-distance must be zero
    assert compute_trajectory_feature_distance(feat_fast, feat_fast) == 0.0


# ---------------------------------------------------------------------------
# 38. Duration Distance Sensitivity
# ---------------------------------------------------------------------------
def test_trajectory_duration_distance_sensitivity() -> None:
    """Verify that duration materially affects trajectory distance."""
    # Identical spatial path, duration differs by 10x (2.0s vs 20.0s)
    t_short = _make_line_trajectory("short", (0.2, 0.3), (0.8, 0.7), n_pts=15, duration_s=2.0)
    t_long = _make_line_trajectory("long", (0.2, 0.3), (0.8, 0.7), n_pts=15, duration_s=20.0)

    feat_short = t_short.extract_features(10)
    feat_long = t_long.extract_features(10)

    dist = compute_trajectory_feature_distance(feat_short, feat_long)
    assert dist > 0.0, f"Expected distance > 0 for differing durations, got {dist}"

    # A larger duration divergence produces a strictly larger distance
    t_very_long = _make_line_trajectory("very_long", (0.2, 0.3), (0.8, 0.7), n_pts=15, duration_s=100.0)
    feat_very_long = t_very_long.extract_features(10)
    dist_larger = compute_trajectory_feature_distance(feat_short, feat_very_long)
    assert dist_larger > dist, f"Expected {dist_larger} > {dist}"


# ---------------------------------------------------------------------------
# 39. Speed and Duration Are Not Passive Fields
# ---------------------------------------------------------------------------
def test_trajectory_speed_duration_not_passive_fields() -> None:
    """Verify production clustering distinguishes trajectories when path is identical but speed/duration differ."""
    # 2 fast trajectories (1.0s) and 2 slow trajectories (10.0s) along exact same path
    t_f1 = _make_line_trajectory("f1", (0.0, 0.0), (1.0, 1.0), n_pts=15, duration_s=1.0)
    t_f2 = _make_line_trajectory("f2", (0.0, 0.0), (1.0, 1.0), n_pts=15, duration_s=1.0)
    t_s1 = _make_line_trajectory("s1", (0.0, 0.0), (1.0, 1.0), n_pts=15, duration_s=10.0)
    t_s2 = _make_line_trajectory("s2", (0.0, 0.0), (1.0, 1.0), n_pts=15, duration_s=10.0)

    feats = [t.extract_features(10) for t in [t_f1, t_f2, t_s1, t_s2]]

    # Within-group distance is 0
    assert compute_trajectory_feature_distance(feats[0], feats[1]) == 0.0
    assert compute_trajectory_feature_distance(feats[2], feats[3]) == 0.0

    # Across-group distance (fast vs slow) is strictly non-zero
    cross_dist = compute_trajectory_feature_distance(feats[0], feats[2])
    assert cross_dist > 0.05, f"Expected cross-group distance > 0.05, got {cross_dist}"


# ---------------------------------------------------------------------------
# 40. Large Duration Does Not Dominate Spatial Distance
# ---------------------------------------------------------------------------
def test_trajectory_large_duration_does_not_dominate_spatial() -> None:
    """Verify that a large duration (e.g. 300s upper bound) does not overwhelm spatial path divergence."""
    # Traj A & B: same path (0.1, 0.1) -> (0.9, 0.1), but B has upper bound duration = 300s vs A's 2s
    t_a = _make_line_trajectory("a", (0.1, 0.1), (0.9, 0.1), n_pts=15, duration_s=2.0)
    t_b = _make_line_trajectory("b", (0.1, 0.1), (0.9, 0.1), n_pts=15, duration_s=300.0)

    # Traj C: spatially distinct path (0.1, 0.6) -> (0.9, 0.6) separated by 0.5 in Y, same duration = 2s
    t_c = _make_line_trajectory("c", (0.1, 0.6), (0.9, 0.6), n_pts=15, duration_s=2.0)

    feat_a = t_a.extract_features(10)
    feat_b = t_b.extract_features(10)
    feat_c = t_c.extract_features(10)

    dist_temporal = compute_trajectory_feature_distance(feat_a, feat_b)
    dist_spatial = compute_trajectory_feature_distance(feat_a, feat_c)

    # Both distances must be well-bounded (<= 1.3)
    assert dist_temporal < 0.35, f"Duration difference should be bounded by ~0.30, got {dist_temporal}"
    assert dist_spatial > 0.30, f"Spatial difference of 0.5 should contribute ~0.35, got {dist_spatial}"

    # Significant spatial difference (0.5 FOV) maintains strong discrimination relative to duration
    assert dist_spatial > dist_temporal, f"Spatial distance {dist_spatial} should exceed duration distance {dist_temporal}"


# ---------------------------------------------------------------------------
# 41. Path Geometry Still Matters
# ---------------------------------------------------------------------------
def test_trajectory_path_still_matters() -> None:
    """Verify that spatially different trajectories with identical speed and duration remain distinguishable."""
    # Two parallel corridors 0.5 apart with identical 2.0s duration and identical speed
    t1 = _make_line_trajectory("corridor1", (0.1, 0.2), (0.9, 0.2), n_pts=15, duration_s=2.0)
    t2 = _make_line_trajectory("corridor2", (0.1, 0.7), (0.9, 0.7), n_pts=15, duration_s=2.0)

    f1 = t1.extract_features(10)
    f2 = t2.extract_features(10)

    # Speeds and durations are identical
    assert pytest.approx(f1.mean_speed, abs=1e-4) == f2.mean_speed
    assert pytest.approx(f1.duration_s, abs=1e-4) == f2.duration_s

    # Path distance alone must strongly separate them
    dist = compute_trajectory_feature_distance(f1, f2)
    # 0.70 * 0.5 = 0.35
    assert pytest.approx(dist, abs=1e-3) == 0.35
    assert dist > 0.30


# ---------------------------------------------------------------------------
# 42. Multi-Modal Feature Distance Determinism
# ---------------------------------------------------------------------------
def test_trajectory_multi_modal_distance_determinism() -> None:
    """Verify that feature distance calculation is 100% deterministic across repeated runs."""
    t1 = _make_line_trajectory("det1", (0.1, 0.2), (0.8, 0.6), n_pts=15, duration_s=3.5)
    t2 = _make_line_trajectory("det2", (0.2, 0.1), (0.7, 0.9), n_pts=15, duration_s=8.0)

    f1 = t1.extract_features(10)
    f2 = t2.extract_features(10)

    initial_dist = compute_trajectory_feature_distance(f1, f2)
    for _ in range(10):
        d = compute_trajectory_feature_distance(f1, f2)
        assert d == initial_dist


# ---------------------------------------------------------------------------
# 43. Multi-Corridor Fast / Slow Discrimination in Production Distance Matrix
# ---------------------------------------------------------------------------
def test_trajectory_fast_slow_same_corridor_discrimination() -> None:
    """Verify that the production clustering distance matrix cleanly reflects both temporal and spatial differences."""
    # 3 fast trajectories on corridor A (y=0.2, duration 1.0s)
    fast_corridor_a = [
        _make_line_trajectory(f"fa_{i}", (0.1, 0.2 + i * 1e-4), (0.9, 0.2 + i * 1e-4), n_pts=15, duration_s=1.0)
        for i in range(3)
    ]
    # 3 slow trajectories on corridor A (y=0.2, duration 10.0s)
    slow_corridor_a = [
        _make_line_trajectory(f"sa_{i}", (0.1, 0.2 + i * 1e-4), (0.9, 0.2 + i * 1e-4), n_pts=15, duration_s=10.0)
        for i in range(3)
    ]
    # 3 trajectories on corridor B (y=0.8, duration 1.0s)
    corridor_b = [
        _make_line_trajectory(f"b_{i}", (0.1, 0.8 + i * 1e-4), (0.9, 0.8 + i * 1e-4), n_pts=15, duration_s=1.0)
        for i in range(3)
    ]

    all_trajs = fast_corridor_a + slow_corridor_a + corridor_b
    features = [t.extract_features(10) for t in all_trajs]

    # Compute production pairwise distance matrix
    n = len(features)
    dm = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_trajectory_feature_distance(features[i], features[j])
            dm[i, j] = d
            dm[j, i] = d

    # 1. Within-group distances should be near 0
    for i in range(3):
        for j in range(3):
            assert dm[i, j] < 1e-3  # Fast A pair
            assert dm[i + 3, j + 3] < 1e-3  # Slow A pair
            assert dm[i + 6, j + 6] < 1e-3  # Corridor B pair

    # 2. Fast A vs Slow A (same spatial path, different temporal profile):
    # Distance must be significant (> 0.08)
    for i in range(3):
        for j in range(3):
            assert dm[i, j + 3] > 0.08, f"Expected temporal separation, got {dm[i, j + 3]}"

    # 3. Corridor A vs Corridor B (spatially separated by 0.6):
    # Distance must be large (> 0.40)
    for i in range(6):
        for j in range(3):
            assert dm[i, j + 6] > 0.40, f"Expected spatial separation, got {dm[i, j + 6]}"



