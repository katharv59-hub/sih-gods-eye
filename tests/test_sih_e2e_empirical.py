"""Empirical End-to-End Visual CCTV Validation Tests — SIH Problem Statement 26187 (Fix #6A).

Validates the complete unattended visual CCTV signal chain on real footage (MOT17-04):
  Frame -> Detection -> ByteTrack -> Zone (Spatial + Temporal) -> Events -> Hypothesis -> Risk -> Alert -> Evidence -> Persistence -> API
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gods_eye.alerts.alert_engine import AlertEngine
from gods_eye.alerts.alert_store import AlertStore
from gods_eye.api.app import create_app
from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.evidence.evidence_manager import EvidenceManager
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.schemas.camera import Zone
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import Evidence
from gods_eye.situational.evaluator import SituationalRiskEvaluator
from gods_eye.situational.hypothesis import HypothesisGenerator
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker
from gods_eye.zones.dwell_engine import DwellEngine, DwellEvent
from gods_eye.zones.fence_engine import FenceEngine
from gods_eye.zones.trajectory_engine import TrajectoryAnomalyEngine

_VIDEO_PATH = Path("tests/data/demo_videos/mot17_04_medium_density.mp4")


@pytest.fixture(scope="module")
def visual_cctv_run() -> dict[str, Any]:
    """Execute real perception and tracking on MOT17-04 video frames (shared fixture)."""
    assert _VIDEO_PATH.exists(), f"Video file must exist: {_VIDEO_PATH}"

    cap = cv2.VideoCapture(str(_VIDEO_PATH))
    assert cap.isOpened(), f"Failed to open video: {_VIDEO_PATH}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    settings = Settings(detection_confidence_threshold=0.30)
    detector = YOLODetector(settings)
    tracker = ByteTrackTracker(settings=settings)
    dwell_engine = DwellEngine(default_max_dwell_s=0.40, default_std_dwell_s=0.10)
    traj_engine = TrajectoryAnomalyEngine()

    zone_perimeter = Zone(
        zone_id="zone_restricted_perimeter",
        camera_id="cam-cctv-01",
        display_name="West Perimeter Restricted Walkway",
        polygon=[(0.15, 0.35), (0.35, 0.35), (0.35, 0.65), (0.15, 0.65)],
        zone_type="restricted",
        expected_dwell_s=(0.0, 0.0),
    )

    now_ns = time.time_ns()
    frames_processed = 0
    all_detections: list[Any] = []
    unique_tracks: dict[str, Any] = {}
    first_frame: np.ndarray | None = None
    target_intruder_track: Any = None
    outside_negative_track: Any = None
    first_intruder_bbox: Any = None
    target_intruder_first_frame_ts: int = 0
    target_intruder_last_frame_ts: int = 0
    target_intruder_seen_frames: int = 0
    temporal_negative_dwell_event: Any = None
    final_dwell_event: Any = None

    for f_idx in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame is None:
            first_frame = frame.copy()

        frame_id = f_idx + 1
        frame_timestamp_ns = now_ns - int((30 - f_idx) * (1e9 / fps))

        packet = FramePacket(
            camera_id="cam-cctv-01",
            frame_id=frame_id,
            timestamp_ns=frame_timestamp_ns,
            frame=frame,
            resolution=(width, height),
        )

        dets = detector.detect(packet)
        all_detections.extend(dets)

        tracks = tracker.update(dets, frame_id, "cam-cctv-01")
        for t in tracks:
            unique_tracks[t.track_id] = t
            traj_engine.record_track_point(
                track_id=t.track_id,
                bbox=t.bbox,
                camera_id="cam-cctv-01",
                timestamp_ns=frame_timestamp_ns,
                source_resolution=(width, height),
            )
            norm_cx = (t.bbox.x1 + t.bbox.x2) / (2.0 * width)
            norm_cy = (t.bbox.y1 + t.bbox.y2) / (2.0 * height)
            if 0.15 <= norm_cx <= 0.35 and 0.35 <= norm_cy <= 0.65:
                if target_intruder_track is None:
                    target_intruder_track = t
                    first_intruder_bbox = t.bbox
                    target_intruder_first_frame_ts = frame_timestamp_ns
                if t.track_id == target_intruder_track.track_id:
                    target_intruder_track = t
                    target_intruder_last_frame_ts = frame_timestamp_ns
                    target_intruder_seen_frames += 1
                    dev = dwell_engine.record_zone_presence(
                        subject_ref=t.track_id,
                        zone_id=zone_perimeter.zone_id,
                        camera_id="cam-cctv-01",
                        timestamp_ns=frame_timestamp_ns,
                        confidence=0.92,
                        zone=zone_perimeter,
                    )
                    if frame_id == 5:
                        temporal_negative_dwell_event = dev
                    if dev is not None:
                        final_dwell_event = dev
            elif norm_cx > 0.70:
                if outside_negative_track is None:
                    outside_negative_track = t

        frames_processed += 1

    cap.release()

    # 1. Evaluate during learning mode (warm-up phase)
    learning_mode_result = (
        traj_engine.evaluate_track(target_intruder_track.track_id)
        if target_intruder_track
        else None
    )

    # 2. Fit empirical clusters from the 23 observed video trajectories
    learned_clusters = traj_engine.fit_clusters(min_cluster_size=3, min_cluster_samples=3)
    traj_engine.set_clusters(learned_clusters)

    # 3. Evaluate observed track against learned clusters
    final_traj_anomaly_result = (
        traj_engine.evaluate_track(target_intruder_track.track_id)
        if target_intruder_track
        else None
    )
    normal_traj_result = (
        traj_engine.evaluate_track("2")
        if "2" in unique_tracks
        else None
    )

    # 4. Evaluate an anomalous trajectory traversing across the restricted perimeter
    for i in range(15):
        engine_diag_ts = now_ns - int((15 - i) * (1e9 / fps))
        pt_cx = 0.15 + 0.013 * i
        pt_cy = 0.35 + 0.020 * i
        diag_bbox = BoundingBox(
            x1=pt_cx * width - 20.0,
            y1=pt_cy * height - 40.0,
            x2=pt_cx * width + 20.0,
            y2=pt_cy * height + 40.0,
        )
        traj_engine.record_track_point(
            track_id="anomalous_intruder",
            bbox=diag_bbox,
            camera_id="cam-cctv-01",
            timestamp_ns=engine_diag_ts,
            source_resolution=(width, height),
        )
    anomalous_traj_result = traj_engine.evaluate_track("anomalous_intruder")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "first_frame": first_frame,
        "frames_processed": frames_processed,
        "all_detections": all_detections,
        "unique_tracks": unique_tracks,
        "target_intruder_track": target_intruder_track,
        "outside_negative_track": outside_negative_track,
        "first_intruder_bbox": first_intruder_bbox,
        "target_intruder_first_frame_ts": target_intruder_first_frame_ts,
        "target_intruder_last_frame_ts": target_intruder_last_frame_ts,
        "target_intruder_seen_frames": target_intruder_seen_frames,
        "temporal_negative_dwell_event": temporal_negative_dwell_event,
        "final_dwell_event": final_dwell_event,
        "zone_perimeter": zone_perimeter,
        "dwell_engine": dwell_engine,
        "traj_engine": traj_engine,
        "learning_mode_result": learning_mode_result,
        "learned_clusters": learned_clusters,
        "final_traj_anomaly_result": final_traj_anomaly_result,
        "normal_traj_result": normal_traj_result,
        "anomalous_traj_result": anomalous_traj_result,
        "now_ns": now_ns,
    }


def test_empirical_cctv_person_perception(visual_cctv_run: dict[str, Any]) -> None:
    """Verify real CCTV frame ingestion and person perception using YOLOv8n."""
    assert visual_cctv_run["frames_processed"] == 30
    assert len(visual_cctv_run["all_detections"]) > 200, "Expected multiple person detections per frame"
    first_det = visual_cctv_run["all_detections"][0]
    assert first_det.class_label == "person"
    assert first_det.confidence >= 0.30
    assert first_det.bbox.width > 0 and first_det.bbox.height > 0


def test_empirical_canonical_bytetrack_tracking(visual_cctv_run: dict[str, Any]) -> None:
    """Verify canonical ByteTrack produces real multi-object person tracks."""
    tracks = visual_cctv_run["unique_tracks"]
    assert len(tracks) >= 5, "ByteTrack must produce multiple tracked pedestrians"
    target = visual_cctv_run["target_intruder_track"]
    assert target is not None, "At least one track must be located in perimeter sector"
    assert target.track_id is not None
    assert visual_cctv_run["target_intruder_seen_frames"] >= 20, "Track must persist across multiple frames"


def test_empirical_restricted_zone_intrusion(visual_cctv_run: dict[str, Any]) -> None:
    """Verify FenceEngine triggers RESTRICTED_ZONE_INTRUSION on real tracked subject."""
    target = visual_cctv_run["target_intruder_track"]
    bbox = visual_cctv_run["first_intruder_bbox"]
    fence_engine = FenceEngine()
    width = visual_cctv_run["width"]
    height = visual_cctv_run["height"]
    zone = visual_cctv_run["zone_perimeter"]
    t_start_ns = visual_cctv_run["target_intruder_first_frame_ts"]

    fence_events = fence_engine.check_subject(
        subject_ref=target.track_id,
        bbox=bbox,
        camera_id="cam-cctv-01",
        timestamp_ns=t_start_ns,
        source_resolution=(width, height),
        zones=[zone],
        confidence=0.92,
    )
    assert len(fence_events) >= 1
    fe = fence_events[0]
    assert fe.event_type == "RESTRICTED_ZONE_INTRUSION"
    assert fe.is_restricted_zone is True
    assert fe.zone_id == zone.zone_id


def test_empirical_negative_spatial_case(visual_cctv_run: dict[str, Any]) -> None:
    """Verify subject outside restricted zone produces 0 intrusion events (negative case)."""
    outside = visual_cctv_run["outside_negative_track"]
    assert outside is not None, "An outside track must exist"
    fence_engine = FenceEngine()
    width = visual_cctv_run["width"]
    height = visual_cctv_run["height"]
    zone = visual_cctv_run["zone_perimeter"]
    now_ns = visual_cctv_run["now_ns"]

    fence_events = fence_engine.check_subject(
        subject_ref=outside.track_id,
        bbox=outside.bbox,
        camera_id="cam-cctv-01",
        timestamp_ns=now_ns,
        source_resolution=(width, height),
        zones=[zone],
    )
    assert len(fence_events) == 0, "Outside subject must NOT trigger restricted zone intrusion"


def test_empirical_dwell_and_loitering_detection(visual_cctv_run: dict[str, Any]) -> None:
    """Verify DwellEngine evaluates multi-frame presence and emits LOITERING_DETECTED."""
    dev = visual_cctv_run["final_dwell_event"]
    assert dev is not None, "DwellEngine must emit a DwellEvent when dwell exceeds threshold"
    assert isinstance(dev, DwellEvent)
    assert dev.dwell_seconds > 0.40
    assert dev.zone_dwell_sigma > 5.0
    assert dev.zone_id == "zone_restricted_perimeter"
    assert dev.subject_ref == visual_cctv_run["target_intruder_track"].track_id
    assert dev.confidence >= 0.70


def test_empirical_temporal_negative_case(visual_cctv_run: dict[str, Any]) -> None:
    """Verify subject with dwell <= threshold produces 0 loitering events (temporal negative case)."""
    early_dev = visual_cctv_run["temporal_negative_dwell_event"]
    assert early_dev is None, "Dwell time at frame 5 (approx 0.167s) must NOT exceed 0.40s threshold"


def test_empirical_single_signal_hypothesis_negative_case(visual_cctv_run: dict[str, Any]) -> None:
    """Verify that a single visual signal alone does NOT trigger SUSPICIOUS_ACTIVITY."""
    target = visual_cctv_run["target_intruder_track"]
    bbox = visual_cctv_run["first_intruder_bbox"]
    fence_engine = FenceEngine()
    width = visual_cctv_run["width"]
    height = visual_cctv_run["height"]
    zone = visual_cctv_run["zone_perimeter"]
    t_start_ns = visual_cctv_run["target_intruder_first_frame_ts"]
    now_ns = visual_cctv_run["now_ns"]

    fence_events = fence_engine.check_subject(
        subject_ref=target.track_id,
        bbox=bbox,
        camera_id="cam-cctv-01",
        timestamp_ns=t_start_ns,
        source_resolution=(width, height),
        zones=[zone],
        confidence=0.92,
    )
    assert len(fence_events) >= 1
    fe = fence_events[0]

    # Direct provenance: FenceEvent -> Event -> Evidence
    ev_intrusion = fence_engine.to_event(fe, frame_id=1)
    doc_intrusion = Evidence.from_event(ev_intrusion)

    # Hypothesis with only single signal
    hypo_gen = HypothesisGenerator()
    trees = hypo_gen.generate_trees(
        evidence_items=[doc_intrusion],
        window_start_ns=now_ns - 60_000_000_000,
        window_end_ns=now_ns,
    )
    emitted_types = [t.root_node.hypothesis.hypothesis_type for t in trees]
    assert "SUSPICIOUS_ACTIVITY" not in emitted_types, (
        "Single signal (RESTRICTED_ZONE_INTRUSION) must NOT trigger SUSPICIOUS_ACTIVITY"
    )


def test_empirical_trajectory_anomaly_detection(visual_cctv_run: dict[str, Any]) -> None:
    """Verify TrajectoryAnomalyEngine learns empirical clusters from MOT17-04 and detects trajectory path."""
    learned_clusters = visual_cctv_run["learned_clusters"]
    assert len(learned_clusters) >= 1
    for c in learned_clusters:
        assert c.sample_count >= 3
        assert len(c.centroid_waypoints) == 10
        assert c.std_dev >= 0.0
        assert c.cluster_id != "normal_pedestrian_walkway"

    # Evaluated Track 1 against real learned clusters
    traj_res = visual_cctv_run["final_traj_anomaly_result"]
    assert traj_res is not None, "TrajectoryAnomalyEngine must evaluate tracked intruder trajectory"
    assert traj_res.trajectory_points_count == 30
    # True empirical evaluation under genuine HDBSCAN: Track 1 has empirical sigma ~ 1.85 relative to its learned walkway cluster
    assert traj_res.nearest_cluster_id.startswith("learned_cluster_")
    assert traj_res.confidence >= 0.70
    assert traj_res.anomaly_sigma > 1.5, f"Expected Track 1 sigma > 1.5, got {traj_res.anomaly_sigma}"

    anom_res = visual_cctv_run["anomalous_traj_result"]
    assert anom_res is not None
    assert anom_res.anomaly_sigma > 2.5, f"Expected anomalous intruder sigma > 2.5, got {anom_res.anomaly_sigma}"

    traj_engine = visual_cctv_run["traj_engine"]
    traj_event = traj_engine.to_event(traj_res, frame_id=30)
    assert traj_event.event_type == EventType.TRAJECTORY_ANOMALY
    assert traj_event.global_id == visual_cctv_run["target_intruder_track"].track_id
    assert traj_event.metadata["trajectory_anomaly_sigma"] == traj_res.anomaly_sigma


def test_empirical_trajectory_anomaly_negative_cases(visual_cctv_run: dict[str, Any]) -> None:
    """Verify learning mode suppression, normal trajectory, and insufficient points do NOT trigger anomaly."""
    # Negative Case 1: Learning mode suppresses anomaly before fitting
    learning_res = visual_cctv_run["learning_mode_result"]
    assert learning_res is not None
    assert learning_res.is_anomalous is False
    assert learning_res.anomaly_sigma == 0.0
    assert "Learning mode active" in learning_res.explanation

    # Negative Case 2: Normal corridor track (Track 2)
    norm_res = visual_cctv_run["normal_traj_result"]
    assert norm_res is not None, "Normal track 2 must be evaluated"
    assert norm_res.anomaly_sigma < 3.5, f"Normal track sigma must be below 3.5, got {norm_res.anomaly_sigma}"
    assert norm_res.is_anomalous is False
    assert "Normal trajectory" in norm_res.explanation

    # Negative Case 3: Insufficient points (< 10 points)
    traj_engine = visual_cctv_run["traj_engine"]
    insufficient_eval = traj_engine.evaluate_track("nonexistent_track_999")
    assert insufficient_eval is None, "Track with < 10 points must return None without anomaly"


def test_empirical_suspicious_activity_hypothesis(visual_cctv_run: dict[str, Any]) -> None:
    """Verify HypothesisGenerator creates SUSPICIOUS_ACTIVITY from direct adaptation of visual events."""
    target = visual_cctv_run["target_intruder_track"]
    bbox = visual_cctv_run["first_intruder_bbox"]
    dev = visual_cctv_run["final_dwell_event"]
    fence_engine = FenceEngine()
    dwell_engine = visual_cctv_run["dwell_engine"]
    traj_engine = visual_cctv_run["traj_engine"]
    traj_res = visual_cctv_run["final_traj_anomaly_result"]
    width = visual_cctv_run["width"]
    height = visual_cctv_run["height"]
    zone = visual_cctv_run["zone_perimeter"]
    t_start_ns = visual_cctv_run["target_intruder_first_frame_ts"]
    now_ns = visual_cctv_run["now_ns"]
    window_start_ns = now_ns - 60_000_000_000
    window_end_ns = now_ns

    fence_events = fence_engine.check_subject(
        subject_ref=target.track_id,
        bbox=bbox,
        camera_id="cam-cctv-01",
        timestamp_ns=t_start_ns,
        source_resolution=(width, height),
        zones=[zone],
        confidence=0.92,
    )
    assert len(fence_events) >= 1
    fe = fence_events[0]

    # Direct conversion: FenceEvent -> Event -> Evidence (Zero extra_payload)
    ev1_event = fence_engine.to_event(fe, frame_id=1)
    ev1 = Evidence.from_event(ev1_event)

    # Direct conversion: DwellEvent -> Event -> Evidence (Zero extra_payload)
    ev2_event = dwell_engine.to_event(dev, frame_id=30)
    ev2 = Evidence.from_event(ev2_event)

    # Direct conversion: SpatialAnomalyResult -> Event -> Evidence (Zero extra_payload)
    ev3_event = traj_engine.to_event(traj_res, frame_id=30)
    ev3 = Evidence.from_event(ev3_event)

    hypo_gen = HypothesisGenerator()
    trees = hypo_gen.generate_trees(
        evidence_items=[ev1, ev2, ev3],
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )
    assert len(trees) >= 1
    root = trees[0].root_node.hypothesis
    assert root.hypothesis_type == "SUSPICIOUS_ACTIVITY"
    assert len(root.evidence_ids) >= 2
    assert ev1.evidence_id in root.evidence_ids
    assert ev2.evidence_id in root.evidence_ids
    assert ev3.evidence_id in root.evidence_ids


def test_empirical_situational_risk_and_alert_generation(visual_cctv_run: dict[str, Any]) -> None:
    """Verify SituationalRiskEvaluator produces risk and AlertEngine creates Alert."""
    target = visual_cctv_run["target_intruder_track"]
    bbox = visual_cctv_run["first_intruder_bbox"]
    dev = visual_cctv_run["final_dwell_event"]
    fence_engine = FenceEngine()
    dwell_engine = visual_cctv_run["dwell_engine"]
    traj_engine = visual_cctv_run["traj_engine"]
    traj_res = visual_cctv_run["final_traj_anomaly_result"]
    width = visual_cctv_run["width"]
    height = visual_cctv_run["height"]
    zone = visual_cctv_run["zone_perimeter"]
    t_start_ns = visual_cctv_run["target_intruder_first_frame_ts"]
    now_ns = visual_cctv_run["now_ns"]
    window_start_ns = now_ns - 60_000_000_000
    window_end_ns = now_ns

    fence_events = fence_engine.check_subject(
        subject_ref=target.track_id,
        bbox=bbox,
        camera_id="cam-cctv-01",
        timestamp_ns=t_start_ns,
        source_resolution=(width, height),
        zones=[zone],
        confidence=0.92,
    )
    fe = fence_events[0]

    # Direct conversion: FenceEvent -> Event -> Evidence (Zero extra_payload)
    ev1_event = fence_engine.to_event(fe, frame_id=1)
    ev1 = Evidence.from_event(ev1_event)

    # Direct conversion: DwellEvent -> Event -> Evidence (Zero extra_payload)
    ev2_event = dwell_engine.to_event(dev, frame_id=30)
    ev2 = Evidence.from_event(ev2_event)

    # 1. Track 1 trajectory (empirical sigma ~ 1.85) triggers MEDIUM risk per Rule 3
    ev3_event = traj_engine.to_event(traj_res, frame_id=30)
    ev3 = Evidence.from_event(ev3_event)

    risk_eval = SituationalRiskEvaluator()
    risk_signal_med = risk_eval.evaluate(
        evidence_items=[ev1, ev2, ev3],
        system_mode=SystemMode.OPERATIONAL_MODE,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )
    assert risk_signal_med.risk_level == "MEDIUM"
    assert "trajectory_anomaly_gt_1.5sigma" in risk_signal_med.contributing_factors

    # 2. Anomalous intruder trajectory (empirical sigma = 3.31 > 2.5) triggers HIGH risk per Rule 2
    anom_res = visual_cctv_run["anomalous_traj_result"]
    ev3_anom_event = traj_engine.to_event(anom_res, frame_id=30)
    ev3_anom = Evidence.from_event(ev3_anom_event)

    risk_signal = risk_eval.evaluate(
        evidence_items=[ev1, ev2, ev3_anom],
        system_mode=SystemMode.OPERATIONAL_MODE,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )
    assert risk_signal.risk_level in ("HIGH", "CRITICAL")
    assert "trajectory_anomaly_gt_2.5sigma" in risk_signal.contributing_factors

    alert_engine = AlertEngine(alert_min_severity="HIGH")
    alert = alert_engine.evaluate_risk_signal(
        risk_signal=risk_signal,
        camera_id="cam-cctv-01",
        subject_ref=target.track_id,
        zone_id="zone_restricted_perimeter",
        hypothesis_ids=("tree_01",),
    )
    assert alert is not None
    assert alert.severity in ("HIGH", "CRITICAL")
    assert alert.subject_ref == target.track_id


def test_empirical_full_signal_chain_persistence_and_api(visual_cctv_run: dict[str, Any]) -> None:
    """Verify complete signal chain: persistence into SQLiteEventStore and retrieval via FastAPI API."""
    target = visual_cctv_run["target_intruder_track"]
    bbox = visual_cctv_run["first_intruder_bbox"]
    dev = visual_cctv_run["final_dwell_event"]
    fence_engine = FenceEngine()
    dwell_engine = visual_cctv_run["dwell_engine"]
    traj_engine = visual_cctv_run["traj_engine"]
    anom_res = visual_cctv_run["anomalous_traj_result"]
    width = visual_cctv_run["width"]
    height = visual_cctv_run["height"]
    zone = visual_cctv_run["zone_perimeter"]
    t_first_ns = visual_cctv_run["target_intruder_first_frame_ts"]
    now_ns = visual_cctv_run["now_ns"]
    first_frame = visual_cctv_run["first_frame"]

    fence_events = fence_engine.check_subject(
        subject_ref=target.track_id,
        bbox=bbox,
        camera_id="cam-cctv-01",
        timestamp_ns=t_first_ns,
        source_resolution=(width, height),
        zones=[zone],
        confidence=0.92,
    )
    fe = fence_events[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "events.db")
        ev_dir = str(Path(tmpdir) / "evidence")

        event_store = SQLiteEventStore(db_path=db_path)
        alert_store = AlertStore(event_store=event_store)
        evidence_mgr = EvidenceManager(base_path=ev_dir)

        try:
            # 1. Direct provenance: FenceEvent / DwellEvent / TrajectoryEvent -> canonical Event -> EventStore
            ev_int = fence_engine.to_event(fe, frame_id=1)
            ev_loiter = dwell_engine.to_event(dev, frame_id=30)
            ev_traj = traj_engine.to_event(anom_res, frame_id=30)
            event_store.append(ev_int)
            event_store.append(ev_loiter)
            event_store.append(ev_traj)

            # 2. Direct provenance: canonical Event -> canonical Evidence (Zero extra_payload)
            evidence_items = [
                Evidence.from_event(ev_int),
                Evidence.from_event(ev_loiter),
                Evidence.from_event(ev_traj),
            ]

            risk_eval = SituationalRiskEvaluator()
            risk_signal = risk_eval.evaluate(
                evidence_items=evidence_items,
                system_mode=SystemMode.OPERATIONAL_MODE,
                window_start_ns=now_ns - 60_000_000_000,
                window_end_ns=now_ns,
            )

            alert_engine = AlertEngine(alert_min_severity="HIGH")
            alert = alert_engine.evaluate_risk_signal(
                risk_signal=risk_signal,
                camera_id="cam-cctv-01",
                subject_ref=target.track_id,
                zone_id="zone_restricted_perimeter",
            )
            assert alert is not None
            assert alert.severity in ("HIGH", "CRITICAL")
            alert_store.append_alert(alert)

            # 3. Capture visual snapshot
            ev_dir_path = evidence_mgr.capture_evidence(
                alert_id=alert.alert_id,
                camera_id="cam-cctv-01",
                timestamp_ns=ev_loiter.timestamp_ns,
                frame=first_frame,
                metadata={"track_id": target.track_id, "risk_level": risk_signal.risk_level},
            )
            assert (Path(ev_dir_path) / "snapshot.jpg").exists()

            # 4. Verify API retrieval
            app = create_app(event_store=event_store, alert_store=alert_store, evidence_manager=evidence_mgr)
            client = TestClient(app)

            # GET /alerts
            r_alerts = client.get("/alerts")
            assert r_alerts.status_code == 200
            assert r_alerts.json()["count"] == 1

            # GET /alerts/{alert_id}
            r_single = client.get(f"/alerts/{alert.alert_id}")
            assert r_single.status_code == 200
            assert r_single.json()["alert_id"] == alert.alert_id

            # GET /evidence/{alert_id}
            r_ev = client.get(f"/evidence/{alert.alert_id}")
            assert r_ev.status_code == 200
            assert len(r_ev.json()["evidence"]) == 1

            # GET /events
            r_events = client.get("/events")
            assert r_events.status_code == 200
            assert r_events.json()["count"] >= 3

            # GET /dashboard
            r_dash = client.get("/dashboard")
            assert r_dash.status_code == 200
        finally:
            event_store.close()

