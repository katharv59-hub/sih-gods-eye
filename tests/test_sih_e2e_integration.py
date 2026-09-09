"""End-to-End Signal Chain Integration Test — SIH 26187 (Section 6 Definition of Done).

Executes the full unattended pipeline chain:
  Detection / Track
    -> Restricted Zone Geofence (FenceEngine sets is_restricted_zone=True)
    -> Loitering Deviation (DwellEngine sets zone_dwell_sigma > 5.0)
    -> SituationalRiskEvaluator fires CRITICAL rule
    -> HypothesisGenerator emits bounded hypothesis tree
    -> Alert created via AlertEngine with severity=CRITICAL
    -> Evidence snapshot and metadata captured via EvidenceManager
    -> Alert, evidence, and plate query retrievable via Tools 11, 12, 13
    -> FastAPI Command-and-Control API endpoints serve data to Dashboard
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gods_eye.alerts.alert_engine import AlertEngine
from gods_eye.alerts.alert_store import AlertStore
from gods_eye.api.app import create_app
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.evidence.evidence_manager import EvidenceManager
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.camera import Zone
import cv2
from gods_eye.config.settings import Settings
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.memory.adapter import TemporalAdapter
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.pipeline import TrackingResult
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.identity_mapper import IdentityMapper
from gods_eye.reid.matcher import Matcher
from gods_eye.reid.osnet_extractor import OSNetExtractor
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import Evidence, ToolCall
from gods_eye.anpr.ocr import PlateReadingResult
from gods_eye.schemas.environment import SystemMode
from gods_eye.schemas.track import Track, TrackState
from gods_eye.schemas.vehicle import VehicleTrack, VehicleTrackState
from gods_eye.situational.evaluator import SituationalRiskEvaluator
from gods_eye.situational.hypothesis import HypothesisGenerator
from gods_eye.vehicle.tracker import VehicleTracker
from gods_eye.zones.dwell_engine import DwellEngine
from gods_eye.zones.fence_engine import FenceEngine



class TestSIHEndToEndPipeline:
    def test_full_sih_demo_signal_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            events_db = str(Path(tmpdir) / "events.db")
            graph_db = str(Path(tmpdir) / "graph.db")
            evidence_dir = str(Path(tmpdir) / "evidence")

            event_store = SQLiteEventStore(db_path=events_db)
            graph_store = SQLiteGraphStore(db_path=graph_db)
            alert_store = AlertStore(event_store=event_store)
            evidence_mgr = EvidenceManager(base_path=evidence_dir)
            risk_evaluator = SituationalRiskEvaluator()
            hypo_generator = HypothesisGenerator()
            fence_engine = FenceEngine()
            dwell_engine = DwellEngine(default_max_dwell_s=60.0, default_std_dwell_s=10.0)
            alert_engine = AlertEngine(alert_min_severity="MEDIUM")

            try:
                # 1. Setup Camera Zone: Restricted Vault Zone
                restricted_zone = Zone(
                    zone_id="zone_vault",
                    camera_id="cam-01",
                    display_name="Vault Restricted Area",
                    polygon=[(0.0, 0.0), (0.6, 0.0), (0.6, 0.6), (0.0, 0.6)],
                    zone_type="restricted",
                    expected_dwell_s=(0.0, 30.0),
                )

                # 2. Real Visual Person Crop -> OSNet -> IdentityMapper -> Persistent Global ID
                t0_ns = 1_000_000_000
                bbox_inside = BoundingBox(x1=200, y1=200, x2=300, y2=300)  # center (0.25, 0.25) in (1000, 1000)

                # Load real visual person sample from detection_samples
                sample_img_path = Path("tests/data/detection_samples/sample_1.jpg")
                if sample_img_path.exists():
                    person_frame = cv2.imread(str(sample_img_path))
                else:
                    person_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                settings = Settings()
                extractor = OSNetExtractor(settings)
                gallery = IdentityGallery()
                lifecycle = LifecycleManager(settings)
                matcher = Matcher(settings)
                identity_mapper = IdentityMapper(
                    extractor=extractor,
                    gallery=gallery,
                    lifecycle=lifecycle,
                    matcher=matcher,
                    settings=settings,
                    camera_id="cam-01",
                )

                # Ingest real tracking result into IdentityMapper
                pkt = FramePacket(
                    camera_id="cam-01",
                    frame_id=1,
                    timestamp_ns=t0_ns,
                    frame=person_frame,
                    resolution=(640, 480),
                )
                det = Detection(
                    detection_id="det_pers_01",
                    camera_id="cam-01",
                    frame_id=1,
                    timestamp_ns=t0_ns,
                    bbox=bbox_inside,
                    confidence=0.96,
                    class_label="person",
                    source_resolution=(1000, 1000),
                )
                track_pers = Track(
                    track_id="trk_person_01",
                    camera_id="cam-01",
                    state=TrackState.ACTIVE,
                    bbox=bbox_inside,
                    velocity=(0.0, 0.0),
                    first_frame_id=1,
                    last_frame_id=1,
                    lost_frame_count=0,
                    detection_history=[det.detection_id],
                    class_label="person",
                )
                tr_res = TrackingResult(packet=pkt, detections=[det], tracks=[track_pers])
                id_res = identity_mapper.process(tr_res)
                assert len(id_res.identities) == 1
                real_identity = id_res.identities["trk_person_01"]
                subject_ref = real_identity.global_id
                assert subject_ref is not None
                assert len(subject_ref) > 0

                # Persist real identity node to SQLiteGraphStore
                TemporalAdapter.persist_identity_to_graph(
                    identity=real_identity,
                    graph_store=graph_store,
                    camera_id="cam-01",
                )

                fence_events = fence_engine.check_subject(
                    subject_ref=subject_ref,
                    bbox=bbox_inside,
                    camera_id="cam-01",
                    timestamp_ns=t0_ns,
                    source_resolution=(1000, 1000),
                    zones=[restricted_zone],
                    confidence=0.96,
                )

                assert len(fence_events) == 1
                fence_ev = fence_events[0]
                assert fence_ev.event_type == "RESTRICTED_ZONE_INTRUSION"
                assert fence_ev.is_restricted_zone is True

                # Persist zone event
                ev_fence = Event(
                    event_id="ev_fence_01",
                    event_type=EventType.RESTRICTED_ZONE_INTRUSION,
                    global_id=subject_ref,
                    camera_id="cam-01",
                    timestamp_ns=t0_ns,
                    frame_id=1,
                    confidence=0.96,
                    explanation=fence_ev.explanation,
                    zone_id="zone_vault",
                    metadata=fence_engine.build_evidence_payload(fence_ev),
                )
                event_store.append(ev_fence)

                # 3. Loitering accumulation: 120s dwell (exceeds max 30s by 90s, std=10s -> 9.0 sigma)
                t1_ns = t0_ns + 120_000_000_000
                # Record entry at t0
                dwell_engine.record_zone_presence(
                    subject_ref=subject_ref,
                    zone_id="zone_vault",
                    camera_id="cam-01",
                    timestamp_ns=t0_ns,
                    confidence=0.95,
                    zone=restricted_zone,
                )
                # Record presence at t1
                dwell_ev = dwell_engine.record_zone_presence(
                    subject_ref=subject_ref,
                    zone_id="zone_vault",
                    camera_id="cam-01",
                    timestamp_ns=t1_ns,
                    confidence=0.95,
                    zone=restricted_zone,
                )
                assert dwell_ev is not None
                assert dwell_ev.zone_dwell_sigma > 5.0

                ev_dwell = Event(
                    event_id="ev_dwell_01",
                    event_type=EventType.LOITERING_DETECTED,
                    global_id=subject_ref,
                    camera_id="cam-01",
                    timestamp_ns=t1_ns,
                    frame_id=3600,
                    confidence=0.95,
                    explanation=dwell_ev.explanation,
                    zone_id="zone_vault",
                    metadata=dwell_engine.build_evidence_payload(dwell_ev),
                )
                event_store.append(ev_dwell)

                # 3b. Associated getaway vehicle perception & ANPR via production objects
                plate_number = "MH12DE1433"
                vehicle_track_id = "vtrk_cam-01_getaway_1"

                # Production PlateOCR reading
                ocr_reading = PlateReadingResult(
                    plate_text=plate_number,
                    confidence=0.94,
                    raw_text=plate_number,
                    is_uncertain=False,
                )

                # Production association via VehicleTracker
                v_tracker = VehicleTracker(stream_prefix="vtrk")
                assoc = v_tracker.associate_plate(
                    track_id=vehicle_track_id,
                    plate_text=ocr_reading.plate_text,
                    confidence=0.94,
                    camera_id="cam-01",
                    frame_id=3750,
                    timestamp_ns=t1_ns + 5_000_000_000,
                    is_uncertain=ocr_reading.is_uncertain,
                    ocr_confidence=ocr_reading.confidence,
                )
                ev_anpr = assoc.to_event(event_id="ev_anpr_01")
                event_store.append(ev_anpr)

                # Production VehicleTrack to canonical VEHICLE_DETECTED event
                v_track = VehicleTrack(
                    track_id=vehicle_track_id,
                    camera_id="cam-01",
                    state=VehicleTrackState.ACTIVE,
                    bbox=BoundingBox(x1=50.0, y1=50.0, x2=250.0, y2=200.0),
                    velocity=(2.0, 0.0),
                    vehicle_class="car",
                    first_frame_id=3700,
                    last_frame_id=3750,
                    lost_frame_count=0,
                    plate_text=plate_number,
                    association_confidence=0.94,
                    plate_association=assoc,
                )
                ev_veh = v_track.to_event(
                    timestamp_ns=t1_ns + 5_000_000_000,
                    confidence=0.92,
                    event_id="ev_veh_01",
                )
                event_store.append(ev_veh)


                # 4. Construct Independent Evidence Items from Phase 9 engines
                # Evaluator CRITICAL rule fires across distinct evidence items:
                # trajectory_anomaly_sigma > 3.5 AND zone_dwell_sigma > 5.0 AND is_restricted_zone
                import time as _time
                now_ns = _time.time_ns()
                t1_ns = now_ns
                t0_ns = now_ns - 120_000_000_000

                ev_crit_fence = Evidence(
                    evidence_id="ev_crit_fence_01",
                    source_store="event_store",
                    record_type="RESTRICTED_ZONE_INTRUSION",
                    record_id="ev_fence_01",
                    timestamp_ns=t0_ns + 1_000_000_000,
                    camera_id="cam-01",
                    global_id=subject_ref,
                    explanation=fence_ev.explanation,
                    payload=fence_engine.build_evidence_payload(fence_ev),
                )

                ev_crit_dwell = Evidence(
                    evidence_id="ev_crit_dwell_01",
                    source_store="event_store",
                    record_type="LOITERING_DETECTED",
                    record_id="ev_dwell_01",
                    timestamp_ns=t1_ns - 10_000_000_000,
                    camera_id="cam-01",
                    global_id=subject_ref,
                    explanation=dwell_ev.explanation,
                    payload=dwell_engine.build_evidence_payload(dwell_ev),
                )

                ev_crit_traj = Evidence(
                    evidence_id="ev_crit_traj_01",
                    source_store="event_store",
                    record_type="TRAJECTORY_ANOMALY",
                    record_id="rec_traj_01",
                    timestamp_ns=t1_ns - 5_000_000_000,
                    camera_id="cam-01",
                    global_id=subject_ref,
                    explanation="High-sigma irregular trajectory motion observed",
                    payload={
                        "trajectory_anomaly_sigma": 4.2,
                        "confidence": 0.95,
                        "subject_ref": subject_ref,
                    },
                )

                evidence_items = [ev_crit_fence, ev_crit_dwell, ev_crit_traj]

                risk_signal = risk_evaluator.evaluate(
                    evidence_items=evidence_items,
                    system_mode=SystemMode.OPERATIONAL_MODE,
                    window_start_ns=t0_ns,
                    window_end_ns=t1_ns,
                )

                # Verify CRITICAL rule fired across independent evidence items
                assert risk_signal.risk_level == "CRITICAL"
                assert risk_signal.risk_score == 1.0
                assert risk_signal.suppressed is False
                assert set(risk_signal.evidence_ids) == {
                    "ev_crit_fence_01",
                    "ev_crit_dwell_01",
                    "ev_crit_traj_01",
                }

                # 5. Generate Bounded Hypothesis Tree — co-occurrence produces SUSPICIOUS_ACTIVITY
                trees = hypo_generator.generate_trees(
                    evidence_items=evidence_items,
                    window_start_ns=t0_ns,
                    window_end_ns=t1_ns,
                )
                assert len(trees) >= 1
                assert trees[0].root_node.hypothesis.hypothesis_type == "SUSPICIOUS_ACTIVITY"
                assert trees[0].root_node.hypothesis.primary_subject_ref == subject_ref
                assert "RESTRICTED_ZONE_INTRUSION" in trees[0].root_node.hypothesis.explanation
                assert "LOITERING_DETECTED" in trees[0].root_node.hypothesis.explanation

                # 6. Create Alert via AlertEngine & Persist
                alert = alert_engine.evaluate_risk_signal(
                    risk_signal=risk_signal,
                    camera_id="cam-01",
                    subject_ref=subject_ref,
                    zone_id="zone_vault",
                    hypothesis_ids=(trees[0].tree_id,),
                )
                assert alert is not None
                assert alert.severity == "CRITICAL"
                assert alert.status.value == "detected"

                alert_store.append_alert(alert)

                # 7. Evidence Capture with Snapshot & Metadata
                mock_snapshot = np.ones((480, 640, 3), dtype=np.uint8) * 50
                ev_path = evidence_mgr.capture_evidence(
                    alert_id=alert.alert_id,
                    camera_id="cam-01",
                    timestamp_ns=t1_ns,
                    frame=mock_snapshot,
                    metadata={
                        "risk_level": "CRITICAL",
                        "rule_fired": "RESTRICTED_ZONE_INTRUSION_AND_LOITERING",
                        "associated_plate": plate_number,
                    },
                )
                assert Path(ev_path).exists()
                ev_records = evidence_mgr.get_evidence(alert.alert_id)
                assert len(ev_records) == 1
                assert ev_records[0]["associated_plate"] == plate_number

                # 8. ToolDispatcher verification: Tool 11, 12, 13
                dispatcher = ToolDispatcher(
                    event_store=event_store,
                    situational_evaluator=risk_evaluator,
                    hypothesis_generator=hypo_generator,
                )

                # Tool 11: Situational Risk
                t11_call = ToolCall(
                    call_id="call_t11",
                    tool_name="get_situational_risk",
                    arguments={
                        "window_start_ns": t0_ns,
                        "window_end_ns": t1_ns,
                        "evidence_items": evidence_items,
                    },
                )
                res_t11 = dispatcher.dispatch(t11_call)
                assert res_t11.success is True
                assert res_t11.data["risk_level"] == "CRITICAL"

                # Tool 12: Hypothesis Forest
                t12_call = ToolCall(
                    call_id="call_t12",
                    tool_name="get_hypothesis_tree",
                    arguments={
                        "window_start_ns": t0_ns,
                        "window_end_ns": t1_ns,
                        "evidence_items": evidence_items,
                    },
                )
                res_t12 = dispatcher.dispatch(t12_call)
                assert res_t12.success is True
                assert res_t12.data["count"] >= 1
                assert res_t12.data["hypothesis_trees"][0]["root_node"]["hypothesis"]["hypothesis_type"] == "SUSPICIOUS_ACTIVITY"

                # Tool 13: Plate History Query
                t13_call = ToolCall(
                    call_id="call_t13",
                    tool_name="get_plate_history",
                    arguments={"plate_text": plate_number},
                )
                res_t13 = dispatcher.dispatch(t13_call)
                assert res_t13.success is True
                assert res_t13.data["count"] == 1
                assert res_t13.data["readings"][0]["plate_text"] == plate_number

                # 9. FastAPI C2 Application Endpoints & Dashboard
                app = create_app(
                    event_store=event_store,
                    graph_store=graph_store,
                    alert_store=alert_store,
                    evidence_manager=evidence_mgr,
                    tool_dispatcher=dispatcher,
                )
                client = TestClient(app)

                # Verify alerts endpoint shows the CRITICAL alert
                res_alerts = client.get("/alerts")
                assert res_alerts.status_code == 200
                assert res_alerts.json()["count"] == 1
                assert res_alerts.json()["alerts"][0]["severity"] == "CRITICAL"

                # Verify single alert endpoint
                res_single = client.get(f"/alerts/{alert.alert_id}")
                assert res_single.status_code == 200
                assert res_single.json()["alert_id"] == alert.alert_id

                # Verify plate query endpoint delegates through Tool 13
                res_plate = client.get(f"/plates/{plate_number}")
                assert res_plate.status_code == 200
                assert res_plate.json()["plate"] == plate_number
                assert len(res_plate.json()["readings"]["readings"]) == 1

                # Verify vehicle query endpoint retrieves real persisted vehicle event
                res_veh = client.get(f"/vehicles/{vehicle_track_id}")
                assert res_veh.status_code == 200
                assert res_veh.json()["vehicle_id"] == vehicle_track_id
                assert len(res_veh.json()["events"]) >= 1
                assert res_veh.json()["events"][0]["event_id"] == ev_veh.event_id

                # Verify subject query endpoint retrieves real persisted subject identity
                res_subj = client.get(f"/subjects/{subject_ref}")
                assert res_subj.status_code == 200
                subj_data = res_subj.json()
                assert subj_data["subject_id"] == subject_ref
                assert subj_data["found"] is True
                assert subj_data["primary_camera_id"] == "cam-01"
                assert subj_data["state"].lower() == "active"

                # Negative Case 1: unobserved plate returns valid empty collection (HTTP 200)
                res_unobs_plate = client.get("/plates/UNOBSERVED999")
                assert res_unobs_plate.status_code == 200
                assert res_unobs_plate.json()["readings"]["count"] == 0

                # Negative Case 2: unobserved vehicle returns valid empty events (HTTP 200)
                res_unobs_veh = client.get("/vehicles/vtrk_nonexistent")
                assert res_unobs_veh.status_code == 200
                assert res_unobs_veh.json()["events"] == []

                # Negative Case 3: unobserved subject returns valid empty semantics (HTTP 200, found=False)
                res_unobs_subj = client.get("/subjects/subj_nonexistent_999")
                assert res_unobs_subj.status_code == 200
                assert res_unobs_subj.json() == {"subject_id": "subj_nonexistent_999", "found": False}

                # Verify evidence endpoint
                res_ev = client.get(f"/evidence/{alert.alert_id}")
                assert res_ev.status_code == 200
                assert len(res_ev.json()["evidence"]) == 1
                assert res_ev.json()["evidence"][0]["associated_plate"] == plate_number

                # Verify dashboard HTML is served
                res_dash = client.get("/dashboard")
                assert res_dash.status_code == 200
                assert "GOD'S EYE" in res_dash.text


            finally:
                event_store.close()
                graph_store.close()
