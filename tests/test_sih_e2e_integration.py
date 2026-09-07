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
from gods_eye.alerts.alert_store import SQLiteAlertStore
from gods_eye.api.app import create_app
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.evidence.evidence_manager import EvidenceManager
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.camera import Zone
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import Evidence, ToolCall
from gods_eye.schemas.environment import SystemMode
from gods_eye.situational.evaluator import SituationalRiskEvaluator
from gods_eye.situational.hypothesis import HypothesisGenerator
from gods_eye.zones.dwell_engine import DwellEngine
from gods_eye.zones.fence_engine import FenceEngine


class TestSIHEndToEndPipeline:
    def test_full_sih_demo_signal_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            events_db = str(Path(tmpdir) / "events.db")
            alerts_db = str(Path(tmpdir) / "alerts.db")
            evidence_dir = str(Path(tmpdir) / "evidence")

            event_store = SQLiteEventStore(db_path=events_db)
            alert_store = SQLiteAlertStore(db_path=alerts_db)
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

                # 2. Subject enters restricted zone
                subject_ref = "subject_infiltrator_01"
                t0_ns = 1_000_000_000
                bbox_inside = BoundingBox(x1=200, y1=200, x2=300, y2=300)  # center (0.25, 0.25) in (1000, 1000)

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

                # Also seed an ANPR reading event for the associated getaway vehicle
                plate_number = "MH12DE1433"
                ev_anpr = Event(
                    event_id="ev_anpr_01",
                    event_type=EventType.ANPR_READING,
                    global_id=None,
                    camera_id="cam-01",
                    timestamp_ns=t1_ns + 5_000_000_000,
                    frame_id=3750,
                    confidence=0.94,
                    explanation=f"Plate {plate_number} identified",
                    metadata={"plate_text": plate_number, "confidence": 0.94},
                )
                event_store.append(ev_anpr)

                # 4. Construct Evidence Items for SituationalRiskEvaluator
                # Evaluator CRITICAL rule:
                # trajectory_anomaly_sigma > 3.5 AND zone_dwell_sigma > 5.0 AND is_restricted_zone
                import time as _time
                now_ns = _time.time_ns()
                t1_ns = now_ns
                t0_ns = now_ns - 120_000_000_000

                evidence_vault = Evidence(
                    evidence_id="ev_crit_01",
                    source_store="event_store",
                    record_type="RESTRICTED_ZONE_INTRUSION",
                    record_id="ev_fence_01",
                    timestamp_ns=t1_ns - 10_000_000_000,
                    camera_id="cam-01",
                    payload={
                        "is_restricted_zone": True,
                        "zone_dwell_sigma": dwell_ev.zone_dwell_sigma,
                        "trajectory_anomaly_sigma": 4.2,  # anomaly motion inside restricted zone
                        "confidence": 0.95,
                        "subject_ref": subject_ref,
                    },
                )

                risk_signal = risk_evaluator.evaluate(
                    evidence_items=[evidence_vault],
                    system_mode=SystemMode.OPERATIONAL_MODE,
                    window_start_ns=t0_ns,
                    window_end_ns=t1_ns,
                )

                # Verify CRITICAL rule fired
                assert risk_signal.risk_level == "CRITICAL"
                assert risk_signal.risk_score == 1.0
                assert risk_signal.suppressed is False

                # 5. Generate Bounded Hypothesis Tree
                trees = hypo_generator.generate_trees(
                    evidence_items=[evidence_vault],
                    window_start_ns=t0_ns,
                    window_end_ns=t1_ns,
                )
                assert len(trees) >= 1
                assert trees[0].root_node.hypothesis.primary_subject_ref == subject_ref

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
                        "evidence_items": [evidence_vault],
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
                        "evidence_items": [evidence_vault],
                    },
                )
                res_t12 = dispatcher.dispatch(t12_call)
                assert res_t12.success is True
                assert res_t12.data["count"] >= 1

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
                alert_store.close()
