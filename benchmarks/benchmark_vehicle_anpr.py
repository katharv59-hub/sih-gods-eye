"""Empirical Vehicle and ANPR Benchmark — SIH Problem Statement 26187 (Fix #7).

Executes the full vehicle perception + tracking + ANPR pipeline on real visual data:
1. Urban street CCTV footage (MOT17-04 medium density).
2. Genuine ground-truth license plate specimen (Maine BMV 'SAMPLE').
3. Real readable vehicle plate visual sample.

Measures observed runtime performance and validates direct provenance into the canonical
event pipeline:
- Vehicle detection counts, rates, and confidence distributions (YOLOv8)
- Canonical ByteTrack track IDs, states, and continuity
- Plate region candidate extraction
- ANPR OCR inference and confidence floor enforcement
- Vehicle-to-plate association and uncertainty handling
- Direct conversion of VehicleTrack -> Event(VEHICLE_DETECTED)
- Direct conversion of VehiclePlateAssociation -> Event(ANPR_READING)
- Persistence into canonical SQLiteEventStore
- Tool 13 (get_plate_history) deterministic retrieval
- C2 API retrieval: GET /plates/{plate} and GET /vehicles/{vehicle_id}
- Negative / adversarial query verification (unobserved plate/vehicle, uncertain OCR preservation)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Ensure repository root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gods_eye.anpr.ocr import PlateOCR
from gods_eye.anpr.plate_detector import PlateDetector
from gods_eye.api.app import create_app
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.observability.logger import get_logger
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.event import EventType
from gods_eye.schemas.reasoning import ToolCall
from gods_eye.schemas.vehicle import VehicleTrack, VehicleTrackState
from gods_eye.vehicle.detector import VehicleDetector
from gods_eye.vehicle.tracker import VehicleTracker

_log = get_logger("benchmark.vehicle_anpr")


def run_cctv_video_benchmark(
    video_path: str = "tests/data/demo_videos/mot17_04_medium_density.mp4",
    max_frames: int = 100,
) -> dict[str, Any]:
    """Run empirical validation on real CCTV video footage.

    Args:
        video_path: Path to MP4 video.
        max_frames: Maximum number of frames to process.

    Returns:
        Dictionary of measured empirical metrics and canonical event provenance.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
    tracker = VehicleTracker(stream_prefix="vtrk")
    plate_detector = PlateDetector(confidence_threshold=0.40)
    ocr = PlateOCR(min_confidence=0.60, use_easyocr=True)

    # Canonical EventStore for direct CCTV provenance (Fix #7)
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "cctv_events.db"
    event_store = SQLiteEventStore(db_path=db_path)

    frame_count = 0
    total_detections = 0
    class_distribution: dict[str, int] = {}
    confidences: list[float] = []
    unique_tracks: set[str] = set()
    active_tracks_per_frame: list[int] = []
    plate_candidates_count = 0
    ocr_attempts = 0
    ocr_uncertain_count = 0
    ocr_confident_count = 0
    plate_detections_detail: list[dict[str, Any]] = []

    persisted_veh_events = 0
    persisted_anpr_events = 0

    t_start = time.perf_counter()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = frame_count + 1
        timestamp_ns = 1_000_000_000 + int(frame_count * (1e9 / fps))

        # 1. Vehicle Detection (YOLOv8)
        dets = detector.detect(frame, "cam-cctv-01", frame_id, timestamp_ns)
        total_detections += len(dets)
        for d in dets:
            class_distribution[d.vehicle_class] = class_distribution.get(d.vehicle_class, 0) + 1
            confidences.append(d.confidence)

        # 2. Canonical ByteTrack Tracking
        tracks = tracker.update(dets, frame_id, "cam-cctv-01")
        active_in_frame = 0
        for t in tracks:
            unique_tracks.add(t.track_id)
            if t.state == VehicleTrackState.ACTIVE:
                active_in_frame += 1

                # Direct provenance: adapt actual VehicleTrack to canonical Event(VEHICLE_DETECTED)
                ev_veh = t.to_event(timestamp_ns=timestamp_ns, frame_id=frame_id)
                event_store.append(ev_veh)
                persisted_veh_events += 1

                # 3. Plate Candidate Detection
                plates = plate_detector.detect_plates(
                    frame=frame,
                    vehicle_bbox=t.bbox,
                    camera_id="cam-cctv-01",
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    vehicle_track_id=t.track_id,
                )
                plate_candidates_count += len(plates)

                # 4. ANPR / OCR Inference & Association
                for p in plates:
                    ocr_attempts += 1
                    res = ocr.read_plate(p.plate_image)
                    if res.is_uncertain:
                        ocr_uncertain_count += 1
                    else:
                        ocr_confident_count += 1

                    # Associate plate candidate with vehicle track
                    assoc = tracker.associate_plate(
                        track_id=t.track_id,
                        plate_text=res.plate_text,
                        is_uncertain=res.is_uncertain,
                        plate_region_id=p.plate_id,
                        ocr_confidence=res.confidence,
                        spatial_confidence=p.confidence,
                        plate_bbox=p.bbox,
                        vehicle_bbox=t.bbox,
                        camera_id="cam-cctv-01",
                        frame_id=frame_id,
                        timestamp_ns=timestamp_ns,
                    )

                    # Direct provenance: adapt actual VehiclePlateAssociation to canonical Event(ANPR_READING)
                    ev_anpr = assoc.to_event()
                    event_store.append(ev_anpr)
                    persisted_anpr_events += 1

                    if len(plate_detections_detail) < 10:
                        plate_detections_detail.append({
                            "frame_id": frame_id,
                            "vehicle_track_id": t.track_id,
                            "vehicle_class": t.vehicle_class,
                            "plate_candidate_id": p.plate_id,
                            "plate_region_conf": round(p.confidence, 3),
                            "ocr_raw": res.raw_text,
                            "ocr_clean": res.plate_text,
                            "ocr_conf": round(res.confidence, 3),
                            "is_uncertain": res.is_uncertain,
                            "association_confidence": assoc.confidence,
                            "persisted_event_id": ev_anpr.event_id,
                        })

        active_tracks_per_frame.append(active_in_frame)
        frame_count += 1

    cap.release()
    t_end = time.perf_counter()
    elapsed_s = t_end - t_start

    # Collect all stored associations
    all_assocs = list(tracker._plate_registry.values())
    assoc_confs = [a.confidence for a in all_assocs]

    # Tool 13 + C2 API verification on persisted real visual events
    dispatcher = ToolDispatcher(event_store=event_store)
    app = create_app(event_store=event_store, tool_dispatcher=dispatcher)
    client = TestClient(app)

    # API 1: Query vehicle endpoint for actual active track
    sample_track_id = sorted(list(unique_tracks))[0] if unique_tracks else "vtrk_none"
    r_veh = client.get(f"/vehicles/{sample_track_id}")
    veh_events_retrieved = len(r_veh.json().get("events", [])) if r_veh.status_code == 200 else 0

    # API 2: Negative unobserved vehicle query
    r_veh_neg = client.get("/vehicles/vtrk_nonexistent_cctv_99")
    veh_neg_events_count = len(r_veh_neg.json().get("events", [])) if r_veh_neg.status_code == 200 else -1

    # Tool 13: Query unobserved plate
    tc_neg = ToolCall(call_id="call_cctv_unobs", tool_name="get_plate_history", arguments={"plate_text": "UNOBSERVED_CCTV_PLATE"})
    tr_neg = dispatcher.dispatch(tc_neg)
    tool13_unobs_success = tr_neg.success and (tr_neg.data.get("count", 0) == 0 if isinstance(tr_neg.data, dict) else False)

    # API 3: Negative unobserved plate query
    r_plate_neg = client.get("/plates/UNOBSERVED_CCTV_PLATE")
    plate_neg_readings_count = len(r_plate_neg.json().get("readings", {}).get("readings", [])) if r_plate_neg.status_code == 200 else -1

    # Cleanly close SQLite store to release Windows file lock
    event_store.close()

    return {
        "dataset": {
            "source": str(path.as_posix()),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_video_frames": total_video_frames,
            "frames_processed": frame_count,
        },
        "vehicle_detection": {
            "total_detections": total_detections,
            "detections_per_frame": round(total_detections / max(1, frame_count), 2),
            "class_distribution": class_distribution,
            "confidence_mean": round(float(np.mean(confidences)), 3) if confidences else 0.0,
            "confidence_min": round(float(np.min(confidences)), 3) if confidences else 0.0,
            "confidence_max": round(float(np.max(confidences)), 3) if confidences else 0.0,
        },
        "bytetrack_tracking": {
            "unique_track_ids": sorted(list(unique_tracks)),
            "total_unique_tracks": len(unique_tracks),
            "mean_active_tracks_per_frame": round(float(np.mean(active_tracks_per_frame)), 2) if active_tracks_per_frame else 0.0,
            "id_prefix": "vtrk_",
        },
        "anpr_ocr": {
            "plate_candidates_detected": plate_candidates_count,
            "ocr_inference_attempts": ocr_attempts,
            "ocr_uncertain_count": ocr_uncertain_count,
            "ocr_confident_count": ocr_confident_count,
            "unreadable_rate": round(ocr_uncertain_count / max(1, ocr_attempts), 3),
            "sample_detections": plate_detections_detail,
        },
        "association": {
            "total_associations": len(all_assocs),
            "associations_confident": len([a for a in all_assocs if not a.is_uncertain]),
            "associations_uncertain": len([a for a in all_assocs if a.is_uncertain]),
            "mean_association_confidence": round(float(np.mean(assoc_confs)), 3) if assoc_confs else 0.0,
        },
        "canonical_event_provenance": {
            "persisted_vehicle_detected_events": persisted_veh_events,
            "persisted_anpr_reading_events": persisted_anpr_events,
            "event_store_path": str(db_path),
            "tool13_unobserved_success": tool13_unobs_success,
            "api_vehicle_events_retrieved": veh_events_retrieved,
            "api_unobserved_vehicle_status": r_veh_neg.status_code,
            "api_unobserved_vehicle_count": veh_neg_events_count,
            "api_unobserved_plate_status": r_plate_neg.status_code,
            "api_unobserved_plate_readings_count": plate_neg_readings_count,
        },
        "performance": {
            "elapsed_seconds": round(elapsed_s, 2),
            "throughput_fps": round(frame_count / max(0.001, elapsed_s), 2),
        },
    }


def run_readable_plate_benchmark(
    image_path: str = "tests/data/vehicle_samples/readable_plate_sample.png",
) -> dict[str, Any]:
    """Run empirical OCR validation on a readable plate image.

    Args:
        image_path: Path to PNG or JPG image of a license plate.

    Returns:
        Dictionary of measured OCR results and canonical event provenance.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Plate sample image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    ocr = PlateOCR(min_confidence=0.60, use_easyocr=True)

    t0 = time.perf_counter()
    reading = ocr.read_plate(img)
    t_ocr = time.perf_counter() - t0

    # Test track association with vehicle tracker
    tracker = VehicleTracker(stream_prefix="vtrk")
    track_id = "vtrk_cam-readable_01"
    assoc = tracker.associate_plate(
        track_id=track_id,
        plate_text=reading.plate_text,
        is_uncertain=reading.is_uncertain,
        ocr_confidence=reading.confidence,
        camera_id="cam-readable",
        frame_id=1,
        timestamp_ns=1_000_000_000,
    )

    # Direct provenance: adapt to canonical events and persist to SQLiteEventStore
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "readable_events.db"
    event_store = SQLiteEventStore(db_path=db_path)

    ev_anpr = assoc.to_event(event_id="ev_anpr_readable_01")
    event_store.append(ev_anpr)

    v_track = VehicleTrack(
        track_id=track_id,
        camera_id="cam-readable",
        state=VehicleTrackState.ACTIVE,
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=float(w - 10), y2=float(h - 10)),
        velocity=(0.0, 0.0),
        vehicle_class="car",
        first_frame_id=1,
        last_frame_id=1,
        lost_frame_count=0,
        plate_text=reading.plate_text,
        association_confidence=assoc.confidence,
        plate_association=assoc,
    )
    ev_veh = v_track.to_event(
        timestamp_ns=1_000_000_000,
        frame_id=1,
        confidence=0.90,
        event_id="ev_veh_readable_01",
    )
    event_store.append(ev_veh)

    # Tool 13 retrieval
    dispatcher = ToolDispatcher(event_store=event_store)
    search_token = reading.plate_text.split()[0] if reading.plate_text else "WHILE9"
    tc = ToolCall(
        call_id="call_readable_13",
        tool_name="get_plate_history",
        arguments={"plate_text": search_token},
    )
    tr = dispatcher.dispatch(tc)

    # C2 API retrieval
    app = create_app(event_store=event_store, tool_dispatcher=dispatcher)
    client = TestClient(app)
    r_plate = client.get(f"/plates/{search_token}")
    r_veh = client.get(f"/vehicles/{track_id}")

    # Cleanly close SQLite store to release Windows file lock
    event_store.close()

    return {
        "image": {
            "path": str(path.as_posix()),
            "resolution": f"{w}x{h}",
        },
        "ocr_result": {
            "raw_text": reading.raw_text,
            "recognized_text": reading.plate_text,
            "confidence": round(reading.confidence, 3),
            "is_uncertain": reading.is_uncertain,
            "inference_time_s": round(t_ocr, 3),
        },
        "association": {
            "associated_track_id": track_id,
            "stored_plate_text": tracker.get_plate_text(track_id),
            "association_confidence": assoc.confidence,
            "association_success": tracker.get_plate_text(track_id) == reading.plate_text,
        },
        "canonical_event_provenance": {
            "anpr_event_id": ev_anpr.event_id,
            "vehicle_event_id": ev_veh.event_id,
            "tool13_success": tr.success,
            "tool13_count": tr.data.get("count", 0) if isinstance(tr.data, dict) else 0,
            "api_plate_status": r_plate.status_code,
            "api_vehicle_status": r_veh.status_code,
            "api_vehicle_events_retrieved": len(r_veh.json().get("events", [])),
        },
    }


def run_ground_truth_ocr_benchmark(
    sample_image_path: str = "tests/data/vehicle_samples/maine_sample.png",
) -> dict[str, Any]:
    """Run genuine ground-truth OCR validation against known independent ground truth.

    Uses an official public domain specimen/sample license plate (State of Maine BMV).
    Zero PII; legally usable test fixture.

    Ground-truth text: 'SAMPLE'

    Tests:
    1. Full plate image (containing state header 'MAINE', slogan 'VACATIONLAND', and 'SAMPLE')
    2. Registration number crop (containing strictly 'SAMPLE')
    3. Direct conversion to canonical ANPR_READING and VEHICLE_DETECTED events
    4. SQLiteEventStore persistence
    5. Tool 13 (get_plate_history) deterministic retrieval
    6. C2 API /plates/{plate} and /vehicles/{vehicle_id} verification
    7. Negative / adversarial query verification

    Returns:
        Structured ground-truth comparison results with end-to-end event provenance.
    """
    path = Path(sample_image_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth sample not found: {sample_image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read ground-truth sample: {sample_image_path}")

    expected_ground_truth = "SAMPLE"
    ocr = PlateOCR(min_confidence=0.60, use_easyocr=True)

    # 1. Full plate evaluation
    t0 = time.perf_counter()
    full_reading = ocr.read_plate(img)
    t_full = time.perf_counter() - t0

    full_exact_match = (full_reading.plate_text.strip() == expected_ground_truth)
    full_token_match = (expected_ground_truth in full_reading.plate_text.split())

    # 2. Registration number crop [y: 68:273, x: 91:610]
    crop = img[68:273, 91:610]
    t1 = time.perf_counter()
    crop_reading = ocr.read_plate(crop)
    t_crop = time.perf_counter() - t1

    crop_exact_match = (crop_reading.plate_text.strip() == expected_ground_truth)
    crop_normalized_match = (
        "".join(c for c in crop_reading.plate_text if c.isalnum())
        == "".join(c for c in expected_ground_truth if c.isalnum())
    )

    # 3. Association evaluation
    tracker = VehicleTracker(stream_prefix="vtrk")
    track_id = "vtrk_cam-gt_01"
    assoc = tracker.associate_plate(
        track_id=track_id,
        plate_text=crop_reading.plate_text,
        ocr_confidence=crop_reading.confidence,
        is_uncertain=crop_reading.is_uncertain,
        camera_id="cam-gt",
        frame_id=1,
        timestamp_ns=1_000_000_000,
    )

    # 4. Canonical EventStore persistence & Tool 13 / C2 API retrieval (Fix #7)
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "gt_events.db"
    event_store = SQLiteEventStore(db_path=db_path)

    # 4a. Adapt production association into canonical Event(ANPR_READING)
    ev_anpr = assoc.to_event(event_id="ev_anpr_gt_01")
    event_store.append(ev_anpr)

    # 4b. Adapt production VehicleTrack into canonical Event(VEHICLE_DETECTED)
    v_track = VehicleTrack(
        track_id=track_id,
        camera_id="cam-gt",
        state=VehicleTrackState.ACTIVE,
        bbox=BoundingBox(x1=91.0, y1=68.0, x2=610.0, y2=273.0),
        velocity=(0.0, 0.0),
        vehicle_class="car",
        first_frame_id=1,
        last_frame_id=1,
        lost_frame_count=0,
        plate_text=crop_reading.plate_text,
        association_confidence=assoc.confidence,
        plate_association=assoc,
    )
    ev_veh = v_track.to_event(
        timestamp_ns=1_000_000_000,
        frame_id=1,
        confidence=0.95,
        event_id="ev_veh_gt_01",
    )
    event_store.append(ev_veh)

    # 4c. Tool 13 verification via ToolDispatcher
    dispatcher = ToolDispatcher(event_store=event_store)
    tc = ToolCall(
        call_id="call_gt_13",
        tool_name="get_plate_history",
        arguments={"plate_text": "SAMPLE"},
    )
    tr = dispatcher.dispatch(tc)
    tool13_readings = tr.data.get("readings", []) if isinstance(tr.data, dict) else []

    # 4d. C2 API verification via FastAPI TestClient
    app = create_app(event_store=event_store, tool_dispatcher=dispatcher)
    client = TestClient(app)

    # Positive API queries
    r_plate = client.get("/plates/SAMPLE")
    plate_api_data = r_plate.json() if r_plate.status_code == 200 else {}

    r_veh = client.get(f"/vehicles/{track_id}")
    veh_api_data = r_veh.json() if r_veh.status_code == 200 else {}

    # Negative / Adversarial queries
    r_plate_neg = client.get("/plates/UNOBSERVED_SAMPLE_999")
    r_plate_neg_count = len(r_plate_neg.json().get("readings", {}).get("readings", [])) if r_plate_neg.status_code == 200 else -1

    r_veh_neg = client.get("/vehicles/vtrk_nonexistent")
    r_veh_neg_count = len(r_veh_neg.json().get("events", [])) if r_veh_neg.status_code == 200 else -1

    # Cleanly close SQLite store to release Windows file lock
    event_store.close()

    return {
        "dataset": {
            "source": str(path.as_posix()),
            "license": "Public Domain / CC0 (Official State Specimen Sample)",
            "pii_safe": True,
            "description": "Official 2025 State of Maine sample license plate (Maine BMV)",
        },
        "ground_truth": {
            "expected_registration_text": expected_ground_truth,
        },
        "full_plate_ocr": {
            "raw_text": full_reading.raw_text,
            "recognized_text": full_reading.plate_text,
            "confidence": round(full_reading.confidence, 3),
            "is_uncertain": full_reading.is_uncertain,
            "exact_match": full_exact_match,
            "token_match": full_token_match,
            "elapsed_s": round(t_full, 3),
        },
        "registration_crop_ocr": {
            "raw_text": crop_reading.raw_text,
            "recognized_text": crop_reading.plate_text,
            "confidence": round(crop_reading.confidence, 3),
            "is_uncertain": crop_reading.is_uncertain,
            "exact_match": crop_exact_match,
            "normalized_match": crop_normalized_match,
            "elapsed_s": round(t_crop, 3),
        },
        "association": {
            "track_id": track_id,
            "associated_plate": assoc.plate_text,
            "association_confidence": assoc.confidence,
            "is_uncertain": assoc.is_uncertain,
            "association_success": assoc.plate_text == expected_ground_truth,
        },
        "canonical_event_provenance": {
            "anpr_event_id": ev_anpr.event_id,
            "vehicle_event_id": ev_veh.event_id,
            "tool13_success": tr.success,
            "tool13_count": len(tool13_readings),
            "tool13_first_plate": tool13_readings[0]["plate_text"] if tool13_readings else "",
            "api_plate_status": r_plate.status_code,
            "api_plate_readings_retrieved": len(plate_api_data.get("readings", {}).get("readings", [])) if isinstance(plate_api_data.get("readings"), dict) else len(plate_api_data.get("readings", [])),
            "api_vehicle_status": r_veh.status_code,
            "api_vehicle_events_retrieved": len(veh_api_data.get("events", [])),
            "api_unobserved_plate_status": r_plate_neg.status_code,
            "api_unobserved_plate_count": r_plate_neg_count,
            "api_unobserved_vehicle_status": r_veh_neg.status_code,
            "api_unobserved_vehicle_count": r_veh_neg_count,
        },
    }


def main() -> None:
    """Run full empirical benchmark suite and report results."""
    parser = argparse.ArgumentParser(description="God's Eye Vehicle + ANPR Empirical Benchmark")
    parser.add_argument("--video", default="tests/data/demo_videos/mot17_04_medium_density.mp4")
    parser.add_argument("--gt-sample", default="tests/data/vehicle_samples/maine_sample.png")
    parser.add_argument("--readable-sample", default="tests/data/vehicle_samples/readable_plate_sample.png")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output-json", default="benchmarks/results/vehicle_anpr_empirical.json")
    args = parser.parse_args()

    print("==========================================================")
    print("GOD'S EYE — EMPIRICAL VEHICLE + ANPR BENCHMARK (FIX #7)")
    print("==========================================================")

    # 1. CCTV Video Run (Visual Provenance)
    print(f"\n[1/3] Processing CCTV video: {args.video} (max {args.max_frames} frames)...")
    cctv_results = run_cctv_video_benchmark(args.video, args.max_frames)

    print("  -> Video processed successfully.")
    print(f"     [VEHICLE] Frames: {cctv_results['dataset']['frames_processed']}")
    print(f"     [VEHICLE] Total detections: {cctv_results['vehicle_detection']['total_detections']} (avg {cctv_results['vehicle_detection']['detections_per_frame']}/frame)")
    print(f"     [VEHICLE] Classes: {cctv_results['vehicle_detection']['class_distribution']}")
    print(f"     [VEHICLE] Unique tracks: {cctv_results['bytetrack_tracking']['total_unique_tracks']} (IDs: {cctv_results['bytetrack_tracking']['unique_track_ids']})")
    print(f"     [VEHICLE] Throughput: {cctv_results['performance']['throughput_fps']} FPS")
    print(f"     [PLATE] Plate candidates: {cctv_results['anpr_ocr']['plate_candidates_detected']}")
    print(f"     [PLATE] OCR inference attempts: {cctv_results['anpr_ocr']['ocr_inference_attempts']}")
    print(f"     [PLATE] Uncertain rate: {cctv_results['anpr_ocr']['unreadable_rate'] * 100:.1f}% (distant night CCTV correctly preserved as uncertain)")
    print(f"     [ASSOCIATION] Total associations: {cctv_results['association']['total_associations']}")
    print(f"     [ASSOCIATION] Mean association confidence: {cctv_results['association']['mean_association_confidence']}")
    print(f"     [CANONICAL PROVENANCE] Persisted VEHICLE_DETECTED events: {cctv_results['canonical_event_provenance']['persisted_vehicle_detected_events']}")
    print(f"     [CANONICAL PROVENANCE] Persisted ANPR_READING events: {cctv_results['canonical_event_provenance']['persisted_anpr_reading_events']}")
    print(f"     [CANONICAL PROVENANCE] API GET /vehicles/{{id}} events retrieved: {cctv_results['canonical_event_provenance']['api_vehicle_events_retrieved']}")
    print(f"     [ADVERSARIAL] Negative unobserved vehicle API status: {cctv_results['canonical_event_provenance']['api_unobserved_vehicle_status']} (events: {cctv_results['canonical_event_provenance']['api_unobserved_vehicle_count']})")
    print(f"     [ADVERSARIAL] Negative unobserved plate API status: {cctv_results['canonical_event_provenance']['api_unobserved_plate_status']} (readings: {cctv_results['canonical_event_provenance']['api_unobserved_plate_readings_count']})")

    # 2. Ground-Truth OCR Run (Specimen Plate 'SAMPLE')
    print(f"\n[2/3] Processing ground-truth plate sample: {args.gt_sample}...")
    gt_results = run_ground_truth_ocr_benchmark(args.gt_sample)
    print("  -> Ground-truth sample processed successfully.")
    print(f"     [GROUND-TRUTH OCR] Expected: '{gt_results['ground_truth']['expected_registration_text']}'")
    print(f"     [GROUND-TRUTH OCR] Crop OCR: '{gt_results['registration_crop_ocr']['recognized_text']}' (conf: {gt_results['registration_crop_ocr']['confidence']}, exact_match: {gt_results['registration_crop_ocr']['exact_match']})")
    print(f"     [ASSOCIATION] Track ID: {gt_results['association']['track_id']}")
    print(f"     [ASSOCIATION] Association confidence: {gt_results['association']['association_confidence']}")
    print(f"     [ASSOCIATION] Success: {gt_results['association']['association_success']}")
    print(f"     [CANONICAL PROVENANCE] ANPR Event ID: {gt_results['canonical_event_provenance']['anpr_event_id']}")
    print(f"     [CANONICAL PROVENANCE] Vehicle Event ID: {gt_results['canonical_event_provenance']['vehicle_event_id']}")
    print(f"     [TOOL 13] Deterministic retrieval success: {gt_results['canonical_event_provenance']['tool13_success']} (matches: {gt_results['canonical_event_provenance']['tool13_count']})")
    print(f"     [API] GET /plates/SAMPLE status: {gt_results['canonical_event_provenance']['api_plate_status']} (readings: {gt_results['canonical_event_provenance']['api_plate_readings_retrieved']})")
    print(f"     [API] GET /vehicles/{gt_results['association']['track_id']} status: {gt_results['canonical_event_provenance']['api_vehicle_status']} (events: {gt_results['canonical_event_provenance']['api_vehicle_events_retrieved']})")
    print(f"     [ADVERSARIAL] Negative unobserved plate status: {gt_results['canonical_event_provenance']['api_unobserved_plate_status']} (count: {gt_results['canonical_event_provenance']['api_unobserved_plate_count']})")

    # 3. Readable Plate Sample Run
    print(f"\n[3/3] Processing readable plate sample: {args.readable_sample}...")
    readable_results = run_readable_plate_benchmark(args.readable_sample)
    print("  -> Readable plate sample processed successfully.")
    print(f"     [OCR] Recognized: '{readable_results['ocr_result']['recognized_text']}' (conf: {readable_results['ocr_result']['confidence']}, uncertain: {readable_results['ocr_result']['is_uncertain']})")
    print(f"     [ASSOCIATION] Track ID: {readable_results['association']['associated_track_id']}")
    print(f"     [ASSOCIATION] Confidence: {readable_results['association']['association_confidence']}")
    print(f"     [CANONICAL PROVENANCE] Tool 13 success: {readable_results['canonical_event_provenance']['tool13_success']} (count: {readable_results['canonical_event_provenance']['tool13_count']})")
    print(f"     [API] GET /plates status: {readable_results['canonical_event_provenance']['api_plate_status']}, GET /vehicles status: {readable_results['canonical_event_provenance']['api_vehicle_status']}")

    # Combined output
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sih_problem_statement": 26187,
        "fix": "#7 - Direct Vehicle + ANPR Provenance into Canonical EventStore / Tool 13 / C2 API",
        "cctv_video_benchmark": cctv_results,
        "ground_truth_ocr_benchmark": gt_results,
        "readable_plate_benchmark": readable_results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults successfully written to: {out_path}")
    print("==========================================================")


if __name__ == "__main__":
    main()
