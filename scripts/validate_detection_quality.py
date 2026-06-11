"""Phase 1B — Real-Person Detection Validation.

Validates detection quality on real human subjects:
1. Captures from webcam (or fallback video)
2. Runs YOLODetector on every frame
3. Reports detection quality metrics and sample outputs
4. Saves annotated sample frames for visual inspection
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import configure_logging
from gods_eye.schemas.detection import Detection

CAPTURE_DURATION_S = 8.0
OUTPUT_DIR = "tests/data/detection_samples"


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    annotated = frame.copy()
    for d in dets:
        x1, y1 = int(d.bbox.x1), int(d.bbox.y1)
        x2, y2 = int(d.bbox.x2), int(d.bbox.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_label} {d.confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def main() -> None:
    configure_logging("INFO")

    settings = Settings.from_env()
    detector = YOLODetector(settings)
    detector.warmup()

    # Try webcam
    cap = cv2.VideoCapture(0)
    source_name = "webcam"
    if not cap.isOpened():
        print("[WARN] No webcam. Cannot validate with real people.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (w, h)

    print("\n" + "=" * 60)
    print("  REAL-PERSON DETECTION VALIDATION")
    print("=" * 60)
    print(f"  Source:     {source_name} ({w}x{h})")
    print(f"  Device:     {detector.device}")
    print(f"  Duration:   {CAPTURE_DURATION_S}s")
    print(f"  Threshold:  {settings.detection_confidence_threshold}")
    print("  Stand in front of the camera now...")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_id = 0
    all_detections: list[list[Detection]] = []
    latencies: list[float] = []
    sample_saved = 0

    deadline = time.perf_counter() + CAPTURE_DURATION_S

    while time.perf_counter() < deadline:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        packet = FramePacket(
            camera_id="webcam-val",
            frame_id=frame_id,
            timestamp_ns=time.time_ns(),
            frame=frame,
            resolution=resolution,
        )

        t0 = time.perf_counter()
        dets = detector.detect(packet)
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        all_detections.append(dets)

        # Save up to 5 annotated sample frames (only those with detections)
        if dets and sample_saved < 5:
            annotated = draw_detections(frame, dets)
            path = os.path.join(OUTPUT_DIR, f"sample_{sample_saved + 1}.jpg")
            cv2.imwrite(path, annotated)
            sample_saved += 1

    cap.release()

    # ── Compute Metrics ──────────────────────────────────────────────────
    total_frames = len(all_detections)
    dets_per_frame = [len(d) for d in all_detections]
    total_dets = sum(dets_per_frame)
    frames_with_dets = sum(1 for d in dets_per_frame if d > 0)
    max_dets = max(dets_per_frame) if dets_per_frame else 0
    avg_dets = total_dets / total_frames if total_frames > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    det_fps = 1000 / avg_latency if avg_latency > 0 else 0

    # Confidence distribution
    all_confs = [d.confidence for frame_dets in all_detections for d in frame_dets]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
    min_conf = min(all_confs) if all_confs else 0
    max_conf = max(all_confs) if all_confs else 0

    # ── Schema Validation ────────────────────────────────────────────────
    schema_ok = True
    schema_errors: list[str] = []

    for frame_dets in all_detections:
        for d in frame_dets:
            if d.class_label != "person":
                schema_ok = False
                schema_errors.append(f"Non-person class: {d.class_label}")
            if not (0.0 <= d.confidence <= 1.0):
                schema_ok = False
                schema_errors.append(f"Invalid confidence: {d.confidence}")
            if d.camera_id != "webcam-val":
                schema_ok = False
                schema_errors.append(f"Wrong camera_id: {d.camera_id}")
            if d.source_resolution != resolution:
                schema_ok = False
                schema_errors.append(f"Wrong resolution: {d.source_resolution}")
            if d.bbox.x1 >= d.bbox.x2 or d.bbox.y1 >= d.bbox.y2:
                schema_ok = False
                schema_errors.append(f"Invalid bbox: {d.bbox}")

    # ── Print Results ────────────────────────────────────────────────────
    print("-" * 60)
    print("  DETECTION QUALITY RESULTS")
    print("-" * 60)
    print(f"  {'Total Frames':<35} {total_frames:>15}")
    print(f"  {'Frames With Detections':<35} {frames_with_dets:>15}")
    print(f"  {'Total Detections':<35} {total_dets:>15}")
    print(f"  {'Avg Detections/Frame':<35} {avg_dets:>15.2f}")
    print(f"  {'Max Detections in a Frame':<35} {max_dets:>15}")
    print(f"  {'Detection FPS':<35} {det_fps:>15.1f}")
    print(f"  {'Avg Latency (ms)':<35} {avg_latency:>15.1f}")
    print()
    print(f"  {'Avg Confidence':<35} {avg_conf:>15.3f}")
    print(f"  {'Min Confidence':<35} {min_conf:>15.3f}")
    print(f"  {'Max Confidence':<35} {max_conf:>15.3f}")
    print(f"  {'Confidence Threshold':<35} {settings.detection_confidence_threshold:>15.2f}")
    print()

    # Schema checks
    print(f"  {'Schema Correctness':<35} {'PASS' if schema_ok else 'FAIL':>15}")
    print(f"  {'Person-Class Filter':<35} {'PASS' if schema_ok else 'FAIL':>15}")
    print(f"  {'All confs >= threshold':<35} ", end="")
    threshold_ok = all(c >= settings.detection_confidence_threshold for c in all_confs) if all_confs else True
    print(f"{'PASS' if threshold_ok else 'FAIL':>14}")

    if schema_errors:
        print(f"\n  Schema Errors:")
        for err in schema_errors[:5]:
            print(f"    - {err}")

    # ── Sample Detection Outputs ─────────────────────────────────────────
    print()
    print("-" * 60)
    print("  SAMPLE DETECTION OUTPUTS (first 5 frames with detections)")
    print("-" * 60)

    samples_shown = 0
    for i, frame_dets in enumerate(all_detections):
        if frame_dets and samples_shown < 5:
            samples_shown += 1
            print(f"\n  Frame {i + 1} ({len(frame_dets)} detection(s)):")
            for d in frame_dets:
                print(f"    - {d.class_label} conf={d.confidence:.3f} "
                      f"bbox=[{d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f}] "
                      f"area={d.bbox.area:.0f}px id={d.detection_id[:8]}...")

    if sample_saved > 0:
        print(f"\n  Annotated samples saved to: {OUTPUT_DIR}/ ({sample_saved} images)")

    print()
    print("-" * 60)
    all_pass = schema_ok and threshold_ok and total_dets > 0
    if total_dets == 0:
        print("  OVERALL: WARN -- No detections. Ensure a person is visible.")
    elif all_pass:
        print("  OVERALL: PASS -- Detection quality validated")
    else:
        print("  OVERALL: FAIL -- See errors above")
    print()


if __name__ == "__main__":
    main()
