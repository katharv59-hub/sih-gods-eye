"""Phase 1C -- Tracking Validation Script.

Validates the tracking subsystem with real inference:
1. Captures webcam frames
2. Runs YOLOv8n detection
3. Feeds detections to ByteTrack
4. Measures: tracking FPS, active tracks, track lifetimes, ID stability
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
from gods_eye.schemas.track import Track, TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

CAPTURE_DURATION_S = 10.0
OUTPUT_DIR = "tests/data/tracking_samples"


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw tracked bounding boxes with IDs on a frame."""
    annotated = frame.copy()
    colors = {
        TrackState.ACTIVE: (0, 255, 0),
        TrackState.LOST: (0, 165, 255),
    }
    for t in tracks:
        if t.state == TrackState.LOST:
            continue  # Don't draw lost tracks
        color = colors.get(t.state, (255, 255, 255))
        x1, y1 = int(t.bbox.x1), int(t.bbox.y1)
        x2, y2 = int(t.bbox.x2), int(t.bbox.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{t.track_id} [{t.state.value}]"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def main() -> None:
    configure_logging("INFO")
    settings = Settings.from_env()

    # Init detector + tracker
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No webcam available.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (w, h)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  TRACKING VALIDATION -- ByteTrack")
    print("=" * 60)
    print(f"  Source:     webcam ({w}x{h})")
    print(f"  Duration:   {CAPTURE_DURATION_S}s")
    print(f"  Detector:   {detector.model_name} on {detector.device}")
    print("  Stand in front of the camera now...\n")

    frame_id = 0
    active_counts: list[int] = []
    lost_counts: list[int] = []
    total_counts: list[int] = []
    latencies: list[float] = []
    track_ids_seen: set[str] = set()
    track_lifetimes: dict[str, int] = {}
    samples_saved = 0

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
        tracks = tracker.update(dets, frame_id=frame_id, camera_id="webcam-val")
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        lost = [t for t in tracks if t.state == TrackState.LOST]
        active_counts.append(len(active))
        lost_counts.append(len(lost))
        total_counts.append(len(tracks))

        for t in active:
            track_ids_seen.add(t.track_id)
            lifetime = t.last_frame_id - t.first_frame_id + 1
            track_lifetimes[t.track_id] = lifetime

        # Save annotated samples
        if active and samples_saved < 5:
            annotated = draw_tracks(frame, tracks)
            path = os.path.join(OUTPUT_DIR, f"track_sample_{samples_saved + 1}.jpg")
            cv2.imwrite(path, annotated)
            samples_saved += 1

    cap.release()

    # Compute metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    tracking_fps = 1000 / avg_latency if avg_latency > 0 else 0
    avg_active = sum(active_counts) / len(active_counts) if active_counts else 0
    max_active = max(active_counts) if active_counts else 0
    unique_tracks = len(track_ids_seen)
    lifetimes = list(track_lifetimes.values())
    avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0
    max_lifetime = max(lifetimes) if lifetimes else 0

    print("-" * 60)
    print("  TRACKING RESULTS")
    print("-" * 60)
    print(f"  {'Total Frames':<35} {frame_id:>15}")
    print(f"  {'Tracking FPS (det+track)':<35} {tracking_fps:>15.1f}")
    print(f"  {'Avg Latency (ms)':<35} {avg_latency:>15.1f}")
    print()
    print(f"  {'Avg Active Tracks/Frame':<35} {avg_active:>15.2f}")
    print(f"  {'Max Active Tracks':<35} {max_active:>15}")
    print(f"  {'Unique Track IDs Created':<35} {unique_tracks:>15}")
    print()
    print(f"  {'Avg Track Lifetime (frames)':<35} {avg_lifetime:>15.1f}")
    print(f"  {'Max Track Lifetime (frames)':<35} {max_lifetime:>15}")
    print()

    # ID stability analysis
    stable_tracks = sum(1 for l in lifetimes if l > 10)
    print(f"  {'Tracks > 10 frames (stable)':<35} {stable_tracks:>15}")
    print(f"  {'Tracks <= 10 frames (short)':<35} {unique_tracks - stable_tracks:>15}")

    if samples_saved > 0:
        print(f"\n  Annotated samples: {OUTPUT_DIR}/ ({samples_saved} images)")

    print()
    print("-" * 60)
    if unique_tracks > 0:
        print("  OVERALL: PASS -- Tracking subsystem validated")
    else:
        print("  OVERALL: WARN -- No tracks created. Ensure a person is visible.")
    print()

    # Known ByteTrack limitations
    print("  KNOWN LIMITATIONS:")
    print("  - ID switches possible on heavy occlusion or fast crossing")
    print("  - No appearance features (fixed in Phase 2 with Re-ID)")
    print("  - Lost tracks use last-known bbox (no motion prediction)")
    print()


if __name__ == "__main__":
    main()
