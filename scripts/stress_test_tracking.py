"""Phase 1C -- Tracking Stress Test.

Two-mode validation:
1. SYNTHETIC: Programmatic multi-person simulation (5-8 people, controlled paths)
   - Precise ID switch measurement
   - Controlled LOST/DEAD transitions
   - Crossing paths, entries, exits
2. WEBCAM: Real-world multi-person detection+tracking (if available)

Reports: FPS, active/max tracks, unique IDs, lifetimes, state transitions,
ID switch approximation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import configure_logging
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker


# ─── Synthetic Person Simulation ─────────────────────────────────────────────


@dataclass
class SimPerson:
    """A simulated person with linear motion."""
    pid: int
    x: float
    y: float
    w: float
    h: float
    vx: float           # pixels/frame
    vy: float
    enter_frame: int
    exit_frame: int
    confidence: float


def build_crowd_scenario(
    total_frames: int = 300,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> list[SimPerson]:
    """Create a multi-person scenario with varied motion patterns."""
    people = [
        # Person 1: walks left-to-right, full duration
        SimPerson(1, 50, 400, 80, 200, 3.0, 0.0, 0, total_frames, 0.92),
        # Person 2: walks right-to-left, full duration
        SimPerson(2, 1800, 350, 70, 190, -2.5, 0.0, 0, total_frames, 0.88),
        # Person 3: enters at frame 30, exits at frame 200
        SimPerson(3, 960, 50, 75, 195, 0.0, 2.0, 30, 200, 0.85),
        # Person 4: stationary person (standing)
        SimPerson(4, 500, 300, 85, 210, 0.0, 0.0, 0, total_frames, 0.95),
        # Person 5: enters late, fast walk
        SimPerson(5, 100, 600, 65, 180, 5.0, -0.5, 100, total_frames, 0.78),
        # Person 6: crosses path of Person 1 (potential ID switch)
        SimPerson(6, 800, 380, 80, 200, -3.0, 0.5, 50, 250, 0.82),
        # Person 7: brief appearance (enters and leaves quickly)
        SimPerson(7, 1400, 500, 70, 185, -8.0, 0.0, 80, 130, 0.90),
        # Person 8: appears in second half
        SimPerson(8, 300, 700, 75, 195, 2.0, -1.0, 150, total_frames, 0.86),
    ]
    return people


def get_frame_detections(
    people: list[SimPerson],
    frame_id: int,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> list[Detection]:
    """Generate detections for all visible people at a given frame."""
    dets: list[Detection] = []
    for p in people:
        if frame_id < p.enter_frame or frame_id >= p.exit_frame:
            continue
        elapsed = frame_id - p.enter_frame
        cx = p.x + p.vx * elapsed
        cy = p.y + p.vy * elapsed
        x1 = cx - p.w / 2
        y1 = cy - p.h / 2
        x2 = cx + p.w / 2
        y2 = cy + p.h / 2
        # Clip to frame
        x1 = max(0.0, min(x1, frame_w - 1.0))
        y1 = max(0.0, min(y1, frame_h - 1.0))
        x2 = max(1.0, min(x2, float(frame_w)))
        y2 = max(1.0, min(y2, float(frame_h)))
        if x2 - x1 < 10 or y2 - y1 < 20:
            continue  # Too small / out of frame
        # Add slight noise to simulate real detection jitter
        noise = np.random.normal(0, 1.5, 4)
        dets.append(Detection(
            detection_id=str(uuid.uuid4()),
            camera_id="synth-stress",
            frame_id=frame_id,
            timestamp_ns=frame_id * 33_000_000,
            bbox=BoundingBox(
                x1=float(x1 + noise[0]),
                y1=float(y1 + noise[1]),
                x2=float(x2 + noise[2]),
                y2=float(y2 + noise[3]),
            ),
            confidence=p.confidence + float(np.random.normal(0, 0.02)),
            class_label="person",
            source_resolution=(frame_w, frame_h),
        ))
    return dets


# ─── Stress Test: Synthetic ──────────────────────────────────────────────────


def run_synthetic_stress() -> None:
    """Run multi-person stress test with synthetic detections."""
    TOTAL_FRAMES = 300
    settings = Settings(
        detection_confidence_threshold=0.25,
        track_lost_timeout=30,
        track_dead_timeout=120,
    )
    tracker = ByteTrackTracker(settings)
    people = build_crowd_scenario(TOTAL_FRAMES)

    print("\n" + "=" * 70)
    print("  SYNTHETIC STRESS TEST -- 8 simulated people, 300 frames")
    print("=" * 70)
    print(f"  Simulated people:     {len(people)}")
    print(f"  Track lost timeout:   {settings.track_lost_timeout} frames")
    print(f"  Track dead timeout:   {settings.track_dead_timeout} frames")
    print()

    active_counts: list[int] = []
    lost_counts: list[int] = []
    total_counts: list[int] = []
    track_ids_per_frame: list[set[str]] = []
    all_track_ids: set[str] = set()
    latencies: list[float] = []
    lost_transitions = 0
    dead_transitions = 0
    prev_active_ids: set[str] = set()
    prev_states: dict[str, TrackState] = {}

    for fid in range(1, TOTAL_FRAMES + 1):
        dets = get_frame_detections(people, fid)
        t0 = time.perf_counter()
        tracks = tracker.update(dets, frame_id=fid, camera_id="synth-stress")
        latency_us = (time.perf_counter() - t0) * 1_000_000
        latencies.append(latency_us)

        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        lost = [t for t in tracks if t.state == TrackState.LOST]
        active_counts.append(len(active))
        lost_counts.append(len(lost))
        total_counts.append(len(tracks))

        current_ids = {t.track_id for t in active}
        track_ids_per_frame.append(current_ids)
        all_track_ids = all_track_ids | current_ids

        # Track state transitions
        current_states: dict[str, TrackState] = {t.track_id: t.state for t in tracks}
        for tid, state in current_states.items():
            prev = prev_states.get(tid)
            if prev == TrackState.ACTIVE and state == TrackState.LOST:
                lost_transitions += 1
        # Dead transitions = tracks that disappeared from prev_states
        for tid in prev_states:
            if tid not in current_states:
                dead_transitions += 1
        prev_states = current_states
        prev_active_ids = current_ids

    # ── Compute Metrics ──────────────────────────────────────────────────
    avg_active = sum(active_counts) / len(active_counts)
    max_active = max(active_counts)
    avg_lost = sum(lost_counts) / len(lost_counts)
    unique_ids = len(all_track_ids)
    avg_latency_us = sum(latencies) / len(latencies)
    tracker_fps = 1_000_000 / avg_latency_us if avg_latency_us > 0 else 0

    # Track lifetime analysis
    # Approximate: count consecutive frames each ID appears as ACTIVE
    id_frame_counts: dict[str, int] = {}
    for frame_ids in track_ids_per_frame:
        for tid in frame_ids:
            id_frame_counts[tid] = id_frame_counts.get(tid, 0) + 1
    lifetimes = list(id_frame_counts.values())
    avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0
    max_lifetime = max(lifetimes) if lifetimes else 0
    min_lifetime = min(lifetimes) if lifetimes else 0

    # ID switch approximation:
    # Ground truth: 8 people → should produce 8 unique IDs
    # Extra IDs = approximate ID switches or fragmentations
    expected_ids = len(people)
    id_switches_approx = max(0, unique_ids - expected_ids)

    # Lifetime distribution buckets
    short = sum(1 for l in lifetimes if l <= 10)
    medium = sum(1 for l in lifetimes if 10 < l <= 50)
    long_ = sum(1 for l in lifetimes if 50 < l <= 200)
    full = sum(1 for l in lifetimes if l > 200)

    # ── Print Results ────────────────────────────────────────────────────
    print("-" * 70)
    print("  SYNTHETIC STRESS TEST RESULTS")
    print("-" * 70)
    print(f"  {'Total Frames':<40} {TOTAL_FRAMES:>15}")
    print(f"  {'Tracker-only FPS':<40} {tracker_fps:>15,.0f}")
    print(f"  {'Avg Tracker Latency (us)':<40} {avg_latency_us:>15.1f}")
    print()
    print(f"  {'Ground Truth People':<40} {expected_ids:>15}")
    print(f"  {'Unique Track IDs Created':<40} {unique_ids:>15}")
    print(f"  {'Approx ID Switches (excess IDs)':<40} {id_switches_approx:>15}")
    print()
    print(f"  {'Avg Active Tracks/Frame':<40} {avg_active:>15.2f}")
    print(f"  {'Max Active Tracks':<40} {max_active:>15}")
    print(f"  {'Avg Lost Tracks/Frame':<40} {avg_lost:>15.2f}")
    print()
    print(f"  {'ACTIVE->LOST Transitions':<40} {lost_transitions:>15}")
    print(f"  {'DEAD Evictions (track removed)':<40} {dead_transitions:>15}")
    print()
    print(f"  {'Avg Track Lifetime (frames)':<40} {avg_lifetime:>15.1f}")
    print(f"  {'Max Track Lifetime':<40} {max_lifetime:>15}")
    print(f"  {'Min Track Lifetime':<40} {min_lifetime:>15}")
    print()
    print("  Track Lifetime Distribution:")
    print(f"    {'Short  (1-10 frames)':<38} {short:>15}")
    print(f"    {'Medium (11-50 frames)':<38} {medium:>15}")
    print(f"    {'Long   (51-200 frames)':<38} {long_:>15}")
    print(f"    {'Full   (200+ frames)':<38} {full:>15}")
    print("-" * 70)
    print()


# ─── Stress Test: Webcam ─────────────────────────────────────────────────────


def run_webcam_stress() -> None:
    """Run real-world detection+tracking stress test on webcam."""
    settings = Settings.from_env()
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[SKIP] No webcam available for real-world stress test.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (w, h)

    DURATION = 10.0
    print("=" * 70)
    print("  WEBCAM STRESS TEST -- Detection + Tracking")
    print("=" * 70)
    print(f"  Source:     webcam ({w}x{h})")
    print(f"  Duration:   {DURATION}s")
    print(f"  Device:     {detector.device}")
    print("  For best results, have 2+ people visible.\n")

    frame_id = 0
    det_latencies: list[float] = []
    trk_latencies: list[float] = []
    active_counts: list[int] = []
    unique_ids: set[str] = set()

    deadline = time.perf_counter() + DURATION
    while time.perf_counter() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        packet = FramePacket(
            camera_id="webcam-stress",
            frame_id=frame_id,
            timestamp_ns=time.time_ns(),
            frame=frame,
            resolution=resolution,
        )

        t0 = time.perf_counter()
        dets = detector.detect(packet)
        t1 = time.perf_counter()
        tracks = tracker.update(dets, frame_id=frame_id, camera_id="webcam-stress")
        t2 = time.perf_counter()

        det_latencies.append((t1 - t0) * 1000)
        trk_latencies.append((t2 - t1) * 1000)

        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        active_counts.append(len(active))
        for t in active:
            unique_ids.add(t.track_id)

    cap.release()

    avg_det = sum(det_latencies) / len(det_latencies) if det_latencies else 0
    avg_trk = sum(trk_latencies) / len(trk_latencies) if trk_latencies else 0
    avg_total = avg_det + avg_trk
    det_fps = 1000 / avg_det if avg_det > 0 else 0
    trk_fps = 1000 / avg_trk if avg_trk > 0 else 0
    combined_fps = 1000 / avg_total if avg_total > 0 else 0

    print("-" * 70)
    print("  WEBCAM STRESS TEST RESULTS")
    print("-" * 70)
    print(f"  {'Total Frames':<40} {frame_id:>15}")
    print()
    print(f"  {'Detection Avg Latency (ms)':<40} {avg_det:>15.1f}")
    print(f"  {'Tracking Avg Latency (ms)':<40} {avg_trk:>15.1f}")
    print(f"  {'Combined Avg Latency (ms)':<40} {avg_total:>15.1f}")
    print()
    print(f"  {'Detection FPS':<40} {det_fps:>15.1f}")
    print(f"  {'Tracking FPS':<40} {trk_fps:>15,.0f}")
    print(f"  {'Combined Throughput FPS':<40} {combined_fps:>15.1f}")
    print()
    print(f"  {'Avg Active Tracks/Frame':<40} ", end="")
    avg_a = sum(active_counts) / len(active_counts) if active_counts else 0
    print(f"{avg_a:>14.2f}")
    print(f"  {'Max Active Tracks':<40} {max(active_counts) if active_counts else 0:>15}")
    print(f"  {'Unique Track IDs':<40} {len(unique_ids):>15}")
    print("-" * 70)
    print()


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    configure_logging("WARNING")  # Suppress info logs during stress test
    run_synthetic_stress()
    run_webcam_stress()
    print("  STRESS TEST COMPLETE")
    print()


if __name__ == "__main__":
    main()
