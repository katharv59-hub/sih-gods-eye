"""Phase 1D -- End-to-end pipeline validation.

Tests with both webcam and video file sources.
Reports: E2E FPS, latencies, queue stats, drops.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.source import VideoFileSource, WebcamSource
from gods_eye.observability.logger import configure_logging
from gods_eye.pipeline import Pipeline, TrackingResult
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

TEST_VIDEO = "tests/data/synthetic_30fps_5s.mp4"


def run_pipeline_test(source_name: str, pipeline: Pipeline, duration: float) -> None:
    results: list[TrackingResult] = []
    lock = threading.Lock()
    latencies: list[float] = []

    original_cb = pipeline._output._callback

    def instrumented_cb(r: IdentityResult) -> None:
        with lock:
            results.append(r)
            lat = (time.time_ns() - r.tracking.packet.timestamp_ns) / 1_000_000
            latencies.append(lat)

    pipeline._output._callback = instrumented_cb

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE VALIDATION -- {source_name}")
    print("=" * 60)

    t0 = time.perf_counter()
    pipeline.start()

    if duration > 0:
        time.sleep(duration)
        pipeline.stop(timeout=5.0)
    else:
        # Wait for source to exhaust
        while pipeline.is_alive:
            time.sleep(0.1)
        time.sleep(0.5)
        pipeline.stop(timeout=5.0)

    elapsed = time.perf_counter() - t0

    with lock:
        n = len(results)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    stats = pipeline.stats
    e2e_fps = n / elapsed if elapsed > 0 else 0

    active_track_counts = []
    with lock:
        for r in results:
            active_track_counts.append(
                sum(1 for t in r.tracking.tracks if t.state == TrackState.ACTIVE)
            )

    avg_tracks = (
        sum(active_track_counts) / len(active_track_counts)
        if active_track_counts
        else 0
    )

    print(f"\n{'-' * 60}")
    print("  RESULTS")
    print("-" * 60)
    print(f"  {'Elapsed Time':<35} {elapsed:>12.2f}s")
    print(f"  {'Frames Captured':<35} {stats['captured']:>12}")
    print(f"  {'Frames Detected':<35} {stats['detected']:>12}")
    print(f"  {'Frames Tracked':<35} {stats['tracked']:>12}")
    print(f"  {'Frames Delivered':<35} {stats['delivered']:>12}")
    print(f"  {'End-to-End FPS':<35} {e2e_fps:>12.1f}")
    print(f"  {'Avg E2E Latency (ms)':<35} {avg_lat:>12.1f}")
    print(f"  {'P95 E2E Latency (ms)':<35} {p95_lat:>12.1f}")
    print(f"  {'Avg Active Tracks':<35} {avg_tracks:>12.2f}")
    print(f"  {'Det Queue Drops':<35} {pipeline._det_q.drop_count:>12}")
    print(f"  {'Track Queue Drops':<35} {pipeline._track_q.drop_count:>12}")
    print("-" * 60)

    ok = stats["delivered"] > 0
    print(f"\n  {'PASS' if ok else 'FAIL'} -- {n} frames delivered E2E\n")


def main() -> None:
    configure_logging("WARNING")
    settings = Settings.from_env()

    print("Loading detector...")
    detector = YOLODetector(settings)
    detector.warmup()

    # --- Video file test ---
    if os.path.exists(TEST_VIDEO):
        tracker1 = ByteTrackTracker(settings)
        source1 = VideoFileSource(TEST_VIDEO)
        p1 = Pipeline(
            "file-test", source1, detector, tracker1, settings,
            on_result=lambda r: None,
        )
        run_pipeline_test("VIDEO FILE", p1, duration=0)
    else:
        print(f"[SKIP] {TEST_VIDEO} not found")

    # --- Webcam test ---
    cap = cv2.VideoCapture(0)
    has_webcam = cap.isOpened()
    cap.release()

    if has_webcam:
        tracker2 = ByteTrackTracker(settings)
        source2 = WebcamSource(0)
        p2 = Pipeline(
            "webcam-test", source2, detector, tracker2, settings,
            on_result=lambda r: None,
        )
        run_pipeline_test("WEBCAM (10s)", p2, duration=10.0)
    else:
        print("[SKIP] No webcam")

    print("  PIPELINE VALIDATION COMPLETE\n")


if __name__ == "__main__":
    main()
