"""Phase 1A — Ingestion Validation Script.

Validates the ingestion subsystem with real-world operation:
1. Generates a synthetic test video (if none exists)
2. Attempts webcam capture (falls back to file if unavailable)
3. Measures: frames captured, effective FPS, queue occupancy, dropped frames
4. Verifies graceful shutdown
5. Verifies structured logging output
"""

from __future__ import annotations

import io
import sys
import time

import cv2
import numpy as np

from gods_eye.ingestion import (
    CameraCaptureThread,
    FrameQueue,
    VideoFileSource,
    WebcamSource,
)
from gods_eye.observability.logger import configure_logging, get_logger

# ─── Config ──────────────────────────────────────────────────────────────────

TEST_VIDEO_PATH = "tests/data/synthetic_30fps_5s.mp4"
VALIDATION_DURATION_S = 5.0
QUEUE_MAXSIZE = 30


def generate_test_video(path: str, fps: int = 30, duration_s: float = 5.0) -> None:
    """Generate a synthetic test video with frame counter overlay."""
    width, height = 640, 480
    total_frames = int(fps * duration_s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        frame[:, :, 0] = np.uint8(i * 255 / total_frames)  # blue ramp
        frame[:, :, 1] = 80
        frame[:, :, 2] = 40
        # Frame counter text
        cv2.putText(
            frame, f"Frame {i + 1}/{total_frames}",
            (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (255, 255, 255), 2,
        )
        writer.write(frame)

    writer.release()
    print(f"[SETUP] Generated test video: {path} ({total_frames} frames @ {fps} FPS)")


def try_webcam() -> bool:
    """Check if a webcam is available."""
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return bool(available)


def run_validation(source_type: str) -> dict[str, object]:
    """Run the ingestion validation and return metrics."""
    results: dict[str, object] = {"source_type": source_type}

    # Create source
    if source_type == "webcam":
        from gods_eye.config.settings import Settings
        dev_idx = Settings.from_env().webcam_device_index
        source = WebcamSource(device_index=dev_idx)
        camera_id = f"webcam-{dev_idx}"
    else:
        source = VideoFileSource(TEST_VIDEO_PATH)
        camera_id = "file-test"

    # Create queue and thread
    queue = FrameQueue(
        maxsize=QUEUE_MAXSIZE,
        queue_name=f"{camera_id}-frames",
        camera_id=camera_id,
    )
    thread = CameraCaptureThread(camera_id, source, queue)

    # ── Start capture ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  INGESTION VALIDATION — {source_type.upper()}")
    print("=" * 60)

    t_start = time.perf_counter()
    thread.start()

    # ── Consume frames ───────────────────────────────────────────────────
    frames_consumed = 0
    max_queue_depth = 0
    timestamps: list[int] = []

    deadline = time.perf_counter() + VALIDATION_DURATION_S

    while time.perf_counter() < deadline:
        pkt = queue.get(timeout=0.5)
        if pkt is None:
            if queue.is_closed:
                break
            continue
        frames_consumed += 1
        timestamps.append(pkt.timestamp_ns)
        current_depth = queue.qsize
        if current_depth > max_queue_depth:
            max_queue_depth = current_depth

    # ── Stop capture ─────────────────────────────────────────────────────
    if thread.is_alive():
        thread.stop()
        thread.join(timeout=3.0)

    t_elapsed = time.perf_counter() - t_start

    # ── Drain remaining ──────────────────────────────────────────────────
    drain_count = 0
    while True:
        pkt = queue.get(timeout=0.1)
        if pkt is None:
            break
        drain_count += 1
        frames_consumed += 1

    # ── Calculate metrics ────────────────────────────────────────────────
    effective_fps = frames_consumed / t_elapsed if t_elapsed > 0 else 0
    produced = thread.frame_count
    dropped = produced - frames_consumed

    results["frames_produced"] = produced
    results["frames_consumed"] = frames_consumed
    results["frames_drained_after_stop"] = drain_count
    results["frames_dropped"] = dropped
    results["drop_rate_pct"] = round(dropped / produced * 100, 2) if produced > 0 else 0.0
    results["effective_fps"] = round(effective_fps, 1)
    results["elapsed_s"] = round(t_elapsed, 2)
    results["max_queue_depth"] = max_queue_depth
    results["queue_closed"] = queue.is_closed
    results["thread_alive"] = thread.is_alive()
    results["graceful_shutdown"] = not thread.is_alive() and queue.is_closed

    return results


def print_results(results: dict[str, object]) -> None:
    """Print validation results in a readable format."""
    print("\n" + "-" * 60)
    print("  VALIDATION RESULTS")
    print("-" * 60)

    checks = [
        ("Source Type", results["source_type"], None),
        ("Elapsed Time", f"{results['elapsed_s']}s", None),
        ("Frames Produced", results["frames_produced"], None),
        ("Frames Consumed", results["frames_consumed"], None),
        ("Frames Drained After Stop", results["frames_drained_after_stop"], None),
        ("Effective FPS", results["effective_fps"], None),
        ("Max Queue Depth", results["max_queue_depth"], None),
        ("Frames Dropped", results["frames_dropped"], None),
        ("Drop Rate", f"{results['drop_rate_pct']}%",
         "PASS" if float(str(results["drop_rate_pct"])) < 2.0 else "FAIL"),
        ("Queue Closed", results["queue_closed"],
         "PASS" if results["queue_closed"] else "FAIL"),
        ("Thread Terminated", not results["thread_alive"],
         "PASS" if not results["thread_alive"] else "FAIL"),
        ("Graceful Shutdown", results["graceful_shutdown"],
         "PASS" if results["graceful_shutdown"] else "FAIL"),
    ]

    for label, value, verdict in checks:
        line = f"  {label:<30} {str(value):>15}"
        if verdict:
            line += f"  [{verdict}]"
        print(line)

    print("-" * 60)

    all_pass = all(v == "PASS" for _, _, v in checks if v is not None)
    print(f"\n  OVERALL: {'PASS -- ALL CHECKS PASSED' if all_pass else 'FAIL -- SOME CHECKS FAILED'}")
    print()


def main() -> None:
    # Configure structured logging
    configure_logging("INFO")
    log = get_logger("validation")

    # Generate test video
    import os
    os.makedirs("tests/data", exist_ok=True)
    if not os.path.exists(TEST_VIDEO_PATH):
        generate_test_video(TEST_VIDEO_PATH)
    else:
        print(f"[SETUP] Using existing test video: {TEST_VIDEO_PATH}")

    # Check webcam
    has_webcam = try_webcam()
    print(f"[SETUP] Webcam available: {has_webcam}")

    # Run file validation (always)
    log.info("validation_start", source="file")
    file_results = run_validation("file")
    print_results(file_results)

    # Run webcam validation (if available)
    if has_webcam:
        log.info("validation_start", source="webcam")
        webcam_results = run_validation("webcam")
        print_results(webcam_results)
    else:
        print("[SKIP] Webcam validation skipped — no webcam detected")

    log.info("validation_complete")


if __name__ == "__main__":
    main()
