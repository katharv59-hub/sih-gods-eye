"""CPU-only mode benchmark — Phase 1 gate criterion.

Runs the full detection → tracking pipeline with GODS_EYE_CPU_ONLY=1
on a 720p synthetic video source for a configurable duration.

Gate criterion: ≥ 6 FPS sustained at 720p.

Usage:
    set GODS_EYE_CPU_ONLY=1
    python -m benchmarks.benchmark_cpu_mode [--duration-s 300]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force CPU-only mode
os.environ["GODS_EYE_CPU_ONLY"] = "1"

from gods_eye.config.settings import Settings  # noqa: E402
from gods_eye.detection.yolo_detector import YOLODetector  # noqa: E402
from gods_eye.ingestion.frame_packet import FramePacket  # noqa: E402
from gods_eye.observability.logger import get_logger  # noqa: E402
from gods_eye.schemas.detection import Detection  # noqa: E402
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker  # noqa: E402

_log = get_logger("benchmark.cpu_mode")

# 720p resolution
_WIDTH = 1280
_HEIGHT = 720


def _generate_synthetic_frame(frame_id: int) -> np.ndarray:
    """Generate a 720p synthetic frame with random noise.

    Uses random noise to simulate worst-case detection load —
    the detector must still process every pixel.
    """
    rng = np.random.default_rng(seed=frame_id % 1000)
    return rng.integers(0, 255, size=(_HEIGHT, _WIDTH, 3), dtype=np.uint8)


def run_benchmark(duration_s: int = 300) -> dict[str, object]:
    """Run CPU-only benchmark for the specified duration.

    Returns a results dict with FPS, frame count, and gate status.
    """
    _log.info("benchmark_start", mode="cpu_only", duration_s=duration_s,
              resolution=f"{_WIDTH}x{_HEIGHT}")

    settings = Settings.from_env()
    assert settings.cpu_only, "GODS_EYE_CPU_ONLY must be set to 1"

    # Initialize detector and tracker
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    _log.info("benchmark_warmup_complete", device=detector.device)

    # Run timed benchmark
    frame_count = 0
    total_det_ms = 0.0
    total_track_ms = 0.0
    fps_samples: list[float] = []
    window_start = time.perf_counter()
    window_frames = 0
    benchmark_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - benchmark_start
        if elapsed >= duration_s:
            break

        # Generate synthetic frame
        frame = _generate_synthetic_frame(frame_count)
        packet = FramePacket(
            camera_id="bench-cpu",
            frame_id=frame_count,
            timestamp_ns=int(elapsed * 1_000_000_000),
            frame=frame,
            resolution=(_WIDTH, _HEIGHT),
        )

        # Detection
        t0 = time.perf_counter()
        detections = detector.detect(packet)
        det_ms = (time.perf_counter() - t0) * 1000
        total_det_ms += det_ms

        # Tracking
        t1 = time.perf_counter()
        tracks = tracker.update(detections, frame_count, "bench-cpu")
        track_ms = (time.perf_counter() - t1) * 1000
        total_track_ms += track_ms

        frame_count += 1
        window_frames += 1

        # FPS sampling every 5 seconds
        window_elapsed = time.perf_counter() - window_start
        if window_elapsed >= 5.0:
            fps = window_frames / window_elapsed
            fps_samples.append(fps)
            _log.info("benchmark_fps_sample", fps=round(fps, 2),
                      frame=frame_count, elapsed_s=round(elapsed, 1))
            window_start = time.perf_counter()
            window_frames = 0

    # Final metrics
    total_elapsed = time.perf_counter() - benchmark_start
    overall_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
    avg_det_ms = total_det_ms / frame_count if frame_count > 0 else 0.0
    avg_track_ms = total_track_ms / frame_count if frame_count > 0 else 0.0
    sustained_fps = min(fps_samples) if fps_samples else overall_fps
    gate_pass = sustained_fps >= 6.0

    results: dict[str, object] = {
        "benchmark": "cpu_mode_phase1",
        "hardware": {
            "mode": "cpu_only",
            "device": detector.device,
            "resolution": f"{_WIDTH}x{_HEIGHT}",
        },
        "duration_s": round(total_elapsed, 2),
        "total_frames": frame_count,
        "overall_fps": round(overall_fps, 2),
        "sustained_fps_min": round(sustained_fps, 2),
        "avg_detection_ms": round(avg_det_ms, 2),
        "avg_tracking_ms": round(avg_track_ms, 2),
        "fps_samples": [round(f, 2) for f in fps_samples],
        "gate_criterion": "sustained_fps >= 6.0 at 720p",
        "gate_result": "PASS" if gate_pass else "FAIL",
    }

    # Print summary
    print("\n" + "=" * 60)
    print("CPU-Only Benchmark Results — Phase 1 Gate")
    print("=" * 60)
    print(f"  Resolution:      {_WIDTH}x{_HEIGHT}")
    print(f"  Duration:        {total_elapsed:.1f}s")
    print(f"  Total Frames:    {frame_count}")
    print(f"  Overall FPS:     {overall_fps:.2f}")
    print(f"  Sustained FPS:   {sustained_fps:.2f} (min of 5s windows)")
    print(f"  Avg Detection:   {avg_det_ms:.1f}ms")
    print(f"  Avg Tracking:    {avg_track_ms:.2f}ms")
    print(f"  Gate (≥6 FPS):   {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only benchmark")
    parser.add_argument(
        "--duration-s", type=int, default=300,
        help="Benchmark duration in seconds (default: 300 = 5 minutes)",
    )
    args = parser.parse_args()

    results = run_benchmark(duration_s=args.duration_s)

    # Write results
    out_path = Path(__file__).parent / "results" / "cpu_mode_phase1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Exit with appropriate code
    if results["gate_result"] == "FAIL":
        _log.error("gate_failed", gate="cpu_mode", sustained_fps=results["sustained_fps_min"])
        sys.exit(1)


if __name__ == "__main__":
    main()
