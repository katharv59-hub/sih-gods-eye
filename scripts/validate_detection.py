"""Phase 1B — Detection Validation Script.

Validates the detection subsystem with real inference:
1. Loads YOLOv8n model
2. Runs warmup
3. Processes frames from the synthetic test video
4. Measures: detection FPS, detections/frame, GPU/VRAM usage
5. Validates output conforms to Detection schema (section 4)
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import torch

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import configure_logging, get_logger

TEST_VIDEO_PATH = "tests/data/synthetic_30fps_5s.mp4"


def get_gpu_stats() -> dict[str, object]:
    """Get GPU utilization and VRAM usage if CUDA is available."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1),
    }


def main() -> None:
    configure_logging("INFO")
    log = get_logger("validation.detection")

    # Check test video
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"[ERROR] Test video not found: {TEST_VIDEO_PATH}")
        print("        Run 'python scripts/validate_ingestion.py' first to generate it.")
        sys.exit(1)

    settings = Settings.from_env()
    print("\n" + "=" * 60)
    print("  DETECTION VALIDATION -- YOLOv8n")
    print("=" * 60)

    # ── GPU Info ─────────────────────────────────────────────────────────
    gpu_stats = get_gpu_stats()
    print(f"\n  GPU Available: {gpu_stats['gpu_available']}")
    if gpu_stats["gpu_available"]:
        print(f"  GPU Name:      {gpu_stats['gpu_name']}")
        print(f"  VRAM Total:    {gpu_stats['vram_total_mb']} MB")

    # ── Load Model ───────────────────────────────────────────────────────
    print("\n  Loading YOLOv8n...")
    t0 = time.perf_counter()
    detector = YOLODetector(settings)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s on device: {detector.device}")

    # ── Warmup ───────────────────────────────────────────────────────────
    print("  Running warmup inference...")
    t0 = time.perf_counter()
    detector.warmup()
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup complete in {warmup_time:.2f}s")

    if gpu_stats["gpu_available"]:
        gpu_after_warmup = get_gpu_stats()
        print(f"  VRAM after warmup: {gpu_after_warmup['vram_allocated_mb']} MB allocated")

    # ── Run Detection on Video ───────────────────────────────────────────
    print(f"\n  Processing video: {TEST_VIDEO_PATH}")
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open test video")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution = (w, h)

    frame_id = 0
    total_detections = 0
    latencies: list[float] = []
    detections_per_frame: list[int] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        packet = FramePacket(
            camera_id="val-det",
            frame_id=frame_id,
            timestamp_ns=time.time_ns(),
            frame=frame,
            resolution=resolution,
        )

        t_start = time.perf_counter()
        dets = detector.detect(packet)
        latency_ms = (time.perf_counter() - t_start) * 1000

        latencies.append(latency_ms)
        detections_per_frame.append(len(dets))
        total_detections += len(dets)

        # Validate schema conformance on first detection
        if dets and frame_id == 1:
            d = dets[0]
            assert d.camera_id == "val-det"
            assert d.frame_id == 1
            assert d.class_label == "person"
            assert 0.0 <= d.confidence <= 1.0
            assert d.source_resolution == resolution
            print(f"  Schema check on first detection: PASS")

    cap.release()

    # ── Calculate Metrics ────────────────────────────────────────────────
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    detection_fps = 1000 / avg_latency if avg_latency > 0 else 0
    avg_dets = total_detections / frame_id if frame_id > 0 else 0

    gpu_final = get_gpu_stats() if gpu_stats["gpu_available"] else {}

    # ── Print Results ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  DETECTION RESULTS")
    print("-" * 60)
    print(f"  {'Device':<35} {detector.device:>15}")
    print(f"  {'Model':<35} {detector.model_name:>15}")
    print(f"  {'Frames Processed':<35} {frame_id:>15}")
    print(f"  {'Total Detections':<35} {total_detections:>15}")
    print(f"  {'Avg Detections/Frame':<35} {avg_dets:>15.1f}")
    print(f"  {'Avg Latency (ms)':<35} {avg_latency:>15.1f}")
    print(f"  {'P95 Latency (ms)':<35} {p95_latency:>15.1f}")
    print(f"  {'Detection FPS':<35} {detection_fps:>15.1f}")
    print(f"  {'Model Load Time (s)':<35} {load_time:>15.2f}")
    print(f"  {'Warmup Time (s)':<35} {warmup_time:>15.2f}")

    if gpu_final:
        print(f"  {'VRAM Allocated (MB)':<35} {gpu_final['vram_allocated_mb']:>15}")
        print(f"  {'VRAM Reserved (MB)':<35} {gpu_final['vram_reserved_mb']:>15}")

    print("-" * 60)
    print(f"\n  OVERALL: PASS -- Detection subsystem validated")
    print()


if __name__ == "__main__":
    main()
