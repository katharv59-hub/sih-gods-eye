"""Generate sample demo frame snapshots for visual review.

Processes a demo video sequence, draws bounding boxes, track IDs,
identity tags, and HUD panel, and saves annotated frames to the artifact directory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.source import VideoFileSource
from gods_eye.observability.logger import configure_logging
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.identity_mapper import IdentityMapper, IdentityResult
from gods_eye.reid.matcher import Matcher
from gods_eye.reid.osnet_extractor import OSNetExtractor
from gods_eye.pipeline import TrackingResult
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker
from run_demo import OverlayRenderer

ARTIFACT_DIR = Path(r"C:\Users\athar\.gemini\antigravity-ide\brain\6a6639a7-c8ba-4734-8649-3457aa7fa69d")


def main() -> None:
    configure_logging("WARNING")
    settings = Settings.from_env()

    video_path = PROJECT_ROOT / "tests" / "data" / "demo_videos" / "mot17_04_medium_density.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "tests" / "data" / "synthetic_30fps_5s.mp4"

    print(f"Processing video for visual snapshots: {video_path.name}")
    source = VideoFileSource(str(video_path))
    if not source.open():
        print(f"Failed to open source: {video_path}")
        sys.exit(1)

    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)
    extractor: EmbeddingExtractor = OSNetExtractor(settings)
    extractor.warmup()

    gallery = IdentityGallery()
    lifecycle = LifecycleManager(settings)
    matcher = Matcher(settings)
    mapper = IdentityMapper(
        extractor=extractor,
        gallery=gallery,
        lifecycle=lifecycle,
        matcher=matcher,
        settings=settings,
        camera_id="demo",
    )
    renderer = OverlayRenderer()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    frame_counter = 0

    fps_history = []
    t_start = time.perf_counter()

    while frame_counter < 120:
        ret, frame = source.read()
        if not ret or frame is None:
            break

        frame_counter += 1
        h, w = frame.shape[:2]
        packet = FramePacket(
            camera_id="demo",
            frame_id=frame_counter,
            timestamp_ns=int(frame_counter * 33_333_333),
            frame=frame,
            resolution=(w, h),
        )

        detections = detector.detect(packet)
        tracks = tracker.update(detections, frame_counter, "demo")

        tracking_result = TrackingResult(
            packet=packet,
            detections=detections,
            tracks=tracks,
        )
        id_result: IdentityResult = mapper.process(tracking_result)

        elapsed = time.perf_counter() - t_start
        current_fps = frame_counter / elapsed if elapsed > 0 else 0.0

        stats = {
            "captured": frame_counter,
            "detected": frame_counter,
            "tracked": frame_counter,
            "delivered": frame_counter,
        }
        queue_depths = {"frame": 0, "det": 0, "track": 0, "identity": 0}

        annotated = renderer.render(frame, tracking_result, stats, 0, queue_depths)

        # Draw identity overlay tags on active tracks
        active_tracks = [t for t in tracks if t.state == TrackState.ACTIVE]
        for t in active_tracks:
            identity = id_result.identities.get(t.track_id)
            if identity:
                x1, y1 = int(t.bbox.x1), int(t.bbox.y1)
                tag = f"UUID:{identity.global_id[:8]}"
                cv2.putText(
                    annotated, tag, (x1, y1 - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2
                )

        # Save snapshot every 30 frames
        if frame_counter in (30, 60, 90, 120):
            saved_count += 1
            output_file = ARTIFACT_DIR / f"demo_frame_{saved_count}.jpg"
            cv2.imwrite(str(output_file), annotated)
            print(f"Saved snapshot #{saved_count}: {output_file}")

    source.release()
    print("Snapshot generation complete.")


if __name__ == "__main__":
    main()
