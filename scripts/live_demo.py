"""God's Eye -- Live Visual Demo.

Webcam -> Detection -> Tracking -> Display

Uses existing ingestion, detection, and tracking subsystems.
Press Q to quit. Graceful shutdown.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import configure_logging
from gods_eye.schemas.track import Track, TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

# Distinct colors for track IDs (BGR)
TRACK_COLORS = [
    (0, 255, 0),    # green
    (255, 128, 0),   # blue-ish
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 128, 255),   # orange
    (128, 255, 0),   # lime
    (255, 0, 128),   # pink
    (0, 255, 128),   # spring
    (128, 0, 255),   # purple
]


def get_color(track_id: str) -> tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    idx = hash(track_id) % len(TRACK_COLORS)
    return TRACK_COLORS[idx]


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    fps: float,
    det_count: int,
    frame_id: int,
) -> np.ndarray:
    """Draw detection boxes, track IDs, confidence, and HUD on frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw tracks
    for t in tracks:
        if t.state != TrackState.ACTIVE:
            continue
        color = get_color(t.track_id)
        x1, y1 = int(t.bbox.x1), int(t.bbox.y1)
        x2, y2 = int(t.bbox.x2), int(t.bbox.y2)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"ID:{t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Track age (frames since first seen)
        age = t.last_frame_id - t.first_frame_id + 1
        info = f"age:{age}"
        cv2.putText(out, info, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # HUD - top-left panel
    panel_h = 90
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (220, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    active = sum(1 for t in tracks if t.state == TrackState.ACTIVE)
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, f"Detections: {det_count}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(out, f"Tracks: {active}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # Bottom bar
    cv2.putText(out, "GOD'S EYE v2.0 | Press Q to quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return out


def main() -> None:
    configure_logging("WARNING")
    settings = Settings.from_env()

    print("Loading YOLOv8n...")
    detector = YOLODetector(settings)
    detector.warmup()

    print("Initializing ByteTrack...")
    tracker = ByteTrackTracker(settings)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No webcam available.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (w, h)

    print(f"Webcam: {w}x{h} | Device: {detector.device}")
    print("Press Q to quit.\n")

    frame_id = 0
    fps = 0.0
    fps_timer = time.perf_counter()
    fps_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            packet = FramePacket(
                camera_id="live-demo",
                frame_id=frame_id,
                timestamp_ns=time.time_ns(),
                frame=frame,
                resolution=resolution,
            )

            dets = detector.detect(packet)
            tracks = tracker.update(dets, frame_id=frame_id, camera_id="live-demo")

            # FPS calc
            fps_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 0.5:
                fps = fps_count / elapsed
                fps_timer = time.perf_counter()
                fps_count = 0

            display = draw_overlay(frame, tracks, fps, len(dets), frame_id)
            cv2.imshow("God's Eye - Live Demo", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_id}")
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
