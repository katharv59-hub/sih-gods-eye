"""God's Eye -- Phase 1 Visual Demo.

Complete end-to-end demonstration:
  Ingestion -> Detection -> Tracking -> Visualization

Usage:
  python run_demo.py              # webcam
  python run_demo.py video.mp4    # video file

Controls:
  Q = quit
  S = screenshot (saved to tests/data/screenshots/)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.ingestion.source import FrameSource, VideoFileSource, WebcamSource
from gods_eye.observability.logger import configure_logging
from gods_eye.pipeline import Pipeline, TrackingResult
from gods_eye.reid.identity_mapper import IdentityResult
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

SCREENSHOT_DIR = "tests/data/screenshots"

# ─── Color Palette ───────────────────────────────────────────────────────────

COLORS = [
    (46, 204, 113),   # emerald
    (52, 152, 219),   # peter river
    (155, 89, 182),   # amethyst
    (231, 76, 60),    # alizarin
    (241, 196, 15),   # sun flower
    (26, 188, 156),   # turquoise
    (230, 126, 34),   # carrot
    (149, 165, 166),  # concrete
    (52, 73, 94),     # wet asphalt
    (192, 57, 43),    # pomegranate
]


def color_for(track_id: str) -> tuple[int, int, int]:
    return COLORS[hash(track_id) % len(COLORS)]


# ─── Overlay Renderer ────────────────────────────────────────────────────────


class OverlayRenderer:
    """Draws bounding boxes, labels, and HUD on frames."""

    def __init__(self) -> None:
        self._fps_history: deque[float] = deque(maxlen=30)
        self._screenshot_count = 0

    def render(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        pipeline_stats: dict[str, int],
        frame_drops: int,
        queue_depths: dict[str, int],
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Draw tracks ──────────────────────────────────────────────
        active_tracks = [t for t in result.tracks if t.state == TrackState.ACTIVE]
        for t in active_tracks:
            c = color_for(t.track_id)
            x1, y1 = int(t.bbox.x1), int(t.bbox.y1)
            x2, y2 = int(t.bbox.x2), int(t.bbox.y2)

            # Box with rounded feel (thick + thin)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

            # Corner accents
            cl = 15
            cv2.line(out, (x1, y1), (x1 + cl, y1), c, 3)
            cv2.line(out, (x1, y1), (x1, y1 + cl), c, 3)
            cv2.line(out, (x2, y1), (x2 - cl, y1), c, 3)
            cv2.line(out, (x2, y1), (x2, y1 + cl), c, 3)
            cv2.line(out, (x1, y2), (x1 + cl, y2), c, 3)
            cv2.line(out, (x1, y2), (x1, y2 - cl), c, 3)
            cv2.line(out, (x2, y2), (x2 - cl, y2), c, 3)
            cv2.line(out, (x2, y2), (x2, y2 - cl), c, 3)

            # ID label (top)
            label = f"ID:{t.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), c, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Age + history (bottom)
            age = t.last_frame_id - t.first_frame_id + 1
            info = f"age:{age} hist:{len(t.detection_history)}"
            cv2.putText(out, info, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        # ── HUD Panel (top-left) ─────────────────────────────────────
        now = time.perf_counter()
        self._fps_history.append(now)
        if len(self._fps_history) > 1:
            elapsed = self._fps_history[-1] - self._fps_history[0]
            fps = (len(self._fps_history) - 1) / elapsed if elapsed > 0 else 0
        else:
            fps = 0.0

        panel_lines = [
            f"FPS: {fps:.1f}",
            f"Detections: {len(result.detections)}",
            f"Tracks: {len(active_tracks)}",
            f"Captured: {pipeline_stats.get('captured', 0)}",
            f"Delivered: {pipeline_stats.get('delivered', 0)}",
            f"Drops: {frame_drops}",
            f"Q frame: {queue_depths.get('frame', 0)}",
            f"Q det:   {queue_depths.get('det', 0)}",
            f"Q track: {queue_depths.get('track', 0)}",
            f"Q ident: {queue_depths.get('identity', 0)}",
        ]

        panel_h = 20 + len(panel_lines) * 22
        panel_w = 200
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        for i, line in enumerate(panel_lines):
            y_pos = 18 + i * 22
            # Color code first line (FPS)
            if i == 0:
                fc = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
            elif i == 5 and frame_drops > 0:
                fc = (0, 165, 255)
            else:
                fc = (220, 220, 220)
            cv2.putText(out, line, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, fc, 1)

        # ── Title bar (bottom) ───────────────────────────────────────
        bar_h = 28
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, out, 0.3, 0, out)
        cv2.putText(out, "GOD'S EYE v2.0  |  Q=Quit  S=Screenshot",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return out

    def save_screenshot(self, frame: np.ndarray) -> str:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self._screenshot_count += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}_{self._screenshot_count}.jpg")
        cv2.imwrite(path, frame)
        return path


# ─── Main Demo ───────────────────────────────────────────────────────────────


def main() -> None:
    configure_logging("WARNING")
    settings = Settings.from_env()

    # Determine source
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"[ERROR] File not found: {video_path}")
            sys.exit(1)
        source: FrameSource = VideoFileSource(video_path)
        source_name = os.path.basename(video_path)
        camera_id = "file"
    else:
        source = WebcamSource(settings.webcam_device_index)
        source_name = f"Webcam (index {settings.webcam_device_index})"
        camera_id = "webcam"

    # Init
    print("God's Eye v2.0 -- Phase 1 Demo")
    print(f"Source: {source_name}")
    print("Loading YOLOv8n...")
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)
    renderer = OverlayRenderer()

    # Shared state for output callback
    latest_result: list[IdentityResult | None] = [None]
    result_lock = threading.Lock()

    def on_result(r: IdentityResult) -> None:
        with result_lock:
            latest_result[0] = r

    pipeline = Pipeline(
        camera_id=camera_id,
        source=source,
        detector=detector,
        tracker=tracker,
        settings=settings,
        on_result=on_result,
    )

    print(f"Device: {detector.device}")
    print("Starting pipeline... Press Q to quit, S for screenshot.\n")
    pipeline.start()

    window_name = f"God's Eye -- {source_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            with result_lock:
                result = latest_result[0]

            if result is None:
                if not pipeline.is_alive:
                    # Source exhausted (video file)
                    time.sleep(0.5)
                    break
                time.sleep(0.01)
                continue

            # Get pipeline stats
            stats = pipeline.stats
            queue_depths = {
                "frame": pipeline._frame_q.qsize,
                "det": pipeline._det_q.qsize,
                "track": pipeline._track_q.qsize,
                "identity": pipeline._identity_q.qsize,
            }
            drops = pipeline._det_q.drop_count + pipeline._track_q.drop_count

            tracking = result.tracking
            display = renderer.render(
                tracking.packet.frame, tracking, stats, drops, queue_depths
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = renderer.save_screenshot(display)
                print(f"[SCREENSHOT] {path}")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down pipeline...")
        pipeline.stop(timeout=5.0)
        cv2.destroyAllWindows()
        stats = pipeline.stats
        print(f"Captured: {stats['captured']}  Detected: {stats['detected']}  "
              f"Tracked: {stats['tracked']}  Delivered: {stats['delivered']}")
        print("Done.")


if __name__ == "__main__":
    main()
