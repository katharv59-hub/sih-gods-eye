"""God's Eye -- Demo Suite Runner.

Runs detection + tracking + overlay rendering against all demo videos
in tests/data/demo_videos/ and produces a summary report.

Usage:
  python scripts/run_demo_suite.py                    # run all videos
  python scripts/run_demo_suite.py --video mot17_09   # run one video
  python scripts/run_demo_suite.py --identity          # enable Re-ID overlay
  python scripts/run_demo_suite.py --headless          # no GUI window
  python scripts/run_demo_suite.py --save-output       # save annotated video

Controls (when not --headless):
  Q = skip to next video
  S = screenshot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker
from gods_eye.reid.identity import Identity

DEMO_VIDEO_DIR = Path("tests/data/demo_videos")
RESULTS_DIR = Path("benchmarks/results")
SCREENSHOT_DIR = Path("tests/data/screenshots")

# ─── Color palette for track rendering ────────────────────────────────────────

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


def _color_for(track_id: str) -> tuple[int, int, int]:
    return COLORS[hash(track_id) % len(COLORS)]


# ─── Per-video statistics ─────────────────────────────────────────────────────


@dataclass
class VideoStats:
    """Accumulates statistics for one video run."""
    video_name: str
    total_frames: int = 0
    total_detections: int = 0
    total_active_tracks: int = 0
    unique_track_ids: set[str] = field(default_factory=set)
    max_simultaneous_tracks: int = 0
    elapsed_s: float = 0.0

    @property
    def avg_fps(self) -> float:
        return self.total_frames / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def avg_detections_per_frame(self) -> float:
        return self.total_detections / self.total_frames if self.total_frames > 0 else 0

    @property
    def avg_tracks_per_frame(self) -> float:
        return self.total_active_tracks / self.total_frames if self.total_frames > 0 else 0

    def to_dict(self) -> dict[str, object]:
        return {
            "video": self.video_name,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "unique_tracks": len(self.unique_track_ids),
            "max_simultaneous_tracks": self.max_simultaneous_tracks,
            "avg_detections_per_frame": round(self.avg_detections_per_frame, 1),
            "avg_tracks_per_frame": round(self.avg_tracks_per_frame, 1),
            "elapsed_s": round(self.elapsed_s, 1),
            "avg_fps": round(self.avg_fps, 1),
        }


# ─── Overlay rendering ───────────────────────────────────────────────────────


def render_overlay(
    frame: np.ndarray,
    tracks: list[object],
    det_count: int,
    frame_id: int,
    fps: float,
    video_name: str,
    *,
    identities: dict[str, Identity] | None = None,
) -> np.ndarray:
    """Draw tracking overlay on a frame.

    When *identities* is provided, shows global UUID short-form
    instead of tracker-local IDs.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    active = [t for t in tracks if t.state == TrackState.ACTIVE]  # type: ignore[union-attr]

    for t in active:
        tid = t.track_id  # type: ignore[union-attr]
        # Resolve display label
        if identities and tid in identities:
            gid = identities[tid].global_id
            label = f"G:{gid[:8]}"
            c = _color_for(gid)  # color by global id for consistency
        else:
            label = f"ID:{tid}"
            c = _color_for(tid)

        x1, y1 = int(t.bbox.x1), int(t.bbox.y1)  # type: ignore[union-attr]
        x2, y2 = int(t.bbox.x2), int(t.bbox.y2)  # type: ignore[union-attr]

        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

        # Corner accents
        cl = 12
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(out, (cx, cy), (cx + cl * dx, cy), c, 3)
            cv2.line(out, (cx, cy), (cx, cy + cl * dy), c, 3)

        # ID label badge
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # HUD panel
    id_count = len(identities) if identities else 0
    panel_lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_id}",
        f"Detections: {det_count}",
        f"Tracks: {len(active)}",
    ]
    if identities is not None:
        panel_lines.append(f"Identities: {id_count}")
    panel_h = 14 + len(panel_lines) * 20
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (200, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    for i, line in enumerate(panel_lines):
        color = (0, 255, 0) if i == 0 and fps >= 20 else (220, 220, 220)
        cv2.putText(out, line, (6, 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Bottom bar
    mode_tag = "IDENTITY" if identities is not None else "TRACKING"
    bar_h = 24
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, out, 0.3, 0, out)
    cv2.putText(out, f"GOD'S EYE [{mode_tag}]  |  {video_name}  |  Q=Next",
                (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return out


# ─── Process one video ────────────────────────────────────────────────────────


def process_video(
    video_path: Path,
    detector: YOLODetector,
    settings: Settings,
    *,
    headless: bool = False,
    save_output: bool = False,
    identity_mode: bool = False,
) -> VideoStats:
    """Run detection + tracking on a single video file."""
    video_name = video_path.stem
    stats = VideoStats(video_name=video_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return stats

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fresh tracker per video
    tracker = ByteTrackTracker(settings)

    # Identity mapper (Phase 2)
    mapper = None
    if identity_mode:
        from gods_eye.reid.identity_mapper import IdentityMapper
        from gods_eye.reid.osnet_extractor import OSNetExtractor
        from gods_eye.reid.identity_gallery import IdentityGallery
        from gods_eye.reid.identity_lifecycle import LifecycleManager
        from gods_eye.reid.matcher import Matcher
        extractor = OSNetExtractor(settings)
        extractor.warmup()
        gallery = IdentityGallery()
        lifecycle = LifecycleManager(settings)
        match = Matcher(settings)
        mapper = IdentityMapper(
            extractor=extractor, gallery=gallery,
            lifecycle=lifecycle, matcher=match, settings=settings,
            camera_id="demo",
        )
        print(f"    Re-ID enabled (device={extractor.device})")

    # Output video writer
    writer = None
    if save_output:
        out_dir = Path("tests/data/demo_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, int(fps_nominal), (w, h))

    window_name = f"God's Eye -- {video_name}"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))

    t_start = time.perf_counter()
    frame_id = 0
    skipped = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1

        # Build FramePacket
        packet = FramePacket(
            camera_id="demo",
            frame_id=frame_id,
            timestamp_ns=int(frame_id * (1e9 / fps_nominal)),
            frame=frame,
            resolution=(w, h),
        )

        # Detect
        detections = detector.detect(packet)
        det_count = len(detections)

        # Track
        tracks = tracker.update(detections, frame_id, "demo")

        # Identity resolution
        id_map: dict[str, Identity] | None = None
        if mapper is not None:
            from gods_eye.pipeline import TrackingResult
            tr = TrackingResult(packet=packet, detections=detections, tracks=tracks)
            id_result = mapper.process(tr)
            id_map = id_result.identities
        active = [t for t in tracks if t.state == TrackState.ACTIVE]

        # Stats
        stats.total_frames += 1
        stats.total_detections += det_count
        stats.total_active_tracks += len(active)
        for t in active:
            stats.unique_track_ids.add(t.track_id)
        stats.max_simultaneous_tracks = max(
            stats.max_simultaneous_tracks, len(active)
        )

        # FPS calc
        elapsed = time.perf_counter() - t_start
        current_fps = frame_id / elapsed if elapsed > 0 else 0

        # Render
        if not headless or save_output:
            display = render_overlay(
                frame, tracks, det_count, frame_id, current_fps, video_name,
                identities=id_map,
            )
            if writer is not None:
                writer.write(display)
            if not headless:
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    skipped = True
                    break
                elif key == ord("s"):
                    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    ss_path = SCREENSHOT_DIR / f"demo_{video_name}_{ts}.jpg"
                    cv2.imwrite(str(ss_path), display)
                    print(f"    [SCREENSHOT] {ss_path}")

        # Progress
        if frame_id % 100 == 0 or frame_id == total_frames:
            pct = frame_id / total_frames * 100 if total_frames > 0 else 0
            print(f"    Frame {frame_id}/{total_frames} "
                  f"({pct:.0f}%) "
                  f"FPS: {current_fps:.1f} "
                  f"Tracks: {len(active)}", end="\r")

    stats.elapsed_s = time.perf_counter() - t_start

    cap.release()
    if writer is not None:
        writer.release()
    if not headless:
        cv2.destroyWindow(window_name)

    status = "SKIPPED" if skipped else "COMPLETE"
    print(f"    [{status}] {stats.total_frames} frames in {stats.elapsed_s:.1f}s "
          f"({stats.avg_fps:.1f} FPS) "
          f"| {len(stats.unique_track_ids)} unique tracks"
          "                              ")

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run God's Eye detection + tracking on demo videos."
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Run only videos matching this substring (e.g. 'mot17_09').",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI window (useful for CI/benchmarking).",
    )
    parser.add_argument(
        "--save-output", action="store_true",
        help="Save annotated output videos to tests/data/demo_outputs/.",
    )
    parser.add_argument(
        "--identity", action="store_true",
        help="Enable Phase 2 Re-ID identity resolution overlay.",
    )
    args = parser.parse_args()

    # Discover videos
    if not DEMO_VIDEO_DIR.exists():
        print(f"[ERROR] Demo video directory not found: {DEMO_VIDEO_DIR}")
        print("Run: python scripts/download_demo_videos.py")
        sys.exit(1)

    videos = sorted(DEMO_VIDEO_DIR.glob("*.mp4"))
    if args.video:
        videos = [v for v in videos if args.video in v.stem]
    if not videos:
        print("[ERROR] No matching demo videos found.")
        sys.exit(1)

    print("=" * 64)
    print("God's Eye -- Demo Suite")
    print("=" * 64)
    print(f"Videos:    {len(videos)}")
    print(f"Headless:  {args.headless}")
    print(f"Save:      {args.save_output}")

    # Init detector once (shared across all videos)
    settings = Settings.from_env()
    print(f"\nInitializing detector (model={settings.detection_model})...")
    detector = YOLODetector(settings)
    detector.warmup()
    print(f"Device: {detector.device}")

    all_stats: list[dict[str, object]] = []

    for i, video_path in enumerate(videos, 1):
        print(f"\n{'=' * 64}")
        print(f"[{i}/{len(videos)}] {video_path.name}")
        print(f"{'=' * 64}")

        stats = process_video(
            video_path, detector, settings,
            headless=args.headless,
            save_output=args.save_output,
            identity_mode=args.identity,
        )
        all_stats.append(stats.to_dict())

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "demo_suite_results.json"
    with open(results_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Summary
    print(f"\n{'=' * 64}")
    print("DEMO SUITE SUMMARY")
    print(f"{'=' * 64}")
    print(f"{'Video':<35s} {'Frames':>6s} {'FPS':>6s} {'Tracks':>7s} {'Max':>4s}")
    print("-" * 64)
    for s in all_stats:
        print(f"  {s['video']:<33s} {s['total_frames']:>6d} "
              f"{s['avg_fps']:>6.1f} {s['unique_tracks']:>7d} "
              f"{s['max_simultaneous_tracks']:>4d}")
    print("-" * 64)
    print(f"\nResults saved: {results_path}")
    if args.save_output:
        print(f"Annotated videos: tests/data/demo_outputs/")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
