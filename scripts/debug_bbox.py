"""Phase 1 Debug — Bounding Box Investigation.

Side-by-side comparison of raw YOLO output vs tracker output.
Green = raw YOLO detection
Blue  = ByteTrack tracker box

Prints exact coordinates and dimensions to terminal.
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
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

OUTPUT_DIR = "tests/data/debug_bbox"
CAPTURE_FRAMES = 30


def main() -> None:
    configure_logging("WARNING")
    settings = Settings.from_env()

    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No webcam")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (w, h)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Frame size: {w}x{h}")
    print(f"Capturing {CAPTURE_FRAMES} frames...\n")
    print(f"{'Frame':>5} | {'Source':<8} | {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6} | "
          f"{'W':>5} {'H':>5} | {'Conf':>5} | Notes")
    print("-" * 90)

    saved = 0
    for fid in range(1, CAPTURE_FRAMES + 1):
        ret, frame = cap.read()
        if not ret:
            break

        packet = FramePacket(
            camera_id="debug",
            frame_id=fid,
            timestamp_ns=time.time_ns(),
            frame=frame,
            resolution=resolution,
        )

        # Raw YOLO detections
        dets = detector.detect(packet)

        # Tracker update
        tracks = tracker.update(dets, frame_id=fid, camera_id="debug")
        active_tracks = [t for t in tracks if t.state == TrackState.ACTIVE]

        # Print coordinates
        for d in dets:
            bw = d.bbox.x2 - d.bbox.x1
            bh = d.bbox.y2 - d.bbox.y1
            print(f"{fid:>5} | {'YOLO':<8} | {d.bbox.x1:>6.1f} {d.bbox.y1:>6.1f} "
                  f"{d.bbox.x2:>6.1f} {d.bbox.y2:>6.1f} | {bw:>5.0f} {bh:>5.0f} | "
                  f"{d.confidence:>5.2f} |")

        for t in active_tracks:
            tw = t.bbox.x2 - t.bbox.x1
            th = t.bbox.y2 - t.bbox.y1
            # Check if tracker box differs from any detection
            match = ""
            for d in dets:
                dx = abs(t.bbox.x1 - d.bbox.x1) + abs(t.bbox.y1 - d.bbox.y1) + \
                     abs(t.bbox.x2 - d.bbox.x2) + abs(t.bbox.y2 - d.bbox.y2)
                if dx < 2.0:
                    match = "MATCH"
                elif dx < 20.0:
                    match = f"DRIFT={dx:.1f}px"
                else:
                    match = f"DIFFERS={dx:.0f}px"
            print(f"{fid:>5} | {'TRACK':<8} | {t.bbox.x1:>6.1f} {t.bbox.y1:>6.1f} "
                  f"{t.bbox.x2:>6.1f} {t.bbox.y2:>6.1f} | {tw:>5.0f} {th:>5.0f} | "
                  f"{'  -  ':>5} | ID:{t.track_id} {match}")

        # Save annotated frame (first 5 with detections)
        if dets and saved < 5:
            annotated = frame.copy()
            # Green = raw YOLO
            for d in dets:
                x1, y1 = int(d.bbox.x1), int(d.bbox.y1)
                x2, y2 = int(d.bbox.x2), int(d.bbox.y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"YOLO {d.confidence:.2f} [{x2-x1}x{y2-y1}]"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Blue = tracker (offset by 2px for visibility)
            for t in active_tracks:
                x1, y1 = int(t.bbox.x1), int(t.bbox.y1)
                x2, y2 = int(t.bbox.x2), int(t.bbox.y2)
                cv2.rectangle(annotated, (x1+2, y1+2), (x2+2, y2+2), (255, 128, 0), 2)
                label = f"TRACK ID:{t.track_id} [{x2-x1}x{y2-y1}]"
                cv2.putText(annotated, label, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

            # Frame info
            cv2.putText(annotated, f"Frame {fid} | GREEN=YOLO  BLUE=TRACK",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            path = os.path.join(OUTPUT_DIR, f"debug_{saved+1}.jpg")
            cv2.imwrite(path, annotated)
            saved += 1

    cap.release()
    print("-" * 90)
    print(f"\nAnnotated frames saved to {OUTPUT_DIR}/ ({saved} images)")


if __name__ == "__main__":
    main()
