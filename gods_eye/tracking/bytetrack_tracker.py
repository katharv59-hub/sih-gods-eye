"""ByteTrack tracker — concrete implementation using supervision.

Wraps supervision.ByteTrack, converts results to canonical Track schema (§4).
Maintains per-track state: lifecycle (ACTIVE/LOST/DEAD), detection history,
frame counters, and last known bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import Track, TrackState
from gods_eye.tracking.tracker import Tracker


@dataclass
class _TrackInfo:
    """Internal per-track state maintained across frames."""

    first_frame_id: int
    last_frame_id: int
    last_bbox: BoundingBox
    lost_frame_count: int = 0
    detection_history: list[str] = field(default_factory=list)


_DETECTION_HISTORY_CAP: int = 50


class ByteTrackTracker(Tracker):
    """ByteTrack-based multi-object tracker.

    Uses supervision.ByteTrack for frame-to-frame association.
    Maintains our own track state registry for §4 Track lifecycle.

    Not thread-safe — must be called from a single thread.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = get_logger("tracking")

        self._tracker = sv.ByteTrack(
            track_activation_threshold=settings.detection_confidence_threshold,
            lost_track_buffer=settings.track_lost_timeout,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        # Internal track state registry: ByteTrack track_id → _TrackInfo
        self._track_info: dict[int, _TrackInfo] = {}
        self._active_count: int = 0

        self._log.info(
            "tracker_init",
            tracker="ByteTrack",
            lost_buffer=settings.track_lost_timeout,
            dead_timeout=settings.track_dead_timeout,
        )

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        camera_id: str,
    ) -> list[Track]:
        """Process one frame of detections through ByteTrack.

        Returns ACTIVE + LOST tracks. DEAD tracks are evicted.
        """
        # Build supervision Detections + detection_id index
        det_ids: list[str] = []

        if detections:
            xyxy = np.array(
                [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
                dtype=np.float64,
            )
            confidence = np.array(
                [d.confidence for d in detections], dtype=np.float64
            )
            det_ids = [d.detection_id for d in detections]
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence)
        else:
            sv_dets = sv.Detections.empty()

        # Run ByteTrack
        tracked = self._tracker.update_with_detections(sv_dets)

        # Process results
        active_ids: set[int] = set()
        tracks: list[Track] = []

        if tracked.tracker_id is not None and len(tracked) > 0:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                active_ids.add(tid)

                bbox = BoundingBox(
                    x1=float(tracked.xyxy[i][0]),
                    y1=float(tracked.xyxy[i][1]),
                    x2=float(tracked.xyxy[i][2]),
                    y2=float(tracked.xyxy[i][3]),
                )

                # Get or create track info
                if tid not in self._track_info:
                    self._track_info[tid] = _TrackInfo(
                        first_frame_id=frame_id,
                        last_frame_id=frame_id,
                        last_bbox=bbox,
                    )

                info = self._track_info[tid]
                info.last_frame_id = frame_id
                info.last_bbox = bbox
                info.lost_frame_count = 0

                # Match detection_id by bbox IoU
                matched_det_id = self._match_detection_id(
                    tracked.xyxy[i], detections, det_ids
                )
                if matched_det_id:
                    info.detection_history.append(matched_det_id)
                    if len(info.detection_history) > _DETECTION_HISTORY_CAP:
                        info.detection_history = info.detection_history[
                            -_DETECTION_HISTORY_CAP:
                        ]

                tracks.append(
                    Track(
                        track_id=str(tid),
                        camera_id=camera_id,
                        state=TrackState.ACTIVE,
                        bbox=bbox,
                        first_frame_id=info.first_frame_id,
                        last_frame_id=frame_id,
                        lost_frame_count=0,
                        detection_history=list(info.detection_history),
                    )
                )

        # Update LOST / evict DEAD
        dead_ids: list[int] = []
        for tid, info in self._track_info.items():
            if tid not in active_ids:
                info.lost_frame_count += 1
                if info.lost_frame_count >= self._settings.track_dead_timeout:
                    dead_ids.append(tid)
                else:
                    tracks.append(
                        Track(
                            track_id=str(tid),
                            camera_id=camera_id,
                            state=TrackState.LOST,
                            bbox=info.last_bbox,
                            first_frame_id=info.first_frame_id,
                            last_frame_id=info.last_frame_id,
                            lost_frame_count=info.lost_frame_count,
                            detection_history=list(info.detection_history),
                        )
                    )

        for tid in dead_ids:
            self._log.info(
                "track_evicted",
                track_id=str(tid),
                state="dead",
                lifetime_frames=(
                    self._track_info[tid].last_frame_id
                    - self._track_info[tid].first_frame_id
                ),
            )
            del self._track_info[tid]

        self._active_count = len(active_ids)
        return tracks

    @staticmethod
    def _match_detection_id(
        tracked_xyxy: np.ndarray,
        detections: list[Detection],
        det_ids: list[str],
    ) -> str:
        """Match a tracked bbox to the closest input detection by IoU."""
        if not detections:
            return ""

        best_iou = 0.0
        best_id = ""
        tx1, ty1, tx2, ty2 = tracked_xyxy

        for det, did in zip(detections, det_ids):
            # Compute IoU
            ix1 = max(tx1, det.bbox.x1)
            iy1 = max(ty1, det.bbox.y1)
            ix2 = min(tx2, det.bbox.x2)
            iy2 = min(ty2, det.bbox.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_t = max(0, tx2 - tx1) * max(0, ty2 - ty1)
            area_d = det.bbox.area
            union = area_t + area_d - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_id = did

        return best_id if best_iou > 0.3 else ""

    def reset(self) -> None:
        """Reset all tracker state."""
        self._tracker.reset()
        self._track_info.clear()
        self._active_count = 0
        self._log.info("tracker_reset")

    @property
    def active_track_count(self) -> int:
        """Number of currently ACTIVE tracks."""
        return self._active_count

    @property
    def total_track_count(self) -> int:
        """Number of ACTIVE + LOST tracks."""
        return len(self._track_info)
