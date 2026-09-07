"""Vehicle Tracker — Phase 8.1 (SIH 26187).

Wraps ByteTrack to track vehicles across frames. Parameterized by detection
stream to avoid instantiating a second tracker object.
"""

from __future__ import annotations

from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.vehicle import VehicleDetection, VehicleTrack, VehicleTrackState

_log = get_logger("vehicle.tracker")


class VehicleTracker:
    """Vehicle tracking using a simple IOU-based tracker.

    Maintains vehicle tracks across frames using bounding box overlap.
    Designed as a lightweight tracker that mirrors person tracking without
    requiring ByteTrack dependency for the vehicle detection stream.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
        max_dead_frames: int = 120,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._max_dead_frames = max_dead_frames
        self._tracks: dict[str, VehicleTrack] = {}
        self._next_id: int = 1

    @staticmethod
    def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute intersection-over-union between two bounding boxes."""
        x1 = max(box_a.x1, box_b.x1)
        y1 = max(box_a.y1, box_b.y1)
        x2 = min(box_a.x2, box_b.x2)
        y2 = min(box_a.y2, box_b.y2)

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = box_a.area
        area_b = box_b.area
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def update(
        self,
        detections: list[VehicleDetection],
        frame_id: int,
        camera_id: str,
    ) -> list[VehicleTrack]:
        """Update vehicle tracks with new detections.

        Args:
            detections: List of VehicleDetection for current frame.
            frame_id: Current frame counter.
            camera_id: Source camera identifier.

        Returns:
            List of active VehicleTrack objects.
        """
        # Match detections to existing tracks by IOU
        matched_track_ids: set[str] = set()
        matched_det_indices: set[int] = set()

        for tid, track in list(self._tracks.items()):
            if track.state == VehicleTrackState.DEAD:
                continue

            best_iou = 0.0
            best_idx: Optional[int] = None

            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                iou_val = self._iou(track.bbox, det.bbox)
                if iou_val > best_iou and iou_val >= self._iou_threshold:
                    best_iou = iou_val
                    best_idx = i

            if best_idx is not None:
                det = detections[best_idx]
                # Compute velocity
                old_cx, old_cy = track.bbox.center
                new_cx, new_cy = det.bbox.center
                velocity = (new_cx - old_cx, new_cy - old_cy)

                hist = track.detection_history + [det.detection_id]
                if len(hist) > 100:
                    hist = hist[-100:]

                self._tracks[tid] = VehicleTrack(
                    track_id=track.track_id,
                    camera_id=camera_id,
                    state=VehicleTrackState.ACTIVE,
                    bbox=det.bbox,
                    velocity=velocity,
                    vehicle_class=det.vehicle_class,
                    first_frame_id=track.first_frame_id,
                    last_frame_id=frame_id,
                    lost_frame_count=0,
                    detection_history=hist,
                    plate_text=track.plate_text,
                )
                matched_track_ids.add(tid)
                matched_det_indices.add(best_idx)

        # Update unmatched tracks
        for tid, track in list(self._tracks.items()):
            if tid in matched_track_ids or track.state == VehicleTrackState.DEAD:
                continue

            new_lost_count = track.lost_frame_count + 1

            if new_lost_count >= self._max_dead_frames:
                new_state = VehicleTrackState.DEAD
            elif new_lost_count >= self._max_lost_frames:
                new_state = VehicleTrackState.LOST
            else:
                new_state = track.state

            self._tracks[tid] = VehicleTrack(
                track_id=track.track_id,
                camera_id=track.camera_id,
                state=new_state,
                bbox=track.bbox,
                velocity=(0.0, 0.0),
                vehicle_class=track.vehicle_class,
                first_frame_id=track.first_frame_id,
                last_frame_id=track.last_frame_id,
                lost_frame_count=new_lost_count,
                detection_history=track.detection_history,
                plate_text=track.plate_text,
            )

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue

            tid = f"vtrk_{camera_id}_{self._next_id}"
            self._next_id += 1

            self._tracks[tid] = VehicleTrack(
                track_id=tid,
                camera_id=camera_id,
                state=VehicleTrackState.ACTIVE,
                bbox=det.bbox,
                velocity=(0.0, 0.0),
                vehicle_class=det.vehicle_class,
                first_frame_id=frame_id,
                last_frame_id=frame_id,
                lost_frame_count=0,
                detection_history=[det.detection_id],
            )

        # Remove dead tracks
        self._tracks = {
            tid: trk for tid, trk in self._tracks.items()
            if trk.state != VehicleTrackState.DEAD
        }

        return [trk for trk in self._tracks.values() if trk.state == VehicleTrackState.ACTIVE]

    def reset(self) -> None:
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_track_count(self) -> int:
        """Return count of active tracks."""
        return sum(
            1 for trk in self._tracks.values() if trk.state == VehicleTrackState.ACTIVE
        )
