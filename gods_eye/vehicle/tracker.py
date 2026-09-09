"""Vehicle Tracker — Phase 8.1 (SIH 26187).

Adapter wrapping the canonical ByteTrackTracker implementation from gods_eye.tracking.
Parameterized by detection stream (prefix 'vtrk') to isolate vehicle tracks from
person tracks and eliminate duplicate tracking algorithms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.vehicle import (
    VEHICLE_CLASSES,
    VehicleClassType,
    VehicleDetection,
    VehiclePlateAssociation,
    VehicleTrack,
    VehicleTrackState,
)
from gods_eye.tracking.bytetrack import ByteTrackTracker

_log = get_logger("vehicle.tracker")


class VehicleTracker:
    """Vehicle tracking adapter wrapping ByteTrackTracker.

    Reuses the canonical ByteTrack implementation from gods_eye.tracking,
    parameterized for the vehicle detection stream. Eliminates any independent
    IOU tracking or duplicate track lifecycle state machines.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        tracker: ByteTrackTracker | None = None,
        stream_prefix: str = "vtrk",
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
        max_dead_frames: int = 120,
    ) -> None:
        """Initialize vehicle tracker adapter wrapping ByteTrack.

        Args:
            settings: Settings snapshot. If None, constructed with timeouts.
            tracker: Optional pre-configured ByteTrackTracker instance.
            stream_prefix: Prefix for ID isolation (default 'vtrk').
            iou_threshold: Activation / confidence threshold.
            max_lost_frames: Frames before ACTIVE -> LOST transition.
            max_dead_frames: Frames before LOST -> DEAD eviction.
        """
        self._stream_prefix = stream_prefix

        if tracker is not None:
            self._tracker = tracker
        else:
            if settings is None:
                settings = Settings(
                    detection_confidence_threshold=iou_threshold,
                    track_lost_timeout=max_lost_frames,
                    track_dead_timeout=max_dead_frames,
                )
            self._tracker = ByteTrackTracker(
                settings=settings,
                stream_prefix=stream_prefix,
            )

        # Metadata registries for vehicle-specific fields
        self._class_registry: dict[str, VehicleClassType] = {}
        self._plate_registry: dict[str, VehiclePlateAssociation] = {}

        _log.info(
            "vehicle_tracker_init",
            backend="ByteTrackTracker",
            stream_prefix=stream_prefix,
        )

    def update(
        self,
        detections: list[VehicleDetection],
        frame_id: int,
        camera_id: str,
    ) -> list[VehicleTrack]:
        """Update vehicle tracks with new detections via ByteTrack.

        Args:
            detections: List of VehicleDetection for current frame.
            frame_id: Current frame counter.
            camera_id: Source camera identifier.

        Returns:
            List of VehicleTrack objects.
        """
        # Convert VehicleDetection to canonical Detection for ByteTrackTracker
        person_style_dets: list[Detection] = []
        det_class_map: dict[str, VehicleClassType] = {}

        for det in detections:
            person_style_dets.append(
                Detection(
                    detection_id=det.detection_id,
                    camera_id=det.camera_id,
                    frame_id=det.frame_id,
                    timestamp_ns=det.timestamp_ns,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_label=det.vehicle_class,
                    source_resolution=det.source_resolution,
                )
            )
            det_class_map[det.detection_id] = det.vehicle_class

        # Delegate to canonical ByteTrackTracker
        tracks = self._tracker.update(person_style_dets, frame_id, camera_id)

        # Convert ByteTrack Track outputs to canonical VehicleTrack representation
        vehicle_tracks: list[VehicleTrack] = []
        for trk in tracks:
            # Resolve vehicle class
            v_class: VehicleClassType = "car"
            if trk.detection_history:
                last_det_id = trk.detection_history[-1]
                if last_det_id in det_class_map:
                    self._class_registry[trk.track_id] = det_class_map[last_det_id]

            if trk.track_id in self._class_registry:
                v_class = self._class_registry[trk.track_id]
            elif hasattr(trk, "class_label") and trk.class_label in VEHICLE_CLASSES:
                v_class = trk.class_label  # type: ignore

            state_val = VehicleTrackState(trk.state.value)
            assoc = self._plate_registry.get(trk.track_id)
            plate_text = assoc.plate_text if assoc is not None else None
            assoc_conf = assoc.confidence if assoc is not None else None

            vehicle_tracks.append(
                VehicleTrack(
                    track_id=trk.track_id,
                    camera_id=trk.camera_id,
                    state=state_val,
                    bbox=trk.bbox,
                    velocity=trk.velocity,
                    vehicle_class=v_class,
                    first_frame_id=trk.first_frame_id,
                    last_frame_id=trk.last_frame_id,
                    lost_frame_count=trk.lost_frame_count,
                    detection_history=list(trk.detection_history),
                    plate_text=plate_text,
                    association_confidence=assoc_conf,
                    plate_association=assoc,
                )
            )

        # Evict stale entries for tracks no longer in ByteTrack
        active_track_ids = {t.track_id for t in tracks}
        self._class_registry = {
            tid: cls for tid, cls in self._class_registry.items() if tid in active_track_ids
        }
        self._plate_registry = {
            tid: plt for tid, plt in self._plate_registry.items() if tid in active_track_ids
        }

        return vehicle_tracks

    def associate_plate(
        self,
        track_id: str,
        plate_text: str,
        confidence: Optional[float] = None,
        camera_id: Optional[str] = None,
        frame_id: Optional[int] = None,
        timestamp_ns: Optional[int] = None,
        is_uncertain: Optional[bool] = None,
        plate_region_id: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        spatial_confidence: Optional[float] = None,
        plate_bbox: Optional[BoundingBox] = None,
        vehicle_bbox: Optional[BoundingBox] = None,
    ) -> VehiclePlateAssociation:
        """Associate a license plate observation with a vehicle track.

        Computes or assigns a defensible association confidence score in [0.0, 1.0]
        derived from spatial containment, OCR reading quality, and track stability.

        Args:
            track_id: Vehicle track identifier.
            plate_text: Recognized plate string, or 'uncertain'.
            confidence: Optional explicit association confidence [0.0, 1.0].
            camera_id: Source camera identifier.
            frame_id: Frame number of observation.
            timestamp_ns: Observation timestamp in Unix nanoseconds.
            is_uncertain: Uncertainty flag. Defaults to True if plate_text == 'uncertain'.
            plate_region_id: Identifier of plate detection region.
            ocr_confidence: Confidence of the OCR reading.
            spatial_confidence: Explicit spatial overlap confidence.
            plate_bbox: Bounding box of plate candidate.
            vehicle_bbox: Bounding box of vehicle track.

        Returns:
            VehiclePlateAssociation record.
        """
        uncertain = is_uncertain
        if uncertain is None:
            uncertain = (plate_text.strip().lower() == "uncertain")

        if confidence is not None:
            assoc_conf = float(np.clip(confidence, 0.0, 1.0))
        else:
            assoc_conf = self._compute_association_confidence(
                track_id=track_id,
                plate_text=plate_text,
                is_uncertain=uncertain,
                ocr_confidence=ocr_confidence,
                spatial_confidence=spatial_confidence,
                plate_bbox=plate_bbox,
                vehicle_bbox=vehicle_bbox,
            )

        assoc = VehiclePlateAssociation(
            track_id=track_id,
            plate_text=plate_text,
            confidence=round(assoc_conf, 3),
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            is_uncertain=uncertain,
            plate_region_id=plate_region_id,
            ocr_confidence=round(ocr_confidence, 3) if ocr_confidence is not None else None,
            spatial_confidence=round(spatial_confidence, 3) if spatial_confidence is not None else None,
        )
        self._plate_registry[track_id] = assoc
        return assoc

    def _compute_association_confidence(
        self,
        track_id: str,
        plate_text: str,
        is_uncertain: bool,
        ocr_confidence: Optional[float] = None,
        spatial_confidence: Optional[float] = None,
        plate_bbox: Optional[BoundingBox] = None,
        vehicle_bbox: Optional[BoundingBox] = None,
    ) -> float:
        """Compute defensible association confidence from observable evidence.

        Evidence components:
        1. Spatial containment & geometry (S):
           - plate_bbox within vehicle_bbox (containment ratio)
           - plate vertical positioning in vehicle lower half
        2. Reading / OCR confidence (O):
           - raw ocr_confidence if available
           - discounted if uncertain or unreadable
        3. Track stability & continuity (T):
           - track active state and history length in underlying tracker

        Composite confidence:
            C = 0.40 * S + 0.35 * O + 0.25 * T
        If uncertain, C is capped at 0.30.
        """
        # 1. Spatial score S
        if spatial_confidence is not None:
            s_score = float(np.clip(spatial_confidence, 0.0, 1.0))
        elif plate_bbox is not None and vehicle_bbox is not None:
            s_score = self._compute_spatial_containment(plate_bbox, vehicle_bbox)
        else:
            s_score = 0.80  # Default assumed spatial crop

        # 2. OCR / Reading score O
        if ocr_confidence is not None:
            o_score = float(np.clip(ocr_confidence, 0.0, 1.0))
        elif is_uncertain or plate_text.strip().lower() == "uncertain":
            o_score = 0.15
        else:
            o_score = 0.85

        # 3. Track stability score T
        t_score = self._compute_track_stability(track_id)

        # Composite score
        raw_conf = 0.40 * s_score + 0.35 * o_score + 0.25 * t_score

        # Enforce uncertainty ceiling: unreadable or uncertain plates cannot have high confidence
        if is_uncertain or plate_text.strip().lower() == "uncertain":
            raw_conf = min(raw_conf, 0.30)

        return float(np.clip(raw_conf, 0.0, 1.0))

    @staticmethod
    def _compute_spatial_containment(plate_bbox: BoundingBox, vehicle_bbox: BoundingBox) -> float:
        """Calculate spatial containment and relative geometry score."""
        ix1 = max(plate_bbox.x1, vehicle_bbox.x1)
        iy1 = max(plate_bbox.y1, vehicle_bbox.y1)
        ix2 = min(plate_bbox.x2, vehicle_bbox.x2)
        iy2 = min(plate_bbox.y2, vehicle_bbox.y2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection_area = iw * ih
        plate_area = max(1, plate_bbox.area)

        # Overlap fraction
        containment = intersection_area / plate_area

        # Geometry heuristic: plate is typically in lower 65% of vehicle
        v_h = max(1, vehicle_bbox.y2 - vehicle_bbox.y1)
        p_cy = (plate_bbox.y1 + plate_bbox.y2) / 2.0
        v_mid = vehicle_bbox.y1 + 0.35 * v_h
        lower_bonus = 0.10 if p_cy >= v_mid else 0.0

        score = 0.90 * containment + lower_bonus
        return float(np.clip(score, 0.0, 1.0))

    def _compute_track_stability(self, track_id: str) -> float:
        """Calculate track continuity and stability score."""
        track_info_map = getattr(self._tracker, "_track_info", {})
        for tid, info in track_info_map.items():
            if track_id.endswith(f"_{tid}") or track_id == str(tid):
                history_len = len(getattr(info, "detection_history", []))
                maturity = min(1.0, history_len / 3.0)
                is_lost = getattr(info, "lost_frame_count", 0) > 0
                base = 0.50 if is_lost else 0.70
                return base + 0.30 * maturity

        return 0.70

    def get_plate_association(self, track_id: str) -> Optional[VehiclePlateAssociation]:
        """Retrieve structured plate association for a vehicle track."""
        return self._plate_registry.get(track_id)

    def get_plate_text(self, track_id: str) -> Optional[str]:
        """Retrieve associated plate text string for a vehicle track."""
        assoc = self._plate_registry.get(track_id)
        return assoc.plate_text if assoc is not None else None

    def reset(self) -> None:
        """Reset all tracker and registry state."""
        self._tracker.reset()
        self._class_registry.clear()
        self._plate_registry.clear()
        _log.info("vehicle_tracker_reset")

    @property
    def active_track_count(self) -> int:
        """Number of currently ACTIVE vehicle tracks."""
        return self._tracker.active_track_count

    @property
    def total_track_count(self) -> int:
        """Number of ACTIVE + LOST vehicle tracks."""
        return self._tracker.total_track_count

    @property
    def underlying_tracker(self) -> ByteTrackTracker:
        """Return the wrapped ByteTrackTracker instance."""
        return self._tracker

    def to_event(
        self,
        track: VehicleTrack,
        timestamp_ns: int,
        frame_id: Optional[int] = None,
        confidence: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> Any:
        """Adapt a VehicleTrack into a canonical Event(VEHICLE_DETECTED)."""
        return track.to_event(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            confidence=confidence,
            event_id=event_id,
        )

    def to_anpr_event(
        self,
        association: VehiclePlateAssociation,
        event_id: Optional[str] = None,
    ) -> Any:
        """Adapt a VehiclePlateAssociation into a canonical Event(ANPR_READING)."""
        return association.to_event(event_id=event_id)

