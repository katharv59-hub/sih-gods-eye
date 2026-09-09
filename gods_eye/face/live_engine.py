"""Live Face Engine — Phase 10 (SIH 26187).

Attaches the YuNet + SFace face recognition core to live video frames and
person tracks from ByteTrack, capturing full-frame and zoomed face crop evidence,
and emitting canonical EventType.FACE_DETECTED events into SQLiteEventStore.

Flow:
    Frame + Person Tracks
          ↓
    YuNet detect_faces (multi-face aware)
          ↓
    Spatial containment association with person tracks
          ↓
    SFace embedding extraction
          ↓
    Cosine similarity matching against GraphStore enrolled gallery
          ↓
    KNOWN (similarity >= threshold) / UNKNOWN
          ↓
    Track-level de-duplication & state-change throttling
          ↓
    Safe-bounds zoomed face crop + snapshot.jpg evidence
          ↓
    EventType.FACE_DETECTED -> SQLiteEventStore
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from gods_eye.events.event_store import BaseEventStore
from gods_eye.evidence.evidence_manager import EvidenceManager
from gods_eye.face.recognizer import FaceRecognitionResult, FaceRecognizer
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.track import Track

_log = get_logger("face.live_engine")


@dataclass
class _TrackFaceState:
    """Internal cache to track recognition state per person track."""

    last_status: str
    last_person_id: Optional[str]
    last_emitted_ns: int


class LiveFaceEngine:
    """Production live face recognition engine connecting cameras/tracks to EventStore & Evidence."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        graph_store: BaseGraphStore,
        event_store: BaseEventStore,
        evidence_manager: EvidenceManager,
        throttle_window_s: float = 5.0,
    ) -> None:
        self._recognizer = recognizer
        self._graph_store = graph_store
        self._event_store = event_store
        self._evidence_manager = evidence_manager
        self._throttle_window_ns = int(throttle_window_s * 1_000_000_000)

        # Cache track_id -> _TrackFaceState for de-duplication
        self._track_states: dict[str, _TrackFaceState] = {}
        # Also cache unassociated face state by rounded bbox / center hash to avoid flooding unassigned faces
        self._untracked_states: dict[str, _TrackFaceState] = {}

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Optional[Sequence[Track]] = None,
        camera_id: str = "CAM-01",
        frame_id: int = 0,
        timestamp_ns: Optional[int] = None,
        identities: Optional[dict[str, Any]] = None,
    ) -> list[FaceRecognitionResult]:
        """Process a live camera frame: detect faces, associate tracks, recognize, save evidence, and emit events.

        Args:
            frame: Live BGR video frame.
            tracks: Active person tracks in the frame from ByteTrack.
            camera_id: Identifier of the source camera.
            frame_id: Monotonic sequence frame number.
            timestamp_ns: Capture timestamp in Unix nanoseconds.
            identities: Optional mapping of track_id to Identity from Re-ID.

        Returns:
            List of FaceRecognitionResult objects for all detected faces.
        """
        if frame is None or frame.size == 0:
            return []

        ts_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()
        faces = self._recognizer.detect_faces(frame)
        if not faces:
            return []

        gallery = self._graph_store.get_enrolled_faces()
        results: list[FaceRecognitionResult] = []

        for face_bbox, det_conf, face_data in faces:
            # 1. Spatial containment association with person tracks
            associated_track: Optional[Track] = None
            face_cx = (face_bbox.x1 + face_bbox.x2) / 2.0
            face_cy = (face_bbox.y1 + face_bbox.y2) / 2.0

            if tracks:
                for trk in tracks:
                    if hasattr(trk, "bbox"):
                        tb = trk.bbox
                        if tb.x1 <= face_cx <= tb.x2 and tb.y1 <= face_cy <= tb.y2:
                            associated_track = trk
                            break

            track_id = str(associated_track.track_id) if associated_track is not None else None
            # Extract track's global_id if present
            track_global_id = getattr(associated_track, "global_id", None)
            if track_global_id is None and track_id is not None and identities is not None:
                id_obj = identities.get(track_id)
                if id_obj is not None:
                    track_global_id = getattr(id_obj, "global_id", None)

            # 2. Extract SFace 128-D embedding & compare against gallery
            try:
                query_emb = self._recognizer.extract_embedding(frame, face_data)
            except Exception as exc:
                _log.warning("embedding_extraction_failed", error=str(exc))
                continue

            best_sim = -1.0
            best_record: Optional[Any] = None

            for record in gallery:
                ref_emb = (
                    record.embedding
                    if hasattr(record, "embedding")
                    else record["embedding"]
                )
                if isinstance(ref_emb, (bytes, bytearray)):
                    ref_emb = np.frombuffer(ref_emb, dtype=np.float32)
                elif isinstance(ref_emb, list):
                    ref_emb = np.array(ref_emb, dtype=np.float32)

                sim = self._recognizer.match(ref_emb, query_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_record = record

            best_sim_bounded = max(0.0, best_sim)
            thresh = self._recognizer.recognition_threshold

            if best_record is not None and best_sim >= thresh:
                status = "KNOWN"
                pid = str(
                    best_record.person_id
                    if hasattr(best_record, "person_id")
                    else best_record["person_id"]
                )
                pname = str(
                    best_record.name
                    if hasattr(best_record, "name")
                    else best_record["name"]
                )
            else:
                status = "UNKNOWN"
                pid = None
                pname = None

            result = FaceRecognitionResult(
                status=status,
                person_id=pid,
                name=pname,
                similarity=best_sim_bounded,
                bbox=face_bbox,
                detection_confidence=det_conf,
                timestamp_ns=ts_ns,
                camera_id=camera_id,
                track_id=track_id,
                global_id=pid if status == "KNOWN" else (str(track_global_id) if track_global_id else None),
            )
            results.append(result)

            # 3. Track-level de-duplication & throttling
            should_emit = self._should_emit_event(track_id, face_bbox, status, pid, ts_ns)
            if not should_emit:
                continue

            # 4. Safe zoomed face crop
            face_crop = self.crop_face_safely(frame, face_bbox)

            # 5. Capture full-frame + face crop evidence
            event_id = f"ev_face_{uuid.uuid4().hex[:12]}"
            evidence_meta = {
                "event_type": EventType.FACE_DETECTED.value,
                "camera_id": camera_id,
                "timestamp_ns": ts_ns,
                "frame_id": frame_id,
                "track_id": track_id,
                "global_id": result.global_id,
                "face_bbox": [face_bbox.x1, face_bbox.y1, face_bbox.x2, face_bbox.y2],
                "recognition_status": status,
                "person_id": pid,
                "name": pname,
                "similarity": best_sim_bounded,
                "detection_confidence": det_conf,
            }

            ev_dir = self._evidence_manager.capture_evidence(
                alert_id=event_id,
                camera_id=camera_id,
                timestamp_ns=ts_ns,
                frame=frame,
                metadata=evidence_meta,
                face_crop=face_crop,
            )

            # 6. Emit canonical EventType.FACE_DETECTED event into EventStore
            explanation = (
                f"Face recognized as {pname} (ID: {pid}) with similarity {best_sim_bounded:.2f}"
                if status == "KNOWN"
                else f"Unknown face detected (similarity: {best_sim_bounded:.2f})"
            )

            canonical_event = Event(
                event_id=event_id,
                event_type=EventType.FACE_DETECTED,
                global_id=result.global_id,
                camera_id=camera_id,
                zone_id=None,
                timestamp_ns=ts_ns,
                frame_id=frame_id,
                confidence=best_sim_bounded if status == "KNOWN" else det_conf,
                explanation=explanation,
                metadata={
                    "status": status,
                    "person_id": pid,
                    "name": pname,
                    "similarity": best_sim_bounded,
                    "face_bbox": [face_bbox.x1, face_bbox.y1, face_bbox.x2, face_bbox.y2],
                    "detection_confidence": det_conf,
                    "evidence_id": event_id,
                    "track_id": track_id,
                    "evidence_dir": ev_dir,
                },
            )

            self._event_store.append(canonical_event)
            _log.info(
                "face_detected_event_emitted",
                event_id=event_id,
                status=status,
                person_id=pid,
                name=pname,
                similarity=best_sim_bounded,
                camera_id=camera_id,
            )

        return results

    def _should_emit_event(
        self,
        track_id: Optional[str],
        face_bbox: BoundingBox,
        status: str,
        person_id: Optional[str],
        timestamp_ns: int,
    ) -> bool:
        """Determines if a face detection event should be emitted or throttled."""
        state_key = track_id if track_id is not None else f"untracked_{int(face_bbox.x1 // 50)}_{int(face_bbox.y1 // 50)}"
        states_dict = self._track_states if track_id is not None else self._untracked_states

        prior = states_dict.get(state_key)
        if prior is None:
            # First observation -> emit
            states_dict[state_key] = _TrackFaceState(
                last_status=status,
                last_person_id=person_id,
                last_emitted_ns=timestamp_ns,
            )
            return True

        # Check if state changed (KNOWN <-> UNKNOWN or identity change)
        state_changed = (prior.last_status != status) or (prior.last_person_id != person_id)
        if state_changed:
            states_dict[state_key] = _TrackFaceState(
                last_status=status,
                last_person_id=person_id,
                last_emitted_ns=timestamp_ns,
            )
            return True

        # Same state: check throttle window
        if (timestamp_ns - prior.last_emitted_ns) >= self._throttle_window_ns:
            states_dict[state_key].last_emitted_ns = timestamp_ns
            return True

        # Throttled
        return False

    @staticmethod
    def crop_face_safely(
        frame: np.ndarray, bbox: BoundingBox, margin_ratio: float = 0.1
    ) -> np.ndarray:
        """Safely crop face region from frame with margin, strictly bounded within image dimensions.

        Args:
            frame: Full BGR image.
            bbox: Face bounding box.
            margin_ratio: Margin to add around face bounding box.

        Returns:
            Cropped BGR face image.
        """
        img_h, img_w = frame.shape[:2]
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1

        pad_x = bw * margin_ratio
        pad_y = bh * margin_ratio

        x1 = max(0, int(np.floor(bbox.x1 - pad_x)))
        y1 = max(0, int(np.floor(bbox.y1 - pad_y)))
        x2 = min(img_w, int(np.ceil(bbox.x2 + pad_x)))
        y2 = min(img_h, int(np.ceil(bbox.y2 + pad_y)))

        if x2 <= x1 or y2 <= y1:
            # Degenerate bounds fallback to 1x1 safe pixel
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return frame[y1:y2, x1:x2].copy()
