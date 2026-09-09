"""Evidence Manager — Phase 10.2 (SIH 26187).

Captures pre/trigger/post frames keyed by alert_id and writes
metadata.json + snapshot.jpg to per-event directories.
Governed by EVIDENCE_RETENTION_DAYS TTL.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gods_eye.observability.logger import get_logger

_log = get_logger("evidence.evidence_manager")

EVIDENCE_RETENTION_DAYS_DEFAULT: int = 30


class EvidenceManager:
    """Evidence capture and storage manager.

    Captures frame snapshots and metadata for alert events.
    Stores evidence in per-alert directories with retention policy.
    """

    def __init__(
        self,
        base_path: str = "data/evidence",
        retention_days: int = EVIDENCE_RETENTION_DAYS_DEFAULT,
    ) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days

    def capture_evidence(
        self,
        alert_id: str,
        camera_id: str,
        timestamp_ns: int,
        frame: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
        face_crop: Optional[np.ndarray] = None,
    ) -> str:
        """Capture evidence for an alert or event.

        Args:
            alert_id: Alert or event identifier this evidence is linked to.
            camera_id: Source camera.
            timestamp_ns: Capture timestamp.
            frame: BGR frame to capture as snapshot.
            metadata: Additional metadata for the evidence record.
            face_crop: Cropped face BGR image to save as face_crop.jpg.

        Returns:
            Evidence directory path.
        """
        evidence_id = f"ev_{uuid.uuid4().hex[:12]}"
        evidence_dir = self._base_path / alert_id / evidence_id
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        meta = {
            "evidence_id": evidence_id,
            "alert_id": alert_id,
            "camera_id": camera_id,
            "timestamp_ns": timestamp_ns,
            "captured_at": time.time_ns(),
            "has_snapshot": frame is not None,
            "has_face_crop": face_crop is not None,
        }
        if metadata:
            meta.update(metadata)

        # Write snapshot
        if frame is not None:
            try:
                import cv2  # type: ignore[import-untyped]

                snapshot_path = evidence_dir / "snapshot.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                meta["snapshot_path"] = str(snapshot_path)
                _log.info(
                    "evidence_snapshot_captured",
                    evidence_id=evidence_id,
                    alert_id=alert_id,
                )
            except ImportError:
                _log.warning("opencv_not_available", msg="Cannot save snapshot")

        # Write face crop
        if face_crop is not None:
            try:
                import cv2  # type: ignore[import-untyped]

                crop_path = evidence_dir / "face_crop.jpg"
                cv2.imwrite(str(crop_path), face_crop)
                meta["face_crop_path"] = str(crop_path)
                _log.info(
                    "evidence_face_crop_captured",
                    evidence_id=evidence_id,
                    alert_id=alert_id,
                )
            except ImportError:
                _log.warning("opencv_not_available", msg="Cannot save face crop")

        meta_path = evidence_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return str(evidence_dir)

    def get_evidence(self, alert_id: str) -> list[dict[str, Any]]:
        """Retrieve all evidence records for an alert.

        Args:
            alert_id: Alert ID to look up.

        Returns:
            List of evidence metadata dicts.
        """
        alert_dir = self._base_path / alert_id
        if not alert_dir.exists():
            return []

        evidence_records: list[dict[str, Any]] = []

        for ev_dir in sorted(alert_dir.iterdir()):
            if not ev_dir.is_dir():
                continue
            meta_path = ev_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    evidence_records.append(json.load(f))

        return evidence_records

    def cleanup_expired(self) -> int:
        """Remove evidence older than retention period.

        Returns:
            Number of evidence directories removed.
        """
        import shutil

        cutoff_ns = time.time_ns() - (self._retention_days * 86_400 * 1_000_000_000)
        removed = 0

        if not self._base_path.exists():
            return 0

        for alert_dir in self._base_path.iterdir():
            if not alert_dir.is_dir():
                continue

            for ev_dir in alert_dir.iterdir():
                if not ev_dir.is_dir():
                    continue
                meta_path = ev_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        if meta.get("captured_at", 0) < cutoff_ns:
                            shutil.rmtree(ev_dir)
                            removed += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Remove empty alert dirs
            if alert_dir.exists() and not any(alert_dir.iterdir()):
                alert_dir.rmdir()

        if removed > 0:
            _log.info("evidence_cleanup_complete", removed=removed)

        return removed
