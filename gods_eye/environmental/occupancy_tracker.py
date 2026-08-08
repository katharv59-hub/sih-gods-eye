"""Occupancy Tracker & Baseline Builder — Phase 3.5 (§16, ADR-006).

Computes 64x64 zone/camera spatial occupancy heatmaps, maps detections to
zones, maintains 168 time-bucketed (day + hour) statistical baselines, and
evaluates scene-level occupancy anomaly signals.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Optional

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.environmental.background_model import BackgroundModeler
from gods_eye.environmental.lighting_classifier import LightingClassifier
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.camera import Zone
from gods_eye.schemas.detection import Detection
from gods_eye.schemas.environment import (
    BackgroundModel,
    LightingCondition,
    OccupancyBaseline,
    SceneState,
)

_log = get_logger("environmental.occupancy_tracker")

_DAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def get_time_bucket(timestamp_ns: int) -> str:
    """Format Unix nanoseconds into a 168-hour time-bucket string (e.g. 'MON_09')."""
    dt = datetime.datetime.fromtimestamp(
        timestamp_ns / 1e9, tz=datetime.timezone.utc
    )
    day_str = _DAY_NAMES[dt.weekday()]
    return f"{day_str}_{dt.hour:02d}"


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon query.

    Coordinates are normalized [0.0, 1.0].
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class OccupancyTracker:
    """Tracks spatial occupancy density and maintains statistical baselines per time-bucket.

    Owned exclusively by ``EnvironmentalWorker`` (single-writer §9).
    """

    def __init__(self, camera_id: str, settings: Settings) -> None:
        self._camera_id = camera_id
        self._min_baseline_samples = settings.warmup_occupancy_samples
        self._anomaly_sigma = settings.occupancy_anomaly_sigma

        # Heatmap grid 64x64
        self._grid_size = 64
        self._heatmap = np.zeros(
            (self._grid_size, self._grid_size), dtype=np.float32
        )

        # Baseline sample storage: key=(zone_id, time_bucket) -> list of occupancy counts
        # zone_id=None means entire camera scene
        self._samples: dict[tuple[Optional[str], str], list[int]] = defaultdict(
            list
        )
        self._last_bucket_update: dict[tuple[Optional[str], str], int] = {}

    def update(
        self,
        detections: list[Detection],
        zones: list[Zone],
        timestamp_ns: int,
        lighting: LightingCondition,
        bg_model_record: BackgroundModel,
        fg_mask: np.ndarray,
    ) -> tuple[SceneState, dict[Optional[str], OccupancyBaseline]]:
        """Process detections, update heatmaps and baselines, evaluate scene state.

        Args:
            detections: List of detections in current frame.
            zones: Camera zones from topology graph.
            timestamp_ns: Unix nanoseconds timestamp.
            lighting: Current lighting condition.
            bg_model_record: Current background model metadata.
            fg_mask: Current foreground motion mask.

        Returns:
            Tuple of (SceneState record, dict of zone_id -> OccupancyBaseline records).
        """
        occupancy_count = len(detections)
        time_bucket = get_time_bucket(timestamp_ns)

        # ── 1. Update Heatmap Grid ─────────────────────────────────────────────
        for det in detections:
            norm_x, norm_y = det.bbox.center
            # Map source resolution to [0,1] normalized coordinates if not already
            if det.source_resolution[0] > 0 and det.source_resolution[1] > 0:
                norm_x /= float(det.source_resolution[0])
                norm_y /= float(det.source_resolution[1])
            cx = int(np.clip(norm_x * self._grid_size, 0, self._grid_size - 1))
            cy = int(np.clip(norm_y * self._grid_size, 0, self._grid_size - 1))
            self._heatmap[cy, cx] += 1.0

        # Occupancy density: normalized area occupied
        occupancy_density = float(
            np.clip(occupancy_count / 50.0, 0.0, 1.0)
        )

        # Foreground motion ratio
        total_pixels = fg_mask.size
        fg_pixels = int(np.count_nonzero(fg_mask))
        fg_ratio = (
            float(fg_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        )

        # ── 2. Zone Mapping ────────────────────────────────────────────────────
        zone_counts: dict[Optional[str], int] = {None: occupancy_count}
        for zone in zones:
            z_count = 0
            for det in detections:
                norm_x, norm_y = det.bbox.center
                if det.source_resolution[0] > 0 and det.source_resolution[1] > 0:
                    norm_x /= float(det.source_resolution[0])
                    norm_y /= float(det.source_resolution[1])
                if point_in_polygon(norm_x, norm_y, zone.polygon):
                    z_count += 1
            zone_counts[zone.zone_id] = z_count

        # ── 3. Update Statistical Baselines ────────────────────────────────────
        baselines: dict[Optional[str], OccupancyBaseline] = {}
        is_scene_anomalous = False
        max_anomaly_conf = 0.0
        anomaly_reasons: list[str] = []

        for z_id, count in zone_counts.items():
            key = (z_id, time_bucket)
            self._samples[key].append(count)
            sample_list = self._samples[key]

            mean_occ = float(np.mean(sample_list))
            std_occ = float(np.std(sample_list)) if len(sample_list) > 1 else 0.0
            is_mature = len(sample_list) >= self._min_baseline_samples

            baseline = OccupancyBaseline(
                camera_id=self._camera_id,
                zone_id=z_id,
                time_bucket=time_bucket,
                mean_occupancy=round(mean_occ, 2),
                std_occupancy=round(std_occ, 2),
                sample_count=len(sample_list),
                last_updated_ns=timestamp_ns,
                is_mature=is_mature,
            )
            baselines[z_id] = baseline

            # Anomaly scoring against mature baseline
            if is_mature and std_occ > 0.01:
                sigma_diff = (count - mean_occ) / std_occ
                if sigma_diff > self._anomaly_sigma:
                    is_scene_anomalous = True
                    conf = min(1.0, 0.70 + (sigma_diff - self._anomaly_sigma) * 0.1)
                    if conf > max_anomaly_conf:
                        max_anomaly_conf = conf
                    z_name = f"zone {z_id}" if z_id else "camera scene"
                    anomaly_reasons.append(
                        f"Occupancy in {z_name} ({count}) exceeds baseline mean ({mean_occ:.1f}) by {sigma_diff:.1f}σ"
                    )

        scene_state = SceneState(
            camera_id=self._camera_id,
            timestamp_ns=timestamp_ns,
            lighting_condition=lighting,
            occupancy_count=occupancy_count,
            occupancy_density=occupancy_density,
            background_model_id=f"{self._camera_id}_v{bg_model_record.version}",
            foreground_ratio=round(fg_ratio, 4),
            is_anomalous=is_scene_anomalous,
            anomaly_confidence=round(max_anomaly_conf, 3),
            anomaly_explanation="; ".join(anomaly_reasons) if anomaly_reasons else "",
        )

        return scene_state, baselines

    @property
    def heatmap(self) -> np.ndarray:
        """Return a copy of the current 64x64 normalized spatial heatmap."""
        norm = np.max(self._heatmap)
        if norm > 0:
            return (self._heatmap / norm).copy()
        return self._heatmap.copy()
