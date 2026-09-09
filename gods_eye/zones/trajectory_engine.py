"""Trajectory Anomaly Engine — Phase 4.5 & Phase 9 (SIH 26187).

Computes spatial trajectory deviation from observed video track history
against established baseline clusters. Feeds trajectory_anomaly_sigma
directly into canonical Event -> Evidence -> SituationalRiskEvaluator.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from gods_eye.behavioral.anomaly import BehavioralAnomalyDetector
from gods_eye.behavioral.clustering import fit_spatial_trajectory_clusters
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.behavioral import (
    SpatialAnomalyResult,
    SpatialPoint,
    SpatialTrajectory,
    SpatialTrajectoryCluster,
)
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.event import Event, EventType

_log = get_logger("zones.trajectory_engine")

TRAJECTORY_MIN_POINTS: int = 10
WARMUP_TRAJECTORY_COUNT: int = 100
MIN_CLUSTER_SAMPLES: int = 20


class TrajectoryAnomalyEngine:
    """Per-camera spatial trajectory tracking and anomaly detection engine (§4.5 & §17 Master Spec).

    Extracts normalized spatial trajectories from real track bounding boxes across frames.
    Operates in LEARNING_MODE during warm-up until clusters are learned from observed tracks.
    When mature clusters exist (member_count >= MIN_CLUSTER_SAMPLES), evaluates deviation:

        sigma = distance(trajectory, nearest_cluster_centroid) / cluster_standard_deviation

    Produces canonical TRAJECTORY_ANOMALY Events and Evidence payloads with zero synthetic injection.
    """

    def __init__(
        self,
        min_points: int = TRAJECTORY_MIN_POINTS,
        sigma_threshold: float = 3.5,
        clusters: Optional[list[SpatialTrajectoryCluster]] = None,
        learning_mode: bool = True,
        warmup_trajectory_count: int = WARMUP_TRAJECTORY_COUNT,
        min_cluster_samples: int = MIN_CLUSTER_SAMPLES,
    ) -> None:
        """Initialize TrajectoryAnomalyEngine.

        Args:
            min_points: Minimum track points required to evaluate (default 10).
            sigma_threshold: Standard deviation threshold for anomaly classification (default 3.5σ).
            clusters: Optional baseline trajectory clusters. If None, initializes with empty clusters.
            learning_mode: Whether learning mode / warm-up is active (default True).
            warmup_trajectory_count: Target trajectory count for warm-up phase (default 100).
            min_cluster_samples: Minimum member count for a cluster to become mature (default 20).
        """
        if min_points < 1:
            raise ValueError(f"min_points MUST be at least 1, got {min_points}")
        if sigma_threshold < 0.0:
            raise ValueError(f"sigma_threshold MUST be non-negative, got {sigma_threshold}")
        if warmup_trajectory_count < 1:
            raise ValueError(f"warmup_trajectory_count MUST be at least 1, got {warmup_trajectory_count}")
        if min_cluster_samples < 1:
            raise ValueError(f"min_cluster_samples MUST be at least 1, got {min_cluster_samples}")

        self.min_points = min_points
        self.sigma_threshold = sigma_threshold
        self.warmup_trajectory_count = warmup_trajectory_count
        self.min_cluster_samples = min_cluster_samples

        if clusters is not None and len(clusters) > 0:
            self._clusters = list(clusters)
            # If any supplied cluster is mature, exit learning mode
            self.learning_mode = not any(c.is_mature for c in self._clusters)
        else:
            self._clusters = []
            self.learning_mode = learning_mode

        self._detector = BehavioralAnomalyDetector(threshold=0.5)

        # {track_id: list[SpatialPoint]}
        self._track_histories: dict[str, list[SpatialPoint]] = {}
        # {track_id: camera_id}
        self._track_cameras: dict[str, str] = {}
        # Completed / eligible trajectories for cluster fitting
        self._learning_samples: list[SpatialTrajectory] = []

    def register_cluster(self, cluster: SpatialTrajectoryCluster) -> None:
        """Register a learned spatial trajectory cluster."""
        self._clusters.append(cluster)
        if cluster.is_mature:
            self.learning_mode = False

    def set_clusters(self, clusters: list[SpatialTrajectoryCluster]) -> None:
        """Replace all baseline spatial trajectory clusters."""
        self._clusters = list(clusters)
        if any(c.is_mature for c in self._clusters):
            self.learning_mode = False

    def add_learning_sample(self, trajectory: SpatialTrajectory) -> bool:
        """Add an observed trajectory sample to the learning pool if eligible."""
        if len(trajectory.points) >= self.min_points:
            self._learning_samples.append(trajectory)
            return True
        return False

    def get_learning_samples(self) -> list[SpatialTrajectory]:
        """Return all eligible learning trajectory samples collected so far."""
        # Include explicit learning samples
        samples = list(self._learning_samples)
        sample_track_ids = {s.track_id for s in samples}

        # Also include completed track histories with >= min_points
        for tid, pts in self._track_histories.items():
            if tid not in sample_track_ids and len(pts) >= self.min_points:
                cam = self._track_cameras.get(tid, "unknown")
                samples.append(SpatialTrajectory(track_id=tid, camera_id=cam, points=list(pts)))
        return samples

    def fit_clusters(
        self,
        min_cluster_size: int = 3,
        min_cluster_samples: Optional[int] = None,
        n_waypoints: int = 10,
    ) -> list[SpatialTrajectoryCluster]:
        """Fit spatial trajectory clusters from collected learning samples (§4.5 & §17 Master Spec).

        Uses HDBSCAN over resampled waypoint features to derive empirical centroid
        waypoints, standard deviation, member count, and maturity.
        """
        eff_min_samples = (
            min_cluster_samples if min_cluster_samples is not None else self.min_cluster_samples
        )
        samples = self.get_learning_samples()
        learned = fit_spatial_trajectory_clusters(
            trajectories=samples,
            min_cluster_size=min_cluster_size,
            min_cluster_samples=eff_min_samples,
            n_waypoints=n_waypoints,
        )
        self._clusters = learned
        if any(c.is_mature for c in self._clusters):
            self.learning_mode = False
        return list(self._clusters)

    def record_track_point(
        self,
        track_id: str,
        bbox: BoundingBox,
        camera_id: str,
        timestamp_ns: int,
        source_resolution: tuple[int, int],
    ) -> Optional[SpatialAnomalyResult]:
        """Record an observed track position from a video frame.

        Args:
            track_id: Subject track identifier.
            bbox: Track bounding box in pixel coordinates.
            camera_id: Source camera identifier.
            timestamp_ns: Unix nanosecond timestamp of observation.
            source_resolution: (width, height) of the video frame.

        Returns:
            SpatialAnomalyResult if evaluated, None otherwise.
        """
        width, height = source_resolution
        if width <= 0 or height <= 0:
            raise ValueError(f"source_resolution must be positive, got {source_resolution}")

        # Compute normalized center of bounding box in [0.0, 1.0]
        norm_cx = (bbox.x1 + bbox.x2) / (2.0 * float(width))
        norm_cy = (bbox.y1 + bbox.y2) / (2.0 * float(height))

        # Clamp to [0.0, 1.0]
        norm_cx = max(0.0, min(1.0, norm_cx))
        norm_cy = max(0.0, min(1.0, norm_cy))

        pt = SpatialPoint(x=norm_cx, y=norm_cy, timestamp_ns=timestamp_ns)

        if track_id not in self._track_histories:
            self._track_histories[track_id] = []
            self._track_cameras[track_id] = camera_id

        self._track_histories[track_id].append(pt)

        # Evaluate if sufficient points exist
        if len(self._track_histories[track_id]) >= self.min_points:
            return self.evaluate_track(track_id)
        return None

    def evaluate_track(
        self, track_id: str, allow_immature: bool = False
    ) -> Optional[SpatialAnomalyResult]:
        """Evaluate a track's spatial trajectory against baseline clusters.

        Args:
            track_id: Track identifier to evaluate.
            allow_immature: Whether to evaluate against immature clusters (default False).

        Returns:
            SpatialAnomalyResult if track exists and has sufficient points, None otherwise.
        """
        pts = self._track_histories.get(track_id, [])
        camera_id = self._track_cameras.get(track_id, "unknown")

        if len(pts) < self.min_points:
            return None

        traj = SpatialTrajectory(
            track_id=track_id,
            camera_id=camera_id,
            points=list(pts),
        )

        if self.learning_mode and not allow_immature:
            last_ts = pts[-1].timestamp_ns if pts else 0
            return SpatialAnomalyResult(
                track_id=track_id,
                camera_id=camera_id,
                timestamp_ns=last_ts,
                anomaly_sigma=0.0,
                nearest_cluster_id=None,
                min_distance=0.0,
                is_anomalous=False,
                confidence=0.0,
                explanation="Learning mode active (warm-up): insufficient learned baseline for anomaly evaluation.",
                trajectory_points_count=len(pts),
            )

        return self._detector.detect_spatial_anomaly(
            trajectory=traj,
            clusters=self._clusters,
            min_points=self.min_points,
            sigma_threshold=self.sigma_threshold,
            allow_immature=allow_immature,
        )

    def build_evidence_payload(self, anomaly_result: SpatialAnomalyResult) -> dict[str, Any]:
        """Build evidence payload dict for SituationalRiskEvaluator.

        Sets trajectory_anomaly_sigma so the evaluator's CRITICAL rule
        (trajectory_anomaly_sigma > 3.5) and HIGH rule can fire without
        synthetic injection.
        """
        return {
            "track_id": anomaly_result.track_id,
            "subject_ref": anomaly_result.track_id,
            "camera_id": anomaly_result.camera_id,
            "trajectory_anomaly_sigma": anomaly_result.anomaly_sigma,
            "min_cluster_distance": anomaly_result.min_distance,
            "nearest_cluster_id": anomaly_result.nearest_cluster_id,
            "is_anomalous": anomaly_result.is_anomalous,
            "confidence": anomaly_result.confidence,
            "trajectory_points_count": anomaly_result.trajectory_points_count,
        }

    def to_event(self, anomaly_result: SpatialAnomalyResult, frame_id: int = 1) -> Event:
        """Adapt a SpatialAnomalyResult into a canonical Event record."""
        return Event(
            event_id=f"ev_traj_{anomaly_result.track_id}_{anomaly_result.timestamp_ns}",
            event_type=EventType.TRAJECTORY_ANOMALY,
            global_id=anomaly_result.track_id,
            camera_id=anomaly_result.camera_id,
            timestamp_ns=anomaly_result.timestamp_ns,
            frame_id=frame_id,
            confidence=anomaly_result.confidence,
            explanation=anomaly_result.explanation,
            zone_id=None,
            metadata=self.build_evidence_payload(anomaly_result),
        )

    def get_track_points_count(self, track_id: str) -> int:
        """Return the number of recorded points for a given track."""
        return len(self._track_histories.get(track_id, []))

    def reset(self) -> None:
        """Clear all recorded track histories and learning samples."""
        self._track_histories.clear()
        self._track_cameras.clear()
        self._learning_samples.clear()
