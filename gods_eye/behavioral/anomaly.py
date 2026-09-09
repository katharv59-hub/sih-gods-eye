"""Behavioral Anomaly Detection Engine — Sub-Phase 6.4 (§6 Master Spec).

Evaluates spatial-temporal trajectory deviation of an observed TrajectorySequence
against established TrajectoryCluster patterns using Normalized Sequence Distance.

Operates deterministically without ML model inference or storage mutations.
"""

from __future__ import annotations

import math
from typing import Optional

from gods_eye.behavioral.extractor import TrajectorySequence, normalized_sequence_distance
from gods_eye.schemas.behavioral import (
    BehavioralAnomalyResult,
    SpatialAnomalyResult,
    SpatialTrajectory,
    SpatialTrajectoryCluster,
    TrajectoryCluster,
)
from gods_eye.schemas.reasoning import QueryStatus


class BehavioralAnomalyDetector:
    """Deterministic detector for behavioral trajectory anomalies."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize detector with configurable anomaly threshold.

        Args:
            threshold: Anomaly score threshold [0.0, 1.0] above which trajectory is flagged as anomalous.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold MUST be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def detect(
        self,
        trajectory: TrajectorySequence,
        clusters: list[TrajectoryCluster],
        threshold_override: Optional[float] = None,
    ) -> BehavioralAnomalyResult:
        """Evaluate behavioral trajectory anomaly relative to established clusters.

        Returns:
            BehavioralAnomalyResult
        """
        eff_threshold = threshold_override if threshold_override is not None else self.threshold
        if not (0.0 <= eff_threshold <= 1.0):
            raise ValueError(f"threshold MUST be between 0.0 and 1.0, got {eff_threshold}")

        # Insufficient evidence check: empty trajectory or no historical clusters
        if not trajectory or not trajectory.camera_sequence or not clusters:
            return BehavioralAnomalyResult(
                status=QueryStatus.INSUFFICIENT_EVIDENCE,
                is_anomalous=False,
                anomaly_score=0.0,
                threshold=eff_threshold,
                nearest_cluster_id=None,
                min_cluster_distance=0.0,
                confidence=0.0,
                explanation="Insufficient historical cluster data to evaluate behavioral anomaly.",
            )

        # Sort clusters deterministically by cluster_id for tie-breaking
        sorted_clusters = sorted(clusters, key=lambda c: (c.cluster_id, c.camera_sequence))

        min_dist = float("inf")
        nearest_cluster: Optional[TrajectoryCluster] = None

        for c in sorted_clusters:
            dist = normalized_sequence_distance(trajectory.camera_sequence, c.camera_sequence)
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = c

        if nearest_cluster is None:
            return BehavioralAnomalyResult(
                status=QueryStatus.INSUFFICIENT_EVIDENCE,
                is_anomalous=False,
                anomaly_score=0.0,
                threshold=eff_threshold,
                nearest_cluster_id=None,
                min_cluster_distance=0.0,
                confidence=0.0,
                explanation="No valid clusters found for spatial distance comparison.",
            )

        anomaly_score = round(min_dist, 4)
        is_anomalous = anomaly_score >= eff_threshold
        confidence = round(nearest_cluster.confidence, 4)

        explanation = (
            f"Trajectory has minimum normalized edit distance of {anomaly_score:.4f} "
            f"to nearest cluster {nearest_cluster.cluster_id}. "
            f"Flagged as {'ANOMALOUS' if is_anomalous else 'NORMAL'} "
            f"against threshold {eff_threshold:.4f}."
        )

        return BehavioralAnomalyResult(
            status=QueryStatus.SUCCESS,
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            threshold=eff_threshold,
            nearest_cluster_id=nearest_cluster.cluster_id,
            min_cluster_distance=anomaly_score,
            confidence=confidence,
            explanation=explanation,
        )

    def detect_spatial_anomaly(
        self,
        trajectory: SpatialTrajectory,
        clusters: list[SpatialTrajectoryCluster],
        min_points: int = 10,
        sigma_threshold: float = 3.5,
        allow_immature: bool = False,
    ) -> SpatialAnomalyResult:
        """Evaluate spatial trajectory deviation against baseline clusters (§4.5 & §17 Master Spec).

        Anomaly calculation follows ADR-010:
            distance(trajectory, nearest_cluster_centroid) / cluster_standard_deviation -> trajectory_anomaly_sigma

        Args:
            trajectory: Observed spatial trajectory with normalized track points.
            clusters: Baseline spatial trajectory clusters.
            min_points: Minimum trajectory points required (TRAJECTORY_MIN_POINTS = 10).
            sigma_threshold: Standard deviation threshold for anomaly flag (default 3.5σ).
            allow_immature: If True, evaluates against immature clusters (default False).

        Returns:
            SpatialAnomalyResult
        """
        pts_count = len(trajectory.points) if trajectory and trajectory.points else 0
        track_id = trajectory.track_id if trajectory else "unknown"
        camera_id = trajectory.camera_id if trajectory else "unknown"
        last_ts = (
            trajectory.points[-1].timestamp_ns
            if trajectory and trajectory.points
            else 0
        )

        if not trajectory or pts_count < min_points:
            return SpatialAnomalyResult(
                track_id=track_id,
                camera_id=camera_id,
                timestamp_ns=last_ts,
                anomaly_sigma=0.0,
                nearest_cluster_id=None,
                min_distance=0.0,
                is_anomalous=False,
                confidence=0.0,
                explanation=(
                    f"Trajectory has {pts_count} points, which is below the minimum required "
                    f"threshold of {min_points} points (TRAJECTORY_MIN_POINTS)."
                ),
                trajectory_points_count=pts_count,
            )

        if not clusters:
            return SpatialAnomalyResult(
                track_id=track_id,
                camera_id=camera_id,
                timestamp_ns=last_ts,
                anomaly_sigma=0.0,
                nearest_cluster_id=None,
                min_distance=0.0,
                is_anomalous=False,
                confidence=0.0,
                explanation="No baseline spatial trajectory clusters available for comparison.",
                trajectory_points_count=pts_count,
            )

        candidate_clusters = (
            [c for c in clusters if c.is_mature] if not allow_immature else clusters
        )
        if not candidate_clusters:
            return SpatialAnomalyResult(
                track_id=track_id,
                camera_id=camera_id,
                timestamp_ns=last_ts,
                anomaly_sigma=0.0,
                nearest_cluster_id=None,
                min_distance=0.0,
                is_anomalous=False,
                confidence=0.0,
                explanation="No mature trajectory clusters available for comparison (warm-up / learning mode active).",
                trajectory_points_count=pts_count,
            )

        # Check for NaN / Inf in trajectory points
        for pt in trajectory.points:
            if math.isnan(pt.x) or math.isnan(pt.y) or math.isinf(pt.x) or math.isinf(pt.y):
                return SpatialAnomalyResult(
                    track_id=track_id,
                    camera_id=camera_id,
                    timestamp_ns=last_ts,
                    anomaly_sigma=0.0,
                    nearest_cluster_id=None,
                    min_distance=0.0,
                    is_anomalous=False,
                    confidence=0.0,
                    explanation="Trajectory contains invalid numerical values (NaN/Inf).",
                    trajectory_points_count=pts_count,
                )

        min_dist = float("inf")
        nearest_cluster: Optional[SpatialTrajectoryCluster] = None

        sorted_clusters = sorted(
            candidate_clusters, key=lambda c: (c.cluster_id, len(c.centroid_waypoints))
        )

        for cluster in sorted_clusters:
            n_waypoints = len(cluster.centroid_waypoints)
            if n_waypoints < 1:
                continue

            resampled = trajectory.resample(n_points=n_waypoints)
            if len(resampled) != n_waypoints:
                continue

            total_dist = 0.0
            for (px, py), (cx, cy) in zip(resampled, cluster.centroid_waypoints):
                total_dist += math.hypot(px - cx, py - cy)
            dist = total_dist / float(n_waypoints)

            if dist < min_dist:
                min_dist = dist
                nearest_cluster = cluster

        if nearest_cluster is None:
            return SpatialAnomalyResult(
                track_id=track_id,
                camera_id=camera_id,
                timestamp_ns=last_ts,
                anomaly_sigma=0.0,
                nearest_cluster_id=None,
                min_distance=0.0,
                is_anomalous=False,
                confidence=0.0,
                explanation="No valid clusters with waypoints found.",
                trajectory_points_count=pts_count,
            )

        effective_std = max(nearest_cluster.std_dev, 1e-4)
        anomaly_sigma = round(min_dist / effective_std, 4)
        is_anomalous = anomaly_sigma >= sigma_threshold

        confidence = min(1.0, 0.70 + 0.01 * min(pts_count, 30))

        explanation = (
            f"Trajectory deviates from nearest known pattern (cluster: {nearest_cluster.cluster_id}) "
            f"by {anomaly_sigma:.1f}σ. "
            f"{'Unusual path toward restricted zone.' if is_anomalous else 'Normal trajectory within expected corridor.'}"
        )

        return SpatialAnomalyResult(
            track_id=track_id,
            camera_id=camera_id,
            timestamp_ns=last_ts,
            anomaly_sigma=anomaly_sigma,
            nearest_cluster_id=nearest_cluster.cluster_id,
            min_distance=round(min_dist, 4),
            is_anomalous=is_anomalous,
            confidence=round(confidence, 4),
            explanation=explanation,
            trajectory_points_count=pts_count,
        )

