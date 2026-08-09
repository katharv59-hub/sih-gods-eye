"""Behavioral Anomaly Detection Engine — Sub-Phase 6.4 (§6 Master Spec).

Evaluates spatial-temporal trajectory deviation of an observed TrajectorySequence
against established TrajectoryCluster patterns using Normalized Sequence Distance.

Operates deterministically without ML model inference or storage mutations.
"""

from __future__ import annotations

from typing import Optional

from gods_eye.behavioral.extractor import TrajectorySequence, normalized_sequence_distance
from gods_eye.schemas.behavioral import BehavioralAnomalyResult, TrajectoryCluster
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
