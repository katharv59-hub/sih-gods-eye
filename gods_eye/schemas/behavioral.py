"""Canonical Behavioral & Predictive Data Schemas — Sub-Phase 6.1 & 6.4 (§5 & §6 Master Spec).

Defines canonical data contracts for Phase 6 Behavioral Intelligence:
- TrajectoryCluster (Dataclass)
- PredictedDestination (Dataclass)
- PredictionResult (Dataclass)
- BehavioralAnomalyResult (Dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from gods_eye.schemas.reasoning import QueryStatus


@dataclass
class TrajectoryCluster:
    """Clustered spatial route sequence pattern with identity privacy minimization (§5 Master Spec).

    Preserves privacy contract: stores aggregate identity counts (unique_identity_count)
    rather than raw lists of global_id UUIDs.
    """

    cluster_id: str
    camera_sequence: list[str]
    occurrence_count: int
    unique_identity_count: int
    mean_duration_s: float
    confidence: float

    def __post_init__(self) -> None:
        """Validate TrajectoryCluster invariants."""
        if not self.cluster_id or not self.cluster_id.strip():
            raise ValueError("cluster_id MUST be a non-empty string")
        if not self.camera_sequence or len(self.camera_sequence) == 0:
            raise ValueError("camera_sequence MUST be a non-empty list of camera IDs")
        for cam in self.camera_sequence:
            if not isinstance(cam, str) or not cam.strip():
                raise ValueError("camera_sequence elements MUST be non-empty strings")
        if self.occurrence_count < 0:
            raise ValueError(f"occurrence_count MUST be non-negative, got {self.occurrence_count}")
        if self.unique_identity_count < 0:
            raise ValueError(
                f"unique_identity_count MUST be non-negative, got {self.unique_identity_count}"
            )
        if self.mean_duration_s < 0.0:
            raise ValueError(f"mean_duration_s MUST be non-negative, got {self.mean_duration_s}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "cluster_id": self.cluster_id,
            "camera_sequence": list(self.camera_sequence),
            "occurrence_count": self.occurrence_count,
            "unique_identity_count": self.unique_identity_count,
            "mean_duration_s": self.mean_duration_s,
            "confidence": self.confidence,
        }


@dataclass
class PredictedDestination:
    """Candidate next location prediction with distinct probability and confidence (§5 Master Spec).

    Note:
    - probability: Mathematical likelihood of transition to destination camera [0.0, 1.0].
    - confidence: Provenance strength of underlying historical evidence [0.0, 1.0].
    """

    camera_id: str
    probability: float
    confidence: float
    confidence_band: tuple[float, float]
    expected_arrival_ns: int

    def __post_init__(self) -> None:
        """Validate PredictedDestination invariants."""
        if not self.camera_id or not self.camera_id.strip():
            raise ValueError("camera_id MUST be a non-empty string")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability MUST be between 0.0 and 1.0, got {self.probability}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")
        lower, upper = self.confidence_band
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError(
                f"confidence_band MUST satisfy 0.0 <= lower <= upper <= 1.0, got {self.confidence_band}"
            )
        if self.expected_arrival_ns <= 0:
            raise ValueError("expected_arrival_ns MUST be a positive Unix nanosecond timestamp")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "camera_id": self.camera_id,
            "probability": self.probability,
            "confidence": self.confidence,
            "confidence_band": list(self.confidence_band),
            "expected_arrival_ns": self.expected_arrival_ns,
        }


@dataclass
class PredictionResult:
    """Probabilistic next-location prediction payload (§5 Master Spec)."""

    global_id: str
    current_camera_id: str
    timestamp_ns: int
    predictions: list[PredictedDestination] = field(default_factory=list)
    overall_confidence: float = 1.0
    evidence_count: int = 0
    model_version: str = "v1.0"
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate PredictionResult invariants."""
        if not self.global_id or not self.global_id.strip():
            raise ValueError("global_id MUST be a non-empty string")
        if not self.current_camera_id or not self.current_camera_id.strip():
            raise ValueError("current_camera_id MUST be a non-empty string")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(
                f"overall_confidence MUST be between 0.0 and 1.0, got {self.overall_confidence}"
            )
        if self.evidence_count < 0:
            raise ValueError(f"evidence_count MUST be non-negative, got {self.evidence_count}")
        if not self.model_version or not self.model_version.strip():
            raise ValueError("model_version MUST be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "global_id": self.global_id,
            "current_camera_id": self.current_camera_id,
            "timestamp_ns": self.timestamp_ns,
            "predictions": [p.to_dict() for p in self.predictions],
            "overall_confidence": self.overall_confidence,
            "evidence_count": self.evidence_count,
            "model_version": self.model_version,
            "explanation": self.explanation,
        }


import math


@dataclass
class BehavioralAnomalyResult:
    """Canonical behavioral anomaly evaluation result (§6 Master Spec)."""

    status: QueryStatus
    is_anomalous: bool
    anomaly_score: float
    threshold: float
    nearest_cluster_id: Optional[str] = None
    min_cluster_distance: float = 0.0
    confidence: float = 1.0
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate BehavioralAnomalyResult invariants."""
        if not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError(f"anomaly_score MUST be between 0.0 and 1.0, got {self.anomaly_score}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold MUST be between 0.0 and 1.0, got {self.threshold}")
        if not (0.0 <= self.min_cluster_distance <= 1.0):
            raise ValueError(
                f"min_cluster_distance MUST be between 0.0 and 1.0, got {self.min_cluster_distance}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "status": self.status.value,
            "is_anomalous": self.is_anomalous,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "nearest_cluster_id": self.nearest_cluster_id,
            "min_cluster_distance": self.min_cluster_distance,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class SpatialPoint:
    """Normalized spatial point in camera FOV [0.0, 1.0]^2 with timestamp (§4.5 Master Spec)."""

    x: float
    y: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        if math.isnan(self.x) or math.isinf(self.x) or math.isnan(self.y) or math.isinf(self.y):
            raise ValueError(f"SpatialPoint coordinates must not be NaN or Inf, got x={self.x}, y={self.y}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")


@dataclass
class TrajectoryFeature:
    """Fixed-length feature vector extracted from a spatial trajectory (§4.5 & ADR-009).

    Contains resampled spatial waypoints, mean normalized speed, and total duration.
    """

    waypoints: list[tuple[float, float]]
    mean_speed: float
    duration_s: float

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("waypoints MUST be non-empty")
        if self.mean_speed < 0.0 or math.isnan(self.mean_speed) or math.isinf(self.mean_speed):
            raise ValueError(f"mean_speed must be non-negative and finite, got {self.mean_speed}")
        if self.duration_s < 0.0 or math.isnan(self.duration_s) or math.isinf(self.duration_s):
            raise ValueError(f"duration_s must be non-negative and finite, got {self.duration_s}")

    def to_vector(self) -> list[float]:
        """Flatten feature representation into a 1D float list for distance computation."""
        vec: list[float] = []
        for x, y in self.waypoints:
            vec.extend([x, y])
        vec.append(self.mean_speed)
        vec.append(self.duration_s)
        return vec


@dataclass
class SpatialTrajectory:
    """Normalized spatial trajectory extracted from track observations (§4.5 & §17 Master Spec)."""

    track_id: str
    camera_id: str
    points: list[SpatialPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def compute_duration_s(self) -> float:
        """Calculate total trajectory duration in seconds."""
        if len(self.points) < 2:
            return 0.0
        delta_ns = self.points[-1].timestamp_ns - self.points[0].timestamp_ns
        return max(0.0, float(delta_ns) / 1e9)

    def compute_mean_speed(self) -> float:
        """Calculate mean speed in normalized FOV distance per second."""
        if len(self.points) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(1, len(self.points)):
            p1, p2 = self.points[i - 1], self.points[i]
            total_dist += math.hypot(p2.x - p1.x, p2.y - p1.y)
        duration_s = self.compute_duration_s()
        if duration_s <= 1e-6:
            return 0.0
        return total_dist / duration_s

    def extract_features(self, n_waypoints: int = 10) -> TrajectoryFeature:
        """Extract canonical TrajectoryFeature (§4.5 & ADR-009)."""
        resampled = self.resample(n_points=n_waypoints)
        return TrajectoryFeature(
            waypoints=resampled,
            mean_speed=self.compute_mean_speed(),
            duration_s=self.compute_duration_s(),
        )

    def resample(self, n_points: int = 10) -> list[tuple[float, float]]:
        """Resample trajectory into fixed N waypoints along cumulative path length.

        Returns:
            List of N (x, y) tuples normalized in [0.0, 1.0].
        """
        if n_points < 1:
            raise ValueError(f"n_points must be at least 1, got {n_points}")
        if not self.points:
            return []
        if len(self.points) == 1 or n_points == 1:
            pt = (self.points[0].x, self.points[0].y)
            return [pt] * n_points

        # Calculate cumulative distance along path
        cum_dist = [0.0]
        for i in range(1, len(self.points)):
            p1 = self.points[i - 1]
            p2 = self.points[i]
            d = math.hypot(p2.x - p1.x, p2.y - p1.y)
            cum_dist.append(cum_dist[-1] + d)

        total_length = cum_dist[-1]
        if total_length <= 1e-9:
            # Subject was stationary
            pt = (self.points[0].x, self.points[0].y)
            return [pt] * n_points

        resampled: list[tuple[float, float]] = []
        step = total_length / (n_points - 1)
        curr_idx = 0

        for j in range(n_points):
            target_d = j * step
            while curr_idx < len(cum_dist) - 2 and cum_dist[curr_idx + 1] < target_d:
                curr_idx += 1

            d0 = cum_dist[curr_idx]
            d1 = cum_dist[curr_idx + 1]
            seg_len = d1 - d0
            p0 = self.points[curr_idx]
            p1 = self.points[curr_idx + 1]

            if seg_len <= 1e-9:
                resampled.append((p0.x, p0.y))
            else:
                alpha = max(0.0, min(1.0, (target_d - d0) / seg_len))
                rx = p0.x + alpha * (p1.x - p0.x)
                ry = p0.y + alpha * (p1.y - p0.y)
                resampled.append((rx, ry))

        return resampled


@dataclass
class SpatialTrajectoryCluster:
    """Canonical spatial trajectory cluster pattern (§4.5 & §17 Master Spec).

    Represents a normal trajectory pattern in camera FOV with empirical centroid waypoints,
    empirical standard deviation, and member count derived solely from observed members.
    """

    cluster_id: str
    centroid_waypoints: list[tuple[float, float]]
    std_dev: float
    sample_count: int = 0
    is_mature: bool = False
    camera_id: str = ""
    mean_speed: float = 0.0
    mean_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.cluster_id or not self.cluster_id.strip():
            raise ValueError("cluster_id MUST be a non-empty string")
        if not self.centroid_waypoints:
            raise ValueError("centroid_waypoints MUST be non-empty")
        for pt in self.centroid_waypoints:
            if len(pt) != 2:
                raise ValueError(f"centroid_waypoints elements must be 2-tuples, got {pt}")
            if math.isnan(pt[0]) or math.isnan(pt[1]) or math.isinf(pt[0]) or math.isinf(pt[1]):
                raise ValueError(f"Centroid coordinates must not be NaN or Inf, got {pt}")
        if self.std_dev < 0.0 or math.isnan(self.std_dev) or math.isinf(self.std_dev):
            raise ValueError(f"std_dev MUST be non-negative and finite, got {self.std_dev}")
        if self.sample_count < 0:
            raise ValueError(f"sample_count MUST be non-negative, got {self.sample_count}")
        if self.mean_speed < 0.0 or math.isnan(self.mean_speed) or math.isinf(self.mean_speed):
            raise ValueError(f"mean_speed must be non-negative and finite, got {self.mean_speed}")
        if self.mean_duration_s < 0.0 or math.isnan(self.mean_duration_s) or math.isinf(self.mean_duration_s):
            raise ValueError(f"mean_duration_s must be non-negative and finite, got {self.mean_duration_s}")


@dataclass
class SpatialAnomalyResult:
    """Canonical spatial trajectory anomaly evaluation result (§4.5 Master Spec)."""

    track_id: str
    camera_id: str
    timestamp_ns: int
    anomaly_sigma: float
    nearest_cluster_id: Optional[str]
    min_distance: float
    is_anomalous: bool
    confidence: float
    explanation: str
    trajectory_points_count: int

    def __post_init__(self) -> None:
        if self.anomaly_sigma < 0.0 or math.isnan(self.anomaly_sigma) or math.isinf(self.anomaly_sigma):
            raise ValueError(f"anomaly_sigma must be non-negative and finite, got {self.anomaly_sigma}")
        if self.min_distance < 0.0 or math.isnan(self.min_distance) or math.isinf(self.min_distance):
            raise ValueError(f"min_distance must be non-negative and finite, got {self.min_distance}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "timestamp_ns": self.timestamp_ns,
            "anomaly_sigma": self.anomaly_sigma,
            "nearest_cluster_id": self.nearest_cluster_id,
            "min_distance": self.min_distance,
            "is_anomalous": self.is_anomalous,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "trajectory_points_count": self.trajectory_points_count,
        }

