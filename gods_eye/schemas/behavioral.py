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
