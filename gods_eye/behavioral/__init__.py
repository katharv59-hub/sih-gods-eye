"""Behavioral Intelligence & Spatial Trajectory Module — Phase 6.

Sub-Phase 6.2 deliverable: Trajectory sequence extraction, temporal features,
and sequence distance primitives.

Sub-Phase 6.3 deliverable: Spatial trajectory clustering engine.

Sub-Phase 6.4 deliverable: Behavioral anomaly detection engine.

Sub-Phase 6.5 deliverable: Markov behavioral predictor.
"""

from gods_eye.behavioral.anomaly import BehavioralAnomalyDetector
from gods_eye.behavioral.clustering import (
    TrajectoryClusteringEngine,
    compute_trajectory_feature_distance,
    fit_spatial_trajectory_clusters,
    hdbscan_cluster,
)
from gods_eye.behavioral.extractor import (
    TrajectorySequence,
    TrajectorySequenceExtractor,
    normalized_sequence_distance,
)
from gods_eye.behavioral.predictor import MarkovBehavioralPredictor
from gods_eye.schemas.behavioral import (
    SpatialAnomalyResult,
    SpatialPoint,
    SpatialTrajectory,
    SpatialTrajectoryCluster,
    TrajectoryFeature,
)

__all__ = [
    "TrajectorySequence",
    "TrajectorySequenceExtractor",
    "normalized_sequence_distance",
    "TrajectoryClusteringEngine",
    "compute_trajectory_feature_distance",
    "fit_spatial_trajectory_clusters",
    "hdbscan_cluster",
    "BehavioralAnomalyDetector",
    "MarkovBehavioralPredictor",
    "SpatialPoint",
    "SpatialTrajectory",
    "SpatialTrajectoryCluster",
    "SpatialAnomalyResult",
    "TrajectoryFeature",
]
