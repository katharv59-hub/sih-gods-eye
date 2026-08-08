"""Environmental Intelligence Layer — Phase 3.5 (§16, ADR-005, ADR-006, ADR-007).

Builds a persistent, adaptive model of each camera's baseline environment.
"""

from gods_eye.environmental.background_model import BackgroundModeler
from gods_eye.environmental.lighting_classifier import LightingClassifier
from gods_eye.environmental.occupancy_tracker import (
    OccupancyTracker,
    get_time_bucket,
    point_in_polygon,
)
from gods_eye.environmental.operational_mode import OperationalModeManager

__all__ = [
    "BackgroundModeler",
    "LightingClassifier",
    "OccupancyTracker",
    "OperationalModeManager",
    "get_time_bucket",
    "point_in_polygon",
]
