"""Environmental Intelligence schemas — §4 Canonical Data Schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SystemMode(Enum):
    """Operational mode governing output trust (§6)."""

    LEARNING_MODE = "learning_mode"
    OPERATIONAL_MODE = "operational_mode"
    DEGRADED_MODE = "degraded_mode"


class LightingCondition(Enum):
    """Ambient lighting conditions (§4)."""

    DAY_BRIGHT = "day_bright"
    DAY_NORMAL = "day_normal"
    DAY_OVERCAST = "day_overcast"
    DUSK_DAWN = "dusk_dawn"
    NIGHT_LIT = "night_lit"
    NIGHT_DARK = "night_dark"
    ARTIFICIAL = "artificial"


@dataclass
class BackgroundModel:
    """Metadata record for a camera's background model (§4)."""

    camera_id: str
    model_type: str  # "gmm" | "running_average"
    version: int
    created_ns: int
    last_updated_ns: int
    warm_up_complete: bool
    frame_count: int
    confidence: float  # [0.0, 1.0] stability score


@dataclass
class SceneState:
    """Instantaneous environmental state of a camera scene (§4)."""

    camera_id: str
    timestamp_ns: int
    lighting_condition: LightingCondition
    occupancy_count: int
    occupancy_density: float  # persons / area [0.0, 1.0]
    background_model_id: str
    foreground_ratio: float  # fraction of frame with motion
    is_anomalous: bool
    anomaly_confidence: float
    anomaly_explanation: str


@dataclass
class OccupancyBaseline:
    """Statistical occupancy baseline for a camera/zone time bucket (§4)."""

    camera_id: str
    zone_id: Optional[str]
    time_bucket: str  # "MON_09", "TUE_14", etc.
    mean_occupancy: float
    std_occupancy: float
    sample_count: int
    last_updated_ns: int
    is_mature: bool  # sample_count >= MIN_BASELINE_SAMPLES
