"""Camera node and zone schemas — §4 Canonical Data Schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gods_eye.schemas.detection import BoundingBox


@dataclass
class Zone:
    """A logical region within a camera's field of view.

    Zones partition a camera's FOV into semantically meaningful areas
    (entrances, exits, dwell areas, transit corridors, restricted areas).
    Coordinates are normalized to [0, 1] within the camera frame.

    Used by:
        - Phase 3.5: Occupancy tracking per zone.
        - Phase 4.5: Trajectory zone visitation, dwell time analysis.
        - Phase 6: Identity behavioral profiles (typical_zones).

    Attributes:
        zone_id: Unique zone identifier (scoped to camera_id).
        camera_id: Camera this zone belongs to.
        display_name: Human-readable name (e.g. "Main Entrance").
        polygon: Normalized [0,1] vertices defining the zone boundary.
        zone_type: Semantic type — one of "entrance", "exit", "dwell",
                   "transit", "restricted".
        expected_dwell_s: (min, max) expected dwell time in seconds.
                          Seeded manually, refined empirically in Phase 4.5.
    """

    zone_id: str
    camera_id: str
    display_name: str
    polygon: list[tuple[float, float]]
    zone_type: str
    expected_dwell_s: tuple[float, float] = (0.0, 0.0)
    # Phase 9 — SIH 26187 zone intelligence extensions
    allowed_classes: list[str] = field(default_factory=lambda: ["person"])
    allowed_time_window: Optional[tuple[int, int]] = None  # (start_hour, end_hour) or None = 24/7
    direction_rules: Optional[str] = None  # e.g. "entry_only", "exit_only", None = bidirectional


@dataclass
class CameraNode:
    """A camera in the surveillance topology graph.

    Attributes:
        camera_id: Unique camera identifier.
        display_name: Human-readable name.
        source_uri: RTSP URL, device index, or file path.
        location_label: Human-readable physical location.
        resolution: (width, height) of the camera feed.
        fps_nominal: Expected frames per second.
        zones: Logical regions within this camera's FOV.
        adjacent_cameras: camera_ids reachable by direct physical path.
        transition_priors: {target_camera_id: probability} — Phase 3.
        overlap_regions: Shared FOV regions with adjacent cameras — Phase 3.
    """

    camera_id: str
    display_name: str
    source_uri: str
    location_label: str
    resolution: tuple[int, int]
    fps_nominal: float
    zones: list[Zone] = field(default_factory=list)
    adjacent_cameras: list[str] = field(default_factory=list)
    transition_priors: dict[str, float] = field(default_factory=dict)
    overlap_regions: list[BoundingBox] = field(default_factory=list)
