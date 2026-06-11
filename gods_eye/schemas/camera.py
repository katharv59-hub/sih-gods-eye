"""Camera node schemas — §4 Canonical Data Schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

from gods_eye.schemas.detection import BoundingBox


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
    adjacent_cameras: list[str] = field(default_factory=list)
    # Phase 3 fields — optional until Phase 3 gate
    transition_priors: dict[str, float] = field(default_factory=dict)
    overlap_regions: list[BoundingBox] = field(default_factory=list)
