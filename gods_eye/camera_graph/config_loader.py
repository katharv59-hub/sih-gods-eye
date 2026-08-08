"""Camera topology configuration loader.

Parses YAML camera configuration files into ``CameraNode`` records
for constructing a ``CameraGraph``.

YAML format::

    cameras:
      - camera_id: "cam_01"
        display_name: "Front Entrance"
        source_uri: "rtsp://192.168.1.10:554/stream"
        location_label: "Building A, Floor 1, Main Entrance"
        resolution: [1920, 1080]
        fps_nominal: 25.0
        adjacent_cameras: ["cam_02", "cam_03"]
        transition_priors:
          cam_02: 0.6
          cam_03: 0.3
        zones:
          - zone_id: "z01"
            display_name: "Door Area"
            zone_type: "entrance"
            polygon: [[0.1, 0.8], [0.4, 0.8], [0.4, 1.0], [0.1, 1.0]]
            expected_dwell_s: [1.0, 10.0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.camera import CameraNode, Zone
from gods_eye.schemas.detection import BoundingBox

_log = get_logger("camera_graph.config_loader")


def load_camera_config(path: str | Path) -> list[CameraNode]:
    """Load camera topology from a YAML configuration file.

    Args:
        path: Path to the YAML camera configuration file.

    Returns:
        List of ``CameraNode`` records ready for ``CameraGraph``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Camera config not found: {config_path}"
        )

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "cameras" not in raw:
        raise ValueError(
            "Camera config must be a YAML dict with a 'cameras' key."
        )

    cameras_raw = raw["cameras"]
    if not isinstance(cameras_raw, list):
        raise ValueError("'cameras' must be a list of camera definitions.")

    nodes: list[CameraNode] = []
    for cam_data in cameras_raw:
        node = _parse_camera_node(cam_data)
        nodes.append(node)

    _log.info("camera_config_loaded", path=str(config_path), count=len(nodes))
    return nodes


def _parse_camera_node(data: dict[str, Any]) -> CameraNode:
    """Parse a single camera node from raw YAML data."""
    required_fields = [
        "camera_id", "display_name", "source_uri",
        "location_label", "resolution", "fps_nominal",
    ]
    for field_name in required_fields:
        if field_name not in data:
            raise ValueError(
                f"Camera definition missing required field: {field_name!r}. "
                f"Got keys: {list(data.keys())}"
            )

    resolution = data["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError(
            f"Camera {data['camera_id']!r}: resolution must be [w, h]."
        )

    # Parse zones
    zones: list[Zone] = []
    for zone_data in data.get("zones", []):
        zones.append(_parse_zone(data["camera_id"], zone_data))

    # Parse overlap regions
    overlaps: list[BoundingBox] = []
    for overlap_data in data.get("overlap_regions", []):
        overlaps.append(BoundingBox(
            x1=overlap_data["x1"], y1=overlap_data["y1"],
            x2=overlap_data["x2"], y2=overlap_data["y2"],
        ))

    return CameraNode(
        camera_id=data["camera_id"],
        display_name=data["display_name"],
        source_uri=data["source_uri"],
        location_label=data["location_label"],
        resolution=tuple(resolution),  # type: ignore[arg-type]
        fps_nominal=float(data["fps_nominal"]),
        zones=zones,
        adjacent_cameras=data.get("adjacent_cameras", []),
        transition_priors=data.get("transition_priors", {}),
        overlap_regions=overlaps,
    )


_VALID_ZONE_TYPES = frozenset({
    "entrance", "exit", "dwell", "transit", "restricted",
})


def _parse_zone(camera_id: str, data: dict[str, Any]) -> Zone:
    """Parse a single zone from raw YAML data."""
    required = ["zone_id", "display_name", "zone_type", "polygon"]
    for field_name in required:
        if field_name not in data:
            raise ValueError(
                f"Zone in camera {camera_id!r} missing field: {field_name!r}."
            )

    zone_type = data["zone_type"]
    if zone_type not in _VALID_ZONE_TYPES:
        raise ValueError(
            f"Zone {data['zone_id']!r} in camera {camera_id!r}: "
            f"invalid zone_type {zone_type!r}. "
            f"Must be one of {sorted(_VALID_ZONE_TYPES)}."
        )

    polygon = [tuple(pt) for pt in data["polygon"]]
    if len(polygon) < 3:
        raise ValueError(
            f"Zone {data['zone_id']!r}: polygon needs ≥ 3 vertices."
        )

    dwell = data.get("expected_dwell_s", [0.0, 0.0])
    if not isinstance(dwell, list) or len(dwell) != 2:
        raise ValueError(
            f"Zone {data['zone_id']!r}: expected_dwell_s must be [min, max]."
        )

    return Zone(
        zone_id=data["zone_id"],
        camera_id=camera_id,
        display_name=data["display_name"],
        polygon=polygon,
        zone_type=zone_type,
        expected_dwell_s=tuple(dwell),  # type: ignore[arg-type]
    )
