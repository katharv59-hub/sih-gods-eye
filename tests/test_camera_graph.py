"""Tests for camera graph — Phase 3, Component 2.

Tests CameraGraph construction, validation, and queries.
"""

from __future__ import annotations

import pytest

from gods_eye.camera_graph.camera_graph import CameraGraph
from gods_eye.schemas.camera import CameraNode, Zone


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_node(
    camera_id: str,
    adjacent: list[str] | None = None,
    priors: dict[str, float] | None = None,
    zones: list[Zone] | None = None,
) -> CameraNode:
    return CameraNode(
        camera_id=camera_id,
        display_name=f"Camera {camera_id}",
        source_uri=f"rtsp://test/{camera_id}",
        location_label=f"Location {camera_id}",
        resolution=(1920, 1080),
        fps_nominal=25.0,
        zones=zones or [],
        adjacent_cameras=adjacent or [],
        transition_priors=priors or {},
    )


def _two_camera_graph() -> CameraGraph:
    """Simple two-camera bidirectional graph."""
    return CameraGraph([
        _make_node("cam_a", adjacent=["cam_b"], priors={"cam_b": 0.7}),
        _make_node("cam_b", adjacent=["cam_a"], priors={"cam_a": 0.6}),
    ])


def _three_camera_chain() -> CameraGraph:
    """Three cameras in a chain: A ↔ B ↔ C."""
    return CameraGraph([
        _make_node("cam_a", adjacent=["cam_b"], priors={"cam_b": 0.8}),
        _make_node("cam_b", adjacent=["cam_a", "cam_c"],
                   priors={"cam_a": 0.4, "cam_c": 0.5}),
        _make_node("cam_c", adjacent=["cam_b"], priors={"cam_b": 0.9}),
    ])


# ── Construction Tests ────────────────────────────────────────────────────


class TestCameraGraphConstruction:

    def test_empty_graph(self) -> None:
        graph = CameraGraph([])
        assert graph.camera_count == 0
        assert graph.all_camera_ids() == []

    def test_single_camera(self) -> None:
        graph = CameraGraph([_make_node("cam_a")])
        assert graph.camera_count == 1
        assert graph.has_camera("cam_a")

    def test_two_camera_graph(self) -> None:
        graph = _two_camera_graph()
        assert graph.camera_count == 2
        assert graph.has_camera("cam_a")
        assert graph.has_camera("cam_b")

    def test_three_camera_chain(self) -> None:
        graph = _three_camera_chain()
        assert graph.camera_count == 3
        assert graph.all_camera_ids() == ["cam_a", "cam_b", "cam_c"]

    def test_zones_stored(self) -> None:
        zone = Zone(
            zone_id="z1", camera_id="cam_a",
            display_name="Entrance", polygon=[(0, 0), (1, 0), (1, 1)],
            zone_type="entrance", expected_dwell_s=(1.0, 10.0),
        )
        graph = CameraGraph([_make_node("cam_a", zones=[zone])])
        node = graph.get_node("cam_a")
        assert node is not None
        assert len(node.zones) == 1
        assert node.zones[0].zone_id == "z1"


# ── Validation Tests ──────────────────────────────────────────────────────


class TestCameraGraphValidation:

    def test_duplicate_camera_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate camera_id"):
            CameraGraph([
                _make_node("cam_a"),
                _make_node("cam_a"),
            ])

    def test_unidirectional_adjacency_raises(self) -> None:
        """A→B exists but B→A is missing."""
        with pytest.raises(ValueError, match="not bidirectional"):
            CameraGraph([
                _make_node("cam_a", adjacent=["cam_b"]),
                _make_node("cam_b"),
            ])

    def test_adjacency_to_nonexistent_camera_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            CameraGraph([
                _make_node("cam_a", adjacent=["cam_z"]),
            ])

    def test_transition_prior_to_nonexistent_camera_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            CameraGraph([
                _make_node("cam_a", priors={"cam_z": 0.5}),
            ])

    def test_transition_priors_exceed_one_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds 1.0"):
            CameraGraph([
                _make_node("cam_a", adjacent=["cam_b", "cam_c"],
                           priors={"cam_b": 0.6, "cam_c": 0.5}),
                _make_node("cam_b", adjacent=["cam_a"]),
                _make_node("cam_c", adjacent=["cam_a"]),
            ])

    def test_negative_prior_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid transition prior"):
            CameraGraph([
                _make_node("cam_a", adjacent=["cam_b"],
                           priors={"cam_b": -0.1}),
                _make_node("cam_b", adjacent=["cam_a"]),
            ])


# ── Query Tests ───────────────────────────────────────────────────────────


class TestCameraGraphQueries:

    def test_get_node(self) -> None:
        graph = _two_camera_graph()
        node = graph.get_node("cam_a")
        assert node is not None
        assert node.camera_id == "cam_a"

    def test_get_node_nonexistent(self) -> None:
        graph = _two_camera_graph()
        assert graph.get_node("cam_z") is None

    def test_has_camera(self) -> None:
        graph = _two_camera_graph()
        assert graph.has_camera("cam_a")
        assert not graph.has_camera("cam_z")

    def test_is_adjacent(self) -> None:
        graph = _three_camera_chain()
        assert graph.is_adjacent("cam_a", "cam_b")
        assert graph.is_adjacent("cam_b", "cam_a")
        assert graph.is_adjacent("cam_b", "cam_c")
        # A and C are NOT adjacent in a chain
        assert not graph.is_adjacent("cam_a", "cam_c")

    def test_is_adjacent_nonexistent_camera(self) -> None:
        graph = _two_camera_graph()
        assert not graph.is_adjacent("cam_a", "cam_z")
        assert not graph.is_adjacent("cam_z", "cam_a")

    def test_get_adjacent(self) -> None:
        graph = _three_camera_chain()
        assert graph.get_adjacent("cam_a") == ["cam_b"]
        assert sorted(graph.get_adjacent("cam_b")) == ["cam_a", "cam_c"]
        assert graph.get_adjacent("cam_c") == ["cam_b"]

    def test_get_adjacent_nonexistent(self) -> None:
        graph = _two_camera_graph()
        assert graph.get_adjacent("cam_z") == []

    def test_get_transition_prior(self) -> None:
        graph = _two_camera_graph()
        assert graph.get_transition_prior("cam_a", "cam_b") == 0.7
        assert graph.get_transition_prior("cam_b", "cam_a") == 0.6

    def test_get_transition_prior_no_prior(self) -> None:
        graph = _three_camera_chain()
        # A→C has no direct prior
        assert graph.get_transition_prior("cam_a", "cam_c") == 0.0

    def test_get_transition_prior_nonexistent(self) -> None:
        graph = _two_camera_graph()
        assert graph.get_transition_prior("cam_z", "cam_a") == 0.0

    def test_repr(self) -> None:
        graph = _two_camera_graph()
        r = repr(graph)
        assert "cameras=2" in r
        assert "cam_a" in r
