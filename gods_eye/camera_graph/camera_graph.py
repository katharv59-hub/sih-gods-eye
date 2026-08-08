"""Camera graph — immutable camera topology for cross-camera intelligence.

The ``CameraGraph`` is loaded once at startup from a YAML configuration
file and is **immutable after initialization** (§9 shared state rules).
All access is read-only and thread-safe without locking.

Responsibilities:
    - Store ``CameraNode`` records indexed by ``camera_id``
    - Adjacency queries: which cameras are physically connected
    - Transition prior lookups: probability of person moving A→B
    - Topology validation at construction time

NOT responsible for:
    - Identity matching (that's the IML + Matcher)
    - Transition detection (that's the IML)
    - Event emission (that's the pipeline boundary)
"""

from __future__ import annotations

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.camera import CameraNode


class CameraGraph:
    """Immutable camera topology graph.

    Thread-safe: no mutation after ``__init__`` completes.
    Read-only queries may be called from any thread (§9).

    Args:
        nodes: Camera nodes defining the topology.  Adjacency
            consistency is validated at construction time.

    Raises:
        ValueError: If adjacency is not bidirectional or transition
            priors sum to > 1.0 for any node.
    """

    def __init__(self, nodes: list[CameraNode]) -> None:
        self._log = get_logger("camera_graph")
        self._nodes: dict[str, CameraNode] = {}

        for node in nodes:
            if node.camera_id in self._nodes:
                raise ValueError(
                    f"Duplicate camera_id: {node.camera_id!r}"
                )
            self._nodes[node.camera_id] = node

        self._validate()
        self._log.info(
            "camera_graph_initialized",
            camera_count=len(self._nodes),
            camera_ids=sorted(self._nodes.keys()),
        )

    # ── Validation ──────────────────────────────────────────────────

    def _validate(self) -> None:
        """Validate topology consistency at construction time."""
        for camera_id, node in self._nodes.items():
            # Adjacency must be bidirectional
            for adj_id in node.adjacent_cameras:
                if adj_id not in self._nodes:
                    raise ValueError(
                        f"Camera {camera_id!r} lists adjacent camera "
                        f"{adj_id!r} which does not exist in the graph."
                    )
                adj_node = self._nodes[adj_id]
                if camera_id not in adj_node.adjacent_cameras:
                    raise ValueError(
                        f"Adjacency is not bidirectional: {camera_id!r} → "
                        f"{adj_id!r} exists, but {adj_id!r} → "
                        f"{camera_id!r} is missing."
                    )

            # Transition priors must reference valid cameras and sum ≤ 1.0
            prior_sum = 0.0
            for target_id, prob in node.transition_priors.items():
                if target_id not in self._nodes:
                    raise ValueError(
                        f"Camera {camera_id!r} has transition prior to "
                        f"{target_id!r} which does not exist in the graph."
                    )
                if prob < 0.0 or prob > 1.0:
                    raise ValueError(
                        f"Camera {camera_id!r} has invalid transition "
                        f"prior {prob} to {target_id!r}. Must be [0, 1]."
                    )
                prior_sum += prob

            if prior_sum > 1.0 + 1e-9:  # small epsilon for float rounding
                raise ValueError(
                    f"Camera {camera_id!r} transition priors sum to "
                    f"{prior_sum:.4f}, which exceeds 1.0."
                )

    # ── Queries (all read-only, thread-safe) ────────────────────────

    def get_node(self, camera_id: str) -> CameraNode | None:
        """Return the CameraNode for the given ID, or None."""
        return self._nodes.get(camera_id)

    def has_camera(self, camera_id: str) -> bool:
        """Check if a camera exists in the graph."""
        return camera_id in self._nodes

    def is_adjacent(self, camera_a: str, camera_b: str) -> bool:
        """Check if two cameras are physically adjacent.

        Returns False if either camera is not in the graph.
        """
        node_a = self._nodes.get(camera_a)
        if node_a is None:
            return False
        return camera_b in node_a.adjacent_cameras

    def get_adjacent(self, camera_id: str) -> list[str]:
        """Return camera_ids adjacent to the given camera.

        Returns empty list if camera not found.
        """
        node = self._nodes.get(camera_id)
        if node is None:
            return []
        return list(node.adjacent_cameras)

    def get_transition_prior(
        self, from_camera: str, to_camera: str,
    ) -> float:
        """Return the transition probability from one camera to another.

        Returns 0.0 if either camera is not in the graph or no prior
        is defined for the pair.
        """
        node = self._nodes.get(from_camera)
        if node is None:
            return 0.0
        return node.transition_priors.get(to_camera, 0.0)

    def all_camera_ids(self) -> list[str]:
        """Return sorted list of all camera IDs in the graph."""
        return sorted(self._nodes.keys())

    @property
    def camera_count(self) -> int:
        """Number of cameras in the graph."""
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"CameraGraph(cameras={self.camera_count}, "
            f"ids={self.all_camera_ids()})"
        )
