"""Unit test suite for Sub-Phase 7.4 HypothesisGenerator engine."""

from __future__ import annotations

import pytest

from gods_eye.schemas.reasoning import Evidence
from gods_eye.schemas.situational import (
    Hypothesis,
    HypothesisNode,
    HypothesisTree,
    IdentityCandidate,
)
from gods_eye.situational.hypothesis import HypothesisGenerator


@pytest.fixture
def generator() -> HypothesisGenerator:
    return HypothesisGenerator(generator_version="v7.4.0")


def make_evidence(
    evidence_id: str = "ev_001",
    timestamp_ns: int = 1000,
    record_type: str = "event",
    camera_id: str = "cam_1",
    global_id: str = "sub_001",
    payload: dict | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_store="event_store",
        record_type=record_type,
        record_id=f"rec_{evidence_id}",
        timestamp_ns=timestamp_ns,
        camera_id=camera_id,
        global_id=global_id,
        explanation="Test evidence for hypothesis engine",
        payload=payload or {},
    )


class TestHypothesisGenerator:
    def test_generator_version_string(self, generator: HypothesisGenerator) -> None:
        assert generator.generator_version == "v7.4.0"

    def test_zero_evidence_returns_empty_tuple(self, generator: HypothesisGenerator) -> None:
        trees = generator.generate_trees([], 1000, 2000)
        assert trees == ()

    def test_valid_hypothesis_generation(self, generator: HypothesisGenerator) -> None:
        ev1 = make_evidence("ev_1", 1200, "trajectory_anomaly", "cam_1", "sub_alpha", {"confidence": 0.85})
        ev2 = make_evidence("ev_2", 1400, "trajectory_anomaly", "cam_2", "sub_alpha", {"confidence": 0.80})

        trees = generator.generate_trees([ev1, ev2], 1000, 2000, max_depth=3, max_branching=5)
        assert len(trees) == 1
        tree = trees[0]
        assert isinstance(tree, HypothesisTree)
        assert tree.max_depth == 3
        assert tree.root_node.depth == 1
        assert tree.root_node.hypothesis.primary_subject_ref == "sub_alpha"
        assert len(tree.root_node.hypothesis.competing_candidates) == 2

    def test_privacy_invariant_no_global_id_in_explanation(self, generator: HypothesisGenerator) -> None:
        ev = make_evidence("ev_1", 1500, "trajectory_anomaly", "cam_1", "sub_secret_uuid")
        trees = generator.generate_trees([ev], 1000, 2000)
        assert len(trees) == 1
        d = trees[0].to_dict()
        d_str = str(d)
        assert "global_id" not in d_str
        assert "subject_global_id" not in d_str

    def test_deterministic_repeated_output(self, generator: HypothesisGenerator) -> None:
        ev1 = make_evidence("ev_1", 1200, "trajectory_anomaly", "cam_1", "sub_alpha", {"confidence": 0.85})
        ev2 = make_evidence("ev_2", 1400, "trajectory_anomaly", "cam_2", "sub_alpha", {"confidence": 0.80})

        t1 = generator.generate_trees([ev1, ev2], 1000, 2000)
        t2 = generator.generate_trees([ev1, ev2], 1000, 2000)
        assert t1 == t2
        assert t1[0].to_dict() == t2[0].to_dict()

    def test_ranking_and_tie_breaking(self, generator: HypothesisGenerator) -> None:
        ev1 = make_evidence("ev_1", 1200, "anomaly_type_b", "cam_1", payload={"confidence": 0.60})
        ev2 = make_evidence("ev_2", 1300, "anomaly_type_a", "cam_1", payload={"confidence": 0.90})

        trees = generator.generate_trees([ev1, ev2], 1000, 2000)
        assert len(trees) == 2
        # Highest composite score S = confidence * identity_confidence first
        assert trees[0].root_node.hypothesis.hypothesis_type == "anomaly_type_a"

    def test_depth_bound_max_3(self, generator: HypothesisGenerator) -> None:
        evs = [make_evidence(f"ev_{i}", 1000 + i * 100, "trajectory_anomaly") for i in range(10)]
        trees = generator.generate_trees(evs, 1000, 3000, max_depth=5)  # Request depth 5, bounded to 3
        assert len(trees) == 1
        assert trees[0].max_depth <= 3

        def _get_max_d(node: HypothesisNode) -> int:
            if not node.children:
                return node.depth
            return max(_get_max_d(c) for c in node.children)

        assert _get_max_d(trees[0].root_node) <= 3

    def test_branching_bound_max_5(self, generator: HypothesisGenerator) -> None:
        evs = [make_evidence(f"ev_{i}", 1000 + i * 50, "trajectory_anomaly") for i in range(10)]
        trees = generator.generate_trees(evs, 1000, 3000, max_branching=10)  # Request 10, bounded to 5
        assert len(trees) == 1
        assert len(trees[0].root_node.children) <= 5

    def test_node_count_bound_max_25(self, generator: HypothesisGenerator) -> None:
        evs = [make_evidence(f"ev_{i}", 1000 + i * 20, "trajectory_anomaly") for i in range(15)]
        trees = generator.generate_trees(evs, 1000, 3000, max_depth=3, max_branching=5)
        for t in trees:
            assert t.total_nodes <= 25

    def test_confidence_pruning(self, generator: HypothesisGenerator) -> None:
        ev_high = make_evidence("ev_1", 1200, "type_a", payload={"confidence": 0.80})
        ev_low = make_evidence("ev_2", 1300, "type_b", payload={"confidence": 0.10})

        trees = generator.generate_trees([ev_high, ev_low], 1000, 2000, min_confidence=0.20)
        assert len(trees) == 1
        assert trees[0].root_node.hypothesis.hypothesis_type == "type_a"

    def test_evidence_provenance_preservation(self, generator: HypothesisGenerator) -> None:
        ev = make_evidence("ev_prov_123", 1500, "type_a")
        trees = generator.generate_trees([ev], 1000, 2000)
        assert len(trees) == 1
        assert "ev_prov_123" in trees[0].root_node.hypothesis.evidence_ids

    def test_competing_identity_candidates(self, generator: HypothesisGenerator) -> None:
        ev = make_evidence("ev_1", 1500, "type_a", payload={"subject_ref": "sub_prime"})
        trees = generator.generate_trees([ev], 1000, 2000)
        assert len(trees) == 1
        cands = trees[0].root_node.hypothesis.competing_candidates
        assert isinstance(cands, tuple)
        assert len(cands) >= 2
        assert cands[0].subject_ref == "sub_prime"

    def test_invalid_temporal_window_rejection(self, generator: HypothesisGenerator) -> None:
        with pytest.raises(ValueError, match="MUST be <= window_end_ns"):
            generator.generate_trees([], 3000, 2000)

        with pytest.raises(ValueError, match="duration MUST NOT exceed 24 hours"):
            generator.generate_trees([], 0, 100_000_000_000_000)
