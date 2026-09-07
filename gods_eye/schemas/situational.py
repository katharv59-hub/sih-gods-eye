"""Canonical Situational Intelligence Schemas — Sub-Phase 7.1 (§4, §5 & §18 Master Spec).

Defines canonical data contracts for Phase 7 Situational Intelligence:
- IdentityCandidate (Dataclass, frozen)
- Hypothesis (Dataclass, frozen)
- SituationalState (Dataclass, frozen)
- RiskSignal (Dataclass, frozen)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from gods_eye.schemas.environment import SystemMode as OperationalMode

ALLOWED_RISK_LEVELS: set[str] = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


@dataclass(frozen=True)
class IdentityCandidate:
    """Opaque identity candidate reference with confidence score."""

    subject_ref: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate IdentityCandidate invariants."""
        if not isinstance(self.subject_ref, str) or not self.subject_ref.strip():
            raise ValueError("subject_ref MUST be a non-empty string")
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "subject_ref": self.subject_ref,
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class Hypothesis:
    """A candidate situational hypothesis derived from evidence."""

    hypothesis_id: str
    hypothesis_type: str
    primary_subject_ref: Optional[str]
    identity_confidence: float
    competing_candidates: tuple[IdentityCandidate, ...]
    confidence: float
    evidence_ids: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        """Validate Hypothesis invariants."""
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id MUST be a non-empty string")
        if not isinstance(self.hypothesis_type, str) or not self.hypothesis_type.strip():
            raise ValueError("hypothesis_type MUST be a non-empty string")

        if self.primary_subject_ref is not None:
            if not isinstance(self.primary_subject_ref, str) or not self.primary_subject_ref.strip():
                raise ValueError("primary_subject_ref MUST be None or a non-empty string")

        if not isinstance(self.identity_confidence, (int, float)) or not (
            0.0 <= float(self.identity_confidence) <= 1.0
        ):
            raise ValueError(
                f"identity_confidence MUST be between 0.0 and 1.0, got {self.identity_confidence}"
            )

        # Convert list inputs to tuples if necessary for immutability
        if isinstance(self.competing_candidates, list):
            object.__setattr__(self, "competing_candidates", tuple(self.competing_candidates))
        if not isinstance(self.competing_candidates, tuple):
            raise ValueError("competing_candidates MUST be an immutable tuple of IdentityCandidate")
        for cand in self.competing_candidates:
            if not isinstance(cand, IdentityCandidate):
                raise ValueError("competing_candidates elements MUST be IdentityCandidate instances")

        if not isinstance(self.confidence, (int, float)) or not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence MUST be between 0.0 and 1.0, got {self.confidence}")

        if isinstance(self.evidence_ids, list):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if not isinstance(self.evidence_ids, tuple):
            raise ValueError("evidence_ids MUST be an immutable tuple of strings")
        for eid in self.evidence_ids:
            if not isinstance(eid, str) or not eid.strip():
                raise ValueError("evidence_ids elements MUST be non-empty strings")

        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("explanation MUST be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_type": self.hypothesis_type,
            "primary_subject_ref": self.primary_subject_ref,
            "identity_confidence": float(self.identity_confidence),
            "competing_candidates": [c.to_dict() for c in self.competing_candidates],
            "confidence": float(self.confidence),
            "evidence_ids": list(self.evidence_ids),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class HypothesisNode:
    """A node in a bounded situational hypothesis tree."""

    node_id: str
    hypothesis: Hypothesis
    parent_node_id: Optional[str]
    children: tuple[HypothesisNode, ...]
    depth: int

    def __post_init__(self) -> None:
        """Validate HypothesisNode invariants."""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id MUST be a non-empty string")

        if not isinstance(self.hypothesis, Hypothesis):
            raise ValueError("hypothesis MUST be a Hypothesis instance")

        if self.parent_node_id is not None:
            if not isinstance(self.parent_node_id, str) or not self.parent_node_id.strip():
                raise ValueError("parent_node_id MUST be None or a non-empty string")

        if isinstance(self.children, list):
            object.__setattr__(self, "children", tuple(self.children))
        if not isinstance(self.children, tuple):
            raise ValueError("children MUST be an immutable tuple of HypothesisNode instances")
        for child in self.children:
            if not isinstance(child, HypothesisNode):
                raise ValueError("children elements MUST be HypothesisNode instances")

        if not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError(f"depth MUST be a positive integer, got {self.depth}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "node_id": self.node_id,
            "hypothesis": self.hypothesis.to_dict(),
            "parent_node_id": self.parent_node_id,
            "children": [c.to_dict() for c in self.children],
            "depth": self.depth,
        }


@dataclass(frozen=True)
class HypothesisTree:
    """A bounded hierarchical tree of situational hypotheses."""

    tree_id: str
    root_node: HypothesisNode
    total_nodes: int
    max_depth: int

    def __post_init__(self) -> None:
        """Validate HypothesisTree invariants."""
        if not isinstance(self.tree_id, str) or not self.tree_id.strip():
            raise ValueError("tree_id MUST be a non-empty string")

        if not isinstance(self.root_node, HypothesisNode):
            raise ValueError("root_node MUST be a HypothesisNode instance")

        if not isinstance(self.total_nodes, int) or self.total_nodes < 1:
            raise ValueError(f"total_nodes MUST be a positive integer, got {self.total_nodes}")

        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise ValueError(f"max_depth MUST be a positive integer, got {self.max_depth}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "tree_id": self.tree_id,
            "root_node": self.root_node.to_dict(),
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
        }


@dataclass(frozen=True)
class SituationalState:
    """Overall multi-camera situational state snapshot over a temporal window."""

    state_id: str
    window_start_ns: int
    window_end_ns: int
    active_identity_count: int
    active_cameras: tuple[str, ...]
    system_mode: OperationalMode
    anomaly_ids: tuple[str, ...]
    overall_risk_level: str
    explanation: str

    def __post_init__(self) -> None:
        """Validate SituationalState invariants."""
        if not isinstance(self.state_id, str) or not self.state_id.strip():
            raise ValueError("state_id MUST be a non-empty string")

        if not isinstance(self.window_start_ns, int) or self.window_start_ns < 0:
            raise ValueError(f"window_start_ns MUST be a non-negative integer, got {self.window_start_ns}")
        if not isinstance(self.window_end_ns, int) or self.window_end_ns < 0:
            raise ValueError(f"window_end_ns MUST be a non-negative integer, got {self.window_end_ns}")
        if self.window_start_ns > self.window_end_ns:
            raise ValueError(
                f"window_start_ns ({self.window_start_ns}) MUST be <= window_end_ns ({self.window_end_ns})"
            )

        if not isinstance(self.active_identity_count, int) or self.active_identity_count < 0:
            raise ValueError(
                f"active_identity_count MUST be a non-negative integer, got {self.active_identity_count}"
            )

        if isinstance(self.active_cameras, list):
            object.__setattr__(self, "active_cameras", tuple(self.active_cameras))
        if not isinstance(self.active_cameras, tuple):
            raise ValueError("active_cameras MUST be an immutable tuple of strings")
        for cam in self.active_cameras:
            if not isinstance(cam, str) or not cam.strip():
                raise ValueError("active_cameras elements MUST be non-empty strings")

        if not isinstance(self.system_mode, OperationalMode):
            raise ValueError(f"system_mode MUST be an OperationalMode instance, got {self.system_mode}")

        if isinstance(self.anomaly_ids, list):
            object.__setattr__(self, "anomaly_ids", tuple(self.anomaly_ids))
        if not isinstance(self.anomaly_ids, tuple):
            raise ValueError("anomaly_ids MUST be an immutable tuple of strings")
        for aid in self.anomaly_ids:
            if not isinstance(aid, str) or not aid.strip():
                raise ValueError("anomaly_ids elements MUST be non-empty strings")

        if self.overall_risk_level not in ALLOWED_RISK_LEVELS:
            raise ValueError(
                f"overall_risk_level MUST be one of {sorted(ALLOWED_RISK_LEVELS)}, got '{self.overall_risk_level}'"
            )

        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("explanation MUST be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "state_id": self.state_id,
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "active_identity_count": self.active_identity_count,
            "active_cameras": list(self.active_cameras),
            "system_mode": self.system_mode.value,
            "anomaly_ids": list(self.anomaly_ids),
            "overall_risk_level": self.overall_risk_level,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class RiskSignal:
    """Situational risk signal record with evidence provenance."""

    signal_id: str
    window_start_ns: int
    window_end_ns: int
    risk_level: str
    risk_score: float
    evidence_ids: tuple[str, ...]
    contributing_factors: tuple[str, ...]
    explanation: str
    suppressed: bool
    suppression_reason: Optional[str]
    evaluator_version: str

    def __post_init__(self) -> None:
        """Validate RiskSignal invariants."""
        if not isinstance(self.signal_id, str) or not self.signal_id.strip():
            raise ValueError("signal_id MUST be a non-empty string")

        if not isinstance(self.window_start_ns, int) or self.window_start_ns < 0:
            raise ValueError(f"window_start_ns MUST be a non-negative integer, got {self.window_start_ns}")
        if not isinstance(self.window_end_ns, int) or self.window_end_ns < 0:
            raise ValueError(f"window_end_ns MUST be a non-negative integer, got {self.window_end_ns}")
        if self.window_start_ns > self.window_end_ns:
            raise ValueError(
                f"window_start_ns ({self.window_start_ns}) MUST be <= window_end_ns ({self.window_end_ns})"
            )

        if self.risk_level not in ALLOWED_RISK_LEVELS:
            raise ValueError(
                f"risk_level MUST be one of {sorted(ALLOWED_RISK_LEVELS)}, got '{self.risk_level}'"
            )

        if not isinstance(self.risk_score, (int, float)) or not (0.0 <= float(self.risk_score) <= 1.0):
            raise ValueError(f"risk_score MUST be between 0.0 and 1.0, got {self.risk_score}")

        if isinstance(self.evidence_ids, list):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if not isinstance(self.evidence_ids, tuple):
            raise ValueError("evidence_ids MUST be an immutable tuple of strings")
        for eid in self.evidence_ids:
            if not isinstance(eid, str) or not eid.strip():
                raise ValueError("evidence_ids elements MUST be non-empty strings")

        if isinstance(self.contributing_factors, list):
            object.__setattr__(self, "contributing_factors", tuple(self.contributing_factors))
        if not isinstance(self.contributing_factors, tuple):
            raise ValueError("contributing_factors MUST be an immutable tuple of strings")
        for factor in self.contributing_factors:
            if not isinstance(factor, str) or not factor.strip():
                raise ValueError("contributing_factors elements MUST be non-empty strings")

        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("explanation MUST be a non-empty string")

        if not isinstance(self.suppressed, bool):
            raise ValueError(f"suppressed MUST be a boolean, got {type(self.suppressed)}")

        if self.suppressed:
            if not isinstance(self.suppression_reason, str) or not self.suppression_reason.strip():
                raise ValueError("suppression_reason MUST be a non-empty string when suppressed is True")
        else:
            if self.suppression_reason is not None:
                raise ValueError("suppression_reason MUST be None when suppressed is False")

        if not isinstance(self.evaluator_version, str) or not self.evaluator_version.strip():
            raise ValueError("evaluator_version MUST be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        return {
            "signal_id": self.signal_id,
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "risk_level": self.risk_level,
            "risk_score": float(self.risk_score),
            "evidence_ids": list(self.evidence_ids),
            "contributing_factors": list(self.contributing_factors),
            "explanation": self.explanation,
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "evaluator_version": self.evaluator_version,
        }
