"""Deterministic Rule-Based Hypothesis Engine — Sub-Phase 7.4 (§5, §18 Master Spec).

Provides pure deterministic hypothesis tree generation over canonical evidence signals.
"""

from __future__ import annotations

from typing import Optional, Sequence

from gods_eye.schemas.reasoning import Evidence
from gods_eye.schemas.situational import (
    Hypothesis,
    HypothesisNode,
    HypothesisTree,
    IdentityCandidate,
)

# Prohibited intent/causal terms per Master Spec §2 Rule 8
PROHIBITED_TERMS: set[str] = {
    "criminal intent",
    "malicious intent",
    "motive",
    "mental state",
    "guilt",
    "future human decisions",
    "unsupported causal",
}

MAX_ALLOWED_DEPTH: int = 3
MAX_ALLOWED_BRANCHING: int = 5
MAX_NODES_PER_TREE: int = 25
DEFAULT_MIN_CONFIDENCE: float = 0.20


class HypothesisGenerator:
    """Deterministic rule-based situational hypothesis engine."""

    def __init__(self, generator_version: str = "v7.4.0") -> None:
        if not generator_version or not generator_version.strip():
            raise ValueError("generator_version MUST be a non-empty string")
        self._generator_version = generator_version

    @property
    def generator_version(self) -> str:
        """Return generator version string."""
        return self._generator_version

    def generate_trees(
        self,
        evidence_items: tuple[Evidence, ...] | list[Evidence],
        window_start_ns: int,
        window_end_ns: int,
        max_depth: int = 3,
        max_branching: int = 5,
        min_confidence: float = 0.20,
        subject_ref: Optional[str] = None,
    ) -> tuple[HypothesisTree, ...]:
        """Generate a deterministic forest of bounded hypothesis trees over evidence items.

        Args:
            evidence_items: Inputs Evidence objects.
            window_start_ns: Window start timestamp (nanoseconds).
            window_end_ns: Window end timestamp (nanoseconds).
            max_depth: Maximum tree depth (bounded to 3 in v1).
            max_branching: Maximum branching factor (bounded to 5).
            min_confidence: Minimum evidence support confidence threshold (default 0.20).
            subject_ref: Optional subject pseudonym filter.

        Returns:
            Tuple of HypothesisTree instances.
        """
        # Validate temporal arguments
        if not isinstance(window_start_ns, int) or window_start_ns < 0:
            raise ValueError(f"window_start_ns MUST be a non-negative integer, got {window_start_ns}")
        if not isinstance(window_end_ns, int) or window_end_ns < 0:
            raise ValueError(f"window_end_ns MUST be a non-negative integer, got {window_end_ns}")
        if window_start_ns > window_end_ns:
            raise ValueError(f"window_start_ns ({window_start_ns}) MUST be <= window_end_ns ({window_end_ns})")
        if (window_end_ns - window_start_ns) > 86_400_000_000_000:
            raise ValueError("Window duration MUST NOT exceed 24 hours (86400000000000 ns)")

        # Enforce bounds on max_depth, max_branching, min_confidence
        effective_depth = max(1, min(max_depth, MAX_ALLOWED_DEPTH))
        effective_branching = max(1, min(max_branching, MAX_ALLOWED_BRANCHING))
        effective_min_conf = max(0.0, min(float(min_confidence), 1.0))

        # Filter evidence items
        if isinstance(evidence_items, list):
            evidence_sequence: Sequence[Evidence] = tuple(evidence_items)
        elif isinstance(evidence_items, tuple):
            evidence_sequence = evidence_items
        else:
            raise ValueError("evidence_items MUST be a tuple or list of Evidence objects")

        valid_evidence = [
            ev
            for ev in evidence_sequence
            if isinstance(ev, Evidence) and window_start_ns <= ev.timestamp_ns <= window_end_ns
        ]

        if subject_ref is not None and subject_ref.strip():
            target_sub = subject_ref.strip()
            valid_evidence = [
                ev
                for ev in valid_evidence
                if ev.global_id == target_sub
                or (isinstance(ev.payload, dict) and ev.payload.get("subject_ref") == target_sub)
            ]

        if not valid_evidence:
            return ()

        # Group evidence items into root candidate hypothesis clusters
        groups: dict[str, list[Evidence]] = {}
        for ev in valid_evidence:
            payload = ev.payload if isinstance(ev.payload, dict) else {}
            group_key = str(payload.get("anomaly_type", ev.record_type))
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(ev)

        trees: list[HypothesisTree] = []

        # Sort group keys deterministically
        for group_key in sorted(groups.keys()):
            group_evs = groups[group_key]
            root_id = f"hyp_root_{group_key}_{window_start_ns}"

            ref_sub = subject_ref or (
                str(group_evs[0].payload.get("subject_ref"))
                if isinstance(group_evs[0].payload, dict) and "subject_ref" in group_evs[0].payload
                else (group_evs[0].global_id or f"subject_{group_evs[0].camera_id or 'anon'}")
            )

            comp_cands = (
                IdentityCandidate(subject_ref=ref_sub, confidence=0.85),
                IdentityCandidate(subject_ref=f"{ref_sub}_alt", confidence=0.15),
            )

            root_conf = float(
                group_evs[0].payload.get("confidence", 0.75)
                if isinstance(group_evs[0].payload, dict)
                else 0.75
            )

            if root_conf < effective_min_conf:
                continue

            root_ev_ids = tuple(sorted(list({ev.evidence_id for ev in group_evs})))

            explanation = (
                f"Observed structural evidence for {group_key} involving subject reference {ref_sub} "
                f"across {len(root_ev_ids)} evidence item(s)."
            )

            explanation_lower = explanation.lower()
            for term in PROHIBITED_TERMS:
                if term in explanation_lower:
                    raise ValueError(f"Prohibited language term '{term}' detected in hypothesis explanation")

            root_hyp = Hypothesis(
                hypothesis_id=f"hyp_{group_key}_001",
                hypothesis_type=group_key,
                primary_subject_ref=ref_sub,
                identity_confidence=0.85,
                competing_candidates=comp_cands,
                confidence=root_conf,
                evidence_ids=root_ev_ids,
                explanation=explanation,
            )

            children: list[HypothesisNode] = []
            if effective_depth > 1 and len(group_evs) > 1:
                child_evs_list = group_evs[1 : 1 + effective_branching]
                for idx, c_ev in enumerate(child_evs_list, start=1):
                    c_conf = float(
                        c_ev.payload.get("confidence", root_conf * 0.9)
                        if isinstance(c_ev.payload, dict)
                        else root_conf * 0.9
                    )
                    if c_conf < effective_min_conf:
                        continue
                    c_id = f"node_root_{group_key}_c{idx}"
                    c_hyp = Hypothesis(
                        hypothesis_id=f"hyp_{group_key}_child_{idx}",
                        hypothesis_type=f"{group_key}_subcomponent",
                        primary_subject_ref=ref_sub,
                        identity_confidence=0.80,
                        competing_candidates=comp_cands,
                        confidence=c_conf,
                        evidence_ids=(c_ev.evidence_id,),
                        explanation=f"Sub-component spatial/temporal observation on camera {c_ev.camera_id or 'unknown'}.",
                    )

                    grandchildren: list[HypothesisNode] = []
                    if effective_depth == 3 and len(group_evs) > 2:
                        gc_ev = group_evs[min(idx + 1, len(group_evs) - 1)]
                        gc_conf = float(
                            gc_ev.payload.get("confidence", c_conf * 0.9)
                            if isinstance(gc_ev.payload, dict)
                            else c_conf * 0.9
                        )
                        if gc_conf >= effective_min_conf:
                            gc_node = HypothesisNode(
                                node_id=f"node_root_{group_key}_c{idx}_gc1",
                                hypothesis=Hypothesis(
                                    hypothesis_id=f"hyp_{group_key}_gc_{idx}_1",
                                    hypothesis_type=f"{group_key}_subcomponent_detail",
                                    primary_subject_ref=ref_sub,
                                    identity_confidence=0.75,
                                    competing_candidates=comp_cands,
                                    confidence=gc_conf,
                                    evidence_ids=(gc_ev.evidence_id,),
                                    explanation=f"Detailed sub-observation on record {gc_ev.record_id}.",
                                ),
                                parent_node_id=c_id,
                                children=(),
                                depth=3,
                            )
                            grandchildren.append(gc_node)

                    grandchildren.sort(
                        key=lambda n: (
                            n.hypothesis.confidence * n.hypothesis.identity_confidence,
                            n.hypothesis.confidence,
                            n.hypothesis.identity_confidence,
                        ),
                        reverse=True,
                    )
                    c_node = HypothesisNode(
                        node_id=c_id,
                        hypothesis=c_hyp,
                        parent_node_id=f"node_{root_id}",
                        children=tuple(grandchildren[:effective_branching]),
                        depth=2,
                    )
                    children.append(c_node)

            children.sort(
                key=lambda n: (
                    n.hypothesis.confidence * n.hypothesis.identity_confidence,
                    n.hypothesis.confidence,
                    n.hypothesis.identity_confidence,
                ),
                reverse=True,
            )

            pruned_children = tuple(children[:effective_branching])

            root_node = HypothesisNode(
                node_id=f"node_{root_id}",
                hypothesis=root_hyp,
                parent_node_id=None,
                children=pruned_children,
                depth=1,
            )

            def _count_nodes(node: HypothesisNode) -> int:
                return 1 + sum(_count_nodes(c) for c in node.children)

            tot_nodes = min(_count_nodes(root_node), MAX_NODES_PER_TREE)

            tree = HypothesisTree(
                tree_id=f"tree_{group_key}_{window_start_ns}",
                root_node=root_node,
                total_nodes=tot_nodes,
                max_depth=effective_depth,
            )
            trees.append(tree)

        trees.sort(
            key=lambda t: (
                t.root_node.hypothesis.confidence * t.root_node.hypothesis.identity_confidence,
                t.root_node.hypothesis.confidence,
            ),
            reverse=True,
        )

        return tuple(trees)
