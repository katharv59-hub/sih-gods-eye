"""Trajectory Sequence & Feature Extractor — Sub-Phase 6.2 (§5 & §6 Master Spec).

Provides a deterministic, read-only transformation layer that converts historical
Spatiotemporal GraphStore observations and transitions into canonical trajectory
sequences, temporal features, and normalized sequence distance primitives.

No probabilistic modeling, clustering, or state mutation occurs in this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from gods_eye.memory.graph_store import BaseGraphStore


@dataclass(frozen=True)
class TrajectorySequence:
    """Canonical representation of an extracted spatial-temporal trajectory.

    Read-only, deterministic snapshot of camera transitions and temporal statistics.
    Exposes aggregate spatial features without raw identity lists.
    """

    camera_sequence: list[str]
    timestamps_ns: list[int]
    total_duration_s: float
    transition_count: int
    unique_camera_count: int
    dwell_durations_s: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize trajectory sequence to dictionary format."""
        return {
            "camera_sequence": list(self.camera_sequence),
            "timestamps_ns": list(self.timestamps_ns),
            "total_duration_s": self.total_duration_s,
            "transition_count": self.transition_count,
            "unique_camera_count": self.unique_camera_count,
            "dwell_durations_s": dict(self.dwell_durations_s),
            "confidence": self.confidence,
        }


def normalized_sequence_distance(seq_a: list[str], seq_b: list[str]) -> float:
    """Compute normalized Levenshtein edit distance between two camera sequences.

    Properties:
    - Bounded in [0.0, 1.0]
    - Returns 0.0 for identical sequences
    - Symmetric: distance(A, B) == distance(B, A)
    - Returns 0.0 if both sequences are empty
    - Does not mutate input lists
    """
    if not seq_a and not seq_b:
        return 0.0
    if not seq_a or not seq_b:
        return 1.0
    if seq_a == seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    # Dynamic programming matrix for Levenshtein distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    edit_dist = dp[m][n]
    max_len = max(m, n)
    return float(edit_dist) / float(max_len)


class TrajectorySequenceExtractor:
    """Deterministic extractor for spatial trajectory sequences and temporal features."""

    @staticmethod
    def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort graph records deterministically by (timestamp_ns, sequence_num, edge_id)."""
        return sorted(
            records,
            key=lambda r: (
                r.get("timestamp_ns", 0),
                r.get("sequence_num", 0),
                r.get("edge_id", ""),
            ),
        )

    def extract_sequence_from_observations(
        self, observations: list[dict[str, Any]]
    ) -> TrajectorySequence:
        """Extract canonical camera sequence and features from observation edges.

        Deduplicates consecutive identical camera observations (frame-level presence)
        to isolate genuine camera node transitions.
        """
        if not observations:
            return TrajectorySequence(
                camera_sequence=[],
                timestamps_ns=[],
                total_duration_s=0.0,
                transition_count=0,
                unique_camera_count=0,
                dwell_durations_s={},
                confidence=1.0,
            )

        sorted_obs = self._sort_records(observations)

        camera_sequence: list[str] = []
        timestamps_ns: list[int] = []
        confidences: list[float] = []

        dwell_map: dict[str, float] = {}
        prev_cam: Optional[str] = None
        prev_ts: Optional[int] = None

        for obs in sorted_obs:
            cam = obs.get("camera_id", "").strip()
            ts = obs.get("timestamp_ns", 0)
            conf = float(obs.get("confidence", 1.0))

            if not cam or ts <= 0:
                continue

            # Calculate dwell time span at previous camera
            if prev_cam is not None and prev_ts is not None and ts >= prev_ts:
                delta_s = (ts - prev_ts) / 1e9
                dwell_map[prev_cam] = dwell_map.get(prev_cam, 0.0) + delta_s

            # Collapse consecutive duplicate camera observations
            if cam != prev_cam:
                camera_sequence.append(cam)
                timestamps_ns.append(ts)

            confidences.append(conf)
            prev_cam = cam
            prev_ts = ts

        if not camera_sequence:
            return TrajectorySequence(
                camera_sequence=[],
                timestamps_ns=[],
                total_duration_s=0.0,
                transition_count=0,
                unique_camera_count=0,
                dwell_durations_s={},
                confidence=1.0,
            )

        total_duration_s = (
            (sorted_obs[-1]["timestamp_ns"] - sorted_obs[0]["timestamp_ns"]) / 1e9
            if len(sorted_obs) > 1 and sorted_obs[-1]["timestamp_ns"] >= sorted_obs[0]["timestamp_ns"]
            else 0.0
        )
        transition_count = max(0, len(camera_sequence) - 1)
        unique_camera_count = len(set(camera_sequence))
        mean_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        return TrajectorySequence(
            camera_sequence=camera_sequence,
            timestamps_ns=timestamps_ns,
            total_duration_s=round(total_duration_s, 6),
            transition_count=transition_count,
            unique_camera_count=unique_camera_count,
            dwell_durations_s={k: round(v, 6) for k, v in dwell_map.items()},
            confidence=round(mean_confidence, 4),
        )

    def extract_sequence_from_transitions(
        self, transitions: list[dict[str, Any]]
    ) -> TrajectorySequence:
        """Extract canonical camera sequence and features from transition edges."""
        if not transitions:
            return TrajectorySequence(
                camera_sequence=[],
                timestamps_ns=[],
                total_duration_s=0.0,
                transition_count=0,
                unique_camera_count=0,
                dwell_durations_s={},
                confidence=1.0,
            )

        sorted_trans = self._sort_records(transitions)

        camera_sequence: list[str] = []
        timestamps_ns: list[int] = []
        confidences: list[float] = []

        total_dur_s = 0.0

        for idx, tr in enumerate(sorted_trans):
            f_cam = tr.get("from_camera_id", "").strip()
            t_cam = tr.get("to_camera_id", "").strip()
            ts = tr.get("timestamp_ns", 0)
            dur = float(tr.get("transition_duration_s", 0.0))
            conf = float(tr.get("confidence", 1.0))

            if idx == 0 and f_cam:
                camera_sequence.append(f_cam)
                timestamps_ns.append(ts)

            if t_cam and (not camera_sequence or camera_sequence[-1] != t_cam):
                camera_sequence.append(t_cam)
                timestamps_ns.append(ts)

            total_dur_s += max(0.0, dur)
            confidences.append(conf)

        if not camera_sequence:
            return TrajectorySequence(
                camera_sequence=[],
                timestamps_ns=[],
                total_duration_s=0.0,
                transition_count=0,
                unique_camera_count=0,
                dwell_durations_s={},
                confidence=1.0,
            )

        transition_count = max(0, len(camera_sequence) - 1)
        unique_camera_count = len(set(camera_sequence))
        mean_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        return TrajectorySequence(
            camera_sequence=camera_sequence,
            timestamps_ns=timestamps_ns,
            total_duration_s=round(total_dur_s, 6),
            transition_count=transition_count,
            unique_camera_count=unique_camera_count,
            dwell_durations_s={},
            confidence=round(mean_confidence, 4),
        )

    def extract_identity_trajectory(
        self,
        graph_store: BaseGraphStore,
        global_id: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> TrajectorySequence:
        """Query GraphStore and extract deterministic TrajectorySequence for identity.

        Read-only query operation.
        """
        obs = graph_store.get_identity_trajectory(
            global_id=global_id, start_ns=start_ns, end_ns=end_ns, limit=limit
        )
        return self.extract_sequence_from_observations(obs)
