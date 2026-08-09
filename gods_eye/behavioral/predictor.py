"""Markov Behavioral Predictor — Sub-Phase 6.5 (§5 Master Spec).

Implements a deterministic, in-memory 1st-order Markov Transition Model to forecast
candidate next-camera destinations and arrival timestamps from historical spatiotemporal
transitions.

No persistent database mutation, deep learning, or online LLM inference occurs.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gods_eye.behavioral.extractor import TrajectorySequence
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.schemas.behavioral import PredictedDestination, PredictionResult


class MarkovBehavioralPredictor:
    """Deterministic 1st-order Markov chain predictor for next-camera destination forecasting."""

    def __init__(self, model_version: str = "markov_v1.0") -> None:
        """Initialize empty in-memory Markov predictor."""
        self.model_version = model_version
        # Transition counts: source_cam -> {dest_cam -> count}
        self._matrix: dict[str, dict[str, int]] = {}
        # Transition duration sums: (source_cam, dest_cam) -> sum_duration_s
        self._duration_sums: dict[tuple[str, str], float] = {}

    def clear(self) -> None:
        """Clear all learned in-memory transition statistics."""
        self._matrix.clear()
        self._duration_sums.clear()

    def fit_from_sequences(self, sequences: Sequence[TrajectorySequence]) -> None:
        """Fit transition model from extracted TrajectorySequence instances."""
        for seq in sequences:
            cams = seq.camera_sequence
            if len(cams) < 2:
                continue

            # Calculate transition duration estimate if timestamps exist
            ts_list = seq.timestamps_ns
            for i in range(len(cams) - 1):
                src = cams[i]
                dst = cams[i + 1]
                if src == dst:
                    # Ignore self-transitions (same camera presence)
                    continue

                if src not in self._matrix:
                    self._matrix[src] = {}
                self._matrix[src][dst] = self._matrix[src].get(dst, 0) + 1

                if len(ts_list) > i + 1 and ts_list[i + 1] > ts_list[i]:
                    dur_s = (ts_list[i + 1] - ts_list[i]) / 1e9
                    key = (src, dst)
                    self._duration_sums[key] = self._duration_sums.get(key, 0.0) + dur_s

    def fit_from_transitions(self, transitions: Sequence[dict[str, Any]]) -> None:
        """Fit transition model directly from GraphStore edge_transitions records."""
        for tr in transitions:
            src = tr.get("from_camera_id", "").strip()
            dst = tr.get("to_camera_id", "").strip()
            dur_s = float(tr.get("transition_duration_s", 0.0))

            if not src or not dst or src == dst:
                continue

            if src not in self._matrix:
                self._matrix[src] = {}
            self._matrix[src][dst] = self._matrix[src].get(dst, 0) + 1

            if dur_s > 0.0:
                key = (src, dst)
                self._duration_sums[key] = self._duration_sums.get(key, 0.0) + dur_s

    def fit_from_graph_store(
        self,
        graph_store: BaseGraphStore,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 10000,
    ) -> None:
        """Read-only query to GraphStore to fit the transition matrix."""
        transitions = graph_store.get_camera_transitions(
            start_ns=start_ns, end_ns=end_ns, limit=limit
        )
        self.fit_from_transitions(transitions)

    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """Return deterministic normalized transition probability matrix: P(dest | src)."""
        prob_matrix: dict[str, dict[str, float]] = {}
        for src, dests in sorted(self._matrix.items()):
            total = sum(dests.values())
            if total <= 0:
                continue
            prob_matrix[src] = {
                dst: round(count / total, 6)
                for dst, count in sorted(dests.items())
            }
        return prob_matrix

    def predict_next(
        self,
        current_camera_id: str,
        global_id: str = "anonymous",
        timestamp_ns: int = 1_700_000_000_000_000_000,
        top_k: int = 3,
        horizon_s: float = 300.0,
    ) -> PredictionResult:
        """Forecast candidate next camera destinations for a given current camera node.

        Deterministically ranks candidate destinations by:
        1. Probability descending
        2. Evidence confidence descending
        3. Camera ID ascending (tie-breaker)

        Args:
            current_camera_id: Source camera node ID
            global_id: Identity key (defaults to "anonymous")
            timestamp_ns: Reference Unix nanosecond timestamp
            top_k: Maximum candidate predictions to return
            horizon_s: Default arrival horizon in seconds if no transition duration is available

        Returns:
            PredictionResult
        """
        if top_k <= 0:
            raise ValueError(f"top_k MUST be positive, got {top_k}")
        if timestamp_ns <= 0:
            raise ValueError("timestamp_ns MUST be a positive Unix nanosecond timestamp")

        src = current_camera_id.strip() if current_camera_id else ""

        # Insufficient evidence check: camera unknown or no outgoing transitions
        if not src or src not in self._matrix or not self._matrix[src]:
            return PredictionResult(
                global_id=global_id,
                current_camera_id=src or "unknown",
                timestamp_ns=timestamp_ns,
                predictions=[],
                overall_confidence=0.0,
                evidence_count=0,
                model_version=self.model_version,
                explanation=f"Insufficient transition evidence for camera '{src or 'unknown'}'.",
            )

        outgoing_map = self._matrix[src]
        total_transitions = sum(outgoing_map.values())
        if total_transitions <= 0:
            return PredictionResult(
                global_id=global_id,
                current_camera_id=src,
                timestamp_ns=timestamp_ns,
                predictions=[],
                overall_confidence=0.0,
                evidence_count=0,
                model_version=self.model_version,
                explanation=f"No outgoing transitions recorded for camera '{src}'.",
            )

        # Compute raw candidate predictions
        candidates: list[tuple[float, float, str, PredictedDestination]] = []

        # Heuristic evidence provenance confidence based on sample count
        overall_confidence = min(1.0, round(total_transitions / 10.0, 4))
        if overall_confidence <= 0.0:
            overall_confidence = 0.1

        for dst, count in outgoing_map.items():
            prob = count / float(total_transitions)
            prob_rounded = round(prob, 4)

            # Destination confidence scaled by candidate sample support
            dst_confidence = min(1.0, round(count / 5.0, 4))
            if dst_confidence <= 0.0:
                dst_confidence = 0.1

            lower_band = round(max(0.0, dst_confidence * 0.8), 4)
            upper_band = round(min(1.0, dst_confidence * 1.2), 4)
            if upper_band < lower_band:
                upper_band = lower_band

            # Estimate expected arrival timestamp
            key = (src, dst)
            if key in self._duration_sums and count > 0:
                avg_dur_s = self._duration_sums[key] / count
            else:
                avg_dur_s = horizon_s

            expected_arrival_ns = timestamp_ns + int(max(1.0, avg_dur_s) * 1e9)

            dest_obj = PredictedDestination(
                camera_id=dst,
                probability=prob_rounded,
                confidence=dst_confidence,
                confidence_band=(lower_band, upper_band),
                expected_arrival_ns=expected_arrival_ns,
            )
            # Tuple for deterministic sorting: (-prob, -confidence, camera_id)
            candidates.append((-prob_rounded, -dst_confidence, dst, dest_obj))

        # Deterministic sorting
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        top_candidates = [item[3] for item in candidates[:top_k]]

        explanation = (
            f"Forecasted top {len(top_candidates)} candidate destination(s) from camera '{src}' "
            f"based on {total_transitions} historical Markov transitions."
        )

        return PredictionResult(
            global_id=global_id,
            current_camera_id=src,
            timestamp_ns=timestamp_ns,
            predictions=top_candidates,
            overall_confidence=overall_confidence,
            evidence_count=total_transitions,
            model_version=self.model_version,
            explanation=explanation,
        )
