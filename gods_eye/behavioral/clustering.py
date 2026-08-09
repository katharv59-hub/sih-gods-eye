"""Spatial Trajectory Clustering Engine — Sub-Phase 6.3 (§5 & §6 Master Spec).

Consumes TrajectorySequence objects produced by Sub-Phase 6.2 and clusters them
using Normalized Sequence Edit Distance + DBSCAN (Density-Based Spatial Clustering).

Outputs canonical Phase 6.1 TrajectoryCluster instances with deterministic ordering,
medoid representative selection, and privacy-minimized aggregate statistics.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from gods_eye.behavioral.extractor import TrajectorySequence, normalized_sequence_distance
from gods_eye.schemas.behavioral import TrajectoryCluster


class TrajectoryClusteringEngine:
    """Deterministic spatial trajectory clustering engine based on sequence distance and DBSCAN."""

    def __init__(self, eps: float = 0.3, min_samples: int = 3) -> None:
        """Initialize clustering engine with DBSCAN parameters.

        Args:
            eps: Maximum normalized sequence distance threshold for neighbor connectivity [0.0, 1.0].
            min_samples: Minimum number of samples required to form a core cluster node.
        """
        if not (0.0 <= eps <= 1.0):
            raise ValueError(f"eps MUST be between 0.0 and 1.0, got {eps}")
        if min_samples < 1:
            raise ValueError(f"min_samples MUST be at least 1, got {min_samples}")

        self.eps = eps
        self.min_samples = min_samples

    def _compute_distance_matrix(
        self, sequences: list[TrajectorySequence]
    ) -> list[list[float]]:
        """Compute pairwise normalized sequence edit distance matrix."""
        n = len(sequences)
        dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = normalized_sequence_distance(
                    sequences[i].camera_sequence, sequences[j].camera_sequence
                )
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        return dist_matrix

    def _dbscan(self, dist_matrix: list[list[float]]) -> list[int]:
        """Deterministic pure-Python DBSCAN over distance matrix.

        Labels:
            -1: Noise point
            0, 1, 2, ...: Cluster ID
        """
        n = len(dist_matrix)
        labels = [-1] * n
        visited = [False] * n
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = [j for j in range(n) if dist_matrix[i][j] <= self.eps]

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
            else:
                labels[i] = cluster_id
                # Expand cluster deterministically
                queue = [idx for idx in neighbors if idx != i]
                idx_in_queue = set(queue)

                q_idx = 0
                while q_idx < len(queue):
                    j = queue[q_idx]
                    q_idx += 1

                    if not visited[j]:
                        visited[j] = True
                        j_neighbors = [k for k in range(n) if dist_matrix[j][k] <= self.eps]
                        if len(j_neighbors) >= self.min_samples:
                            for k in j_neighbors:
                                if k not in idx_in_queue and k != i:
                                    queue.append(k)
                                    idx_in_queue.add(k)

                    if labels[j] == -1:
                        labels[j] = cluster_id

                cluster_id += 1

        return labels

    def cluster(
        self,
        input_data: list[Union[TrajectorySequence, tuple[str, TrajectorySequence]]],
    ) -> list[TrajectoryCluster]:
        """Cluster spatial trajectory sequences into canonical TrajectoryCluster instances.

        Input can be:
        - list[TrajectorySequence]
        - list[tuple[global_id, TrajectorySequence]]

        Returns:
            list[TrajectoryCluster] sorted deterministically by representative camera sequence.
        """
        if not input_data:
            return []

        # Parse inputs while preserving optional global_id keys internally
        entries: list[tuple[Optional[str], TrajectorySequence]] = []
        for item in input_data:
            if isinstance(item, tuple):
                entries.append((item[0], item[1]))
            else:
                entries.append((None, item))

        # Filter out empty camera sequence trajectories
        valid_entries = [e for e in entries if e[1].camera_sequence]
        if not valid_entries:
            return []

        # Sort valid entries deterministically before clustering to guarantee ordering
        sorted_entries = sorted(
            valid_entries,
            key=lambda e: (
                e[1].camera_sequence,
                e[1].total_duration_s,
                e[0] or "",
            ),
        )

        sequences = [e[1] for e in sorted_entries]
        dist_matrix = self._compute_distance_matrix(sequences)
        labels = self._dbscan(dist_matrix)

        # Group indices by cluster label
        cluster_groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label != -1:
                cluster_groups.setdefault(label, []).append(idx)

        if not cluster_groups:
            return []

        result_clusters: list[TrajectoryCluster] = []

        # Process each raw DBSCAN cluster
        raw_cluster_items: list[tuple[list[str], list[int]]] = []
        for label, member_indices in cluster_groups.items():
            # Compute medoid (representative camera sequence)
            medoid_idx = member_indices[0]
            best_sum_dist = float("inf")
            for idx in member_indices:
                sum_dist = sum(dist_matrix[idx][other] for other in member_indices)
                if sum_dist < best_sum_dist:
                    best_sum_dist = sum_dist
                    medoid_idx = idx

            rep_seq = sequences[medoid_idx].camera_sequence
            raw_cluster_items.append((rep_seq, member_indices))

        # Sort clusters deterministically by representative camera sequence
        raw_cluster_items.sort(key=lambda item: (item[0], len(item[1])))

        for cluster_idx, (rep_seq, member_indices) in enumerate(raw_cluster_items, start=1):
            cluster_id = f"cluster_{cluster_idx:03d}"
            occurrence_count = len(member_indices)

            # Privacy invariant: aggregate unique identity count only
            identities = {sorted_entries[idx][0] for idx in member_indices if sorted_entries[idx][0]}
            unique_identity_count = len(identities) if identities else 1

            durations = [sequences[idx].total_duration_s for idx in member_indices]
            mean_duration_s = sum(durations) / len(durations) if durations else 0.0

            confidences = [sequences[idx].confidence for idx in member_indices]
            confidence = sum(confidences) / len(confidences) if confidences else 1.0

            cluster_obj = TrajectoryCluster(
                cluster_id=cluster_id,
                camera_sequence=rep_seq,
                occurrence_count=occurrence_count,
                unique_identity_count=unique_identity_count,
                mean_duration_s=round(mean_duration_s, 4),
                confidence=round(confidence, 4),
            )
            result_clusters.append(cluster_obj)

        return result_clusters
