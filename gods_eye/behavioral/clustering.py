"""Spatial Trajectory Clustering Engine — Sub-Phase 6.3 (§5 & §6 Master Spec).

Consumes TrajectorySequence objects produced by Sub-Phase 6.2 and clusters them
using Normalized Sequence Edit Distance + DBSCAN (Density-Based Spatial Clustering).

Outputs canonical Phase 6.1 TrajectoryCluster instances with deterministic ordering,
medoid representative selection, and privacy-minimized aggregate statistics.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from gods_eye.behavioral.extractor import TrajectorySequence, normalized_sequence_distance
from gods_eye.schemas.behavioral import (
    SpatialTrajectory,
    SpatialTrajectoryCluster,
    TrajectoryCluster,
    TrajectoryFeature,
)

# Trajectory feature scaling constants (§17 Master Spec)
TRAJECTORY_MAX_DURATION_S: float = 300.0  # Upper bound duration (§17 configuration key)
TRAJECTORY_REF_SPEED: float = 1.0        # Reference traversal speed (1.0 normalized FOV / s)

# Trajectory feature distance weights (ADR-009)
WEIGHT_SPATIAL: float = 0.70    # 70% path geometry
WEIGHT_SPEED: float = 0.15      # 15% pacing / velocity
WEIGHT_DURATION: float = 0.15   # 15% dwell / temporal persistence


def compute_trajectory_feature_distance(
    feat_a: TrajectoryFeature,
    feat_b: TrajectoryFeature,
    w_spatial: float = WEIGHT_SPATIAL,
    w_speed: float = WEIGHT_SPEED,
    w_duration: float = WEIGHT_DURATION,
) -> float:
    """Compute multi-modal distance between two canonical TrajectoryFeatures (§4.5 & ADR-009).

    Incorporates resampled 2D waypoints, mean speed, and total duration with principled
    bounded normalization to prevent scale disparity:
      - Spatial: Mean Euclidean distance across corresponding waypoints in [0.0, sqrt(2)].
      - Speed: Rational soft-saturation v / (v + V_ref) bounded in [0.0, 1.0).
      - Duration: Concave logarithmic scaling log1p(min(t, T_max)) / log1p(T_max) bounded in [0.0, 1.0].

    Returns:
        Symmetric non-negative distance in [0.0, ~1.3].
    """
    # 1. Spatial path distance (mean Euclidean waypoint difference)
    n_wp = len(feat_a.waypoints)
    waypoint_diffs = [
        math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        for p1, p2 in zip(feat_a.waypoints, feat_b.waypoints)
    ]
    d_spatial = sum(waypoint_diffs) / float(n_wp) if n_wp > 0 else 0.0

    # 2. Normalized speed distance (bounded rational saturation)
    v_a = feat_a.mean_speed / (feat_a.mean_speed + TRAJECTORY_REF_SPEED)
    v_b = feat_b.mean_speed / (feat_b.mean_speed + TRAJECTORY_REF_SPEED)
    d_speed = abs(v_a - v_b)

    # 3. Normalized duration distance (bounded logarithmic scaling per Weber-Fechner law)
    log_max_dur = math.log1p(TRAJECTORY_MAX_DURATION_S)
    dur_a = math.log1p(min(feat_a.duration_s, TRAJECTORY_MAX_DURATION_S)) / log_max_dur
    dur_b = math.log1p(min(feat_b.duration_s, TRAJECTORY_MAX_DURATION_S)) / log_max_dur
    d_duration = abs(dur_a - dur_b)

    # 4. Multi-modal weighted combination
    return float(w_spatial * d_spatial + w_speed * d_speed + w_duration * d_duration)


class _CondensedClusterNode:
    """Node in the HDBSCAN condensed cluster tree (§4.5 & ADR-008)."""

    def __init__(self, cid: int, birth_lambda: float, dendro_node: int, initial_points: list[int]) -> None:
        self.cid = cid
        self.birth_lambda = birth_lambda
        self.death_lambda = birth_lambda
        self.dendro_node = dendro_node
        self.children: list[int] = []
        self.points: set[int] = set(initial_points)
        self.dropouts: dict[int, float] = {}


def hdbscan_cluster(
    dist_matrix: Union[np.ndarray, list[list[float]]],
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    allow_single_cluster: bool = False,
) -> list[int]:
    """Cluster pairwise distance matrix using HDBSCAN (Campello et al., 2013).

    Implements formal HDBSCAN density hierarchy & cluster stability per Master Spec §4.5 / ADR-008:
    1. Core distance computation: d_core(p) = distance to k-th nearest neighbor.
    2. Mutual reachability distance: d_mreach(a, b) = max(d_core(a), d_core(b), d(a, b)).
    3. Minimum spanning tree / single-linkage dendrogram via SciPy linkage.
    4. Lambda density space conversion: lambda = 1 / delta_merge.
    5. Condensed cluster tree construction: true splits vs member dropouts.
    6. Cluster stability integration: S(C) = sum_{p in C} (lambda_p - lambda_birth(C)).
    7. Flat cluster extraction via Excess of Mass (EOM) dynamic programming.
    8. Deterministic label assignment and noise identification (-1).

    Args:
        dist_matrix: N x N symmetric pairwise distance matrix.
        min_cluster_size: Minimum member threshold required to form a cluster.
        min_samples: Core neighborhood size m (1-based, including self; defaults to min_cluster_size).
        allow_single_cluster: If True, allows root cluster to compete in EOM
            selection (canonical HDBSCAN default is False).

    Returns:
        List of cluster labels of length N (-1 for noise, 0, 1, 2... for clusters).
    """
    n = len(dist_matrix)
    if min_cluster_size < 2:
        raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")
    if min_samples is not None and min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    if n < min_cluster_size:
        return [-1] * n

    dists = np.array(dist_matrix, dtype=float)
    if dists.shape != (n, n):
        raise ValueError(f"dist_matrix must be square of shape (N, N), got {dists.shape}")

    # 1. Core distance: distance to m-th nearest neighbor including self (0-indexed: index m - 1)
    effective_min_samples = min_samples if min_samples is not None else min_cluster_size
    k_idx = max(0, min(effective_min_samples - 1, n - 1))
    sorted_dists = np.sort(dists, axis=1)
    core_dists = sorted_dists[:, k_idx]

    # 2. Mutual reachability distance: d_mreach(a, b) = max(core(a), core(b), d(a, b))
    mreach = np.maximum(np.maximum(core_dists[:, None], core_dists[None, :]), dists)
    np.fill_diagonal(mreach, 0.0)

    # 3. Hierarchy construction: single-linkage over condensed mutual reachability matrix
    condensed_dist = squareform(mreach, checks=False)
    z_linkage = sch.linkage(condensed_dist, method="single")

    # Map member points for each dendrogram node (0..n-1 are leaves, n..2n-2 are merges)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    for idx, row in enumerate(z_linkage):
        node_id = n + idx
        members[node_id] = members[int(row[0])] + members[int(row[1])]

    # 4. Condensed Cluster Tree Construction
    root_dendro = 2 * n - 2
    nodes: dict[int, _CondensedClusterNode] = {}
    cluster_counter = 0

    nodes[0] = _CondensedClusterNode(0, 0.0, root_dendro, members[root_dendro])
    dendro_to_cluster: dict[int, int] = {root_dendro: 0}

    for idx in range(n - 2, -1, -1):
        u = n + idx
        c1 = int(z_linkage[idx, 0])
        c2 = int(z_linkage[idx, 1])
        delta = z_linkage[idx, 2]
        lam = 1.0 / delta if delta > 1e-9 else 1e9

        cid = dendro_to_cluster.get(u)
        if cid is None:
            continue

        s1 = len(members[c1])
        s2 = len(members[c2])

        if s1 >= min_cluster_size and s2 >= min_cluster_size:
            # Case A: True cluster split — parent ends at lambda; two new clusters are born
            nodes[cid].death_lambda = lam
            cluster_counter += 1
            cid1 = cluster_counter
            nodes[cid1] = _CondensedClusterNode(cid1, lam, c1, members[c1])

            cluster_counter += 1
            cid2 = cluster_counter
            nodes[cid2] = _CondensedClusterNode(cid2, lam, c2, members[c2])

            nodes[cid].children.extend([cid1, cid2])
            dendro_to_cluster[c1] = cid1
            dendro_to_cluster[c2] = cid2

        elif s1 >= min_cluster_size and s2 < min_cluster_size:
            # Case B: Points in c2 fall out of active cluster as noise/dropouts at lambda
            dendro_to_cluster[c1] = cid
            for p in members[c2]:
                nodes[cid].dropouts[p] = lam

        elif s2 >= min_cluster_size and s1 < min_cluster_size:
            # Case C: Points in c1 fall out of active cluster as noise/dropouts at lambda
            dendro_to_cluster[c2] = cid
            for p in members[c1]:
                nodes[cid].dropouts[p] = lam

        else:
            # Case D: Both children < min_cluster_size — active cluster completely dissolves
            nodes[cid].death_lambda = lam
            for p in members[u]:
                if p not in nodes[cid].dropouts:
                    nodes[cid].dropouts[p] = lam

    # 5. Cluster Stability Calculation: S(C) = sum_{p in C} (lambda_p - lambda_birth(C))
    stabilities: dict[int, float] = {}
    for cid, node in nodes.items():
        stab = 0.0
        for p in node.points:
            lam_p = node.dropouts.get(p, node.death_lambda)
            stab += max(0.0, lam_p - node.birth_lambda)
        stabilities[cid] = stab

    # 6. Flat Cluster Selection via Excess of Mass (EOM)
    # Bottom-up dynamic programming traversal from highest cid down to root 0
    subtree_stability: dict[int, float] = {}
    selected_clusters: set[int] = set()

    for cid in range(cluster_counter, -1, -1):
        node = nodes[cid]
        if not node.children:
            subtree_stability[cid] = stabilities[cid]
            selected_clusters.add(cid)
        else:
            child_sum = sum(subtree_stability[ch] for ch in node.children)
            if (cid != 0 or allow_single_cluster) and stabilities[cid] >= child_sum:
                # Parent has greater or equal Excess of Mass than children (canonical parent-on-tie)
                subtree_stability[cid] = stabilities[cid]
                selected_clusters.add(cid)

                # Deselect all descendants in subtree
                def _deselect_descendants(parent_id: int) -> None:
                    for ch in nodes[parent_id].children:
                        if ch in selected_clusters:
                            selected_clusters.remove(ch)
                        _deselect_descendants(ch)

                _deselect_descendants(cid)
            else:
                subtree_stability[cid] = child_sum

    if not allow_single_cluster and 0 in selected_clusters:
        selected_clusters.remove(0)

    # 7. Cluster Membership & Noise Assignment
    # Canonical HDBSCAN semantics (McInnes _hdbscan_tree.pyx / Campello et al., 2013):
    # - Non-root selected clusters (cid != 0): all members in the cluster's subtree belong to it.
    # - Root cluster (cid == 0, allow_single_cluster=True): points departing prior to the cluster's
    #   maximum persistence density (death_lambda / parent_max_lambda) are noise; points persisting
    #   until death form the single cluster core.
    labels = [-1] * n
    sorted_selected = sorted(selected_clusters, key=lambda c: min(nodes[c].points))

    for label_idx, cid in enumerate(sorted_selected):
        node = nodes[cid]
        for p in node.points:
            lam_p = node.dropouts.get(p, node.death_lambda)
            if cid != 0 or len(node.dropouts) == 0:
                labels[p] = label_idx
            else:
                if lam_p >= node.death_lambda - 1e-5:
                    labels[p] = label_idx

    return labels


def fit_spatial_trajectory_clusters(
    trajectories: list[SpatialTrajectory],
    min_cluster_size: int = 3,
    min_cluster_samples: int = 20,
    n_waypoints: int = 10,
) -> list[SpatialTrajectoryCluster]:
    """Fit spatial trajectory clusters from observed trajectories (§4.5 & §17 Master Spec).

    Derives empirical centroid waypoints, empirical standard deviation, member count,
    and maturity solely from actual trajectory members.

    Args:
        trajectories: List of observed SpatialTrajectory objects.
        min_cluster_size: Minimum member trajectories required to form a cluster.
        min_cluster_samples: Minimum member count required for a cluster to be considered mature.
        n_waypoints: Number of resampled waypoints along cumulative path length.

    Returns:
        List of SpatialTrajectoryCluster instances with 100% empirical attributes.
    """
    valid_trajectories = [t for t in trajectories if len(t.points) >= 10]
    if len(valid_trajectories) < min_cluster_size:
        return []

    # 1. Extract canonical TrajectoryFeature for each valid trajectory (§4.5 & ADR-009)
    features = [t.extract_features(n_waypoints=n_waypoints) for t in valid_trajectories]
    resampled_paths = [f.waypoints for f in features]
    n = len(valid_trajectories)

    # 2. Pairwise multi-modal trajectory feature distance matrix
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_trajectory_feature_distance(features[i], features[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # 3. Cluster using HDBSCAN — multi-cluster preference with single-cluster fallback.
    # Policy: prefer subclusters (allow_single_cluster=False, canonical default).
    # If no subclusters found (all noise), accept single cohesive cluster.
    # This ensures single-corridor datasets produce a learned cluster while
    # multi-corridor datasets discover distinct subclusters.
    labels = hdbscan_cluster(dist_matrix, min_cluster_size=min_cluster_size,
                             allow_single_cluster=False)
    if all(label == -1 for label in labels):
        labels = hdbscan_cluster(dist_matrix, min_cluster_size=min_cluster_size,
                                 allow_single_cluster=True)

    # Group member indices by cluster label
    cluster_groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label != -1:
            cluster_groups.setdefault(label, []).append(idx)

    if not cluster_groups:
        return []

    learned_clusters: list[SpatialTrajectoryCluster] = []

    for cluster_label, member_indices in sorted(cluster_groups.items()):
        member_count = len(member_indices)
        member_paths = [resampled_paths[idx] for idx in member_indices]
        member_trajs = [valid_trajectories[idx] for idx in member_indices]

        # Empirical Centroid: Mean coordinates across all members at each waypoint
        centroid_waypoints: list[tuple[float, float]] = []
        for wp_idx in range(n_waypoints):
            mean_x = sum(path[wp_idx][0] for path in member_paths) / float(member_count)
            mean_y = sum(path[wp_idx][1] for path in member_paths) / float(member_count)
            centroid_waypoints.append((round(mean_x, 4), round(mean_y, 4)))

        # Empirical Standard Deviation: Root-mean-square distance of member trajectories to centroid
        member_distances: list[float] = []
        for path in member_paths:
            dist_to_centroid = sum(
                math.hypot(p[0] - c[0], p[1] - c[1])
                for p, c in zip(path, centroid_waypoints)
            ) / float(n_waypoints)
            member_distances.append(dist_to_centroid)

        # Sample standard deviation (or RMSD from centroid)
        if member_count > 1:
            mean_dist = sum(member_distances) / float(member_count)
            variance = sum((d - mean_dist) ** 2 for d in member_distances) / float(member_count - 1)
            empirical_std = math.sqrt(variance)
        else:
            empirical_std = 0.0

        # Speeds and durations
        speeds = [t.compute_mean_speed() for t in member_trajs]
        durations = [t.compute_duration_s() for t in member_trajs]
        mean_speed = sum(speeds) / float(member_count) if speeds else 0.0
        mean_duration = sum(durations) / float(member_count) if durations else 0.0

        # Camera ID from members
        camera_id = member_trajs[0].camera_id if member_trajs else ""

        # Maturity: cluster becomes mature only when member_count >= min_cluster_samples
        is_mature = member_count >= min_cluster_samples

        cluster_id = f"learned_cluster_{camera_id}_{cluster_label + 1:03d}" if camera_id else f"learned_cluster_{cluster_label + 1:03d}"

        cluster = SpatialTrajectoryCluster(
            cluster_id=cluster_id,
            centroid_waypoints=centroid_waypoints,
            std_dev=round(empirical_std, 4),
            sample_count=member_count,
            is_mature=is_mature,
            camera_id=camera_id,
            mean_speed=round(mean_speed, 4),
            mean_duration_s=round(mean_duration, 4),
        )
        learned_clusters.append(cluster)

    return learned_clusters


class TrajectoryClusteringEngine:
    """Deterministic spatial trajectory clustering engine based on sequence distance and DBSCAN/HDBSCAN."""

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

    def cluster_spatial(
        self,
        trajectories: list[SpatialTrajectory],
        min_cluster_size: int = 3,
        min_cluster_samples: int = 20,
        n_waypoints: int = 10,
    ) -> list[SpatialTrajectoryCluster]:
        """Cluster spatial trajectories using HDBSCAN per Master Spec §4.5."""
        return fit_spatial_trajectory_clusters(
            trajectories=trajectories,
            min_cluster_size=min_cluster_size,
            min_cluster_samples=min_cluster_samples,
            n_waypoints=n_waypoints,
        )

