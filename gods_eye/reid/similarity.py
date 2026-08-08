"""Cosine similarity engine — safe, defensive embedding comparison.

All functions in this module are pure (no side effects, no state).
Zero-vector safety is enforced: any embedding with norm < epsilon
returns -1.0 similarity to prevent NaN propagation and false matches.

ADR-002: Cosine similarity chosen over Euclidean (magnitude-sensitive)
and Siamese metrics (requires training infrastructure).
"""

from __future__ import annotations

import numpy as np

# Minimum L2 norm threshold. Below this, the vector is treated as
# invalid (likely a zero-vector from a failed crop extraction).
_NORM_EPSILON: float = 1e-6


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector, shape (D,).
        b: Second embedding vector, shape (D,).

    Returns:
        Cosine similarity in [-1.0, 1.0].
        Returns -1.0 if either vector has norm < epsilon
        (zero-vector safety guard).
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < _NORM_EPSILON or norm_b < _NORM_EPSILON:
        return -1.0

    score = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [-1.0, 1.0] to guard against floating point drift
    return max(-1.0, min(1.0, score))


def cosine_similarity_matrix(
    query: np.ndarray, gallery: np.ndarray, *, normalized: bool = False,
) -> np.ndarray:
    """Compute cosine similarity between one query and N gallery vectors.

    Args:
        query: Single embedding vector, shape (D,).
        gallery: Gallery matrix, shape (N, D).
        normalized: If True, assumes both query and gallery rows are
            already L2-normalized (skips norm computation — ~50x faster
            at 10K entries).  The extractor contract guarantees this
            for all embeddings in the God's Eye pipeline.

    Returns:
        1D array of shape (N,) with similarity scores in [-1.0, 1.0].
        Returns -1.0 for any gallery entry with norm < epsilon
        (only when ``normalized=False``).
        Returns empty array if gallery is empty.
    """
    if gallery.shape[0] == 0:
        return np.array([], dtype=np.float32)

    if normalized:
        # Fast path: dot product only (vectors are unit-length)
        scores = gallery @ query
        np.clip(scores, -1.0, 1.0, out=scores)
        return np.asarray(scores, dtype=np.float32)

    # Defensive path: compute norms for safety
    query_norm = float(np.linalg.norm(query))
    if query_norm < _NORM_EPSILON:
        return np.full(gallery.shape[0], -1.0, dtype=np.float32)

    # Compute gallery norms
    gallery_norms = np.linalg.norm(gallery, axis=1)  # (N,)

    # Dot product: query · each gallery row
    dots = gallery @ query  # (N,)

    # Build result with zero-vector safety
    scores = np.full(gallery.shape[0], -1.0, dtype=np.float32)
    valid_mask = gallery_norms > _NORM_EPSILON
    scores[valid_mask] = (
        dots[valid_mask] / (query_norm * gallery_norms[valid_mask])
    ).astype(np.float32)

    # Clamp to [-1.0, 1.0]
    np.clip(scores, -1.0, 1.0, out=scores)

    return scores


def top_k_indices(
    scores: np.ndarray, k: int
) -> list[int]:
    """Return indices of the top-k highest similarity scores.

    Args:
        scores: 1D array of similarity scores.
        k: Number of top results to return.

    Returns:
        List of indices sorted by score descending.
        Returns fewer than k if array is smaller.
    """
    if scores.shape[0] == 0:
        return []
    k = min(k, scores.shape[0])
    # argpartition is O(n) vs argsort O(n log n), then sort only top-k
    top_indices = np.argpartition(scores, -k)[-k:]
    # Sort the top-k by score descending
    sorted_top = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [int(i) for i in sorted_top]
