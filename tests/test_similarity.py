"""Tests for the similarity engine and matcher (Phase 2, Task 2).

Tests validate:
- Cosine similarity: identical, orthogonal, opposite, arbitrary vectors
- Zero-vector safety: no NaN, returns -1.0
- Batch similarity matrix: shape, correctness, zero-vector rows
- Top-k ranking: ordering, k limits, empty input
- Matcher: threshold behavior, top-k filtering, no-match, MatchResult fields
"""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.reid.matcher import MatchResult, Matcher
from gods_eye.reid.similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    top_k_indices,
)


# ═══════════════════════════════════════════════════════════════════════════════
# COSINE SIMILARITY — SCALAR
# ═══════════════════════════════════════════════════════════════════════════════


class TestCosineSimilarity:
    """Tests for pairwise cosine similarity."""

    def test_identical_vectors_return_1(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_identical_normalized_embedding(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(512).astype(np.float32)
        a = a / np.linalg.norm(a)
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_return_0(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_neg1(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_arbitrary_known_value(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cosine_similarity(a, b) == pytest.approx(float(expected), abs=1e-5)

    def test_output_range_clamped(self) -> None:
        """Result is always in [-1.0, 1.0] even with float drift."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        score = cosine_similarity(a, b)
        assert -1.0 <= score <= 1.0

    def test_magnitude_invariant(self) -> None:
        """Cosine similarity is invariant to vector magnitude."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_small = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        b_large = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        assert cosine_similarity(a, b_small) == pytest.approx(
            cosine_similarity(a, b_large), abs=1e-5
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ZERO-VECTOR SAFETY
# ═══════════════════════════════════════════════════════════════════════════════


class TestZeroVectorSafety:
    """Zero-vector guard: must return -1.0, never NaN."""

    def test_valid_vs_zero_returns_neg1(self) -> None:
        valid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        zero = np.zeros(3, dtype=np.float32)
        assert cosine_similarity(valid, zero) == -1.0

    def test_zero_vs_valid_returns_neg1(self) -> None:
        valid = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        zero = np.zeros(3, dtype=np.float32)
        assert cosine_similarity(zero, valid) == -1.0

    def test_zero_vs_zero_returns_neg1(self) -> None:
        zero = np.zeros(3, dtype=np.float32)
        assert cosine_similarity(zero, zero) == -1.0

    def test_near_zero_returns_neg1(self) -> None:
        near_zero = np.array([1e-8, 1e-8, 1e-8], dtype=np.float32)
        valid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(near_zero, valid) == -1.0

    def test_no_nan_ever(self) -> None:
        """Exhaustive check: no combination produces NaN."""
        vectors = [
            np.zeros(512, dtype=np.float32),
            np.ones(512, dtype=np.float32),
            np.random.default_rng(0).standard_normal(512).astype(np.float32),
        ]
        for a in vectors:
            for b in vectors:
                score = cosine_similarity(a, b)
                assert not np.isnan(score), f"NaN detected for {a[:3]}... vs {b[:3]}..."
                assert -1.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH SIMILARITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════════


class TestCosineSimilarityMatrix:
    """Tests for batch cosine_similarity_matrix."""

    def test_shape_matches_gallery_size(self) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gallery = np.eye(3, dtype=np.float32)
        scores = cosine_similarity_matrix(query, gallery)
        assert scores.shape == (3,)

    def test_identity_query_against_basis(self) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gallery = np.eye(3, dtype=np.float32)
        scores = cosine_similarity_matrix(query, gallery)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] == pytest.approx(0.0, abs=1e-5)
        assert scores[2] == pytest.approx(0.0, abs=1e-5)

    def test_empty_gallery_returns_empty(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        gallery = np.zeros((0, 2), dtype=np.float32)
        scores = cosine_similarity_matrix(query, gallery)
        assert scores.shape == (0,)

    def test_zero_query_returns_all_neg1(self) -> None:
        query = np.zeros(3, dtype=np.float32)
        gallery = np.eye(3, dtype=np.float32)
        scores = cosine_similarity_matrix(query, gallery)
        assert all(s == -1.0 for s in scores)

    def test_zero_gallery_row_returns_neg1(self) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gallery = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Zero row
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        scores = cosine_similarity_matrix(query, gallery)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] == -1.0  # Zero row
        assert scores[2] == pytest.approx(0.0, abs=1e-5)

    def test_no_nan_in_matrix(self) -> None:
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64).astype(np.float32)
        gallery = np.vstack([
            rng.standard_normal(64).astype(np.float32),
            np.zeros(64, dtype=np.float32),
            rng.standard_normal(64).astype(np.float32),
        ])
        scores = cosine_similarity_matrix(query, gallery)
        assert not np.any(np.isnan(scores))

    def test_matrix_agrees_with_scalar(self) -> None:
        """Matrix version produces same values as scalar version."""
        rng = np.random.default_rng(99)
        query = rng.standard_normal(128).astype(np.float32)
        gallery = rng.standard_normal((5, 128)).astype(np.float32)

        matrix_scores = cosine_similarity_matrix(query, gallery)
        scalar_scores = [cosine_similarity(query, gallery[i]) for i in range(5)]

        for i in range(5):
            assert matrix_scores[i] == pytest.approx(scalar_scores[i], abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# TOP-K RANKING
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopKIndices:
    """Tests for top_k_indices ranking utility."""

    def test_returns_correct_order(self) -> None:
        scores = np.array([0.3, 0.9, 0.1, 0.7], dtype=np.float32)
        result = top_k_indices(scores, k=3)
        assert result == [1, 3, 0]  # 0.9, 0.7, 0.3

    def test_k_larger_than_array(self) -> None:
        scores = np.array([0.5, 0.8], dtype=np.float32)
        result = top_k_indices(scores, k=10)
        assert len(result) == 2
        assert result[0] == 1  # Highest first

    def test_empty_scores(self) -> None:
        scores = np.array([], dtype=np.float32)
        assert top_k_indices(scores, k=5) == []

    def test_single_element(self) -> None:
        scores = np.array([0.42], dtype=np.float32)
        assert top_k_indices(scores, k=1) == [0]

    def test_k_equals_1(self) -> None:
        scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        result = top_k_indices(scores, k=1)
        assert result == [1]


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHER
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatcher:
    """Tests for the stateless Matcher."""

    @pytest.fixture()
    def matcher(self) -> Matcher:
        settings = Settings(reid_match_threshold=0.75, reid_top_k=3)
        return Matcher(settings)

    def test_exact_match_above_threshold(self, matcher: Matcher) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([
            [1.0, 0.0, 0.0],  # Identical = 1.0
            [0.0, 1.0, 0.0],  # Orthogonal = 0.0
        ], dtype=np.float32)
        result = matcher.match(query, candidates)
        assert result.matched is True
        assert result.best_index == 0
        assert result.best_score == pytest.approx(1.0, abs=1e-5)

    def test_no_match_below_threshold(self, matcher: Matcher) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([
            [0.0, 1.0, 0.0],  # 0.0 similarity
            [0.0, 0.0, 1.0],  # 0.0 similarity
        ], dtype=np.float32)
        result = matcher.match(query, candidates)
        assert result.matched is False
        assert result.best_index == -1
        assert result.best_score == -1.0
        assert result.top_k == []

    def test_empty_candidates(self, matcher: Matcher) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.zeros((0, 3), dtype=np.float32)
        result = matcher.match(query, candidates)
        assert result.matched is False
        assert result.best_index == -1

    def test_top_k_ordering(self, matcher: Matcher) -> None:
        """Top-k results are sorted by score descending."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([
            [0.9, 0.1, 0.0],   # High
            [0.0, 1.0, 0.0],   # Orthogonal
            [0.95, 0.05, 0.0], # Highest
        ], dtype=np.float32)
        result = matcher.match(query, candidates)
        assert result.matched is True
        assert len(result.top_k) >= 1
        # Top-k scores should be in descending order
        scores_in_topk = [s for _, s in result.top_k]
        assert scores_in_topk == sorted(scores_in_topk, reverse=True)

    def test_zero_vector_candidate_rejected(self, matcher: Matcher) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([
            [0.0, 0.0, 0.0],  # Zero vector
        ], dtype=np.float32)
        result = matcher.match(query, candidates)
        assert result.matched is False

    def test_zero_vector_query_no_match(self, matcher: Matcher) -> None:
        query = np.zeros(3, dtype=np.float32)
        candidates = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        result = matcher.match(query, candidates)
        assert result.matched is False

    def test_is_match_above_threshold(self, matcher: Matcher) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert matcher.is_match(a, b) is True

    def test_is_match_below_threshold(self, matcher: Matcher) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert matcher.is_match(a, b) is False

    def test_is_match_zero_vector(self, matcher: Matcher) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        zero = np.zeros(3, dtype=np.float32)
        assert matcher.is_match(a, zero) is False

    def test_match_result_is_frozen(self, matcher: Matcher) -> None:
        result = MatchResult(matched=False, best_index=-1, best_score=-1.0, top_k=[])
        with pytest.raises(AttributeError):
            result.matched = True  # type: ignore[misc]

    def test_threshold_property(self, matcher: Matcher) -> None:
        assert matcher.threshold == 0.75

    def test_candidate_ids_does_not_affect_matching(self, matcher: Matcher) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        r1 = matcher.match(query, candidates)
        r2 = matcher.match(query, candidates, candidate_ids=["id-abc"])
        assert r1.matched == r2.matched
        assert r1.best_score == pytest.approx(r2.best_score, abs=1e-6)
