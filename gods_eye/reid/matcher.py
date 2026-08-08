"""Re-ID matcher — stateless matching primitives for identity resolution.

Operates on embeddings supplied to it. No persistent storage,
no gallery management, no identity lifecycle.

Responsibilities:
- Select best matching candidate from a set of embeddings
- Apply threshold validation
- Return top-k ranked candidates
- Log match decisions with structured logging
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.reid.similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    top_k_indices,
)


@dataclass(frozen=True)
class MatchResult:
    """Result of a single match query against candidates.

    Attributes:
        matched: Whether a match above threshold was found.
        best_index: Index of the best candidate (-1 if no match).
        best_score: Similarity score of the best candidate (-1.0 if no match).
        top_k: List of (index, score) tuples for the top-k candidates,
               sorted by score descending. Only includes candidates
               above threshold.
    """

    matched: bool
    best_index: int
    best_score: float
    top_k: list[tuple[int, float]]


class Matcher:
    """Stateless Re-ID matcher.

    Compares a query embedding against a set of candidate embeddings
    and returns ranked matches above the configured threshold.

    Not tied to any persistent storage — operates purely on
    the embeddings provided to each method call.
    """

    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.reid_match_threshold
        self._top_k = settings.reid_top_k
        self._log = get_logger("reid.matcher")

    def match(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        candidate_ids: list[str] | None = None,
    ) -> MatchResult:
        """Find the best matching candidate for a query embedding.

        Args:
            query: Query embedding vector, shape (D,).
            candidates: Candidate embeddings matrix, shape (N, D).
            candidate_ids: Optional IDs for logging. Does not affect
                           matching logic.

        Returns:
            MatchResult with best match info and top-k ranked candidates.
        """
        if candidates.shape[0] == 0:
            self._log.debug("matcher_no_candidates")
            return MatchResult(
                matched=False, best_index=-1, best_score=-1.0, top_k=[]
            )

        # Compute similarity against all candidates
        scores = cosine_similarity_matrix(query, candidates, normalized=True)

        # Get top-k indices (by score, descending)
        ranked_indices = top_k_indices(scores, self._top_k)

        # Filter to above-threshold candidates
        above_threshold: list[tuple[int, float]] = [
            (idx, float(scores[idx]))
            for idx in ranked_indices
            if float(scores[idx]) >= self._threshold
        ]

        if not above_threshold:
            self._log.debug(
                "matcher_no_match",
                best_score=round(float(scores[ranked_indices[0]]), 4)
                if ranked_indices
                else -1.0,
                threshold=self._threshold,
            )
            return MatchResult(
                matched=False, best_index=-1, best_score=-1.0, top_k=[]
            )

        best_idx, best_score = above_threshold[0]

        self._log.debug(
            "matcher_match_found",
            best_index=best_idx,
            best_score=round(best_score, 4),
            threshold=self._threshold,
            candidates_above_threshold=len(above_threshold),
            candidate_id=candidate_ids[best_idx]
            if candidate_ids
            else None,
        )

        return MatchResult(
            matched=True,
            best_index=best_idx,
            best_score=best_score,
            top_k=above_threshold,
        )

    def is_match(self, query: np.ndarray, candidate: np.ndarray) -> bool:
        """Check if two embeddings match above the threshold.

        Simple pairwise check — no ranking, no top-k.

        Args:
            query: First embedding, shape (D,).
            candidate: Second embedding, shape (D,).

        Returns:
            True if cosine similarity >= threshold.
        """
        score = cosine_similarity(query, candidate)
        return score >= self._threshold

    @property
    def threshold(self) -> float:
        """Current match threshold."""
        return self._threshold
