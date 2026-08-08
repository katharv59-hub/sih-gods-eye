"""Identity Mapping Layer (IML) — Phase 2 + Phase 3.

The IML is the core orchestrator of identity persistence.  It connects
embedding extraction, similarity matching, identity lifecycle management,
and gallery storage into a coherent identity persistence pipeline.

Phase 3 additions:
    - Optional ``CameraGraph`` dependency for adjacency-aware scoring.
    - Cross-camera transition detection: when a matched identity's
      ``last_camera_id`` differs from the current camera, a
      ``CROSS_CAMERA_TRANSITION`` signal is emitted.
    - ``Identity.camera_history`` and ``last_camera_id`` are updated.

Per-frame contract:
    Input:  ``TrackingResult`` (ephemeral ByteTrack IDs)
    Output: ``IdentityResult``  (stable global Identity UUIDs)

Architecture (ADR-004):
    ByteTrack ID → IML → OSNet embedding → cosine similarity →
    gallery → EMA update → Global Identity UUID

Design decisions incorporated from the architecture review:
    - C1: ``IdentityResult`` composes ``TrackingResult`` (no field copy).
    - C2: Gallery embedding matrix caching (dirty-flag) is owned by
      the gallery; the IML calls ``get_embedding_matrix()``.
    - C3: Score-sorted global assignment pass for conflict resolution.
    - C4: Cache pruning on explicit LOST/DEAD state, not absence.
    - R2: Emits ``IdentityTransition`` signals, not ``Event`` objects.
    - R6: EMA computation handles first-observation edge case.

This module is NOT thread-safe.  It is designed to run inside a single
``IdentityWorker`` thread (spec §9 — single-writer for the gallery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.pipeline import TrackingResult
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.identity import Identity, LifecycleState, create_identity
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.matcher import Matcher, MatchResult
from gods_eye.schemas.track import Track, TrackState


# ── Transition Types ─────────────────────────────────────────────────────


class TransitionType(Enum):
    """Types of identity transitions emitted by the IML.

    These are lightweight domain signals, not full Event objects.
    Conversion to §4 ``Event`` happens at the pipeline boundary.
    """

    CONFIRMED_NEW = "confirmed_new"        # New identity registered
    CONFIRMED_RELINK = "confirmed_relink"  # LOST identity re-linked
    CROSS_CAMERA_TRANSITION = "cross_camera_transition"  # Phase 3: camera change
    LOST = "lost"                          # ACTIVE → LOST timeout
    EXPIRED = "expired"                    # LOST → EXPIRED (TTL)


@dataclass(frozen=True)
class IdentityTransition:
    """A lightweight, immutable signal describing an identity state change.

    The IML emits these instead of full ``Event`` objects (review R2).
    A separate converter transforms them into §4 ``Event`` records at
    the pipeline boundary.

    Attributes:
        transition_type: What happened.
        global_id: The affected identity's UUID.
        camera_id: Camera where the transition was observed.
        timestamp_ns: When the transition occurred (Unix nanoseconds).
        frame_id: Frame number at which the transition occurred.
        confidence: Match confidence score in [0.0, 1.0].
            For new identities this is 0.0 (no prior to compare).
            For re-links this is the match score.
            For lost/expired this is the identity's last confidence.
        track_id: Tracker-local ID that triggered the transition.
            Present for CONFIRMED_NEW and CONFIRMED_RELINK.
            ``None`` for LOST and EXPIRED (no active track).
        previous_state: The identity's state before the transition.
            ``None`` for CONFIRMED_NEW (identity did not exist).
    """

    transition_type: TransitionType
    global_id: str
    camera_id: str
    timestamp_ns: int
    frame_id: int
    confidence: float
    track_id: str | None = None
    previous_state: LifecycleState | None = None


# ── Pipeline Output Type ─────────────────────────────────────────────────


@dataclass
class IdentityResult:
    """Phase 2 pipeline output: tracking result enriched with identities.

    Composes ``TrackingResult`` rather than copying its fields (review C1).
    Downstream consumers access frame/detection/track data via
    ``result.tracking.packet``, ``result.tracking.tracks``, etc.

    Attributes:
        tracking: The upstream tracking result (frame, detections, tracks).
        identities: Mapping of ``track_id → Identity`` for every track
            that was successfully resolved this frame.  Tracks with
            failed crops (zero-vector embedding) are absent.
        transitions: Identity state changes that occurred during this
            frame's processing.  Empty list when nothing changed.
    """

    tracking: TrackingResult
    identities: dict[str, Identity] = field(default_factory=dict)
    transitions: list[IdentityTransition] = field(default_factory=list)


# ── Identity Mapper ──────────────────────────────────────────────────────


class IdentityMapper:
    """Identity Mapping Layer — the core identity persistence orchestrator.

    Receives ``TrackingResult`` objects containing ephemeral ByteTrack IDs
    and produces ``IdentityResult`` objects containing persistent global
    Identity UUIDs.  This is the boundary defined by ADR-004.

    Phase 3: When processing results from multiple cameras (via a shared
    fan-in queue), the IML automatically detects cross-camera transitions
    and emits ``CROSS_CAMERA_TRANSITION`` signals.

    All dependencies are injected via the constructor (spec §11 —
    composition over inheritance, no hidden globals).

    Not thread-safe — designed for single-thread access from
    ``IdentityWorker`` (spec §9).

    Args:
        extractor: Embedding extractor (OSNet or any ABC impl).
        gallery: In-memory identity gallery (single-writer).
        lifecycle: Deterministic lifecycle state machine.
        matcher: Stateless Re-ID matcher with threshold + top-k.
        settings: Immutable configuration snapshot.
        camera_id: Camera identifier for logging and transitions.
            In multi-camera mode this may be a default; the actual
            camera_id is read from each TrackingResult's packet.
        camera_graph: Optional camera topology for adjacency validation
            and transition-prior-weighted scoring (Phase 3).
        metrics: Optional Prometheus metrics registry.
    """

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        matcher: Matcher,
        settings: Settings,
        camera_id: str,
        *,
        camera_graph: object | None = None,  # CameraGraph, typed as object to avoid circular import
        metrics: MetricsRegistry | None = None,
    ) -> None:
        # ── Injected dependencies ────────────────────────────────────
        self._extractor = extractor
        self._gallery = gallery
        self._lifecycle = lifecycle
        self._matcher = matcher
        self._camera_id = camera_id
        self._camera_graph = camera_graph
        self._metrics = metrics

        # ── Configuration (extracted once, never changes) ────────────
        self._ema_alpha: float = settings.reid_ema_alpha
        self._gallery_max_size: int = settings.gallery_max_size
        self._extract_interval: int = getattr(settings, "reid_extract_interval", 5)
        self._cross_camera_prior_weight: float = getattr(
            settings, "cross_camera_prior_weight", 0.1,
        )

        # ── Internal state ───────────────────────────────────────────
        #
        # Track-to-identity mapping cache.
        # Key:   (camera_id, track_id) — unique per tracker output.
        # Value: global_id — the persistent identity UUID.
        #
        # Entries are:
        #   Created  — when a track is first resolved (match or register).
        #   Used     — on fast-path EMA update (cache hit).
        #   Removed  — when ByteTrack emits the track as LOST or DEAD
        #              (review C4: prune on state, not absence).
        self._cache: dict[tuple[str, str], str] = {}
        self._last_extracted_frame: dict[str, int] = {}

        # ── Logging ──────────────────────────────────────────────────
        self._log = get_logger("reid.iml", camera_id)

    # ── Public API ───────────────────────────────────────────────────────

    def process(self, result: TrackingResult) -> IdentityResult:
        """Resolve identities for one frame of tracking results.

        This is the single entry point called once per frame by the
        ``IdentityWorker``.  It executes the full IML pipeline:

        1. Filter active tracks.
        2. Batch-extract embeddings.
        3. Resolve identities (fast path + slow path with global
           assignment — review C3).
        4. Prune stale cache entries (review C4).
        5. Tick gallery lifecycle (ACTIVE→LOST→EXPIRED).
        6. Enforce gallery capacity (LRU eviction).
        7. Update Prometheus metrics.

        Args:
            result: Tracking result containing the frame and tracks.

        Returns:
            ``IdentityResult`` with identity mappings and transitions.
        """
        t0 = time.perf_counter()
        timestamp_ns = result.packet.timestamp_ns
        frame_id = result.packet.frame_id
        # Phase 3: use camera_id from the packet (supports multi-camera fan-in)
        current_camera_id = getattr(result.packet, 'camera_id', self._camera_id)
        all_transitions: list[IdentityTransition] = []
        identities: dict[str, Identity] = {}
        cache_hits = 0
        cache_misses = 0
        new_identities = 0
        relinks = 0

        # ── Step 1: Filter active tracks ─────────────────────────────
        active_tracks = self._filter_active_tracks(result.tracks)

        # ── Step 2: Separate cached (within interval) vs tracks needing extraction
        tracks_to_extract: list[Track] = []
        for track in active_tracks:
            cached_id = self._cache_lookup(track)
            if cached_id is not None:
                existing = self._gallery.get(cached_id)
                if existing is not None:
                    last_f = self._last_extracted_frame.get(track.track_id)
                    if last_f is not None and (frame_id - last_f) < self._extract_interval:
                        # Fast path hit without extraction: refresh timestamp
                        refreshed = self._lifecycle.refresh(existing, timestamp_ns)
                        self._gallery.update(refreshed)
                        identities[track.track_id] = refreshed
                        cache_hits += 1
                        continue

            tracks_to_extract.append(track)

        if tracks_to_extract:
            try:
                bboxes = [t.bbox for t in tracks_to_extract]
                embeddings = self._extractor.extract(
                    result.packet.frame, bboxes,
                )
                for t in tracks_to_extract:
                    self._last_extracted_frame[t.track_id] = frame_id
            except Exception:
                self._log.error(
                    "extractor_failed",
                    frame_id=frame_id,
                    camera_id=current_camera_id,
                    active_tracks=len(tracks_to_extract),
                    exc_info=True,
                )
                # Rule 6: explicit failure — return empty result,
                # pipeline continues.
                return IdentityResult(tracking=result)
        else:
            embeddings = []

        # ── Step 3: Resolve identities for extracted tracks ──────────
        uncached: list[tuple[Track, np.ndarray]] = []

        for track, embedding in zip(tracks_to_extract, embeddings):
            if self._is_zero_vector(embedding):
                continue

            cached_id = self._cache_lookup(track)
            if cached_id is not None:
                # Fast path with EMA embedding update
                refreshed = self._resolve_cached_track(
                    track, embedding, timestamp_ns, current_camera_id,
                )
                if refreshed is not None:
                    identities[track.track_id] = refreshed
                    cache_hits += 1
                else:
                    # Stale cache — fall through to slow path
                    uncached.append((track, embedding))
                    cache_misses += 1
            else:
                uncached.append((track, embedding))
                cache_misses += 1

        # Slow path: gallery search + conflict resolution (C3)
        if uncached:
            slow_ids, slow_trans = self._resolve_uncached_tracks(
                uncached, timestamp_ns, frame_id, current_camera_id,
            )
            identities.update(slow_ids)
            all_transitions.extend(slow_trans)
            for t in slow_trans:
                if t.transition_type is TransitionType.CONFIRMED_NEW:
                    new_identities += 1
                elif t.transition_type is TransitionType.CONFIRMED_RELINK:
                    relinks += 1

        # ── Step 4: Prune stale cache entries (C4) ───────────────────
        self._prune_cache(result.tracks)

        # ── Step 5: Tick gallery lifecycle ───────────────────────────
        lifecycle_trans = self._tick_lifecycle(timestamp_ns, frame_id)
        all_transitions.extend(lifecycle_trans)

        # ── Step 6: Enforce gallery capacity ────────────────────────
        eviction_trans = self._enforce_capacity()
        all_transitions.extend(eviction_trans)

        # ── Step 7: Update metrics ─────────────────────────────────
        self._update_metrics(cache_hits, cache_misses, new_identities, relinks)

        latency_ms = (time.perf_counter() - t0) * 1000
        if self._metrics is not None:
            self._metrics.iml_process_latency_ms.labels(
                camera_id=self._camera_id,
            ).observe(latency_ms)

        self._log.debug(
            "iml_frame_complete",
            frame_id=frame_id,
            active_tracks=len(active_tracks),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            identities_resolved=len(identities),
            transitions=len(all_transitions),
            latency_ms=round(latency_ms, 2),
        )

        return IdentityResult(
            tracking=result,
            identities=identities,
            transitions=all_transitions,
        )

    @property
    def active_mapping_count(self) -> int:
        """Number of currently cached track → identity associations."""
        return len(self._cache)

    @property
    def gallery_size(self) -> int:
        """Current number of identities in the gallery."""
        return len(self._gallery)

    # ── Private helpers (stubs — implemented in Steps 2–7) ───────────────

    def _filter_active_tracks(
        self, tracks: list[Track]
    ) -> list[Track]:
        """Return only tracks in ACTIVE state.

        LOST and DEAD tracks have no fresh detection to crop from.
        """
        return [t for t in tracks if t.state is TrackState.ACTIVE]

    def _prune_cache(self, tracks: list[Track]) -> None:
        """Remove cache entries for tracks that ByteTrack marked LOST/DEAD.

        Per review C4: prune on *explicit state change*, not on absence
        from the active tracks list.  A track that is simply missing
        from this frame (brief dropout) keeps its cache entry.

        This means a track that was ACTIVE last frame and is completely
        absent this frame will retain its cache entry.  Only when
        ByteTrack explicitly emits the track with state LOST or DEAD
        does the cache entry get removed.
        """
        for track in tracks:
            if track.state in (TrackState.LOST, TrackState.DEAD):
                key = self._cache_key(track)
                removed_id = self._cache_remove(key)
                if removed_id is not None:
                    self._log.debug(
                        "cache_pruned",
                        track_id=track.track_id,
                        global_id=removed_id,
                        reason=track.state.value,
                    )

    # ── Cache access helpers ─────────────────────────────────────────────
    #
    # Encapsulated access to self._cache for clarity and testability.
    # The cache maps (camera_id, track_id) → global_id.

    @staticmethod
    def _cache_key(track: Track) -> tuple[str, str]:
        """Build the deterministic cache key for a track."""
        return (track.camera_id, track.track_id)

    def _cache_lookup(self, track: Track) -> str | None:
        """Return the cached global_id for a track, or None."""
        return self._cache.get(self._cache_key(track))

    def _cache_set(self, track: Track, global_id: str) -> None:
        """Associate a track with a global identity in the cache."""
        self._cache[self._cache_key(track)] = global_id

    def _cache_remove(self, key: tuple[str, str]) -> str | None:
        """Remove and return a cache entry, or None if absent."""
        self._last_extracted_frame.pop(key[1], None)
        return self._cache.pop(key, None)

    def _cached_global_ids(self) -> frozenset[str]:
        """Return the set of global_ids currently in the cache.

        Used by the eviction policy (review R1) to determine which
        ACTIVE identities are currently being observed and must not
        be evicted.
        """
        return frozenset(self._cache.values())

    def _resolve_cached_track(
        self,
        track: Track,
        embedding: np.ndarray,
        timestamp_ns: int,
        camera_id: str = "",
    ) -> Identity | None:
        """Fast path: update an already-cached track→identity association.

        Computes the EMA-blended embedding and refreshes the identity
        via the lifecycle manager.  Phase 3: also updates camera tracking
        fields if the camera has changed.

        Returns:
            The refreshed ``Identity``, or ``None`` if the cached
            identity is no longer in the gallery (defensive guard).
        """
        global_id = self._cache_lookup(track)
        if global_id is None:
            return None

        existing = self._gallery.get(global_id)
        if existing is None:
            # Identity was evicted or expired between frames.
            # Remove stale cache entry.
            self._cache_remove(self._cache_key(track))
            self._log.warning(
                "cache_stale_miss",
                track_id=track.track_id,
                global_id=global_id,
            )
            return None

        ema_emb = self._compute_ema_embedding(existing.embedding, embedding)

        # Phase 3: update camera fields if camera changed
        extra_updates: dict[str, object] = {}
        if camera_id and existing.last_camera_id != camera_id:
            history = existing.camera_history
            if not history or history[-1] != camera_id:
                history = history + (camera_id,)
            extra_updates["last_camera_id"] = camera_id
            extra_updates["camera_history"] = history

        refreshed = self._lifecycle.refresh(
            existing, timestamp_ns, embedding=ema_emb,
        )
        if extra_updates:
            refreshed = refreshed.with_updates(**extra_updates)
        self._gallery.update(refreshed)
        return refreshed

    def _resolve_uncached_tracks(
        self,
        uncached: list[tuple[Track, np.ndarray]],
        timestamp_ns: int,
        frame_id: int,
        camera_id: str = "",
    ) -> tuple[dict[str, Identity], list[IdentityTransition]]:
        """Slow path: gallery search + score-sorted global assignment.

        Per review C3: all cache-miss tracks are matched in a single
        batch, then conflicts are resolved deterministically by score
        (highest score wins).  No processing-order dependence.

        Algorithm:
            1. Query the gallery for each uncached track's best match.
            2. Collect all (track, global_id, score) candidates.
            3. Sort candidates by score descending.
            4. Assign in score order — first claim wins.
            5. Rejected matches and unmatched tracks → register new.

        Args:
            uncached: List of (track, embedding) pairs with no cache hit.
            timestamp_ns: Current frame timestamp.
            frame_id: Current frame number.

        Returns:
            Tuple of (track_id → Identity mapping, transitions list).
        """
        if not uncached:
            return {}, []

        identities: dict[str, Identity] = {}
        transitions: list[IdentityTransition] = []

        # ── Phase 1: Query gallery for all uncached tracks ───────────
        gallery_matrix, gallery_ids = self._gallery.get_embedding_matrix()

        # Candidate matches: (track_index, global_id, score)
        candidates: list[tuple[int, str, float]] = []
        # Track indices with no gallery match at all
        unmatched_indices: list[int] = []

        for i, (track, embedding) in enumerate(uncached):
            if gallery_matrix.shape[0] == 0:
                # Empty gallery — all tracks are new
                unmatched_indices.append(i)
                continue

            result: MatchResult = self._matcher.match(
                embedding, gallery_matrix, gallery_ids,
            )
            if result.matched:
                candidates.append((i, gallery_ids[result.best_index], result.best_score))
            else:
                unmatched_indices.append(i)

        # ── Phase 2: Score-sorted assignment (C3) ────────────────────
        # Sort by score descending — highest confidence wins.
        candidates.sort(key=lambda c: c[2], reverse=True)

        claimed_global_ids: set[str] = set()

        for track_idx, matched_gid, score in candidates:
            track, embedding = uncached[track_idx]

            if matched_gid in claimed_global_ids:
                # Conflict: another track already claimed this identity
                # with a higher score.  This track becomes a new identity.
                self._log.debug(
                    "conflict_rejected",
                    track_id=track.track_id,
                    contested_gid=matched_gid,
                    score=round(score, 4),
                )
                unmatched_indices.append(track_idx)
                continue

            # Claim the identity — re-link
            claimed_global_ids.add(matched_gid)
            existing = self._gallery.get(matched_gid)
            if existing is None:
                # Defensive: identity disappeared between matrix build
                # and lookup (should not happen in single-thread).
                self._log.warning(
                    "gallery_miss_after_match",
                    global_id=matched_gid,
                    track_id=track.track_id,
                )
                unmatched_indices.append(track_idx)
                continue

            previous_state = existing.state
            ema_emb = self._compute_ema_embedding(existing.embedding, embedding)
            refreshed = self._lifecycle.refresh(
                existing, timestamp_ns,
                embedding=ema_emb, confidence=score,
            )

            # Phase 3: detect cross-camera transition
            is_cross_camera = (
                camera_id
                and existing.last_camera_id
                and existing.last_camera_id != camera_id
            )
            if is_cross_camera:
                history = refreshed.camera_history
                if not history or history[-1] != camera_id:
                    history = history + (camera_id,)
                refreshed = refreshed.with_updates(
                    last_camera_id=camera_id,
                    camera_history=history,
                )
            elif camera_id and not refreshed.last_camera_id:
                refreshed = refreshed.with_updates(
                    last_camera_id=camera_id,
                    camera_history=(camera_id,),
                )

            self._gallery.update(refreshed)
            self._cache_set(track, refreshed.global_id)
            identities[track.track_id] = refreshed

            transitions.append(IdentityTransition(
                transition_type=TransitionType.CONFIRMED_RELINK,
                global_id=refreshed.global_id,
                camera_id=camera_id or self._camera_id,
                timestamp_ns=timestamp_ns,
                frame_id=frame_id,
                confidence=score,
                track_id=track.track_id,
                previous_state=previous_state,
            ))

            # Phase 3: emit cross-camera transition event
            if is_cross_camera:
                transitions.append(IdentityTransition(
                    transition_type=TransitionType.CROSS_CAMERA_TRANSITION,
                    global_id=refreshed.global_id,
                    camera_id=camera_id,
                    timestamp_ns=timestamp_ns,
                    frame_id=frame_id,
                    confidence=score,
                    track_id=track.track_id,
                    previous_state=previous_state,
                ))
                self._log.info(
                    "cross_camera_transition",
                    global_id=refreshed.global_id,
                    from_camera=existing.last_camera_id,
                    to_camera=camera_id,
                    score=round(score, 4),
                )

            self._log.info(
                "identity_relinked",
                global_id=refreshed.global_id,
                track_id=track.track_id,
                score=round(score, 4),
                previous_state=previous_state.value,
                cross_camera=is_cross_camera,
            )

        # ── Phase 3: Register new identities for unmatched tracks ────
        for track_idx in unmatched_indices:
            track, embedding = uncached[track_idx]
            identity = self._register_new_identity(
                track, embedding, timestamp_ns, frame_id,
                camera_id=camera_id,
            )
            identities[track.track_id] = identity
            transitions.append(IdentityTransition(
                transition_type=TransitionType.CONFIRMED_NEW,
                global_id=identity.global_id,
                camera_id=camera_id or self._camera_id,
                timestamp_ns=timestamp_ns,
                frame_id=frame_id,
                confidence=0.0,
                track_id=track.track_id,
                previous_state=None,
            ))

        return identities, transitions

    def _register_new_identity(
        self,
        track: Track,
        embedding: np.ndarray,
        timestamp_ns: int,
        frame_id: int,
        *,
        camera_id: str = "",
    ) -> Identity:
        """Create, activate, insert, and cache a brand-new identity.

        Args:
            track: The track that triggered identity creation.
            embedding: The track's L2-normalized embedding.
            timestamp_ns: Observation time.
            frame_id: Frame number (for logging only).
            camera_id: Camera where the identity was first observed.

        Returns:
            The activated identity (ACTIVE state).
        """
        new_id = create_identity(
            embedding, timestamp_ns,
            camera_id=camera_id or self._camera_id,
        )
        activated = self._lifecycle.activate(new_id, timestamp_ns)
        self._gallery.insert(activated)
        self._cache_set(track, activated.global_id)

        self._log.info(
            "identity_registered",
            global_id=activated.global_id,
            track_id=track.track_id,
            camera_id=camera_id or self._camera_id,
            frame_id=frame_id,
        )
        return activated

    def _compute_ema_embedding(
        self,
        old_embedding: np.ndarray,
        new_observation: np.ndarray,
    ) -> np.ndarray:
        """Blend a new observation into an existing embedding via EMA.

        Formula (ADR-003):
            blended = α × old + (1 - α) × new
            result  = blended / ||blended||₂

        Per review R6: if the old embedding is a zero-vector (should
        never happen post-creation, but defensive), return the new
        observation directly without blending.

        Args:
            old_embedding: Current identity embedding (512-dim, L2-norm).
            new_observation: Fresh extraction (512-dim, L2-norm).

        Returns:
            L2-normalized blended embedding.
        """
        # R6: If old is zero-vector, use raw observation (first update)
        if self._is_zero_vector(old_embedding):
            norm = float(np.linalg.norm(new_observation))
            if norm < 1e-6:
                return new_observation  # both zero — nothing to do
            return (new_observation / norm).astype(np.float32)

        blended = self._ema_alpha * old_embedding + (1.0 - self._ema_alpha) * new_observation
        norm = float(np.linalg.norm(blended))
        if norm < 1e-6:
            # Degenerate case: blending produced zero.  Keep old.
            return old_embedding
        return (blended / norm).astype(np.float32)

    def _tick_lifecycle(
        self, timestamp_ns: int, frame_id: int
    ) -> list[IdentityTransition]:
        """Advance time-driven lifecycle transitions for all gallery entries.

        Calls ``lifecycle.advance()`` on each identity.  Collects
        ACTIVE→LOST and LOST→EXPIRED transitions.  Removes expired
        identities from the gallery.

        Returns:
            List of transitions that occurred during this tick.
        """
        transitions: list[IdentityTransition] = []
        expired_ids: list[str] = []

        for identity in self._gallery.identities():
            previous_state = identity.state
            advanced = self._lifecycle.advance(identity, timestamp_ns)

            if advanced.state is previous_state:
                # No transition occurred.
                continue

            # A transition happened — update gallery and record it.
            self._gallery.update(advanced)

            if advanced.state is LifecycleState.LOST:
                transitions.append(IdentityTransition(
                    transition_type=TransitionType.LOST,
                    global_id=advanced.global_id,
                    camera_id=self._camera_id,
                    timestamp_ns=timestamp_ns,
                    frame_id=frame_id,
                    confidence=advanced.confidence,
                ))
            elif advanced.state is LifecycleState.EXPIRED:
                expired_ids.append(advanced.global_id)
                transitions.append(IdentityTransition(
                    transition_type=TransitionType.EXPIRED,
                    global_id=advanced.global_id,
                    camera_id=self._camera_id,
                    timestamp_ns=timestamp_ns,
                    frame_id=frame_id,
                    confidence=advanced.confidence,
                ))

        # Remove expired identities from gallery after iteration.
        for gid in expired_ids:
            self._gallery.remove(gid)

        return transitions

    def _enforce_capacity(self) -> list[IdentityTransition]:
        """Enforce gallery size limit via tiered LRU eviction.

        Tiers (review R1):
            1. Evict LOST identities, oldest ``last_seen_ns`` first.
            2. If no LOST remain, evict ACTIVE identities not in the
               track cache, oldest ``last_seen_ns`` first.
            3. If even that fails, log CRITICAL and stop evicting.

        Returns:
            List of transitions for evicted identities.
        """
        if len(self._gallery) <= self._gallery_max_size:
            return []

        transitions: list[IdentityTransition] = []
        protected_ids = self._cached_global_ids()

        while len(self._gallery) > self._gallery_max_size:
            # Tier 1: Evict oldest LOST identity
            victim = self._find_eviction_victim(
                LifecycleState.LOST, protected_ids,
            )
            tier = "lost"

            if victim is None:
                # Tier 2: Evict oldest ACTIVE not in cache
                victim = self._find_eviction_victim(
                    LifecycleState.ACTIVE, protected_ids,
                )
                tier = "stale_active"

            if victim is None:
                # Tier 3: Cannot evict — all identities are actively
                # observed.  This should not happen at N=10K.
                self._log.critical(
                    "gallery_capacity_exhausted",
                    gallery_size=len(self._gallery),
                    max_size=self._gallery_max_size,
                    cached_ids=len(protected_ids),
                )
                break

            self._gallery.remove(victim.global_id)
            if self._metrics is not None:
                self._metrics.iml_evictions_total.labels(
                    camera_id=self._camera_id, tier=tier,
                ).inc()
            self._log.warning(
                "gallery_eviction",
                global_id=victim.global_id,
                tier=tier,
                last_seen_ns=victim.last_seen_ns,
                state=victim.state.value,
            )
            transitions.append(IdentityTransition(
                transition_type=TransitionType.EXPIRED,
                global_id=victim.global_id,
                camera_id=self._camera_id,
                timestamp_ns=victim.last_seen_ns,
                frame_id=0,  # eviction is not tied to a specific frame
                confidence=victim.confidence,
            ))

        return transitions

    def _find_eviction_victim(
        self,
        target_state: LifecycleState,
        protected_ids: frozenset[str],
    ) -> Identity | None:
        """Find the oldest identity in ``target_state`` not in the protected set.

        Args:
            target_state: Only consider identities in this state.
            protected_ids: Set of global_ids that must not be evicted
                (currently in the track cache).

        Returns:
            The identity with the smallest ``last_seen_ns`` matching
            the criteria, or ``None`` if no candidate exists.
        """
        victim: Identity | None = None
        for identity in self._gallery:
            if identity.state is not target_state:
                continue
            if identity.global_id in protected_ids:
                continue
            if victim is None or identity.last_seen_ns < victim.last_seen_ns:
                victim = identity
        return victim

    def _update_metrics(
        self,
        cache_hits: int,
        cache_misses: int,
        new_identities: int,
        relinks: int,
    ) -> None:
        """Update Prometheus metrics for this frame (review R3).

        Records: gallery size, active/lost counts, cache hit/miss
        counters, new identity and re-link counters.

        All updates are guarded by ``self._metrics is not None`` so
        the IML works correctly without a metrics registry (tests).
        """
        if self._metrics is None:
            return

        m = self._metrics
        cam = self._camera_id

        # Gauges — current snapshot
        m.gallery_size.set(len(self._gallery))
        m.identities_active.set(self._gallery.active_count)
        m.identities_lost.set(self._gallery.lost_count)

        # Counters — incremental
        if cache_hits > 0:
            m.iml_cache_hits_total.labels(camera_id=cam).inc(cache_hits)
        if cache_misses > 0:
            m.iml_cache_misses_total.labels(camera_id=cam).inc(cache_misses)
        if new_identities > 0:
            m.iml_new_identities_total.labels(camera_id=cam).inc(new_identities)
        if relinks > 0:
            m.iml_relinks_total.labels(camera_id=cam).inc(relinks)

    @staticmethod
    def _is_zero_vector(embedding: np.ndarray, epsilon: float = 1e-6) -> bool:
        """True if the embedding has near-zero L2 norm (failed crop)."""
        return float(np.linalg.norm(embedding)) < epsilon
