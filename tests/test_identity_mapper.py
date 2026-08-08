"""Tests for the Identity Mapper Layer (IML) — Phase 2, Task 7.

Covers:
    - Data types: IdentityResult, IdentityTransition, TransitionType
    - Track cache: filter, prune (C4), lookup, set, remove
    - EMA embedding computation (ADR-003, R6 zero-vector guard)
    - Fast-path resolution (cache hit → refresh)
    - Slow-path resolution + deterministic conflict resolution (C3)
    - Lifecycle tick (ACTIVE→LOST→EXPIRED)
    - Gallery capacity enforcement / tiered eviction (R1)
    - Full process() integration (end-to-end per-frame pipeline)
"""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.pipeline import TrackingResult
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.identity import (
    Identity,
    LifecycleState,
    create_identity,
)
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.identity_mapper import (
    IdentityMapper,
    IdentityResult,
    IdentityTransition,
    TransitionType,
)
from gods_eye.reid.matcher import Matcher
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import Track, TrackState

_NS_PER_S: int = 1_000_000_000


# ── Helpers ──────────────────────────────────────────────────────────────────


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    """Deterministic unit-norm vector from a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _perturbed(base: np.ndarray, noise: float = 0.01, seed: int = 99) -> np.ndarray:
    """Return a slightly perturbed copy of ``base`` (cosine ≈ 1.0)."""
    rng = np.random.RandomState(seed)
    v = base + rng.randn(*base.shape).astype(np.float32) * noise
    return v / np.linalg.norm(v)


def _bbox() -> BoundingBox:
    return BoundingBox(x1=10, y1=20, x2=110, y2=220)


def _track(track_id: str, state: TrackState = TrackState.ACTIVE) -> Track:
    return Track(
        track_id=track_id,
        camera_id="cam0",
        state=state,
        bbox=_bbox(),
        velocity=(0.0, 0.0),
        first_frame_id=1,
        last_frame_id=5,
        lost_frame_count=0 if state is TrackState.ACTIVE else 10,
    )


def _detection() -> Detection:
    return Detection(
        detection_id="det-1",
        camera_id="cam0",
        frame_id=1,
        timestamp_ns=1000,
        bbox=_bbox(),
        confidence=0.9,
        class_label="person",
        source_resolution=(640, 480),
    )


def _packet(frame_id: int = 1, ts: int = 1000) -> FramePacket:
    return FramePacket(
        camera_id="cam0",
        frame_id=frame_id,
        timestamp_ns=ts,
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        resolution=(640, 480),
    )


class MockExtractor(EmbeddingExtractor):
    """Test double — returns pre-configured embeddings by bbox index."""

    def __init__(self, embeddings: list[np.ndarray] | None = None) -> None:
        self._embeddings = embeddings or []
        self._call_count = 0

    def extract(
        self, frame: np.ndarray, bboxes: list[BoundingBox]
    ) -> list[np.ndarray]:
        self._call_count += 1
        if self._embeddings:
            return self._embeddings[: len(bboxes)]
        return [_unit_vec(i + self._call_count * 1000) for i in range(len(bboxes))]

    def warmup(self) -> None:
        pass

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "mock"


class FailingExtractor(MockExtractor):
    """Extractor that always raises."""

    def extract(
        self, frame: np.ndarray, bboxes: list[BoundingBox]
    ) -> list[np.ndarray]:
        raise RuntimeError("GPU OOM")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def gallery() -> IdentityGallery:
    return IdentityGallery()


@pytest.fixture()
def lifecycle(settings: Settings) -> LifecycleManager:
    return LifecycleManager(settings)


@pytest.fixture()
def matcher(settings: Settings) -> Matcher:
    return Matcher(settings)


@pytest.fixture()
def mapper(
    gallery: IdentityGallery,
    lifecycle: LifecycleManager,
    matcher: Matcher,
    settings: Settings,
) -> IdentityMapper:
    return IdentityMapper(
        extractor=MockExtractor(),
        gallery=gallery,
        lifecycle=lifecycle,
        matcher=matcher,
        settings=settings,
        camera_id="cam0",
    )


# ── Data Types ───────────────────────────────────────────────────────────────


class TestDataTypes:
    def test_transition_type_values(self) -> None:
        assert TransitionType.CONFIRMED_NEW.value == "confirmed_new"
        assert TransitionType.CONFIRMED_RELINK.value == "confirmed_relink"
        assert TransitionType.LOST.value == "lost"
        assert TransitionType.EXPIRED.value == "expired"

    def test_identity_transition_frozen(self) -> None:
        t = IdentityTransition(
            transition_type=TransitionType.CONFIRMED_NEW,
            global_id="abc",
            camera_id="cam0",
            timestamp_ns=1000,
            frame_id=1,
            confidence=0.0,
        )
        with pytest.raises(AttributeError):
            t.global_id = "changed"  # type: ignore[misc]

    def test_identity_result_composes_tracking(self) -> None:
        packet = _packet()
        tr = TrackingResult(packet=packet, detections=[], tracks=[])
        ir = IdentityResult(tracking=tr)
        assert ir.tracking is tr
        assert ir.identities == {}
        assert ir.transitions == []


# ── Track Cache ──────────────────────────────────────────────────────────────


class TestTrackCache:
    def test_filter_active_tracks(self, mapper: IdentityMapper) -> None:
        active = _track("A", TrackState.ACTIVE)
        lost = _track("L", TrackState.LOST)
        dead = _track("D", TrackState.DEAD)
        result = mapper._filter_active_tracks([active, lost, dead])
        assert len(result) == 1
        assert result[0].track_id == "A"

    def test_filter_empty_list(self, mapper: IdentityMapper) -> None:
        assert mapper._filter_active_tracks([]) == []

    def test_cache_key_deterministic(self) -> None:
        t = _track("T1")
        assert IdentityMapper._cache_key(t) == ("cam0", "T1")

    def test_cache_set_and_lookup(self, mapper: IdentityMapper) -> None:
        t = _track("T1")
        assert mapper._cache_lookup(t) is None
        mapper._cache_set(t, "gid-1")
        assert mapper._cache_lookup(t) == "gid-1"

    def test_prune_on_lost_state(self, mapper: IdentityMapper) -> None:
        """C4: prune when ByteTrack marks track LOST."""
        t_active = _track("T1", TrackState.ACTIVE)
        mapper._cache_set(t_active, "gid-1")

        t_lost = _track("T1", TrackState.LOST)
        mapper._prune_cache([t_lost])
        assert mapper._cache_lookup(t_active) is None

    def test_prune_on_dead_state(self, mapper: IdentityMapper) -> None:
        """C4: prune when ByteTrack marks track DEAD."""
        t = _track("T1", TrackState.ACTIVE)
        mapper._cache_set(t, "gid-1")

        t_dead = _track("T1", TrackState.DEAD)
        mapper._prune_cache([t_dead])
        assert mapper._cache_lookup(t) is None

    def test_no_prune_on_active_state(self, mapper: IdentityMapper) -> None:
        """C4: ACTIVE tracks keep their cache entries."""
        t = _track("T1", TrackState.ACTIVE)
        mapper._cache_set(t, "gid-1")
        mapper._prune_cache([t])
        assert mapper._cache_lookup(t) == "gid-1"

    def test_cached_global_ids(self, mapper: IdentityMapper) -> None:
        mapper._cache_set(_track("T1"), "gid-1")
        mapper._cache_set(_track("T2"), "gid-2")
        assert mapper._cached_global_ids() == frozenset({"gid-1", "gid-2"})


# ── EMA Computation ─────────────────────────────────────────────────────────


class TestEMA:
    def test_ema_blending_favors_old(self, mapper: IdentityMapper) -> None:
        """With α=0.9, result should be closer to old than new."""
        old = _unit_vec(1)
        new = _unit_vec(2)
        blended = mapper._compute_ema_embedding(old, new)
        sim_old = float(np.dot(blended, old))
        sim_new = float(np.dot(blended, new))
        assert sim_old > sim_new

    def test_ema_result_is_l2_normalized(self, mapper: IdentityMapper) -> None:
        blended = mapper._compute_ema_embedding(_unit_vec(1), _unit_vec(2))
        assert abs(float(np.linalg.norm(blended)) - 1.0) < 1e-5

    def test_ema_zero_old_returns_new(self, mapper: IdentityMapper) -> None:
        """R6: zero-vector old → return new directly."""
        zero = np.zeros(512, dtype=np.float32)
        new = _unit_vec(1)
        result = mapper._compute_ema_embedding(zero, new)
        assert np.allclose(result, new, atol=1e-5)

    def test_ema_identical_inputs_unchanged(self, mapper: IdentityMapper) -> None:
        v = _unit_vec(42)
        result = mapper._compute_ema_embedding(v, v)
        assert np.allclose(result, v, atol=1e-5)


# ── Fast Path (Cache Hit) ───────────────────────────────────────────────────


class TestFastPath:
    def test_resolve_cached_track(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
    ) -> None:
        emb = _unit_vec(10)
        identity = create_identity(emb, 1000, global_id="fp-1")
        active = lifecycle.activate(identity, 1000)
        gallery.insert(active)
        mapper._cache_set(_track("T1"), "fp-1")

        new_emb = _unit_vec(11)
        refreshed = mapper._resolve_cached_track(_track("T1"), new_emb, 2000)
        assert refreshed is not None
        assert refreshed.global_id == "fp-1"
        assert refreshed.observation_count == 2

    def test_stale_cache_returns_none(
        self, mapper: IdentityMapper, gallery: IdentityGallery
    ) -> None:
        """Cached identity removed from gallery → returns None, cleans cache."""
        mapper._cache_set(_track("T1"), "gone-id")
        result = mapper._resolve_cached_track(_track("T1"), _unit_vec(1), 2000)
        assert result is None
        assert mapper._cache_lookup(_track("T1")) is None


# ── Slow Path + Conflict Resolution ─────────────────────────────────────────


class TestSlowPath:
    def test_empty_gallery_registers_new(
        self, mapper: IdentityMapper
    ) -> None:
        emb = _unit_vec(20)
        track = _track("T1")
        ids, trans = mapper._resolve_uncached_tracks(
            [(track, emb)], 1000, 1,
        )
        assert len(ids) == 1
        assert ids["T1"].state is LifecycleState.ACTIVE
        assert trans[0].transition_type is TransitionType.CONFIRMED_NEW

    def test_relink_to_lost_identity(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
    ) -> None:
        base = _unit_vec(30)
        ident = create_identity(base, 500, global_id="lost-1")
        active = lifecycle.activate(ident, 500)
        lost = lifecycle.mark_lost(active, 600)
        gallery.insert(lost)

        similar = _perturbed(base, noise=0.01)
        ids, trans = mapper._resolve_uncached_tracks(
            [(_track("T1"), similar)], 2000, 10,
        )
        assert ids["T1"].global_id == "lost-1"
        assert trans[0].transition_type is TransitionType.CONFIRMED_RELINK
        assert trans[0].previous_state is LifecycleState.LOST

    def test_conflict_resolution_deterministic(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
    ) -> None:
        """C3: two tracks match same identity — higher score wins."""
        base = _unit_vec(40)
        ident = create_identity(base, 500, global_id="contested")
        active = lifecycle.activate(ident, 500)
        lost = lifecycle.mark_lost(active, 600)
        gallery.insert(lost)

        close = _perturbed(base, noise=0.005, seed=1)   # higher score
        far = _perturbed(base, noise=0.05, seed=2)      # lower score

        ids, trans = mapper._resolve_uncached_tracks(
            [(_track("W"), close), (_track("L"), far)], 3000, 20,
        )
        assert len(ids) == 2
        relinks = [t for t in trans if t.transition_type is TransitionType.CONFIRMED_RELINK]
        news = [t for t in trans if t.transition_type is TransitionType.CONFIRMED_NEW]
        assert len(relinks) == 1
        assert len(news) == 1
        assert relinks[0].global_id == "contested"

    def test_empty_uncached_list(self, mapper: IdentityMapper) -> None:
        ids, trans = mapper._resolve_uncached_tracks([], 1000, 1)
        assert ids == {}
        assert trans == []


# ── Lifecycle Tick ───────────────────────────────────────────────────────────


class TestLifecycleTick:
    def test_active_to_lost(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        settings: Settings,
    ) -> None:
        emb = _unit_vec(50)
        ident = create_identity(emb, 1000, global_id="tick-1")
        active = lifecycle.activate(ident, 1000)
        gallery.insert(active)

        future = 1000 + settings.identity_lost_timeout_s * _NS_PER_S
        trans = mapper._tick_lifecycle(future, 999)
        assert len(trans) == 1
        assert trans[0].transition_type is TransitionType.LOST
        assert gallery.get("tick-1") is not None
        assert gallery.get("tick-1").state is LifecycleState.LOST  # type: ignore[union-attr]

    def test_lost_to_expired_removed(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        settings: Settings,
    ) -> None:
        emb = _unit_vec(51)
        ident = create_identity(emb, 1000, global_id="tick-2")
        active = lifecycle.activate(ident, 1000)
        lost = lifecycle.mark_lost(active, 2000)
        gallery.insert(lost)

        future = 2000 + settings.identity_ttl_s * _NS_PER_S
        trans = mapper._tick_lifecycle(future, 9999)
        assert len(trans) == 1
        assert trans[0].transition_type is TransitionType.EXPIRED
        assert gallery.get("tick-2") is None  # removed


# ── Eviction Policy ─────────────────────────────────────────────────────────


class TestEviction:
    def test_no_eviction_under_capacity(self, mapper: IdentityMapper) -> None:
        assert mapper._enforce_capacity() == []

    def test_tier1_evicts_oldest_lost(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
    ) -> None:
        mapper._gallery_max_size = 2
        for i in range(3):
            emb = _unit_vec(60 + i)
            ident = create_identity(emb, 1000 + i, global_id=f"ev-{i}")
            active = lifecycle.activate(ident, 1000 + i)
            if i == 0:
                lost = lifecycle.mark_lost(active, 2000)
                gallery.insert(lost)
            else:
                gallery.insert(active)

        trans = mapper._enforce_capacity()
        assert len(gallery) == 2
        assert len(trans) == 1
        assert trans[0].global_id == "ev-0"  # oldest LOST evicted

    def test_tier2_evicts_stale_active(
        self,
        mapper: IdentityMapper,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
    ) -> None:
        mapper._gallery_max_size = 2
        for i in range(3):
            emb = _unit_vec(70 + i)
            ident = create_identity(emb, 1000 + i * 100, global_id=f"act-{i}")
            active = lifecycle.activate(ident, 1000 + i * 100)
            gallery.insert(active)

        # Protect act-2 via cache
        mapper._cache_set(_track("T0"), "act-2")

        trans = mapper._enforce_capacity()
        assert len(gallery) == 2
        assert gallery.get("act-2") is not None  # protected
        assert trans[0].global_id == "act-0"  # oldest unprotected


# ── Full process() Integration ───────────────────────────────────────────────


class TestProcess:
    def test_first_frame_registers_new_identities(
        self,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        matcher: Matcher,
        settings: Settings,
    ) -> None:
        mapper = IdentityMapper(
            extractor=MockExtractor(),
            gallery=gallery,
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="cam0",
        )
        tr = TrackingResult(
            packet=_packet(1, 1000),
            detections=[_detection()],
            tracks=[_track("T1")],
        )
        result = mapper.process(tr)
        assert isinstance(result, IdentityResult)
        assert result.tracking is tr
        assert len(result.identities) == 1
        assert "T1" in result.identities
        assert result.identities["T1"].state is LifecycleState.ACTIVE
        assert len(result.transitions) == 1
        assert result.transitions[0].transition_type is TransitionType.CONFIRMED_NEW

    def test_second_frame_cache_hit(
        self,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        matcher: Matcher,
        settings: Settings,
    ) -> None:
        mapper = IdentityMapper(
            extractor=MockExtractor(),
            gallery=gallery,
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="cam0",
        )
        track = _track("T1")
        tr1 = TrackingResult(
            packet=_packet(1, 1000),
            detections=[_detection()],
            tracks=[track],
        )
        r1 = mapper.process(tr1)
        gid = r1.identities["T1"].global_id

        tr2 = TrackingResult(
            packet=_packet(2, 2000),
            detections=[_detection()],
            tracks=[track],
        )
        r2 = mapper.process(tr2)
        assert r2.identities["T1"].global_id == gid
        assert r2.identities["T1"].observation_count == 2
        assert len(r2.transitions) == 0

    def test_empty_tracks(
        self,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        matcher: Matcher,
        settings: Settings,
    ) -> None:
        mapper = IdentityMapper(
            extractor=MockExtractor(),
            gallery=gallery,
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="cam0",
        )
        tr = TrackingResult(
            packet=_packet(1, 1000),
            detections=[],
            tracks=[],
        )
        result = mapper.process(tr)
        assert len(result.identities) == 0
        assert len(result.transitions) == 0

    def test_extractor_failure_graceful(
        self,
        lifecycle: LifecycleManager,
        matcher: Matcher,
        settings: Settings,
    ) -> None:
        mapper = IdentityMapper(
            extractor=FailingExtractor(),
            gallery=IdentityGallery(),
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="cam0",
        )
        tr = TrackingResult(
            packet=_packet(1, 1000),
            detections=[_detection()],
            tracks=[_track("T1")],
        )
        result = mapper.process(tr)
        assert isinstance(result, IdentityResult)
        assert len(result.identities) == 0
        assert result.tracking is tr

    def test_lost_track_pruned_from_cache(
        self,
        gallery: IdentityGallery,
        lifecycle: LifecycleManager,
        matcher: Matcher,
        settings: Settings,
    ) -> None:
        mapper = IdentityMapper(
            extractor=MockExtractor(),
            gallery=gallery,
            lifecycle=lifecycle,
            matcher=matcher,
            settings=settings,
            camera_id="cam0",
        )
        track = _track("T1")
        tr1 = TrackingResult(
            packet=_packet(1, 1000),
            detections=[_detection()],
            tracks=[track],
        )
        mapper.process(tr1)
        assert mapper._cache_lookup(track) is not None

        lost_track = _track("T1", TrackState.LOST)
        tr2 = TrackingResult(
            packet=_packet(2, 2000),
            detections=[],
            tracks=[lost_track],
        )
        mapper.process(tr2)
        assert mapper._cache_lookup(track) is None
