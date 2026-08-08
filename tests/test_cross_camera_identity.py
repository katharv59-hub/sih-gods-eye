"""Tests for cross-camera identity linking — Phase 3, Component 4.

Tests that the IML correctly:
- Detects cross-camera transitions
- Updates camera_history and last_camera_id
- Emits CROSS_CAMERA_TRANSITION transitions
- Passes camera_id to create_identity
"""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.reid.identity import Identity, LifecycleState, create_identity
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.identity_mapper import (
    IdentityMapper,
    IdentityResult,
    IdentityTransition,
    TransitionType,
)
from gods_eye.reid.matcher import Matcher
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.track import Track, TrackState


# ── Test Helpers ──────────────────────────────────────────────────────────


class StubExtractor:
    """Returns pre-set embeddings for each extraction call."""

    def __init__(self, embeddings: list[np.ndarray] | None = None) -> None:
        self._embeddings = embeddings or []
        self._call_count = 0

    def extract(
        self, frame: np.ndarray, bboxes: list[BoundingBox],
    ) -> list[np.ndarray]:
        if self._embeddings:
            result = self._embeddings[
                self._call_count : self._call_count + len(bboxes)
            ]
            self._call_count += len(bboxes)
            return result
        # Default: random L2-normalized embeddings
        out = []
        for _ in bboxes:
            v = np.random.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            out.append(v)
        return out

    @property
    def embedding_dim(self) -> int:
        return 512


def _make_embedding(seed: int = 0) -> np.ndarray:
    """Create a deterministic L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_track(
    track_id: str = "t1",
    camera_id: str = "cam_a",
    state: TrackState = TrackState.ACTIVE,
) -> Track:
    return Track(
        track_id=track_id,
        camera_id=camera_id,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        state=state,
        velocity=(0.0, 0.0),
        first_frame_id=0,
        last_frame_id=0,
        lost_frame_count=0,
    )


class FakeTrackingResult:
    """Minimal TrackingResult-like object for IML tests."""

    def __init__(
        self,
        tracks: list[Track],
        camera_id: str = "cam_a",
        frame_id: int = 1,
        timestamp_ns: int = 1_000_000_000,
    ) -> None:
        self.tracks = tracks
        self.packet = type("P", (), {
            "frame": np.zeros((480, 640, 3), dtype=np.uint8),
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
            "camera_id": camera_id,
        })()


# ── Identity Model Tests ─────────────────────────────────────────────────


class TestIdentityCameraFields:

    def test_create_identity_with_camera_id(self) -> None:
        emb = _make_embedding()
        identity = create_identity(emb, 1_000_000_000, camera_id="cam_a")
        assert identity.last_camera_id == "cam_a"
        assert identity.camera_history == ("cam_a",)

    def test_create_identity_without_camera_id(self) -> None:
        """Backward compatibility: no camera_id results in empty fields."""
        emb = _make_embedding()
        identity = create_identity(emb, 1_000_000_000)
        assert identity.last_camera_id == ""
        assert identity.camera_history == ()

    def test_with_updates_camera_fields(self) -> None:
        emb = _make_embedding()
        identity = create_identity(emb, 1_000_000_000, camera_id="cam_a")
        updated = identity.with_updates(
            last_camera_id="cam_b",
            camera_history=("cam_a", "cam_b"),
        )
        assert updated.last_camera_id == "cam_b"
        assert updated.camera_history == ("cam_a", "cam_b")
        # Original unchanged (frozen)
        assert identity.last_camera_id == "cam_a"
        assert identity.camera_history == ("cam_a",)


# ── Transition Type Tests ────────────────────────────────────────────────


class TestTransitionTypes:

    def test_cross_camera_transition_exists(self) -> None:
        assert TransitionType.CROSS_CAMERA_TRANSITION.value == "cross_camera_transition"


# ── Cross-Camera IML Integration Tests ───────────────────────────────────


class TestCrossCameraIdentity:
    """Tests that the IML detects cross-camera transitions."""

    def _make_mapper(
        self,
        embeddings: list[np.ndarray] | None = None,
    ) -> IdentityMapper:
        settings = Settings()
        return IdentityMapper(
            extractor=StubExtractor(embeddings),
            gallery=IdentityGallery(),
            lifecycle=LifecycleManager(settings),
            matcher=Matcher(settings),
            settings=settings,
            camera_id="default",
        )

    def test_new_identity_gets_camera_id(self) -> None:
        """When a new identity is created, it records the source camera."""
        emb = _make_embedding(42)
        mapper = self._make_mapper(embeddings=[emb])

        track = _make_track("t1", camera_id="cam_a")
        result = mapper.process(FakeTrackingResult(
            tracks=[track], camera_id="cam_a",
        ))

        assert len(result.identities) == 1
        identity = list(result.identities.values())[0]
        assert identity.last_camera_id == "cam_a"
        assert "cam_a" in identity.camera_history

    def test_cross_camera_relink_emits_transition(self) -> None:
        """When an identity matches from a different camera,
        CROSS_CAMERA_TRANSITION is emitted."""
        # Use the same embedding so cosine similarity ≈ 1.0
        emb = _make_embedding(42)
        mapper = self._make_mapper(embeddings=[emb, emb])

        # Step 1: Register identity on cam_a
        track_a = _make_track("t1", camera_id="cam_a")
        result1 = mapper.process(FakeTrackingResult(
            tracks=[track_a], camera_id="cam_a",
            frame_id=1, timestamp_ns=1_000_000_000,
        ))
        assert len(result1.identities) == 1
        gid = list(result1.identities.values())[0].global_id

        # Step 2: Track goes LOST on cam_a (prune cache)
        lost_track = _make_track("t1", camera_id="cam_a", state=TrackState.LOST)
        mapper.process(FakeTrackingResult(
            tracks=[lost_track], camera_id="cam_a",
            frame_id=2, timestamp_ns=2_000_000_000,
        ))

        # Step 3: Same person appears on cam_b
        track_b = _make_track("t2", camera_id="cam_b")
        result3 = mapper.process(FakeTrackingResult(
            tracks=[track_b], camera_id="cam_b",
            frame_id=3, timestamp_ns=3_000_000_000,
        ))

        # Should have matched the same identity
        assert len(result3.identities) == 1
        relinked = list(result3.identities.values())[0]
        assert relinked.global_id == gid

        # Should have CROSS_CAMERA_TRANSITION transition
        cross_transitions = [
            t for t in result3.transitions
            if t.transition_type is TransitionType.CROSS_CAMERA_TRANSITION
        ]
        assert len(cross_transitions) == 1
        ct = cross_transitions[0]
        assert ct.global_id == gid
        assert ct.camera_id == "cam_b"

        # camera_history should include both cameras
        assert relinked.last_camera_id == "cam_b"
        assert "cam_a" in relinked.camera_history
        assert "cam_b" in relinked.camera_history

    def test_same_camera_no_cross_transition(self) -> None:
        """When identity relinks on the same camera, no cross-camera event."""
        emb = _make_embedding(42)
        mapper = self._make_mapper(embeddings=[emb, emb])

        # Register on cam_a
        track_a = _make_track("t1", camera_id="cam_a")
        mapper.process(FakeTrackingResult(
            tracks=[track_a], camera_id="cam_a",
            frame_id=1, timestamp_ns=1_000_000_000,
        ))

        # Lose on cam_a
        lost_track = _make_track("t1", camera_id="cam_a", state=TrackState.LOST)
        mapper.process(FakeTrackingResult(
            tracks=[lost_track], camera_id="cam_a",
            frame_id=2, timestamp_ns=2_000_000_000,
        ))

        # Reappear on cam_a (same camera)
        track_a2 = _make_track("t2", camera_id="cam_a")
        result = mapper.process(FakeTrackingResult(
            tracks=[track_a2], camera_id="cam_a",
            frame_id=3, timestamp_ns=3_000_000_000,
        ))

        cross_transitions = [
            t for t in result.transitions
            if t.transition_type is TransitionType.CROSS_CAMERA_TRANSITION
        ]
        assert len(cross_transitions) == 0
