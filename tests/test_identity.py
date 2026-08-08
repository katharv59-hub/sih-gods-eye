"""Tests for the identity persistence layer (Phase 2, Task 4).

Covers:
    - UUID uniqueness
    - Identity creation + immutability
    - Lifecycle transitions (NEW→ACTIVE→LOST→EXPIRED, re-link)
    - Gallery insert / update / remove / enumerate
    - Duplicate protection
    - Store serialization round-trips (in-memory + JSON file)
    - Expiration logic (time-driven advance)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.reid.identity import (
    Identity,
    LifecycleState,
    create_identity,
    new_global_id,
)
from gods_eye.reid.identity_gallery import (
    DuplicateIdentityError,
    IdentityGallery,
    IdentityNotFoundError,
)
from gods_eye.reid.identity_lifecycle import (
    InvalidTransitionError,
    LifecycleManager,
)
from gods_eye.reid.identity_store import (
    InMemoryIdentityStore,
    JsonFileIdentityStore,
    identity_from_dict,
    identity_to_dict,
)

_NS_PER_S: int = 1_000_000_000


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def settings() -> Settings:
    """Default settings: lost_timeout=300s, ttl=86400s."""
    return Settings()


@pytest.fixture()
def embedding() -> np.ndarray:
    """A deterministic unit-norm 512-dim embedding."""
    vec = np.arange(512, dtype=np.float32) + 1.0
    return vec / np.linalg.norm(vec)


@pytest.fixture()
def identity(embedding: np.ndarray) -> Identity:
    """A fresh NEW identity at t=1000ns."""
    return create_identity(embedding, timestamp_ns=1_000, confidence=0.9)


@pytest.fixture()
def manager(settings: Settings) -> LifecycleManager:
    return LifecycleManager(settings)


# ── UUID uniqueness ──────────────────────────────────────────────────────────

def test_new_global_id_is_unique() -> None:
    ids = {new_global_id() for _ in range(10_000)}
    assert len(ids) == 10_000


def test_created_identities_have_unique_ids(embedding: np.ndarray) -> None:
    ids = {create_identity(embedding, timestamp_ns=i).global_id for i in range(1_000)}
    assert len(ids) == 1_000


# ── Identity creation + immutability ─────────────────────────────────────────

def test_create_identity_fields(identity: Identity) -> None:
    assert identity.state is LifecycleState.NEW
    assert identity.observation_count == 1
    assert identity.created_ns == 1_000
    assert identity.last_seen_ns == 1_000
    assert identity.confidence == 0.9
    assert identity.expire_at_ns is None
    assert identity.embedding.shape == (512,)


def test_create_identity_explicit_id(embedding: np.ndarray) -> None:
    ident = create_identity(embedding, timestamp_ns=5, global_id="fixed-id")
    assert ident.global_id == "fixed-id"


def test_identity_is_frozen(identity: Identity) -> None:
    with pytest.raises(Exception):
        identity.confidence = 0.1  # type: ignore[misc]


def test_embedding_is_read_only(identity: Identity) -> None:
    with pytest.raises(ValueError):
        identity.embedding[0] = 99.0


def test_embedding_is_copied_not_aliased(embedding: np.ndarray) -> None:
    src = embedding.copy()
    ident = create_identity(src, timestamp_ns=1)
    src[0] = 123.456  # mutate the caller's array
    assert ident.embedding[0] != 123.456


def test_with_updates_returns_new_object(identity: Identity) -> None:
    updated = identity.with_updates(confidence=0.5)
    assert updated is not identity
    assert updated.confidence == 0.5
    assert identity.confidence == 0.9  # original unchanged


def test_is_terminal(identity: Identity) -> None:
    assert not identity.is_terminal
    expired = identity.with_updates(state=LifecycleState.EXPIRED)
    assert expired.is_terminal


# ── Lifecycle transitions ────────────────────────────────────────────────────

def test_activate(manager: LifecycleManager, identity: Identity) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    assert active.state is LifecycleState.ACTIVE
    assert active.last_seen_ns == 2_000


def test_activate_requires_new(manager: LifecycleManager, identity: Identity) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    with pytest.raises(InvalidTransitionError):
        manager.activate(active, timestamp_ns=3_000)


def test_refresh_increments_observation(
    manager: LifecycleManager, identity: Identity
) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    refreshed = manager.refresh(active, timestamp_ns=3_000)
    assert refreshed.observation_count == 2
    assert refreshed.last_seen_ns == 3_000
    assert refreshed.state is LifecycleState.ACTIVE


def test_refresh_updates_embedding_and_confidence(
    manager: LifecycleManager, identity: Identity, embedding: np.ndarray
) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    new_emb = np.ones(512, dtype=np.float32)
    refreshed = manager.refresh(
        active, timestamp_ns=3_000, embedding=new_emb, confidence=0.42
    )
    assert refreshed.confidence == 0.42
    assert np.allclose(refreshed.embedding, 1.0)


def test_mark_lost_sets_expiry(
    manager: LifecycleManager, identity: Identity, settings: Settings
) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    lost = manager.mark_lost(active, timestamp_ns=5_000)
    assert lost.state is LifecycleState.LOST
    assert lost.expire_at_ns == 5_000 + settings.identity_ttl_s * _NS_PER_S


def test_mark_lost_requires_active(
    manager: LifecycleManager, identity: Identity
) -> None:
    with pytest.raises(InvalidTransitionError):
        manager.mark_lost(identity, timestamp_ns=2_000)  # identity is NEW


def test_relink_lost_to_active(
    manager: LifecycleManager, identity: Identity
) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    lost = manager.mark_lost(active, timestamp_ns=5_000)
    relinked = manager.refresh(lost, timestamp_ns=6_000)
    assert relinked.state is LifecycleState.ACTIVE
    assert relinked.expire_at_ns is None
    assert relinked.observation_count == 2


def test_expire_from_lost(manager: LifecycleManager, identity: Identity) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    lost = manager.mark_lost(active, timestamp_ns=5_000)
    expired = manager.expire(lost)
    assert expired.state is LifecycleState.EXPIRED
    assert expired.is_terminal


def test_expire_requires_lost(manager: LifecycleManager, identity: Identity) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    with pytest.raises(InvalidTransitionError):
        manager.expire(active)


def test_refresh_expired_forbidden(
    manager: LifecycleManager, identity: Identity
) -> None:
    active = manager.activate(identity, timestamp_ns=2_000)
    lost = manager.mark_lost(active, timestamp_ns=5_000)
    expired = manager.expire(lost)
    with pytest.raises(InvalidTransitionError):
        manager.refresh(expired, timestamp_ns=6_000)


# ── Expiration logic (time-driven advance) ───────────────────────────────────

def test_advance_active_to_lost(
    manager: LifecycleManager, identity: Identity, settings: Settings
) -> None:
    active = manager.activate(identity, timestamp_ns=0)
    just_past = settings.identity_lost_timeout_s * _NS_PER_S
    advanced = manager.advance(active, current_ns=just_past)
    assert advanced.state is LifecycleState.LOST


def test_advance_active_stays_active_before_timeout(
    manager: LifecycleManager, identity: Identity, settings: Settings
) -> None:
    active = manager.activate(identity, timestamp_ns=0)
    before = settings.identity_lost_timeout_s * _NS_PER_S - 1
    advanced = manager.advance(active, current_ns=before)
    assert advanced.state is LifecycleState.ACTIVE


def test_advance_lost_to_expired(
    manager: LifecycleManager, identity: Identity, settings: Settings
) -> None:
    active = manager.activate(identity, timestamp_ns=0)
    lost = manager.mark_lost(active, timestamp_ns=0)
    deadline = settings.identity_ttl_s * _NS_PER_S
    advanced = manager.advance(lost, current_ns=deadline)
    assert advanced.state is LifecycleState.EXPIRED


def test_advance_lost_stays_lost_before_ttl(
    manager: LifecycleManager, identity: Identity, settings: Settings
) -> None:
    active = manager.activate(identity, timestamp_ns=0)
    lost = manager.mark_lost(active, timestamp_ns=0)
    advanced = manager.advance(lost, current_ns=settings.identity_ttl_s * _NS_PER_S - 1)
    assert advanced.state is LifecycleState.LOST


def test_advance_noop_when_nothing_due(
    manager: LifecycleManager, identity: Identity
) -> None:
    active = manager.activate(identity, timestamp_ns=0)
    advanced = manager.advance(active, current_ns=100)
    assert advanced is active


def test_should_predicates(
    manager: LifecycleManager, identity: Identity, settings: Settings
) -> None:
    active = manager.activate(identity, timestamp_ns=0)
    assert not manager.should_mark_lost(active, current_ns=0)
    assert manager.should_mark_lost(
        active, current_ns=settings.identity_lost_timeout_s * _NS_PER_S
    )
    assert not manager.should_expire(active, current_ns=10**18)  # not LOST


# ── Gallery insert / update / remove / enumerate ─────────────────────────────

def test_gallery_insert_and_get(identity: Identity) -> None:
    gallery = IdentityGallery()
    gallery.insert(identity)
    assert gallery.get(identity.global_id) is identity
    assert len(gallery) == 1


def test_gallery_duplicate_protection(identity: Identity) -> None:
    gallery = IdentityGallery()
    gallery.insert(identity)
    with pytest.raises(DuplicateIdentityError):
        gallery.insert(identity)


def test_gallery_update(manager: LifecycleManager, identity: Identity) -> None:
    gallery = IdentityGallery()
    gallery.insert(identity)
    active = manager.activate(identity, timestamp_ns=2_000)
    gallery.update(active)
    stored = gallery.get(identity.global_id)
    assert stored is not None
    assert stored.state is LifecycleState.ACTIVE


def test_gallery_update_missing_raises(identity: Identity) -> None:
    gallery = IdentityGallery()
    with pytest.raises(IdentityNotFoundError):
        gallery.update(identity)


def test_gallery_remove(identity: Identity) -> None:
    gallery = IdentityGallery()
    gallery.insert(identity)
    removed = gallery.remove(identity.global_id)
    assert removed is identity
    assert identity.global_id not in gallery
    assert len(gallery) == 0


def test_gallery_remove_missing_raises() -> None:
    gallery = IdentityGallery()
    with pytest.raises(IdentityNotFoundError):
        gallery.remove("nope")


def test_gallery_upsert(manager: LifecycleManager, identity: Identity) -> None:
    gallery = IdentityGallery()
    gallery.upsert(identity)  # acts as insert
    active = manager.activate(identity, timestamp_ns=2_000)
    gallery.upsert(active)  # acts as update
    assert len(gallery) == 1
    stored = gallery.get(identity.global_id)
    assert stored is not None and stored.state is LifecycleState.ACTIVE


def test_gallery_enumerate(embedding: np.ndarray) -> None:
    gallery = IdentityGallery()
    made = [create_identity(embedding, timestamp_ns=i) for i in range(5)]
    for ident in made:
        gallery.insert(ident)
    assert len(gallery.identities()) == 5
    assert set(gallery.ids()) == {m.global_id for m in made}
    assert {i.global_id for i in gallery} == {m.global_id for m in made}


def test_gallery_get_absent_returns_none() -> None:
    gallery = IdentityGallery()
    assert gallery.get("absent") is None


def test_gallery_clear(embedding: np.ndarray) -> None:
    gallery = IdentityGallery()
    for i in range(3):
        gallery.insert(create_identity(embedding, timestamp_ns=i))
    gallery.clear()
    assert len(gallery) == 0


# ── Store serialization ──────────────────────────────────────────────────────

def test_serialization_round_trip(identity: Identity) -> None:
    restored = identity_from_dict(identity_to_dict(identity))
    assert restored.global_id == identity.global_id
    assert restored.state is identity.state
    assert restored.created_ns == identity.created_ns
    assert restored.last_seen_ns == identity.last_seen_ns
    assert restored.observation_count == identity.observation_count
    assert restored.confidence == identity.confidence
    assert restored.expire_at_ns == identity.expire_at_ns
    assert np.allclose(restored.embedding, identity.embedding)


def test_serialization_bad_version_raises(identity: Identity) -> None:
    data = identity_to_dict(identity)
    data["schema_version"] = 999
    with pytest.raises(ValueError):
        identity_from_dict(data)


def test_serialization_malformed_raises() -> None:
    with pytest.raises(ValueError):
        identity_from_dict({"schema_version": 1})  # missing fields


def test_in_memory_store_crud(identity: Identity) -> None:
    store = InMemoryIdentityStore()
    assert not store.exists(identity.global_id)
    store.save(identity)
    assert store.exists(identity.global_id)
    loaded = store.load(identity.global_id)
    assert loaded is not None and loaded.global_id == identity.global_id
    assert len(store.load_all()) == 1
    assert store.delete(identity.global_id) is True
    assert store.delete(identity.global_id) is False
    assert store.load(identity.global_id) is None


def test_in_memory_store_save_overwrites(
    manager: LifecycleManager, identity: Identity
) -> None:
    store = InMemoryIdentityStore()
    store.save(identity)
    active = manager.activate(identity, timestamp_ns=2_000)
    store.save(active)
    loaded = store.load(identity.global_id)
    assert loaded is not None and loaded.state is LifecycleState.ACTIVE
    assert len(store.load_all()) == 1


def test_json_file_store_crud(identity: Identity, tmp_path: Path) -> None:
    store = JsonFileIdentityStore(tmp_path)
    assert not store.exists(identity.global_id)
    store.save(identity)
    assert store.exists(identity.global_id)
    loaded = store.load(identity.global_id)
    assert loaded is not None
    assert np.allclose(loaded.embedding, identity.embedding)
    assert store.delete(identity.global_id) is True
    assert store.delete(identity.global_id) is False
    assert store.load(identity.global_id) is None


def test_json_file_store_persists_across_instances(
    identity: Identity, tmp_path: Path
) -> None:
    JsonFileIdentityStore(tmp_path).save(identity)
    # New instance pointing at the same directory sees the record.
    reopened = JsonFileIdentityStore(tmp_path)
    loaded = reopened.load(identity.global_id)
    assert loaded is not None and loaded.global_id == identity.global_id


def test_json_file_store_load_all(embedding: np.ndarray, tmp_path: Path) -> None:
    store = JsonFileIdentityStore(tmp_path)
    made = [create_identity(embedding, timestamp_ns=i) for i in range(4)]
    for ident in made:
        store.save(ident)
    assert {i.global_id for i in store.load_all()} == {m.global_id for m in made}


# ── End-to-end lifecycle through gallery + store ─────────────────────────────

def test_full_lifecycle_persisted(
    manager: LifecycleManager, identity: Identity, tmp_path: Path
) -> None:
    """NEW → ACTIVE → LOST → EXPIRED, mirrored into gallery and store."""
    gallery = IdentityGallery()
    store = JsonFileIdentityStore(tmp_path)

    gallery.insert(identity)
    store.save(identity)

    active = manager.activate(identity, timestamp_ns=1_000)
    gallery.update(active)
    store.save(active)

    lost = manager.mark_lost(active, timestamp_ns=2_000)
    gallery.update(lost)
    store.save(lost)

    expired = manager.expire(lost)
    gallery.update(expired)
    store.save(expired)

    reloaded = store.load(identity.global_id)
    assert reloaded is not None
    assert reloaded.state is LifecycleState.EXPIRED
    stored = gallery.get(identity.global_id)
    assert stored is not None and stored.is_terminal
