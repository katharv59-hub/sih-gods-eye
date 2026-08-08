"""Immutable identity model — Phase 2, Task 4.

Defines the persistent identity object and its lifecycle states. An
``Identity`` is the minimal record required for persistent identity
management: a globally unique UUID, an appearance embedding, timing
metadata, an observation count, a lifecycle state, and a confidence score.

Design notes:
    - The model is IMMUTABLE (frozen dataclass). Every "mutation"
      (activate, refresh, expire) returns a NEW ``Identity`` via the
      lifecycle manager. This keeps state transitions deterministic and
      free of aliasing bugs.
    - The model carries NO tracking information. Tracker IDs are owned by
      ByteTrack and deliberately excluded here (spec ADR-004 — the
      identity layer is permanently separated from tracker-local IDs).
    - There is no matching, similarity, or gallery-search logic here.

This module is standalone: it does not depend on the Re-ID extractor,
similarity engine, or matcher.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np


class LifecycleState(Enum):
    """Deterministic lifecycle state of a persistent identity.

    State machine (owned by :mod:`gods_eye.reid.identity_lifecycle`)::

        NEW ── activate ──▶ ACTIVE
        ACTIVE ── refresh ──▶ ACTIVE            (re-observed, stays active)
        ACTIVE ── mark_lost ──▶ LOST            (not seen within lost timeout)
        LOST ── refresh ──▶ ACTIVE              (re-observed before expiry)
        LOST ── expire ──▶ EXPIRED              (TTL elapsed while lost)
        EXPIRED ── (none) ──▶ EXPIRED           (terminal)

    Values are lowercase strings for stable serialization.
    """

    NEW = "new"          # Just created, not yet confirmed active
    ACTIVE = "active"    # Currently being observed
    LOST = "lost"        # Not observed within the lost timeout window
    EXPIRED = "expired"  # TTL elapsed — terminal, record tombstoned


# The set of states from which no further forward transition is possible.
TERMINAL_STATES: frozenset[LifecycleState] = frozenset({LifecycleState.EXPIRED})


def _freeze_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a read-only float32 copy of an embedding vector.

    Copying + clearing the writeable flag preserves the immutability
    guarantee of the frozen dataclass: callers cannot mutate the stored
    embedding in place after construction.
    """
    frozen = np.array(embedding, dtype=np.float32, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class Identity:
    """A globally unique, persistent identity.

    Immutable by construction. Use the lifecycle manager
    (:class:`gods_eye.reid.identity_lifecycle.LifecycleManager`) to derive
    new states rather than mutating an instance.

    Attributes:
        global_id: Globally unique identifier (UUID4 hex string).
        embedding: Appearance embedding vector (read-only float32 array).
        created_ns: Creation timestamp, Unix nanoseconds.
        last_seen_ns: Timestamp of most recent observation, Unix nanoseconds.
        observation_count: Number of times this identity has been observed.
        state: Current lifecycle state.
        confidence: Match/appearance confidence score in [0.0, 1.0].
        last_camera_id: Camera where this identity was most recently observed.
            Empty string when camera is unknown (Phase 1/2 compatibility).
        camera_history: Ordered list of camera_ids this identity has visited.
            Used by Phase 5 NLQ (get_camera_path) and Phase 6 profiles.
        expire_at_ns: Absolute deadline (Unix ns) after which a LOST
            identity expires. ``None`` while ACTIVE/NEW.
    """

    global_id: str
    embedding: np.ndarray = field(repr=False)
    created_ns: int
    last_seen_ns: int
    observation_count: int
    state: LifecycleState
    confidence: float
    last_camera_id: str = ""
    camera_history: tuple[str, ...] = ()
    expire_at_ns: int | None = None

    def __post_init__(self) -> None:
        # Enforce a read-only, float32 embedding regardless of what the
        # caller passed. ``object.__setattr__`` is required because the
        # dataclass is frozen.
        object.__setattr__(self, "embedding", _freeze_embedding(self.embedding))

    # ── Convenience predicates (no side effects) ────────────────────────

    @property
    def is_terminal(self) -> bool:
        """True if the identity is in a terminal (EXPIRED) state."""
        return self.state in TERMINAL_STATES

    def with_updates(self, **changes: object) -> Identity:
        """Return a copy of this identity with the given fields replaced.

        Thin, type-checked wrapper over :func:`dataclasses.replace`. Used
        by the lifecycle manager to derive successor states without
        mutating the original. The embedding is re-frozen by
        ``__post_init__`` on the returned copy.
        """
        return replace(self, **changes)  # type: ignore[arg-type]


def new_global_id() -> str:
    """Generate a fresh globally unique identifier (UUID4 hex)."""
    return uuid.uuid4().hex


def create_identity(
    embedding: np.ndarray,
    timestamp_ns: int,
    *,
    confidence: float = 0.0,
    global_id: str | None = None,
    camera_id: str = "",
) -> Identity:
    """Create a fresh identity in the NEW state.

    Args:
        embedding: Appearance embedding vector. Copied and frozen.
        timestamp_ns: Creation/first-observation time, Unix nanoseconds.
            Used for both ``created_ns`` and ``last_seen_ns``.
        confidence: Initial confidence score in [0.0, 1.0].
        global_id: Optional explicit UUID. A fresh UUID4 is generated
            when omitted. Primarily an escape hatch for deserialization.
        camera_id: Camera where the identity was first observed.
            Empty string for backward compatibility with Phase 1/2.

    Returns:
        A new immutable :class:`Identity` with ``observation_count == 1``
        and ``state == LifecycleState.NEW``.
    """
    camera_history = (camera_id,) if camera_id else ()
    return Identity(
        global_id=global_id if global_id is not None else new_global_id(),
        embedding=embedding,
        created_ns=timestamp_ns,
        last_seen_ns=timestamp_ns,
        observation_count=1,
        state=LifecycleState.NEW,
        confidence=confidence,
        last_camera_id=camera_id,
        camera_history=camera_history,
        expire_at_ns=None,
    )
