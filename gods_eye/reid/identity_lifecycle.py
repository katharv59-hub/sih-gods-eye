"""Identity lifecycle manager — Phase 2, Task 4.

Owns the deterministic state machine for a persistent identity:

    NEW ── activate ──▶ ACTIVE
    ACTIVE / LOST ── refresh ──▶ ACTIVE
    ACTIVE ── mark_lost ──▶ LOST
    LOST ── expire ──▶ EXPIRED (terminal)

Every transition is a pure function of the input identity and the current
time: it returns a NEW immutable :class:`~gods_eye.reid.identity.Identity`
and never mutates the argument. Illegal transitions raise
:class:`InvalidTransitionError` so bugs surface loudly rather than
silently corrupting lifecycle state.

Timeouts come from :class:`~gods_eye.config.settings.Settings`:
    - ``identity_lost_timeout_s`` : ACTIVE → LOST after this idle gap.
    - ``identity_ttl_s``          : LOST → EXPIRED after this long lost.

This module performs NO matching, similarity, or persistence. It only
computes state.
"""

from __future__ import annotations

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.reid.identity import Identity, LifecycleState

_NS_PER_S: int = 1_000_000_000


class InvalidTransitionError(Exception):
    """Raised when a lifecycle transition is not permitted from a state."""


class LifecycleManager:
    """Deterministic lifecycle transitions for identities.

    Stateless aside from configuration and a logger. All timeouts are
    resolved once at construction from :class:`Settings`.
    """

    def __init__(self, settings: Settings) -> None:
        self._lost_timeout_ns: int = settings.identity_lost_timeout_s * _NS_PER_S
        self._ttl_ns: int = settings.identity_ttl_s * _NS_PER_S
        self._log = get_logger("reid.lifecycle")

    # ── Explicit transitions ────────────────────────────────────────────

    def activate(self, identity: Identity, timestamp_ns: int) -> Identity:
        """Confirm a NEW identity into the ACTIVE state.

        Args:
            identity: Identity in the NEW state.
            timestamp_ns: Observation time, Unix nanoseconds.

        Returns:
            A new ACTIVE identity with refreshed ``last_seen_ns``.

        Raises:
            InvalidTransitionError: If ``identity`` is not NEW.
        """
        if identity.state is not LifecycleState.NEW:
            raise InvalidTransitionError(
                f"activate requires NEW state, got {identity.state.value}"
            )
        result = identity.with_updates(
            state=LifecycleState.ACTIVE,
            last_seen_ns=timestamp_ns,
            expire_at_ns=None,
        )
        self._log.info(
            "identity_activated",
            global_id=identity.global_id,
            observation_count=result.observation_count,
        )
        return result

    def refresh(
        self,
        identity: Identity,
        timestamp_ns: int,
        *,
        embedding: np.ndarray | None = None,
        confidence: float | None = None,
    ) -> Identity:
        """Register a fresh observation, returning an ACTIVE identity.

        Valid from NEW, ACTIVE, or LOST. A LOST identity is re-linked back
        to ACTIVE. Increments ``observation_count`` and clears any pending
        expiry deadline. Optionally updates the embedding and confidence.

        Args:
            identity: Identity to refresh. Must not be terminal (EXPIRED).
            timestamp_ns: Observation time, Unix nanoseconds.
            embedding: Optional replacement embedding.
            confidence: Optional replacement confidence score.

        Returns:
            A new ACTIVE identity.

        Raises:
            InvalidTransitionError: If ``identity`` is EXPIRED.
        """
        if identity.state is LifecycleState.EXPIRED:
            raise InvalidTransitionError("cannot refresh an EXPIRED identity")

        was_lost = identity.state is LifecycleState.LOST
        changes: dict[str, object] = {
            "state": LifecycleState.ACTIVE,
            "last_seen_ns": timestamp_ns,
            "observation_count": identity.observation_count + 1,
            "expire_at_ns": None,
        }
        if embedding is not None:
            changes["embedding"] = embedding
        if confidence is not None:
            changes["confidence"] = confidence

        result = identity.with_updates(**changes)
        self._log.info(
            "identity_refreshed",
            global_id=identity.global_id,
            relinked=was_lost,
            observation_count=result.observation_count,
        )
        return result

    def mark_lost(self, identity: Identity, timestamp_ns: int) -> Identity:
        """Transition an ACTIVE identity to LOST and set its expiry deadline.

        Args:
            identity: Identity in the ACTIVE state.
            timestamp_ns: Current time, Unix nanoseconds. The expiry
                deadline is set to ``timestamp_ns + identity_ttl_s``.

        Returns:
            A new LOST identity with ``expire_at_ns`` populated.

        Raises:
            InvalidTransitionError: If ``identity`` is not ACTIVE.
        """
        if identity.state is not LifecycleState.ACTIVE:
            raise InvalidTransitionError(
                f"mark_lost requires ACTIVE state, got {identity.state.value}"
            )
        result = identity.with_updates(
            state=LifecycleState.LOST,
            expire_at_ns=timestamp_ns + self._ttl_ns,
        )
        self._log.info(
            "identity_lost",
            global_id=identity.global_id,
            expire_at_ns=result.expire_at_ns,
        )
        return result

    def expire(self, identity: Identity) -> Identity:
        """Transition a LOST identity to the terminal EXPIRED state.

        Args:
            identity: Identity in the LOST state.

        Returns:
            A new EXPIRED identity.

        Raises:
            InvalidTransitionError: If ``identity`` is not LOST.
        """
        if identity.state is not LifecycleState.LOST:
            raise InvalidTransitionError(
                f"expire requires LOST state, got {identity.state.value}"
            )
        result = identity.with_updates(state=LifecycleState.EXPIRED)
        self._log.info("identity_expired", global_id=identity.global_id)
        return result

    # ── Time-driven predicates ──────────────────────────────────────────

    def should_mark_lost(self, identity: Identity, current_ns: int) -> bool:
        """True if an ACTIVE identity has been idle past the lost timeout."""
        if identity.state is not LifecycleState.ACTIVE:
            return False
        return (current_ns - identity.last_seen_ns) >= self._lost_timeout_ns

    def should_expire(self, identity: Identity, current_ns: int) -> bool:
        """True if a LOST identity has reached its expiry deadline."""
        if identity.state is not LifecycleState.LOST:
            return False
        return identity.expire_at_ns is not None and current_ns >= identity.expire_at_ns

    def advance(self, identity: Identity, current_ns: int) -> Identity:
        """Apply any due time-driven transition for a single identity.

        Deterministic, idempotent per timestamp. Applies at most one
        forward step:
            - ACTIVE past lost timeout    → LOST
            - LOST past expiry deadline   → EXPIRED
        Returns the identity unchanged if no transition is due.

        Args:
            identity: Identity to evaluate.
            current_ns: Current time, Unix nanoseconds.

        Returns:
            The (possibly transitioned) identity.
        """
        if self.should_mark_lost(identity, current_ns):
            return self.mark_lost(identity, current_ns)
        if self.should_expire(identity, current_ns):
            return self.expire(identity)
        return identity
