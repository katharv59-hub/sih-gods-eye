"""Identity gallery — Phase 2, Task 4.

An in-memory collection of :class:`~gods_eye.reid.identity.Identity`
objects keyed by ``global_id``. The gallery is a pure container: it
stores and retrieves identities and nothing more.

Deliberately EXCLUDED (belongs elsewhere):
    - matching / similarity search  → Re-ID matcher (Task 2)
    - lifecycle transitions         → LifecycleManager (Task 4)
    - persistence                   → IdentityStore (Task 4)

Because :class:`Identity` is immutable, "update" means replacing the
stored object for a given ``global_id`` with a new one. The gallery
enforces duplicate protection on insert and referential consistency on
update/remove.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from gods_eye.observability.logger import get_logger
from gods_eye.reid.identity import Identity, LifecycleState


class DuplicateIdentityError(Exception):
    """Raised when inserting an identity whose ``global_id`` already exists."""


class IdentityNotFoundError(Exception):
    """Raised when updating/removing a ``global_id`` not in the gallery."""


class IdentityGallery:
    """In-memory store of identities indexed by ``global_id``.

    Single-writer by design (spec §3.2.4 — accessed from one thread). No
    internal locking; cross-thread access must be synchronized by the
    caller.

    The gallery maintains an **incremental embedding matrix** (review C2)
    so that ``get_embedding_matrix()`` returns in O(1) instead of
    rebuilding a full N×D array on every call.

    - **insert**: appends a row (amortized O(1) with pre-allocation).
    - **update**: overwrites the row in-place (O(1)).
    - **remove**: swap-deletes with the last row (O(1)).
    """

    _INITIAL_CAPACITY: int = 256
    _EMBEDDING_DIM: int = 512

    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}
        self._log = get_logger("reid.gallery")

        # ── Incremental embedding matrix ──────────────────────────────
        # Pre-allocated buffer; only rows [0, _size) are valid.
        self._capacity: int = self._INITIAL_CAPACITY
        self._size: int = 0
        self._matrix: np.ndarray = np.zeros(
            (self._capacity, self._EMBEDDING_DIM), dtype=np.float32,
        )
        self._id_to_row: dict[str, int] = {}
        self._row_to_id: list[str] = []   # length == _size

    # ── Matrix helpers ──────────────────────────────────────────────────

    def _grow_if_needed(self) -> None:
        """Double the buffer capacity when full."""
        if self._size < self._capacity:
            return
        new_cap = self._capacity * 2
        new_matrix = np.zeros(
            (new_cap, self._EMBEDDING_DIM), dtype=np.float32,
        )
        new_matrix[: self._size] = self._matrix[: self._size]
        self._matrix = new_matrix
        self._capacity = new_cap

    def _append_row(self, global_id: str, embedding: np.ndarray) -> None:
        """Append a new row to the matrix."""
        self._grow_if_needed()
        row = self._size
        self._matrix[row] = embedding
        self._id_to_row[global_id] = row
        self._row_to_id.append(global_id)
        self._size += 1

    def _update_row(self, global_id: str, embedding: np.ndarray) -> None:
        """Overwrite an existing row in-place."""
        row = self._id_to_row[global_id]
        self._matrix[row] = embedding

    def _remove_row(self, global_id: str) -> None:
        """Swap-delete: move the last row into the vacated slot."""
        row = self._id_to_row.pop(global_id)
        last = self._size - 1

        if row != last:
            # Swap last row into the gap
            last_id = self._row_to_id[last]
            self._matrix[row] = self._matrix[last]
            self._id_to_row[last_id] = row
            self._row_to_id[row] = last_id

        self._row_to_id.pop()  # shrink by 1
        self._matrix[last] = 0.0  # zero out vacated slot
        self._size -= 1

    # ── Mutating operations ─────────────────────────────────────────────

    def insert(self, identity: Identity) -> None:
        """Insert a new identity.

        Args:
            identity: Identity to store.

        Raises:
            DuplicateIdentityError: If ``identity.global_id`` already exists.
        """
        if identity.global_id in self._identities:
            raise DuplicateIdentityError(
                f"identity {identity.global_id} already exists"
            )
        self._identities[identity.global_id] = identity
        self._append_row(identity.global_id, identity.embedding)
        self._log.debug(
            "gallery_insert",
            global_id=identity.global_id,
            size=len(self._identities),
        )

    def update(self, identity: Identity) -> None:
        """Replace an existing identity with a new immutable version.

        Args:
            identity: Identity whose ``global_id`` must already be present.

        Raises:
            IdentityNotFoundError: If ``identity.global_id`` is not stored.
        """
        if identity.global_id not in self._identities:
            raise IdentityNotFoundError(
                f"identity {identity.global_id} not found"
            )
        self._identities[identity.global_id] = identity
        self._update_row(identity.global_id, identity.embedding)
        self._log.debug(
            "gallery_update",
            global_id=identity.global_id,
            state=identity.state.value,
        )

    def upsert(self, identity: Identity) -> None:
        """Insert the identity, or replace it if the id already exists.

        Convenience for callers that do not care whether the identity is
        new. Never raises on duplication.
        """
        if identity.global_id in self._identities:
            self._identities[identity.global_id] = identity
            self._update_row(identity.global_id, identity.embedding)
        else:
            self._identities[identity.global_id] = identity
            self._append_row(identity.global_id, identity.embedding)
        self._log.debug(
            "gallery_upsert",
            global_id=identity.global_id,
            size=len(self._identities),
        )

    def remove(self, global_id: str) -> Identity:
        """Remove and return the identity with ``global_id``.

        Args:
            global_id: Identifier to remove.

        Returns:
            The removed identity.

        Raises:
            IdentityNotFoundError: If ``global_id`` is not stored.
        """
        try:
            identity = self._identities.pop(global_id)
        except KeyError:
            raise IdentityNotFoundError(f"identity {global_id} not found") from None
        self._remove_row(global_id)
        self._log.debug(
            "gallery_remove", global_id=global_id, size=len(self._identities)
        )
        return identity

    def clear(self) -> None:
        """Remove all identities from the gallery."""
        self._identities.clear()
        self._size = 0
        self._id_to_row.clear()
        self._row_to_id.clear()
        self._matrix[:] = 0.0
        self._log.debug("gallery_clear")

    # ── Read-only accessors ─────────────────────────────────────────────

    def get(self, global_id: str) -> Identity | None:
        """Return the identity for ``global_id``, or ``None`` if absent."""
        return self._identities.get(global_id)

    def contains(self, global_id: str) -> bool:
        """True if an identity with ``global_id`` is stored."""
        return global_id in self._identities

    def identities(self) -> list[Identity]:
        """Return a snapshot list of all stored identities."""
        return list(self._identities.values())

    def ids(self) -> list[str]:
        """Return a snapshot list of all stored ``global_id`` values."""
        return list(self._identities.keys())

    def get_embedding_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return the gallery embedding matrix and corresponding IDs.

        Returns a **view** of the internal buffer — O(1), no copy.
        The matrix rows correspond 1:1 to the returned ID list.

        Returns:
            Tuple of:
                - ``(N, D)`` float32 numpy array of all identity
                  embeddings, where N is the gallery size and D is the
                  embedding dimensionality.
                - List of ``global_id`` strings in the same row order
                  as the matrix.
            If the gallery is empty, returns a ``(0, 0)`` array and
            an empty list.
        """
        if self._size == 0:
            return np.zeros((0, 0), dtype=np.float32), []
        return self._matrix[: self._size], list(self._row_to_id)

    @property
    def active_count(self) -> int:
        """Number of identities in ACTIVE state."""
        return sum(
            1 for i in self._identities.values()
            if i.state is LifecycleState.ACTIVE
        )

    @property
    def lost_count(self) -> int:
        """Number of identities in LOST state (review R4)."""
        return sum(
            1 for i in self._identities.values()
            if i.state is LifecycleState.LOST
        )

    # ── Dunder conveniences ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._identities)

    def __contains__(self, global_id: object) -> bool:
        return global_id in self._identities

    def __iter__(self) -> Iterator[Identity]:
        return iter(list(self._identities.values()))
