"""Identity persistence abstraction — Phase 2, Task 4.

Defines the :class:`IdentityStore` interface for persisting identities and
two lightweight reference implementations:

    - :class:`InMemoryIdentityStore` : dict-backed, non-durable (tests).
    - :class:`JsonFileIdentityStore` : one JSON file per identity on disk.

The interface exists so a production database (Postgres, Redis, a vector
store, …) can be dropped in later without touching callers (spec Rule 4 —
modularity). This module intentionally ships NO production database.

Serialization is handled here via :func:`identity_to_dict` /
:func:`identity_from_dict`, which round-trip the immutable
:class:`~gods_eye.reid.identity.Identity` — including its embedding — to
plain JSON-compatible structures.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from gods_eye.observability.logger import get_logger
from gods_eye.reid.identity import Identity, LifecycleState

# Serialization schema version — bump on any breaking field change so old
# records can be detected and migrated rather than silently mis-read.
SCHEMA_VERSION: int = 1


# ── Serialization ───────────────────────────────────────────────────────

def identity_to_dict(identity: Identity) -> dict[str, Any]:
    """Serialize an identity to a JSON-compatible dictionary.

    The embedding is stored as a plain list of floats; the lifecycle
    state as its string value.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "global_id": identity.global_id,
        "embedding": [float(x) for x in identity.embedding.tolist()],
        "created_ns": identity.created_ns,
        "last_seen_ns": identity.last_seen_ns,
        "observation_count": identity.observation_count,
        "state": identity.state.value,
        "confidence": identity.confidence,
        "expire_at_ns": identity.expire_at_ns,
    }


def identity_from_dict(data: dict[str, Any]) -> Identity:
    """Reconstruct an identity from a dictionary produced by
    :func:`identity_to_dict`.

    Raises:
        ValueError: If the schema version is unsupported or a required
            field is missing/invalid.
    """
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported identity schema_version: {version!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    try:
        embedding = np.array(data["embedding"], dtype=np.float32)
        return Identity(
            global_id=str(data["global_id"]),
            embedding=embedding,
            created_ns=int(data["created_ns"]),
            last_seen_ns=int(data["last_seen_ns"]),
            observation_count=int(data["observation_count"]),
            state=LifecycleState(data["state"]),
            confidence=float(data["confidence"]),
            expire_at_ns=(
                None if data["expire_at_ns"] is None else int(data["expire_at_ns"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed identity record: {exc}") from exc


# ── Store interface ─────────────────────────────────────────────────────

class IdentityStore(ABC):
    """Abstract persistence interface for identities.

    Implementations must be safe to call repeatedly and must treat
    ``global_id`` as the primary key. Replacement semantics: :meth:`save`
    overwrites any existing record with the same id.
    """

    @abstractmethod
    def save(self, identity: Identity) -> None:
        """Persist (insert or overwrite) a single identity."""
        ...

    @abstractmethod
    def load(self, global_id: str) -> Identity | None:
        """Load one identity by id, or return ``None`` if absent."""
        ...

    @abstractmethod
    def load_all(self) -> list[Identity]:
        """Load every persisted identity."""
        ...

    @abstractmethod
    def delete(self, global_id: str) -> bool:
        """Delete one identity by id.

        Returns:
            True if a record was removed, False if it did not exist.
        """
        ...

    @abstractmethod
    def exists(self, global_id: str) -> bool:
        """True if an identity with ``global_id`` is persisted."""
        ...


# ── In-memory implementation ────────────────────────────────────────────

class InMemoryIdentityStore(IdentityStore):
    """Non-durable dict-backed store.

    Round-trips through :func:`identity_to_dict` /
    :func:`identity_from_dict` so it exercises the same serialization path
    as durable stores — useful for tests and defaults.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._log = get_logger("reid.store.memory")

    def save(self, identity: Identity) -> None:
        self._records[identity.global_id] = identity_to_dict(identity)
        self._log.debug("store_save", global_id=identity.global_id)

    def load(self, global_id: str) -> Identity | None:
        record = self._records.get(global_id)
        return identity_from_dict(record) if record is not None else None

    def load_all(self) -> list[Identity]:
        return [identity_from_dict(r) for r in self._records.values()]

    def delete(self, global_id: str) -> bool:
        existed = self._records.pop(global_id, None) is not None
        self._log.debug("store_delete", global_id=global_id, existed=existed)
        return existed

    def exists(self, global_id: str) -> bool:
        return global_id in self._records


# ── JSON-file implementation ────────────────────────────────────────────

class JsonFileIdentityStore(IdentityStore):
    """Durable store writing one ``<global_id>.json`` file per identity.

    A minimal reference persistence backend — NOT a production database.
    Writes are atomic per record (temp file + ``os.replace``) to avoid
    torn reads. The directory is created on construction if missing.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log = get_logger("reid.store.jsonfile")

    def _path_for(self, global_id: str) -> Path:
        return self._dir / f"{global_id}.json"

    def save(self, identity: Identity) -> None:
        path = self._path_for(identity.global_id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(identity_to_dict(identity))
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)  # atomic on the same filesystem
        self._log.debug("store_save", global_id=identity.global_id, path=str(path))

    def load(self, global_id: str) -> Identity | None:
        path = self._path_for(global_id)
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return identity_from_dict(data)

    def load_all(self) -> list[Identity]:
        out: list[Identity] = []
        for path in sorted(self._dir.glob("*.json")):
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            out.append(identity_from_dict(data))
        return out

    def delete(self, global_id: str) -> bool:
        path = self._path_for(global_id)
        if not path.exists():
            return False
        path.unlink()
        self._log.debug("store_delete", global_id=global_id)
        return True

    def exists(self, global_id: str) -> bool:
        return self._path_for(global_id).exists()
