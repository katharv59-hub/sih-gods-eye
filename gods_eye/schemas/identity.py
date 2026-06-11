"""Identity schemas — §4 Canonical Data Schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class IdentityStatus(Enum):
    """Lifecycle state of a global identity.

    State Transitions:
        ACTIVE → LOST   : last_seen_ns older than IDENTITY_LOST_TIMEOUT_S seconds
        LOST   → ACTIVE : Re-ID match above REID_MATCH_THRESHOLD
        LOST   → PURGED : current_time > purge_at_ns (set at IDENTITY_TTL_S from last_seen)
        PURGED → (none) : terminal state — embeddings zeroed, record tombstoned
    """

    ACTIVE = "active"    # Currently visible in at least one camera
    LOST = "lost"        # Not seen for > IDENTITY_LOST_TIMEOUT seconds
    PURGED = "purged"    # TTL expired — record tombstoned, embeddings deleted


@dataclass
class Identity:
    """A globally unique identity that persists across cameras and sessions.

    Attributes:
        global_id: UUID — stable across cameras and sessions.
        status: Current lifecycle state.
        embedding: Current EMA-updated embedding (OSNet dim).
        embedding_history: Rolling window, max EMBEDDING_HISTORY_LEN.
        first_seen_ns: Unix nanoseconds.
        last_seen_ns: Unix nanoseconds.
        last_camera_id: Most recent camera where this identity was observed.
        camera_history: Ordered list of camera_ids visited.
        track_id_history: [(camera_id, track_id), ...] mapping.
        confidence: Mean match confidence over lifetime.
        purge_at_ns: Set when status transitions to LOST.
    """

    global_id: str
    status: IdentityStatus
    embedding: np.ndarray
    embedding_history: list[np.ndarray] = field(default_factory=list)
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    last_camera_id: str = ""
    camera_history: list[str] = field(default_factory=list)
    track_id_history: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    purge_at_ns: Optional[int] = None
