"""ByteTrack tracking module re-export for SIH specification compliance.

Re-exports ByteTrackTracker and internal types from bytetrack_tracker.py
so that imports from gods_eye.tracking.bytetrack resolve directly to the
canonical ByteTrack implementation.
"""

from __future__ import annotations

from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker, _TrackInfo

__all__ = [
    "ByteTrackTracker",
    "_TrackInfo",
]
