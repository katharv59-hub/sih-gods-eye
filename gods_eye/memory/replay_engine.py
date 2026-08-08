"""Deterministic Replay Engine — Phase 4.4 (§14).

Provides deterministic read-only historical replay streams over authoritative EventStore
and GraphStore data using composite ordering key (timestamp_ns, sequence_num, record_id).
"""

from __future__ import annotations

from typing import Iterator, Optional

from gods_eye.events.event_store import BaseEventStore
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.event import EventType
from gods_eye.schemas.timeline import ReplayFrame

_log = get_logger("memory.replay_engine")


class DeterministicReplayEngine:
    """Read-only deterministic historical replay engine."""

    def __init__(self, event_store: BaseEventStore, graph_store: BaseGraphStore) -> None:
        self._event_store = event_store
        self._graph_store = graph_store

    def stream_replay(
        self,
        start_ns: int,
        end_ns: int,
        camera_id: Optional[str] = None,
        global_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
    ) -> Iterator[ReplayFrame]:
        """Stream deterministically ordered replay frames within the timestamp window [start_ns, end_ns].

        Strict composite key sorting: (timestamp_ns, sequence_num, record_id).
        """
        frames: list[ReplayFrame] = []

        # 1. Fetch Events
        events = self._event_store.query_events(
            event_type=event_type,
            camera_id=camera_id,
            global_id=global_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=10_000,
        )
        for idx, e in enumerate(events):
            frames.append(
                ReplayFrame(
                    timestamp_ns=e.timestamp_ns,
                    sequence_num=idx + 1,
                    record_id=e.event_id,
                    record_type="event",
                    data={
                        "event_type": e.event_type.value,
                        "global_id": e.global_id,
                        "camera_id": e.camera_id,
                        "confidence": e.confidence,
                        "explanation": e.explanation,
                        "metadata": e.metadata,
                    },
                )
            )

        # 2. Fetch Graph Trajectory Observations
        if global_id is not None:
            trajs = self._graph_store.get_identity_trajectory(
                global_id, start_ns=start_ns, end_ns=end_ns, limit=10_000
            )
            for obs in trajs:
                frames.append(
                    ReplayFrame(
                        timestamp_ns=obs["timestamp_ns"],
                        sequence_num=obs["sequence_num"],
                        record_id=obs["edge_id"],
                        record_type="observation",
                        data=obs,
                    )
                )

        # 3. Fetch Camera Transitions
        transitions = self._graph_store.get_camera_transitions(
            from_camera_id=camera_id,
            global_id=global_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=10_000,
        )
        for tr in transitions:
            frames.append(
                ReplayFrame(
                    timestamp_ns=tr["timestamp_ns"],
                    sequence_num=tr["sequence_num"],
                    record_id=tr["edge_id"],
                    record_type="transition",
                    data=tr,
                )
            )

        # 4. Deterministic Sort by Composite Key (timestamp_ns, sequence_num, record_id)
        frames.sort(key=lambda f: (f.timestamp_ns, f.sequence_num, f.record_id))

        yield from frames
