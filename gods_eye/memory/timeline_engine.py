"""Timeline Reconstruction Engine — Phase 4.4 (§14).

Deterministically reconstructs identity, camera, and zone spatiotemporal timelines
from authoritative EventStore and GraphStore projections without mutation.
"""

from __future__ import annotations

from typing import Optional

from gods_eye.events.event_store import BaseEventStore
from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.timeline import (
    CameraTimeline,
    IdentityTimeline,
    VisitSegment,
    ZoneTimeline,
)

_log = get_logger("memory.timeline_engine")


class TimelineReconstructionEngine:
    """Read-only timeline reconstruction engine derived from EventStore and GraphStore."""

    def __init__(self, event_store: BaseEventStore, graph_store: BaseGraphStore) -> None:
        self._event_store = event_store
        self._graph_store = graph_store

    def reconstruct_identity_timeline(
        self,
        global_id: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> IdentityTimeline:
        """Reconstruct the complete spatiotemporal history for a global identity."""
        obs_list = self._graph_store.get_identity_trajectory(
            global_id, start_ns=start_ns, end_ns=end_ns
        )
        transitions = self._graph_store.get_camera_transitions(
            global_id=global_id, start_ns=start_ns, end_ns=end_ns
        )
        events = self._event_store.query_events(
            global_id=global_id, start_ns=start_ns, end_ns=end_ns
        )

        visits: list[VisitSegment] = []
        if obs_list:
            current_cam = obs_list[0]["camera_id"]
            group: list[dict] = [obs_list[0]]

            for obs in obs_list[1:]:
                if obs["camera_id"] == current_cam:
                    group.append(obs)
                else:
                    visits.append(self._create_visit_segment(global_id, current_cam, group, is_complete=True))
                    current_cam = obs["camera_id"]
                    group = [obs]

            # Last segment: complete if followed by transition or end_ns bound reached
            is_last_complete = len(transitions) > 0 or len(obs_list) > 1
            visits.append(self._create_visit_segment(global_id, current_cam, group, is_complete=is_last_complete))

        min_ts = start_ns if start_ns is not None else (obs_list[0]["timestamp_ns"] if obs_list else 0)
        max_ts = end_ns if end_ns is not None else (obs_list[-1]["timestamp_ns"] if obs_list else 0)

        return IdentityTimeline(
            global_id=global_id,
            start_ns=min_ts,
            end_ns=max_ts,
            visits=visits,
            transitions=transitions,
            events=events,
        )

    def reconstruct_camera_timeline(
        self,
        camera_id: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> CameraTimeline:
        """Reconstruct spatial activity history for a specific camera."""
        events = self._event_store.query_events(
            camera_id=camera_id, start_ns=start_ns, end_ns=end_ns
        )
        # Fetch transitions involving camera
        trans_from = self._graph_store.get_camera_transitions(
            from_camera_id=camera_id, start_ns=start_ns, end_ns=end_ns
        )
        trans_to = self._graph_store.get_camera_transitions(
            to_camera_id=camera_id, start_ns=start_ns, end_ns=end_ns
        )

        active_identities = list({
            e.global_id for e in events if e.global_id is not None
        }.union({
            t["global_id"] for t in trans_from + trans_to
        }))

        min_ts = start_ns if start_ns is not None else 0
        max_ts = end_ns if end_ns is not None else 0

        return CameraTimeline(
            camera_id=camera_id,
            start_ns=min_ts,
            end_ns=max_ts,
            active_identities=active_identities,
            visit_segments=[],
            events=events,
        )

    def reconstruct_zone_timeline(
        self,
        zone_id: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> ZoneTimeline:
        """Reconstruct spatial occupancy timeline for a zone."""
        occupancy_edges = self._graph_store.get_zone_occupancy(
            zone_id=zone_id, start_ns=start_ns, end_ns=end_ns
        )
        unique_identities = list({edge["global_id"] for edge in occupancy_edges})

        min_ts = start_ns if start_ns is not None else 0
        max_ts = end_ns if end_ns is not None else 0

        return ZoneTimeline(
            zone_id=zone_id,
            start_ns=min_ts,
            end_ns=max_ts,
            occupancy_segments=occupancy_edges,
            unique_identities=unique_identities,
        )

    def _create_visit_segment(
        self, global_id: str, camera_id: str, group: list[dict], is_complete: bool
    ) -> VisitSegment:
        import json
        first_ts = group[0]["timestamp_ns"]
        last_ts = group[-1]["timestamp_ns"]
        dwell_s = (last_ts - first_ts) / 1e9 if len(group) > 1 else 0.0

        meta_str = group[0].get("metadata")
        meta_dict = json.loads(meta_str) if isinstance(meta_str, str) else (meta_str or {})

        return VisitSegment(
            global_id=global_id,
            camera_id=camera_id,
            zone_id=meta_dict.get("zone_id"),
            start_ns=first_ts,
            end_ns=last_ts if is_complete else None,
            dwell_s=dwell_s,
            observation_count=len(group),
            is_complete=is_complete,
        )
