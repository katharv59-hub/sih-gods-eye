"""Unit tests for Spatial Intelligence Engines (Phase 9 — SIH 26187).

Tests:
- FenceEngine (geofence, crossing, is_restricted_zone wire)
- DwellEngine (dwell sigma calculation, thresholding)
- NightMovementEngine (lighting condition correlation, 0.80 confidence floor)
"""

from __future__ import annotations

import pytest

from gods_eye.schemas.camera import Zone
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.environment import LightingCondition
from gods_eye.zones.dwell_engine import DwellEngine
from gods_eye.zones.fence_engine import FenceEngine
from gods_eye.zones.night_rules import NightMovementEngine


class TestFenceEngine:
    def test_point_in_polygon(self) -> None:
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert FenceEngine._point_in_polygon((0.5, 0.5), poly) is True
        assert FenceEngine._point_in_polygon((1.5, 0.5), poly) is False
        assert FenceEngine._point_in_polygon((-0.1, 0.5), poly) is False

    def test_check_subject_fence_crossing(self) -> None:
        engine = FenceEngine()
        zone = Zone(
            zone_id="zone_transit",
            camera_id="cam-01",
            display_name="Corridor",
            polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            zone_type="transit",
            expected_dwell_s=(0.0, 60.0),
        )

        bbox = BoundingBox(x1=400, y1=400, x2=600, y2=600)  # center (500, 500) -> (0.5, 0.5)
        events = engine.check_subject(
            subject_ref="person_01",
            bbox=bbox,
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            source_resolution=(1000, 1000),
            zones=[zone],
            confidence=0.95,
        )

        assert len(events) == 1
        assert events[0].event_type == "VIRTUAL_FENCE_CROSSED"
        assert events[0].is_restricted_zone is False
        assert events[0].zone_id == "zone_transit"

    def test_check_subject_restricted_zone_wire(self) -> None:
        engine = FenceEngine()
        restricted_zone = Zone(
            zone_id="zone_vault",
            camera_id="cam-01",
            display_name="Vault Periphery",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
            zone_type="restricted",
            expected_dwell_s=(0.0, 0.0),
        )

        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)  # center (150, 150) -> (0.15, 0.15)
        events = engine.check_subject(
            subject_ref="person_02",
            bbox=bbox,
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            source_resolution=(1000, 1000),
            zones=[restricted_zone],
            confidence=0.92,
        )

        assert len(events) == 1
        assert events[0].event_type == "RESTRICTED_ZONE_INTRUSION"
        assert events[0].is_restricted_zone is True

        # Check the evidence payload wire
        payload = engine.build_evidence_payload(events[0])
        assert payload["is_restricted_zone"] is True
        assert payload["zone_id"] == "zone_vault"


class TestDwellEngine:
    def test_record_presence_under_threshold(self) -> None:
        engine = DwellEngine(default_max_dwell_s=60.0)
        zone = Zone(
            zone_id="zone_atm",
            camera_id="cam-01",
            display_name="ATM Lobby",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            zone_type="dwell",
            expected_dwell_s=(10.0, 60.0),
        )

        # Initial entry
        ev1 = engine.record_zone_presence(
            subject_ref="person_10",
            zone_id="zone_atm",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            confidence=0.9,
            zone=zone,
        )
        assert ev1 is None

        # 30 seconds later (under 60s max)
        ev2 = engine.record_zone_presence(
            subject_ref="person_10",
            zone_id="zone_atm",
            camera_id="cam-01",
            timestamp_ns=31_000_000_000,
            confidence=0.9,
            zone=zone,
        )
        assert ev2 is None

    def test_record_presence_exceeds_threshold(self) -> None:
        engine = DwellEngine(default_max_dwell_s=60.0)
        zone = Zone(
            zone_id="zone_atm",
            camera_id="cam-01",
            display_name="ATM Lobby",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            zone_type="dwell",
            expected_dwell_s=(10.0, 60.0),
        )

        # Entry at t = 0
        engine.record_zone_presence(
            subject_ref="person_10",
            zone_id="zone_atm",
            camera_id="cam-01",
            timestamp_ns=0,
            confidence=0.9,
            zone=zone,
        )

        # 120 seconds later (well above 60s)
        ev = engine.record_zone_presence(
            subject_ref="person_10",
            zone_id="zone_atm",
            camera_id="cam-01",
            timestamp_ns=120_000_000_000,
            confidence=0.95,
            zone=zone,
        )

        assert ev is not None
        assert ev.dwell_seconds == 120.0
        assert ev.zone_dwell_sigma > 0.0

        # Check evidence payload wire
        payload = engine.build_evidence_payload(ev)
        assert payload["zone_dwell_sigma"] == ev.zone_dwell_sigma
        assert payload["dwell_seconds"] == 120.0


class TestNightMovementEngine:
    def test_night_movement_daytime_rejected(self) -> None:
        engine = NightMovementEngine()
        ev = engine.evaluate(
            subject_ref="person_01",
            zone_id="zone_vault",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            lighting_condition=LightingCondition.DAY_NORMAL,
            associated_event_type="RESTRICTED_ZONE_INTRUSION",
            event_confidence=0.95,
        )
        assert ev is None

    def test_night_movement_low_confidence_rejected(self) -> None:
        engine = NightMovementEngine(min_confidence=0.80)
        # Night lighting but confidence is below 0.80
        ev = engine.evaluate(
            subject_ref="person_01",
            zone_id="zone_vault",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            lighting_condition=LightingCondition.NIGHT_DARK,
            associated_event_type="RESTRICTED_ZONE_INTRUSION",
            event_confidence=0.75,
        )
        assert ev is None

    def test_night_movement_valid(self) -> None:
        engine = NightMovementEngine(min_confidence=0.80)
        ev = engine.evaluate(
            subject_ref="person_01",
            zone_id="zone_vault",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            lighting_condition=LightingCondition.NIGHT_DARK,
            associated_event_type="RESTRICTED_ZONE_INTRUSION",
            event_confidence=0.92,
        )
        assert ev is not None
        assert ev.lighting_condition == LightingCondition.NIGHT_DARK
        assert ev.associated_event_type == "RESTRICTED_ZONE_INTRUSION"
        assert ev.confidence >= 0.80
