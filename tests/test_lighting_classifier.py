"""Unit tests for lighting classifier — Phase 3.5 Component 6."""

from __future__ import annotations

import datetime

import numpy as np

from gods_eye.environmental.lighting_classifier import LightingClassifier
from gods_eye.schemas.environment import LightingCondition


class TestLightingClassifier:

    def test_classify_day_bright(self) -> None:
        classifier = LightingClassifier()
        # High intensity image at 12:00 UTC
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 245
        dt = datetime.datetime(2026, 8, 7, 12, 0, tzinfo=datetime.timezone.utc)
        timestamp_ns = int(dt.timestamp() * 1e9)

        condition = classifier.classify(frame, timestamp_ns)
        assert condition is LightingCondition.DAY_BRIGHT

    def test_classify_night_dark(self) -> None:
        classifier = LightingClassifier()
        # Dark image at 02:00 UTC
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 10
        dt = datetime.datetime(2026, 8, 7, 2, 0, tzinfo=datetime.timezone.utc)
        timestamp_ns = int(dt.timestamp() * 1e9)

        condition = classifier.classify(frame, timestamp_ns)
        assert condition is LightingCondition.NIGHT_DARK

    def test_classify_day_overcast(self) -> None:
        classifier = LightingClassifier()
        # Medium low intensity image during daytime
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        dt = datetime.datetime(2026, 8, 7, 14, 0, tzinfo=datetime.timezone.utc)
        timestamp_ns = int(dt.timestamp() * 1e9)

        condition = classifier.classify(frame, timestamp_ns)
        assert condition is LightingCondition.DAY_OVERCAST
