"""Lighting Condition Classifier — Phase 3.5 (§16, ADR-007).

Histogram-based rule classifier + time-of-day lookup to categorize frame
ambient lighting conditions into 7 states. Provides environmental context
for occupancy baseline selection and anomaly scoring.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np

from gods_eye.schemas.environment import LightingCondition


class LightingClassifier:
    """Fast, explainable lighting classifier using intensity histograms and time-of-day.

    Stateless rule-based classifier per ADR-007. Execution latency < 0.5ms.
    """

    def classify(
        self, frame: np.ndarray, timestamp_ns: int
    ) -> LightingCondition:
        """Classify ambient lighting condition of a video frame.

        Args:
            frame: Raw BGR or grayscale image array (HxWxC or HxW).
            timestamp_ns: Capture timestamp in Unix nanoseconds.

        Returns:
            ``LightingCondition`` enum classification.
        """
        # Convert to grayscale if BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = (
                0.299 * frame[:, :, 2]
                + 0.587 * frame[:, :, 1]
                + 0.114 * frame[:, :, 0]
            ).astype(np.uint8)
        else:
            gray = frame

        # Compute mean intensity, 95th percentile, and standard deviation
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        p95_val = float(np.percentile(gray, 95))

        # Time-of-day lookup
        dt = datetime.datetime.fromtimestamp(
            timestamp_ns / 1e9, tz=datetime.timezone.utc
        )
        hour = dt.hour

        is_night_time = hour < 6 or hour >= 20
        is_dusk_dawn = (6 <= hour < 8) or (18 <= hour < 20)

        # ── Decision Rules (ADR-007) ──────────────────────────────────────────
        if is_night_time:
            if mean_val < 30.0 and p95_val < 70.0:
                return LightingCondition.NIGHT_DARK
            if mean_val >= 30.0 or std_val > 40.0:
                return LightingCondition.NIGHT_LIT
            return LightingCondition.NIGHT_DARK

        if is_dusk_dawn:
            if mean_val < 50.0:
                return LightingCondition.DUSK_DAWN
            if std_val < 25.0:
                return LightingCondition.DAY_OVERCAST

        # Daytime rules (08:00 to 18:00)
        if mean_val > 180.0 and p95_val > 240.0:
            return LightingCondition.DAY_BRIGHT

        if mean_val < 80.0 and std_val < 30.0:
            return LightingCondition.DAY_OVERCAST

        if std_val < 18.0:
            # Low contrast uniform illumination (indoor fluorescent/LED)
            return LightingCondition.ARTIFICIAL

        return LightingCondition.DAY_NORMAL
