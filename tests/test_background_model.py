"""Unit tests for background modeler — Phase 3.5 Component 6."""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.environmental.background_model import BackgroundModeler
from gods_eye.schemas.environment import BackgroundModel


class TestBackgroundModeler:

    def test_background_modeler_initialization(self) -> None:
        settings = Settings(cpu_only=True)
        modeler = BackgroundModeler("cam_test", settings)
        assert modeler._camera_id == "cam_test"

    def test_apply_generates_mask_and_model(self) -> None:
        settings = Settings(cpu_only=True, warmup_bg_frames=10)
        modeler = BackgroundModeler("cam_test", settings)

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        mask, model = modeler.apply(frame, 1_000_000_000)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)
        assert isinstance(model, BackgroundModel)
        assert model.camera_id == "cam_test"
        assert model.frame_count == 1
        assert model.confidence > 0.0

    def test_confidence_computation_and_warmup(self) -> None:
        settings = Settings(
            cpu_only=True, warmup_bg_frames=5, warmup_bg_confidence=0.60
        )
        modeler = BackgroundModeler("cam_test", settings)

        frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
        for i in range(5):
            mask, model = modeler.apply(frame, (i + 1) * 1_000_000_000)

        assert model.frame_count == 5
        assert model.confidence >= 0.60
        assert model.warm_up_complete is True

    def test_reset_increments_version(self) -> None:
        settings = Settings(cpu_only=True)
        modeler = BackgroundModeler("cam_test", settings)

        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        _, model1 = modeler.apply(frame, 1_000_000_000)
        assert model1.version == 1

        modeler.reset()
        _, model2 = modeler.apply(frame, 2_000_000_000)
        assert model2.version == 2
        assert model2.frame_count == 1
