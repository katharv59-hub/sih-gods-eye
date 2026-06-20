"""Centralized settings — all tunables from the spec live here.

Values are resolved from environment variables with sensible defaults.
Threshold changes require a documented re-benchmark (spec §14).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


def _env_tuple_str(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return tuple(s.strip() for s in raw.split(",") if s.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable configuration snapshot.

    Construct via ``Settings.from_env()`` to resolve from environment.
    """

    # ── Execution environment ──────────────────────────────────────────
    cpu_only: bool = False

    # ── Queue sizes (§8 Concurrency Model) ─────────────────────────────
    frame_queue_size: int = 30
    detection_queue_size: int = 30
    track_queue_size: int = 30
    event_queue_size: int = 100

    # ── Detection ──────────────────────────────────────────────────────
    detection_model: str = "yolov8n.pt"
    detection_confidence_threshold: float = 0.25
    detection_imgsz: int = 640
    detection_target_classes: tuple[str, ...] = ("person",)

    # ── Tracking ───────────────────────────────────────────────────────
    track_lost_timeout: int = 30       # frames before ACTIVE → LOST
    track_dead_timeout: int = 120      # frames before LOST → DEAD

    # ── Re-ID (§14 Configuration Keys) ─────────────────────────────────
    reid_match_threshold: float = 0.75
    reid_ema_alpha: float = 0.9
    identity_lost_timeout_s: int = 300
    identity_ttl_s: int = 86400
    embedding_history_len: int = 50
    gallery_max_size: int = 10_000

    # ── Observability (§7) ─────────────────────────────────────────────
    metrics_port: int = 9090
    log_level: str = "INFO"

    # ── Alerting thresholds (§7) ───────────────────────────────────────
    alert_fps_min: float = 20.0
    alert_fps_window_s: float = 5.0
    alert_queue_depth_max: int = 50
    alert_frame_drop_rate_max: float = 0.05
    alert_gpu_memory_max_pct: float = 0.90

    # ── Security (§9) ─────────────────────────────────────────────────
    api_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> Settings:
        """Resolve ALL settings from environment variables."""
        return cls(
            cpu_only=_env_bool("GODS_EYE_CPU_ONLY", False),
            # Queue sizes
            frame_queue_size=_env_int("GODS_EYE_FRAME_QUEUE_SIZE", 30),
            detection_queue_size=_env_int("GODS_EYE_DETECTION_QUEUE_SIZE", 30),
            track_queue_size=_env_int("GODS_EYE_TRACK_QUEUE_SIZE", 30),
            event_queue_size=_env_int("GODS_EYE_EVENT_QUEUE_SIZE", 100),
            # Detection
            detection_model=os.environ.get("GODS_EYE_DETECTION_MODEL", "yolov8n.pt"),
            detection_confidence_threshold=_env_float(
                "GODS_EYE_DETECTION_CONFIDENCE", 0.25
            ),
            detection_imgsz=_env_int("GODS_EYE_DETECTION_IMGSZ", 640),
            detection_target_classes=_env_tuple_str(
                "GODS_EYE_DETECTION_TARGET_CLASSES", ("person",)
            ),
            # Tracking
            track_lost_timeout=_env_int("GODS_EYE_TRACK_LOST_TIMEOUT", 30),
            track_dead_timeout=_env_int("GODS_EYE_TRACK_DEAD_TIMEOUT", 120),
            # Re-ID
            reid_match_threshold=_env_float("GODS_EYE_REID_MATCH_THRESHOLD", 0.75),
            reid_ema_alpha=_env_float("GODS_EYE_REID_EMA_ALPHA", 0.9),
            identity_lost_timeout_s=_env_int("GODS_EYE_IDENTITY_LOST_TIMEOUT_S", 300),
            identity_ttl_s=_env_int("GODS_EYE_IDENTITY_TTL_S", 86400),
            embedding_history_len=_env_int("GODS_EYE_EMBEDDING_HISTORY_LEN", 50),
            gallery_max_size=_env_int("GODS_EYE_GALLERY_MAX_SIZE", 10_000),
            # Observability
            metrics_port=_env_int("GODS_EYE_METRICS_PORT", 9090),
            log_level=os.environ.get("GODS_EYE_LOG_LEVEL", "INFO"),
            # Alerting
            alert_fps_min=_env_float("GODS_EYE_ALERT_FPS_MIN", 20.0),
            alert_fps_window_s=_env_float("GODS_EYE_ALERT_FPS_WINDOW_S", 5.0),
            alert_queue_depth_max=_env_int("GODS_EYE_ALERT_QUEUE_DEPTH_MAX", 50),
            alert_frame_drop_rate_max=_env_float(
                "GODS_EYE_ALERT_FRAME_DROP_RATE_MAX", 0.05
            ),
            alert_gpu_memory_max_pct=_env_float(
                "GODS_EYE_ALERT_GPU_MEMORY_MAX_PCT", 0.90
            ),
            # Security
            api_token=os.environ.get("GODS_EYE_API_TOKEN"),
        )
