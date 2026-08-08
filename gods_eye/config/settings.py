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
    reid_model: str = "osnet_x1_0"
    reid_weights_path: Optional[str] = None
    reid_match_threshold: float = 0.75
    reid_top_k: int = 5
    reid_ema_alpha: float = 0.9
    reid_extract_interval: int = 5     # frames between OSNet feature extractions for cached tracks
    identity_lost_timeout_s: int = 300
    identity_ttl_s: int = 86400
    embedding_history_len: int = 50
    gallery_max_size: int = 10_000

    # ── Multi-Camera / Phase 3 ─────────────────────────────────────────
    camera_config_path: Optional[str] = None   # Path to cameras YAML topology
    cross_camera_prior_weight: float = 0.1     # Weight for transition prior scoring

    # ── Environmental Intelligence / Phase 3.5 (§16) ───────────────────
    warmup_bg_frames: int = 10_000
    warmup_bg_confidence: float = 0.85
    warmup_occupancy_samples: int = 14
    occupancy_anomaly_sigma: float = 3.0
    anomaly_min_confidence: float = 0.70
    bg_learning_rate: float = 0.01
    force_degraded_mode: bool = False

    # ── Temporal Memory / Phase 4 ──────────────────────────────────────
    event_store_path: str = "data/events.db"
    graph_store_path: str = "data/graph.db"
    event_log_ttl_s: int = 604_800             # 7 days
    temporal_batch_size: int = 100

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
            reid_model=os.environ.get("GODS_EYE_REID_MODEL", "osnet_x1_0"),
            reid_weights_path=os.environ.get("GODS_EYE_REID_WEIGHTS_PATH"),
            reid_match_threshold=_env_float("GODS_EYE_REID_MATCH_THRESHOLD", 0.75),
            reid_top_k=_env_int("GODS_EYE_REID_TOP_K", 5),
            reid_ema_alpha=_env_float("GODS_EYE_REID_EMA_ALPHA", 0.9),
            reid_extract_interval=_env_int("GODS_EYE_REID_EXTRACT_INTERVAL", 5),
            identity_lost_timeout_s=_env_int("GODS_EYE_IDENTITY_LOST_TIMEOUT_S", 300),
            identity_ttl_s=_env_int("GODS_EYE_IDENTITY_TTL_S", 86400),
            embedding_history_len=_env_int("GODS_EYE_EMBEDDING_HISTORY_LEN", 50),
            gallery_max_size=_env_int("GODS_EYE_GALLERY_MAX_SIZE", 10_000),
            # Multi-Camera / Phase 3
            camera_config_path=os.environ.get("GODS_EYE_CAMERA_CONFIG_PATH"),
            cross_camera_prior_weight=_env_float(
                "GODS_EYE_CROSS_CAMERA_PRIOR_WEIGHT", 0.1,
            ),
            # Environmental Intelligence / Phase 3.5
            warmup_bg_frames=_env_int("GODS_EYE_WARMUP_BG_FRAMES", 10_000),
            warmup_bg_confidence=_env_float("GODS_EYE_WARMUP_BG_CONFIDENCE", 0.85),
            warmup_occupancy_samples=_env_int(
                "GODS_EYE_WARMUP_OCCUPANCY_SAMPLES", 14
            ),
            occupancy_anomaly_sigma=_env_float(
                "GODS_EYE_OCCUPANCY_ANOMALY_SIGMA", 3.0
            ),
            anomaly_min_confidence=_env_float(
                "GODS_EYE_ANOMALY_MIN_CONFIDENCE", 0.70
            ),
            bg_learning_rate=_env_float("GODS_EYE_BG_LEARNING_RATE", 0.01),
            force_degraded_mode=_env_bool("GODS_EYE_FORCE_DEGRADED_MODE", False),
            # Temporal Memory / Phase 4
            event_store_path=os.environ.get("GODS_EYE_EVENT_STORE_PATH", "data/events.db"),
            event_log_ttl_s=_env_int("GODS_EYE_EVENT_LOG_TTL_S", 604_800),
            temporal_batch_size=_env_int("GODS_EYE_TEMPORAL_BATCH_SIZE", 100),
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
