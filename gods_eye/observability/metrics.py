"""Prometheus metrics registry — §7 Observability Contract.

All 8 required metrics are defined here plus identity-layer metrics.
Exposed via HTTP endpoint on localhost:METRICS_PORT.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server


class MetricsRegistry:
    """Singleton-style registry for all God's Eye Prometheus metrics.

    Instantiate once at startup. Pass to subsystems that need to record metrics.
    """

    def __init__(self) -> None:
        # ── Required metrics — All subsystems (§7) ─────────────────────
        self.fps = Gauge(
            "gods_eye_fps",
            "Current frames per second",
            ["subsystem", "camera_id"],
        )
        self.latency_ms = Histogram(
            "gods_eye_latency_ms",
            "Processing latency in milliseconds",
            ["subsystem", "camera_id"],
            buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
        )
        self.cpu_utilization = Gauge(
            "gods_eye_cpu_utilization",
            "CPU utilization percentage",
            ["subsystem"],
        )
        self.ram_bytes = Gauge(
            "gods_eye_ram_bytes",
            "RAM usage in bytes",
            ["subsystem"],
        )
        self.gpu_utilization = Gauge(
            "gods_eye_gpu_utilization",
            "GPU utilization percentage",
            ["subsystem"],
        )
        self.vram_bytes = Gauge(
            "gods_eye_vram_bytes",
            "VRAM usage in bytes",
            ["subsystem"],
        )
        self.queue_depth = Gauge(
            "gods_eye_queue_depth",
            "Current queue depth",
            ["queue_name"],
        )
        self.frame_drops_total = Counter(
            "gods_eye_frame_drops_total",
            "Total frames dropped",
            ["camera_id", "reason"],
        )

        # ── Identity layer metrics (§7) ───────────────────────────────
        self.identities_active = Gauge(
            "gods_eye_identities_active",
            "Number of active identities",
        )
        self.identities_lost = Gauge(
            "gods_eye_identities_lost",
            "Number of lost identities",
        )
        self.identities_purged_total = Counter(
            "gods_eye_identities_purged_total",
            "Total identities purged",
        )
        self.reid_match_confidence = Histogram(
            "gods_eye_reid_match_confidence",
            "Re-ID match confidence scores",
            ["camera_id"],
            buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
        )
        self.reid_false_merge_total = Counter(
            "gods_eye_reid_false_merge_total",
            "Total false identity merges",
        )
        self.gallery_size = Gauge(
            "gods_eye_gallery_size",
            "Current identity gallery size",
        )

    def start_server(self, port: int = 9090) -> None:
        """Start the Prometheus metrics HTTP endpoint."""
        start_http_server(port)
