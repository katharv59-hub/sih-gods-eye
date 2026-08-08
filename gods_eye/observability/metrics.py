"""Prometheus metrics registry — §7 Observability Contract.

All 8 required metrics are defined here plus identity-layer metrics.
Exposed via HTTP endpoint on localhost:METRICS_PORT.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, REGISTRY, start_http_server


class MetricsRegistry:
    """Singleton-style registry for all God's Eye Prometheus metrics.

    Instantiate once at startup. Pass to subsystems that need to record metrics.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else REGISTRY

        # ── Required metrics — All subsystems (§7) ─────────────────────
        self.fps = Gauge(
            "gods_eye_fps",
            "Current frames per second",
            ["subsystem", "camera_id"],
            registry=reg,
        )
        self.latency_ms = Histogram(
            "gods_eye_latency_ms",
            "Processing latency in milliseconds",
            ["subsystem", "camera_id"],
            buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
            registry=reg,
        )
        self.cpu_utilization = Gauge(
            "gods_eye_cpu_utilization",
            "CPU utilization percentage",
            ["subsystem"],
            registry=reg,
        )
        self.ram_bytes = Gauge(
            "gods_eye_ram_bytes",
            "RAM usage in bytes",
            ["subsystem"],
            registry=reg,
        )
        self.gpu_utilization = Gauge(
            "gods_eye_gpu_utilization",
            "GPU utilization percentage",
            ["subsystem"],
            registry=reg,
        )
        self.vram_bytes = Gauge(
            "gods_eye_vram_bytes",
            "VRAM usage in bytes",
            ["subsystem"],
            registry=reg,
        )
        self.queue_depth = Gauge(
            "gods_eye_queue_depth",
            "Current queue depth",
            ["queue_name"],
            registry=reg,
        )
        self.frame_drops_total = Counter(
            "gods_eye_frame_drops_total",
            "Total frames dropped",
            ["camera_id", "reason"],
            registry=reg,
        )

        # ── Identity layer metrics (§7) ───────────────────────────────
        self.identities_active = Gauge(
            "gods_eye_identities_active",
            "Number of active identities",
            registry=reg,
        )
        self.identities_lost = Gauge(
            "gods_eye_identities_lost",
            "Number of lost identities",
            registry=reg,
        )
        self.identities_purged_total = Counter(
            "gods_eye_identities_purged_total",
            "Total identities purged",
            registry=reg,
        )
        self.reid_match_confidence = Histogram(
            "gods_eye_reid_match_confidence",
            "Re-ID match confidence scores",
            ["camera_id"],
            buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=reg,
        )
        self.reid_false_merge_total = Counter(
            "gods_eye_reid_false_merge_total",
            "Total false identity merges",
            registry=reg,
        )
        self.gallery_size = Gauge(
            "gods_eye_gallery_size",
            "Current identity gallery size",
            registry=reg,
        )

        # ── Temporal memory boundary metrics (Phase 4.1) ─────────────
        self.temporal_queue_drops_total = Counter(
            "gods_eye_temporal_queue_drops_total",
            "Total temporal observations rejected at admission control",
            ["priority"],
            registry=reg,
        )
        self.temporal_queue_watermark_active = Gauge(
            "gods_eye_temporal_queue_watermark_active",
            "Gauge set to 1 when temporal queue depth >= 80% high watermark",
            registry=reg,
        )

        # ── Event storage metrics (Phase 4.2) ─────────────────────────
        self.events_written_total = Counter(
            "gods_eye_events_written_total",
            "Total events persisted to event store",
            ["event_type"],
            registry=reg,
        )
        self.event_write_latency_ms = Histogram(
            "gods_eye_event_write_latency_ms",
            "Event store append latency in milliseconds",
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100),
            registry=reg,
        )
        self.event_store_errors_total = Counter(
            "gods_eye_event_store_errors_total",
            "Total event store operation failures",
            ["operation"],
            registry=reg,
        )
        self.events_purged_total = Counter(
            "gods_eye_events_purged_total",
            "Total events purged from store by retention policy",
            registry=reg,
        )

        # ── Spatiotemporal graph metrics (Phase 4.3) ───────────────────
        self.graph_writes_total = Counter(
            "gods_eye_graph_writes_total",
            "Total graph records persisted to graph store",
            ["record_type"],
            registry=reg,
        )
        self.graph_write_latency_ms = Histogram(
            "gods_eye_graph_write_latency_ms",
            "Graph store append latency in milliseconds",
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100),
            registry=reg,
        )
        self.graph_errors_total = Counter(
            "gods_eye_graph_errors_total",
            "Total graph store operation failures",
            ["operation"],
            registry=reg,
        )
        self.graph_records_purged_total = Counter(
            "gods_eye_graph_records_purged_total",
            "Total graph records purged from store by retention policy",
            registry=reg,
        )

    def start_server(self, port: int = 9090) -> None:
        """Start the Prometheus metrics HTTP endpoint."""
        start_http_server(port)
