"""Gallery operation latency benchmark — Phase 2, Task 9.

Measures search, insert, update, and embedding matrix rebuild latency
at gallery sizes: 100, 1000, 5000, 10000.

Gate criteria (§14):
    - Gallery search p99 ≤ 5ms at 10,000 identities

Output:
    benchmarks/results/gallery_latency.json
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import configure_logging
from gods_eye.reid.identity import create_identity, LifecycleState
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.matcher import Matcher


@dataclass
class LatencyStats:
    """Latency statistics for a single operation at a given gallery size."""

    gallery_size: int
    operation: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    timestamp: str
    gate_passed: bool
    search_p99_at_10k_ms: float
    gate_threshold_ms: float
    stats: list[LatencyStats] = field(default_factory=list)


def _random_embedding(dim: int = 512) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _compute_stats(
    timings_ns: list[float], gallery_size: int, operation: str
) -> LatencyStats:
    """Compute percentile statistics from nanosecond timings."""
    timings_ms = [t * 1000 for t in timings_ns]  # s -> ms
    timings_ms.sort()
    n = len(timings_ms)
    return LatencyStats(
        gallery_size=gallery_size,
        operation=operation,
        count=n,
        mean_ms=round(statistics.mean(timings_ms), 4),
        p50_ms=round(timings_ms[n // 2], 4),
        p95_ms=round(timings_ms[int(n * 0.95)], 4),
        p99_ms=round(timings_ms[int(n * 0.99)], 4),
        min_ms=round(timings_ms[0], 4),
        max_ms=round(timings_ms[-1], 4),
    )


def bench_search(
    gallery: IdentityGallery,
    matcher: Matcher,
    gallery_size: int,
    num_queries: int = 500,
) -> LatencyStats:
    """Benchmark gallery search (get_embedding_matrix + matcher.match)."""
    timings: list[float] = []

    for _ in range(num_queries):
        query = _random_embedding()
        t0 = time.perf_counter()
        matrix, ids = gallery.get_embedding_matrix()
        if matrix.shape[0] > 0:
            matcher.match(query, matrix, ids)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    return _compute_stats(timings, gallery_size, "search")


def bench_insert(
    lifecycle: LifecycleManager,
    gallery_size: int,
    num_inserts: int = 200,
) -> tuple[IdentityGallery, LatencyStats]:
    """Benchmark gallery insert, returning the populated gallery."""
    gallery = IdentityGallery()
    timings: list[float] = []

    # Pre-populate to target size - num_inserts
    pre_fill = max(0, gallery_size - num_inserts)
    for i in range(pre_fill):
        emb = _random_embedding()
        ident = create_identity(emb, 1000 + i)
        active = lifecycle.activate(ident, 1000 + i)
        gallery.insert(active)

    # Benchmark the last num_inserts
    for i in range(num_inserts):
        emb = _random_embedding()
        ident = create_identity(emb, 2000000 + i)
        active = lifecycle.activate(ident, 2000000 + i)

        t0 = time.perf_counter()
        gallery.insert(active)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    return gallery, _compute_stats(timings, gallery_size, "insert")


def bench_update(
    gallery: IdentityGallery,
    lifecycle: LifecycleManager,
    gallery_size: int,
    num_updates: int = 500,
) -> LatencyStats:
    """Benchmark gallery update (EMA-like refresh + update)."""
    ids = gallery.ids()
    timings: list[float] = []

    for i in range(num_updates):
        gid = ids[i % len(ids)]
        identity = gallery.get(gid)
        if identity is None:
            continue

        new_emb = _random_embedding()
        t0 = time.perf_counter()
        refreshed = lifecycle.refresh(
            identity, 5000000 + i, embedding=new_emb,
        )
        gallery.update(refreshed)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    return _compute_stats(timings, gallery_size, "update")


def bench_matrix_rebuild(
    gallery: IdentityGallery,
    gallery_size: int,
    num_rebuilds: int = 100,
) -> LatencyStats:
    """Benchmark embedding matrix cache rebuild."""
    timings: list[float] = []

    for _ in range(num_rebuilds):
        # Force dirty flag by doing a trivial update
        gallery._dirty = True  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        gallery.get_embedding_matrix()
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    return _compute_stats(timings, gallery_size, "matrix_rebuild")


def main() -> None:
    """Run the full gallery latency benchmark."""
    print("=" * 60)
    print("Gallery Latency Benchmark - Phase 2, Task 9")
    print("=" * 60)

    # Suppress debug logging to avoid I/O overhead in timing measurements
    configure_logging("WARNING")

    settings = Settings()
    lifecycle = LifecycleManager(settings)
    matcher = Matcher(settings)

    SIZES = [100, 1_000, 5_000, 10_000]
    GATE_THRESHOLD_MS = 5.0
    all_stats: list[LatencyStats] = []
    search_p99_at_10k = 0.0

    for size in SIZES:
        print(f"\n--- Gallery Size: {size:,} ---")

        # Insert benchmark (also populates gallery)
        gallery, insert_stats = bench_insert(lifecycle, size)
        all_stats.append(insert_stats)
        print(f"  INSERT  p99={insert_stats.p99_ms:.4f}ms  "
              f"mean={insert_stats.mean_ms:.4f}ms")

        # Search benchmark
        search_stats = bench_search(gallery, matcher, size)
        all_stats.append(search_stats)
        print(f"  SEARCH  p99={search_stats.p99_ms:.4f}ms  "
              f"mean={search_stats.mean_ms:.4f}ms")

        if size == 10_000:
            search_p99_at_10k = search_stats.p99_ms

        # Update benchmark
        update_stats = bench_update(gallery, lifecycle, size)
        all_stats.append(update_stats)
        print(f"  UPDATE  p99={update_stats.p99_ms:.4f}ms  "
              f"mean={update_stats.mean_ms:.4f}ms")

        # Matrix rebuild benchmark
        rebuild_stats = bench_matrix_rebuild(gallery, size)
        all_stats.append(rebuild_stats)
        print(f"  REBUILD p99={rebuild_stats.p99_ms:.4f}ms  "
              f"mean={rebuild_stats.mean_ms:.4f}ms")

    # Gate evaluation
    gate_passed = search_p99_at_10k <= GATE_THRESHOLD_MS

    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print(f"  Search p99 at 10K: {search_p99_at_10k:.4f} ms")
    print(f"  Threshold:         {GATE_THRESHOLD_MS:.1f} ms")
    print(f"  Result:            {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 60)

    # Save results
    result = BenchmarkResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        gate_passed=gate_passed,
        search_p99_at_10k_ms=search_p99_at_10k,
        gate_threshold_ms=GATE_THRESHOLD_MS,
        stats=[asdict(s) for s in all_stats],  # type: ignore[arg-type]
    )

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gallery_latency.json"
    output_file.write_text(json.dumps(asdict(result), indent=2))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
