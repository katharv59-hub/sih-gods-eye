"""Phase 6.7 Behavioral Intelligence Performance & Accuracy Benchmark.

Measures:
1. Engine latency (p50, p95, p99) for TrajectoryClusteringEngine & MarkovBehavioralPredictor.
2. ToolDispatcher latency for Tool 9 & Tool 10 across dataset scales (100, 1,000, 10,000 transitions).
3. Complete ReasoningEngine & Planner latency.
4. Deterministic held-out prediction accuracy (Top-1, Top-3, Coverage).
5. Outputs machine-readable JSON results.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from gods_eye.behavioral.clustering import TrajectoryClusteringEngine
from gods_eye.behavioral.extractor import TrajectorySequence, TrajectorySequenceExtractor
from gods_eye.behavioral.predictor import MarkovBehavioralPredictor
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.reasoning.engine import ReasoningEngine
from gods_eye.reasoning.planner import NLQPlanner, RuleBasedPlanner
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.reasoning import ToolCall, UserQuery


def run_latency_benchmark(fn, warmup: int = 20, iterations: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    durations_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    durations_ms.sort()
    n = len(durations_ms)
    p50 = durations_ms[int(n * 0.50)]
    p95 = durations_ms[int(n * 0.95)]
    p99 = durations_ms[int(n * 0.99)]

    return {"p50_ms": round(p50, 4), "p95_ms": round(p95, 4), "p99_ms": round(p99, 4)}


def benchmark_held_out_accuracy(num_transitions: int = 1000) -> dict[str, Any]:
    gs = SQLiteGraphStore(":memory:")
    cams = [f"cam_{i:02d}" for i in range(1, 11)]
    for c in cams:
        gs.upsert_camera_node(c)

    # Deterministic transition generation: cam_i -> cam_{i+1} (80%), cam_i -> cam_{i+2} (20%)
    split_ts = 500_000_000_000
    for i in range(num_transitions):
        src_idx = i % 8
        dst_idx = (src_idx + 1) if (i % 10 < 8) else (src_idx + 2)
        src = cams[src_idx]
        dst = cams[dst_idx % 10]
        ts = (i + 1) * 1_000_000
        gid = f"gid_{i % 50}"
        gs.add_transition_edge(f"tr_{i}", gid, src, dst, ts, 1.0)

    # Train range: ts <= 500 * 1_000_000 (first 500 transitions)
    # Test range: ts > 500 * 1_000_000 (last 500 transitions)
    split_ts = 500 * 1_000_000
    predictor = MarkovBehavioralPredictor()
    predictor.fit_from_graph_store(gs, end_ns=split_ts)

    test_transitions = gs.get_camera_transitions(start_ns=split_ts + 1)
    evaluated = 0
    correct_top1 = 0
    correct_top3 = 0

    for tr in test_transitions:
        src = tr["from_camera_id"]
        actual_dst = tr["to_camera_id"]
        res = predictor.predict_next(src, timestamp_ns=tr["timestamp_ns"], top_k=3)
        if res.predictions:
            evaluated += 1
            top1 = res.predictions[0].camera_id
            top3 = [p.camera_id for p in res.predictions]
            if top1 == actual_dst:
                correct_top1 += 1
            if actual_dst in top3:
                correct_top3 += 1

    total_test = len(test_transitions)
    coverage = evaluated / total_test if total_test > 0 else 0.0
    acc_top1 = correct_top1 / evaluated if evaluated > 0 else 0.0
    acc_top3 = correct_top3 / evaluated if evaluated > 0 else 0.0

    gs.close()
    return {
        "num_transitions": num_transitions,
        "total_test_cases": total_test,
        "evaluated_predictions": evaluated,
        "coverage": round(coverage, 4),
        "top1_accuracy": round(acc_top1, 4),
        "top3_accuracy": round(acc_top3, 4),
        "dataset_type": "Deterministic Synthetic Transition Graph",
    }


def benchmark_scale(num_transitions: int) -> dict[str, Any]:
    es = SQLiteEventStore(":memory:")
    gs = SQLiteGraphStore(":memory:")
    cams = [f"cam_{i:02d}" for i in range(1, 11)]
    for c in cams:
        gs.upsert_camera_node(c)

    for i in range(num_transitions):
        src = cams[i % 8]
        dst = cams[(i + 1) % 8]
        gs.add_transition_edge(f"t_{num_transitions}_{i}", f"gid_{i % 20}", src, dst, 1000 + i, 1.0)

    dispatcher = ToolDispatcher(event_store=es, graph_store=gs)
    planner = NLQPlanner()
    engine = ReasoningEngine(planner=planner, dispatcher=dispatcher)

    call9 = ToolCall("c9", "get_trajectory_clusters", {})
    call10 = ToolCall("c10", "get_behavioral_prediction", {"current_camera_id": "cam_01"})
    q9 = UserQuery("q9", "Show common routes for cam_01", 1000)
    q10 = UserQuery("q10", "Where is someone likely to go from cam_01?", 1000)

    bench_t9 = run_latency_benchmark(lambda: dispatcher.dispatch(call9))
    bench_t10 = run_latency_benchmark(lambda: dispatcher.dispatch(call10))
    bench_planner = run_latency_benchmark(lambda: planner.create_plan(q10))
    bench_e2e = run_latency_benchmark(lambda: engine.process_query(q10))

    es.close()
    gs.close()

    return {
        "num_transitions": num_transitions,
        "tool9_dispatch": bench_t9,
        "tool10_dispatch": bench_t10,
        "planner_latency": bench_planner,
        "reasoning_engine_e2e": bench_e2e,
    }


def run_all_benchmarks() -> dict[str, Any]:
    sys_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    acc_results = benchmark_held_out_accuracy(num_transitions=1000)
    scale_100 = benchmark_scale(100)
    scale_1000 = benchmark_scale(1000)
    scale_10000 = benchmark_scale(10000)

    output = {
        "environment": sys_info,
        "held_out_prediction_accuracy": acc_results,
        "scale_benchmarks": {
            "n_100": scale_100,
            "n_1000": scale_1000,
            "n_10000": scale_10000,
        },
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_phase6_behavioral.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_all_benchmarks()
