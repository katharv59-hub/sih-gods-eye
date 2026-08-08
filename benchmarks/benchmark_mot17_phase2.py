"""MOT17-04 benchmark with Identity Resolution — Phase 2 gate criterion.

Extends the Phase 1 MOT17 benchmark by adding the IdentityMapper (IML)
to the pipeline.  Evaluates using *global identity IDs* instead of raw
ByteTrack track IDs, measuring the ID-switch reduction provided by
identity persistence.

Phase 2 Gate Criteria (evaluated here):
    - ID switches ≤ 70 on MOT17-04 (with identity resolution)
    - FPS regression < 10% vs Phase 1 (tracking-only)

Also measures (informational):
    - MOTA, IDF1, precision, recall
    - Identity relink rate
    - False merge estimate (new IDs per GT identity)
    - Gallery size over time
    - Pipeline FPS (with and without IML)

Usage:
    python -m benchmarks.benchmark_mot17_phase2
    python -m benchmarks.benchmark_mot17_phase2 --data-root path/to/MOT17-04
    python -m benchmarks.benchmark_mot17_phase2 --cpu-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Monkeypatch np.asfarray for NumPy 2.x compatibility with legacy motmetrics
if not hasattr(np, "asfarray"):
    def asfarray(a, dtype=np.float64):  # type: ignore[no-untyped-def]
        return np.asarray(a, dtype=dtype)
    np.asfarray = asfarray  # type: ignore[attr-defined]

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import configure_logging, get_logger
from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.identity_gallery import IdentityGallery
from gods_eye.reid.identity_lifecycle import LifecycleManager
from gods_eye.reid.identity_mapper import (
    IdentityMapper,
    IdentityResult,
    TransitionType,
)
from gods_eye.reid.matcher import Matcher
from gods_eye.reid.osnet_extractor import OSNetExtractor
from gods_eye.pipeline import TrackingResult
from gods_eye.schemas.detection import Detection
from gods_eye.schemas.track import Track, TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

_log = get_logger("benchmark.mot17_phase2")

_DEFAULT_DATA_ROOT = Path("tests/data/MOT17/train/MOT17-04-FRCNN")


# ── Dataset helpers (shared with Phase 1 benchmark) ─────────────────────────

def _parse_seqinfo(seq_dir: Path) -> dict[str, str]:
    import configparser
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found at {ini_path}")
    parser = configparser.ConfigParser()
    parser.read(str(ini_path))
    return dict(parser["Sequence"])


_MIN_VISIBILITY: float = 0.25


def _load_ground_truth(
    gt_path: Path,
    min_visibility: float = _MIN_VISIBILITY,
) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Load MOT ground truth: {frame_id: [(track_id, x1, y1, x2, y2), ...]}."""
    gt: dict[int, list[tuple[int, float, float, float, float]]] = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            flag = int(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])
            if flag != 1 or class_id != 1 or visibility < min_visibility:
                continue
            gt.setdefault(frame_id, []).append((
                track_id, x, y, x + w, y + h
            ))
    return gt


def _load_ignore_regions(
    gt_path: Path,
    min_visibility: float = _MIN_VISIBILITY,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Load ignore regions for MOTChallenge evaluation protocol."""
    ignore: dict[int, list[tuple[float, float, float, float]]] = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            flag = int(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])
            if flag == 1 and class_id == 1 and visibility > min_visibility:
                continue
            ignore.setdefault(frame_id, []).append((x, y, x + w, y + h))
    return ignore


# ── Metrics computation ─────────────────────────────────────────────────────


def _iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ioa(pred_box: tuple[float, ...], ign_box: tuple[float, ...]) -> float:
    """Intersection over prediction area (for ignore regions)."""
    x1 = max(pred_box[0], ign_box[0])
    y1 = max(pred_box[1], ign_box[1])
    x2 = min(pred_box[2], ign_box[2])
    y2 = min(pred_box[3], ign_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    return inter / pred_area if pred_area > 0 else 0.0


def _filter_predictions_with_ignore(
    predictions: dict[int, list[tuple[str, float, float, float, float]]],
    gt: dict[int, list[tuple[int, float, float, float, float]]],
    ignore: dict[int, list[tuple[float, float, float, float]]],
    ioa_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[int, list[tuple[str, float, float, float, float]]]:
    """Filter predictions per MOTChallenge ignore-region protocol."""
    filtered: dict[int, list[tuple[str, float, float, float, float]]] = {}
    for fid, preds in predictions.items():
        gt_objs = gt.get(fid, [])
        ign_boxes = ignore.get(fid, [])
        if not ign_boxes:
            filtered[fid] = preds
            continue
        kept: list[tuple[str, float, float, float, float]] = []
        for p in preds:
            p_box = (p[1], p[2], p[3], p[4])
            matches_gt = any(
                _iou(p_box, (g[1], g[2], g[3], g[4])) >= iou_threshold
                for g in gt_objs
            )
            if matches_gt:
                kept.append(p)
                continue
            in_ignore = any(
                _ioa(p_box, ign) >= ioa_threshold for ign in ign_boxes
            )
            if not in_ignore:
                kept.append(p)
        filtered[fid] = kept
    return filtered


def _compute_mot_metrics(
    gt: dict[int, list[tuple[int, float, float, float, float]]],
    predictions: dict[int, list[tuple[str, float, float, float, float]]],
    ignore: dict[int, list[tuple[float, float, float, float]]] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute MOT metrics using py-motmetrics."""
    try:
        import motmetrics as mm
    except ImportError:
        print("ERROR: motmetrics not installed. Run: pip install motmetrics")
        sys.exit(1)

    if ignore is not None:
        predictions = _filter_predictions_with_ignore(
            predictions, gt, ignore,
            ioa_threshold=0.5, iou_threshold=iou_threshold,
        )

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(list(gt.keys()) + list(predictions.keys())))

    # motmetrics requires numeric IDs (internally casts to float).
    # Global identity IDs are UUID hex strings — map them to stable ints.
    _pred_id_map: dict[str, int] = {}
    _next_pred_int = 100_000  # Offset to avoid collisions with GT int IDs

    def _to_numeric_id(string_id: str) -> int:
        nonlocal _next_pred_int
        if string_id not in _pred_id_map:
            _pred_id_map[string_id] = _next_pred_int
            _next_pred_int += 1
        return _pred_id_map[string_id]

    for fid in all_frames:
        gt_objs = gt.get(fid, [])
        pred_objs = predictions.get(fid, [])
        gt_ids = [g[0] for g in gt_objs]
        pred_ids = [_to_numeric_id(p[0]) for p in pred_objs]

        if len(gt_objs) == 0 and len(pred_objs) == 0:
            acc.update([], [], [])
            continue

        gt_boxes = np.array(
            [[g[1], g[2], g[3], g[4]] for g in gt_objs]
        ) if gt_objs else np.empty((0, 4))
        pred_boxes = np.array(
            [[p[1], p[2], p[3], p[4]] for p in pred_objs]
        ) if pred_objs else np.empty((0, 4))

        # Convert x1y1x2y2 to tlwh
        if len(gt_boxes) > 0:
            gt_tlwh = np.column_stack([
                gt_boxes[:, 0], gt_boxes[:, 1],
                gt_boxes[:, 2] - gt_boxes[:, 0],
                gt_boxes[:, 3] - gt_boxes[:, 1],
            ])
        else:
            gt_tlwh = np.empty((0, 4))

        if len(pred_boxes) > 0:
            pred_tlwh = np.column_stack([
                pred_boxes[:, 0], pred_boxes[:, 1],
                pred_boxes[:, 2] - pred_boxes[:, 0],
                pred_boxes[:, 3] - pred_boxes[:, 1],
            ])
        else:
            pred_tlwh = np.empty((0, 4))

        dists = mm.distances.iou_matrix(
            gt_tlwh, pred_tlwh, max_iou=1.0 - iou_threshold
        )
        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "mota", "idf1", "num_switches", "num_misses",
        "num_false_positives", "precision", "recall",
    ], name="MOT17-04")

    return {
        "mota": float(summary["mota"].iloc[0]),
        "idf1": float(summary["idf1"].iloc[0]),
        "num_switches": int(summary["num_switches"].iloc[0]),
        "num_misses": int(summary["num_misses"].iloc[0]),
        "num_false_positives": int(summary["num_false_positives"].iloc[0]),
        "precision": float(summary["precision"].iloc[0]),
        "recall": float(summary["recall"].iloc[0]),
    }


# ── Pipeline runner ─────────────────────────────────────────────────────────


def _load_sequence_packets(
    seq_dir: Path, seq_info: dict[str, str]
) -> list[FramePacket]:
    """Pre-load sequence images into memory to isolate pipeline execution FPS."""
    img_dir = seq_dir / seq_info.get("imdir", "img1")
    seq_length = int(seq_info.get("seqlength", "1050"))
    img_ext = seq_info.get("imext", ".jpg")
    packets: list[FramePacket] = []
    for fid in range(1, seq_length + 1):
        img_name = f"{fid:06d}{img_ext}"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        packets.append(FramePacket(
            camera_id="mot17-04",
            frame_id=fid,
            timestamp_ns=int(fid * 33_333_333),
            frame=frame,
            resolution=(w, h),
        ))
    return packets


def _run_phase2_pipeline(
    packets: list[FramePacket],
    settings: Settings,
) -> tuple[
    dict[int, list[tuple[str, float, float, float, float]]],
    dict[str, Any],
]:
    """Run detection → tracking → identity resolution on pre-loaded packets."""
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    extractor: EmbeddingExtractor = OSNetExtractor(settings)
    extractor.warmup()
    gallery = IdentityGallery()
    lifecycle = LifecycleManager(settings)
    matcher = Matcher(settings)
    mapper = IdentityMapper(
        extractor=extractor,
        gallery=gallery,
        lifecycle=lifecycle,
        matcher=matcher,
        settings=settings,
        camera_id="mot17-04",
    )

    seq_length = len(packets)
    predictions: dict[int, list[tuple[str, float, float, float, float]]] = {}
    total_new = 0
    total_relinks = 0
    total_lost = 0
    total_expired = 0
    max_gallery_size = 0

    _log.info("phase2_pipeline_start", seq_length=seq_length)
    t_start = time.perf_counter()

    for packet in packets:
        fid = packet.frame_id
        detections = detector.detect(packet)
        tracks = tracker.update(detections, fid, "mot17-04")

        tracking_result = TrackingResult(
            packet=packet,
            detections=detections,
            tracks=tracks,
        )
        id_result: IdentityResult = mapper.process(tracking_result)

        frame_preds: list[tuple[str, float, float, float, float]] = []
        for track in tracks:
            if track.state is not TrackState.ACTIVE:
                continue
            identity = id_result.identities.get(track.track_id)
            pred_id = identity.global_id if identity else track.track_id
            frame_preds.append((
                pred_id,
                track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2,
            ))
        predictions[fid] = frame_preds

        for t in id_result.transitions:
            if t.transition_type is TransitionType.CONFIRMED_NEW:
                total_new += 1
            elif t.transition_type is TransitionType.CONFIRMED_RELINK:
                total_relinks += 1
            elif t.transition_type is TransitionType.LOST:
                total_lost += 1
            elif t.transition_type is TransitionType.EXPIRED:
                total_expired += 1

        max_gallery_size = max(max_gallery_size, mapper.gallery_size)

        if fid % 100 == 0:
            elapsed = time.perf_counter() - t_start
            fps = fid / elapsed if elapsed > 0 else 0
            _log.info(
                "progress", frame=fid, total=seq_length,
                fps=round(fps, 1),
                gallery_size=mapper.gallery_size,
                cache_size=mapper.active_mapping_count,
            )

    elapsed = time.perf_counter() - t_start
    pipeline_fps = seq_length / elapsed if elapsed > 0 else 0

    _log.info(
        "phase2_pipeline_complete",
        frames=seq_length,
        elapsed_s=round(elapsed, 1),
        fps=round(pipeline_fps, 1),
    )

    identity_stats = {
        "total_new_identities": total_new,
        "total_relinks": total_relinks,
        "total_lost": total_lost,
        "total_expired": total_expired,
        "max_gallery_size": max_gallery_size,
        "final_gallery_size": mapper.gallery_size,
        "final_cache_size": mapper.active_mapping_count,
        "pipeline_fps": round(pipeline_fps, 2),
        "elapsed_s": round(elapsed, 2),
    }

    return predictions, identity_stats


def _run_phase1_pipeline(
    packets: list[FramePacket],
    settings: Settings,
) -> tuple[
    dict[int, list[tuple[str, float, float, float, float]]],
    float,
]:
    """Run detection → tracking only (Phase 1 baseline) for comparison."""
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    seq_length = len(packets)
    predictions: dict[int, list[tuple[str, float, float, float, float]]] = {}

    t_start = time.perf_counter()
    for packet in packets:
        fid = packet.frame_id
        detections = detector.detect(packet)
        tracks = tracker.update(detections, fid, "mot17-04")

        frame_preds: list[tuple[str, float, float, float, float]] = []
        for t in tracks:
            if t.state is TrackState.ACTIVE:
                frame_preds.append((
                    t.track_id,
                    t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2,
                ))
        predictions[fid] = frame_preds

    elapsed = time.perf_counter() - t_start
    fps = seq_length / elapsed if elapsed > 0 else 0
    return predictions, fps


# ── Main ────────────────────────────────────────────────────────────────────


def run_benchmark(
    data_root: Path | None = None,
    cpu_only: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 2 MOT17-04 benchmark with identity resolution.

    Returns results dict with metrics, gate pass/fail, and identity stats.
    """
    seq_dir = data_root or _DEFAULT_DATA_ROOT

    if not seq_dir.exists():
        print(f"\n{'='*60}")
        print("ERROR: MOT17-04 dataset not found!")
        print(f"Expected at: {seq_dir.resolve()}")
        print()
        print("Download instructions:")
        print("  1. Go to https://motchallenge.net/data/MOT17/")
        print("  2. Download MOT17.zip (~5.5 GB)")
        print("  3. Extract to tests/data/MOT17/")
        print(f"  4. Verify: {seq_dir / 'img1'} exists")
        print(f"  5. Verify: {seq_dir / 'gt' / 'gt.txt'} exists")
        print(f"{'='*60}\n")
        sys.exit(1)

    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found at {gt_path}")
        sys.exit(1)

    seq_info = _parse_seqinfo(seq_dir)
    _log.info("sequence_info", **seq_info)

    gt = _load_ground_truth(gt_path)
    ignore = _load_ignore_regions(gt_path)

    settings_kwargs: dict[str, Any] = {}
    if cpu_only:
        settings_kwargs["cpu_only"] = True
    settings = Settings(**settings_kwargs)

    # Pre-load all frames into memory once — shared by both pipelines
    # for fair FPS comparison (isolates I/O from pipeline execution).
    packets = _load_sequence_packets(seq_dir, seq_info)
    _log.info("packets_loaded", count=len(packets))

    # ── Phase 1 baseline (tracking only) ─────────────────────────────
    print("\n--- Phase 1 Baseline (tracking only) ---")
    p1_predictions, p1_fps = _run_phase1_pipeline(packets, settings)
    p1_metrics = _compute_mot_metrics(gt, p1_predictions, ignore=ignore)

    # ── Phase 2 (tracking + identity resolution) ─────────────────────
    print("\n--- Phase 2 (tracking + identity resolution) ---")
    p2_predictions, id_stats = _run_phase2_pipeline(packets, settings)
    p2_metrics = _compute_mot_metrics(gt, p2_predictions, ignore=ignore)

    # ── Gate evaluation ──────────────────────────────────────────────
    p2_id_switches = p2_metrics["num_switches"]
    p1_id_switches = p1_metrics["num_switches"]
    ids_pass = p2_id_switches <= 70

    fps_regression = (
        (p1_fps - id_stats["pipeline_fps"]) / p1_fps * 100
        if p1_fps > 0 else 0
    )
    fps_pass = fps_regression < 10.0

    # Relink rate: ratio of relinks to (relinks + new identities)
    total_assignments = id_stats["total_new_identities"] + id_stats["total_relinks"]
    relink_rate = (
        id_stats["total_relinks"] / total_assignments * 100
        if total_assignments > 0 else 0
    )

    # Unique predicted IDs vs unique GT IDs (rough false merge indicator)
    unique_pred_ids: set[str] = set()
    for preds in p2_predictions.values():
        for p in preds:
            unique_pred_ids.add(p[0])
    unique_gt_ids: set[int] = set()
    for gts in gt.values():
        for g in gts:
            unique_gt_ids.add(g[0])

    results: dict[str, Any] = {
        "benchmark": "mot17_phase2",
        "sequence": "MOT17-04-FRCNN",
        "phase1_baseline": {
            "metrics": p1_metrics,
            "fps": round(p1_fps, 2),
        },
        "phase2_results": {
            "metrics": p2_metrics,
            "identity_stats": id_stats,
            "relink_rate_pct": round(relink_rate, 2),
            "unique_predicted_ids": len(unique_pred_ids),
            "unique_gt_ids": len(unique_gt_ids),
        },
        "gate_criteria": {
            "id_switches": {
                "threshold": 70,
                "phase1_actual": p1_id_switches,
                "phase2_actual": p2_id_switches,
                "reduction": p1_id_switches - p2_id_switches,
                "result": "PASS" if ids_pass else "FAIL",
            },
            "fps_regression": {
                "threshold_pct": 10.0,
                "phase1_fps": round(p1_fps, 2),
                "phase2_fps": id_stats["pipeline_fps"],
                "regression_pct": round(fps_regression, 2),
                "result": "PASS" if fps_pass else "FAIL",
            },
        },
        "gate_result": "PASS" if (ids_pass and fps_pass) else "FAIL",
    }

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("MOT17-04 Phase 2 Benchmark Results")
    print(f"{'='*65}")
    print()
    print("  Phase 1 (tracking only):")
    print(f"    MOTA:            {p1_metrics['mota']:.4f}")
    print(f"    ID Switches:     {p1_id_switches}")
    print(f"    FPS:             {p1_fps:.2f}")
    print()
    print("  Phase 2 (tracking + identity):")
    print(f"    MOTA:            {p2_metrics['mota']:.4f}")
    print(f"    IDF1:            {p2_metrics['idf1']:.4f}")
    print(f"    ID Switches:     {p2_id_switches}  "
          f"{'PASS ✓' if ids_pass else 'FAIL ✗'} (gate ≤ 70)")
    print(f"    Reduction:       {p1_id_switches - p2_id_switches} fewer "
          f"({(1 - p2_id_switches / max(p1_id_switches, 1)) * 100:.1f}%)")
    print(f"    FPS:             {id_stats['pipeline_fps']:.2f}  "
          f"({'PASS ✓' if fps_pass else 'FAIL ✗'} regression "
          f"{fps_regression:.1f}%, gate < 10%)")
    print()
    print("  Identity Stats:")
    print(f"    New identities:  {id_stats['total_new_identities']}")
    print(f"    Re-links:        {id_stats['total_relinks']}")
    print(f"    Relink rate:     {relink_rate:.1f}%")
    print(f"    Lost:            {id_stats['total_lost']}")
    print(f"    Expired:         {id_stats['total_expired']}")
    print(f"    Max gallery:     {id_stats['max_gallery_size']}")
    print(f"    Unique pred IDs: {len(unique_pred_ids)}  "
          f"(GT: {len(unique_gt_ids)})")
    print()
    print(f"  Overall Gate:      "
          f"{'PASS ✓' if results['gate_result'] == 'PASS' else 'FAIL ✗'}")
    print(f"{'='*65}")

    return results


def main() -> None:
    configure_logging("WARNING")

    parser = argparse.ArgumentParser(
        description="MOT17-04 Phase 2 benchmark (with identity resolution)"
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Path to MOT17-04 sequence directory",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU-only mode",
    )
    args = parser.parse_args()

    results = run_benchmark(
        data_root=args.data_root,
        cpu_only=args.cpu_only,
    )

    out_path = Path(__file__).parent / "results" / "mot17_phase2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if results["gate_result"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
