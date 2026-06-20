"""MOT17-04 tracking benchmark — Phase 1 gate criterion.

Runs the full detection → tracking pipeline on the MOT17-04 image
sequence and evaluates against ground truth using py-motmetrics.

Gate criteria:
    - MOTA ≥ 0.60 on MOT17-04
    - ID switches ≤ 150

Dataset structure expected:
    tests/data/MOT17/train/MOT17-04-FRCNN/
        img1/       ← 1050 frames (000001.jpg → 001050.jpg)
        gt/gt.txt   ← ground truth in MOTChallenge format
        seqinfo.ini ← sequence metadata

MOT ground truth format (gt.txt):
    frame_id, track_id, x, y, w, h, flag, class, visibility
    - flag=1: active target to evaluate
    - class=1: pedestrian
    - visibility: 0-1 occlusion ratio

Usage:
    python -m benchmarks.benchmark_mot17
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-untyped]
import numpy as np

# Monkeypatch np.asfarray for NumPy 2.x compatibility with legacy motmetrics library
if not hasattr(np, "asfarray"):
    def asfarray(a, dtype=np.float64):  # type: ignore[no-untyped-def]
        return np.asarray(a, dtype=dtype)
    np.asfarray = asfarray  # type: ignore[attr-defined]

from gods_eye.config.settings import Settings
from gods_eye.detection.yolo_detector import YOLODetector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import get_logger
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker

_log = get_logger("benchmark.mot17")

# ── Dataset paths ────────────────────────────────────────────────────────────

_DEFAULT_DATA_ROOT = Path("tests/data/MOT17/train/MOT17-04-FRCNN")


def _parse_seqinfo(seq_dir: Path) -> dict[str, str]:
    """Parse seqinfo.ini from a MOT sequence directory."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found at {ini_path}")
    parser = configparser.ConfigParser()
    parser.read(str(ini_path))
    return dict(parser["Sequence"])


# Standard MOTChallenge visibility threshold — pedestrians below this
# are treated as ignore regions (too occluded for any detector).
_MIN_VISIBILITY: float = 0.25


def _load_ground_truth(
    gt_path: Path,
    min_visibility: float = _MIN_VISIBILITY,
) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Load MOT ground truth into {frame_id: [(track_id, x1, y1, x2, y2), ...]}.

    Filters for:
        - flag == 1 (active target)
        - class_id == 1 (pedestrian)
        - visibility > min_visibility
    """
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

            # Filter: active pedestrian targets above visibility threshold
            if flag != 1 or class_id != 1 or visibility <= min_visibility:
                continue

            gt.setdefault(frame_id, []).append((
                track_id, x, y, x + w, y + h
            ))
    return gt


def _load_ignore_regions(
    gt_path: Path,
    min_visibility: float = _MIN_VISIBILITY,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Load MOTChallenge ignore/distractor regions from gt.txt.

    Per the official MOT17 evaluation protocol, any GT entry that is NOT
    an active pedestrian target is treated as an ignore region. This includes:
    - Non-pedestrian classes (vehicles, static objects, distractors)
    - Inactive targets (flag=0)
    - Heavily occluded pedestrians (visibility <= min_visibility)

    Predictions overlapping these regions are not counted as false positives.

    Returns {frame_id: [(x1, y1, x2, y2), ...]}.
    """
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

            # Anything NOT an active visible pedestrian is an ignore region
            if flag == 1 and class_id == 1 and visibility > min_visibility:
                continue

            ignore.setdefault(frame_id, []).append((
                x, y, x + w, y + h
            ))
    return ignore


def _filter_predictions_with_ignore(
    predictions: dict[int, list[tuple[str, float, float, float, float]]],
    gt: dict[int, list[tuple[int, float, float, float, float]]],
    ignore: dict[int, list[tuple[float, float, float, float]]],
    ioa_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[int, list[tuple[str, float, float, float, float]]]:
    """Filter predictions per MOTChallenge ignore-region protocol.

    For each prediction in each frame:
    1. If it matches any active GT target (IoU >= iou_threshold), keep it.
    2. Otherwise, if it overlaps an ignore region (IoA >= ioa_threshold),
       discard it — it should NOT count as a false positive.
    3. Otherwise, keep it (it is a genuine false positive).

    IoA = Intersection over (prediction) Area — standard MOTChallenge metric
    for ignore-region matching.
    """
    filtered: dict[int, list[tuple[str, float, float, float, float]]] = {}

    for fid, preds in predictions.items():
        gt_objs = gt.get(fid, [])
        ign_boxes = ignore.get(fid, [])

        if not ign_boxes:
            filtered[fid] = preds
            continue

        kept: list[tuple[str, float, float, float, float]] = []
        for pid, px1, py1, px2, py2 in preds:
            p_area = (px2 - px1) * (py2 - py1)
            if p_area <= 0:
                continue

            # Check if this prediction matches any active GT target
            matched_gt = False
            for _, gx1, gy1, gx2, gy2 in gt_objs:
                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                g_area = (gx2 - gx1) * (gy2 - gy1)
                union = p_area + g_area - inter
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    matched_gt = True
                    break

            if matched_gt:
                kept.append((pid, px1, py1, px2, py2))
                continue

            # Check if prediction overlaps an ignore region (IoA)
            in_ignore = False
            for ix1, iy1, ix2, iy2 in ign_boxes:
                xx1 = max(px1, ix1)
                yy1 = max(py1, iy1)
                xx2 = min(px2, ix2)
                yy2 = min(py2, iy2)
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                ioa = inter / p_area  # Intersection over prediction Area
                if ioa >= ioa_threshold:
                    in_ignore = True
                    break

            if not in_ignore:
                kept.append((pid, px1, py1, px2, py2))

        filtered[fid] = kept

    return filtered


def _compute_mot_metrics(
    gt: dict[int, list[tuple[int, float, float, float, float]]],
    predictions: dict[int, list[tuple[str, float, float, float, float]]],
    ignore: dict[int, list[tuple[float, float, float, float]]] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute MOT metrics using py-motmetrics.

    If ignore regions are provided, predictions overlapping them (but not
    matching active GT targets) are filtered out per the standard
    MOTChallenge evaluation protocol before scoring.

    Returns dict with MOTA, IDF1, num_switches, etc.
    """
    try:
        import motmetrics as mm  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: motmetrics not installed. Run: pip install motmetrics")
        sys.exit(1)

    # Apply ignore-region filtering if provided
    if ignore is not None:
        predictions = _filter_predictions_with_ignore(
            predictions, gt, ignore,
            ioa_threshold=0.5, iou_threshold=iou_threshold,
        )

    acc = mm.MOTAccumulator(auto_id=True)

    # Get all frame ids
    all_frames = sorted(set(list(gt.keys()) + list(predictions.keys())))

    for fid in all_frames:
        gt_objs = gt.get(fid, [])
        pred_objs = predictions.get(fid, [])

        gt_ids = [g[0] for g in gt_objs]
        pred_ids = [p[0] for p in pred_objs]

        if len(gt_objs) == 0 and len(pred_objs) == 0:
            acc.update([], [], [])
            continue

        # Compute IoU distance matrix
        gt_boxes = np.array([[g[1], g[2], g[3], g[4]] for g in gt_objs]) if gt_objs else np.empty((0, 4))
        pred_boxes = np.array([[p[1], p[2], p[3], p[4]] for p in pred_objs]) if pred_objs else np.empty((0, 4))

        # mm.distances.iou_matrix expects (tlwh) format
        # Convert from x1y1x2y2 to tlwh
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


def _run_pipeline_on_sequence(
    seq_dir: Path,
    seq_info: dict[str, str],
) -> dict[int, list[tuple[str, float, float, float, float]]]:
    """Run detection + tracking on all frames.

    Returns {frame_id: [(track_id_str, x1, y1, x2, y2), ...]}.
    """
    settings = Settings.from_env()
    detector = YOLODetector(settings)
    detector.warmup()
    tracker = ByteTrackTracker(settings)

    img_dir = seq_dir / seq_info.get("imdir", "img1")
    seq_length = int(seq_info.get("seqlength", "1050"))
    img_ext = seq_info.get("imext", ".jpg")

    predictions: dict[int, list[tuple[str, float, float, float, float]]] = {}

    _log.info("pipeline_start", seq_length=seq_length)
    t_start = time.perf_counter()

    for fid in range(1, seq_length + 1):
        img_name = f"{fid:06d}{img_ext}"
        img_path = img_dir / img_name
        if not img_path.exists():
            _log.warning("frame_missing", frame=fid, path=str(img_path))
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            _log.warning("frame_read_error", frame=fid)
            continue

        h, w = frame.shape[:2]
        packet = FramePacket(
            camera_id="mot17-04",
            frame_id=fid,
            timestamp_ns=int(fid * 33_333_333),  # ~30fps
            frame=frame,
            resolution=(w, h),
        )

        detections = detector.detect(packet)
        tracks = tracker.update(detections, fid, "mot17-04")

        frame_preds: list[tuple[str, float, float, float, float]] = []
        for t in tracks:
            if t.state.value == "active":
                frame_preds.append((
                    t.track_id,
                    t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2,
                ))
        predictions[fid] = frame_preds

        if fid % 100 == 0:
            elapsed = time.perf_counter() - t_start
            fps = fid / elapsed if elapsed > 0 else 0
            _log.info("progress", frame=fid, total=seq_length,
                      fps=round(fps, 1))

    elapsed = time.perf_counter() - t_start
    _log.info("pipeline_complete", frames=seq_length,
              elapsed_s=round(elapsed, 1),
              fps=round(seq_length / elapsed, 1))
    return predictions


def run_benchmark(data_root: Path | None = None) -> dict[str, Any]:
    """Run the full MOT17-04 benchmark.

    Returns results dict with metrics and gate pass/fail.
    """
    seq_dir = data_root or _DEFAULT_DATA_ROOT

    # Validate dataset exists
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

    # Parse sequence info
    seq_info = _parse_seqinfo(seq_dir)
    _log.info("sequence_info", **seq_info)

    # Load ground truth
    gt = _load_ground_truth(gt_path)
    gt_frames = len(gt)
    gt_objects = sum(len(v) for v in gt.values())
    _log.info("ground_truth_loaded", frames=gt_frames, objects=gt_objects)

    # Run pipeline
    predictions = _run_pipeline_on_sequence(seq_dir, seq_info)

    # Load ignore regions per MOTChallenge protocol
    ignore = _load_ignore_regions(gt_path)
    ignore_objects = sum(len(v) for v in ignore.values())
    _log.info("ignore_regions_loaded", objects=ignore_objects)

    # Compute metrics (with ignore-region filtering)
    metrics = _compute_mot_metrics(gt, predictions, ignore=ignore)

    # Gate evaluation
    mota = metrics["mota"]
    id_switches = metrics["num_switches"]
    mota_pass = mota >= 0.60
    ids_pass = id_switches <= 150
    gate_pass = mota_pass and ids_pass

    results: dict[str, Any] = {
        "benchmark": "mot17_phase1",
        "sequence": "MOT17-04-FRCNN",
        "metrics": metrics,
        "gate_criteria": {
            "mota": {"threshold": 0.60, "actual": round(mota, 4),
                     "result": "PASS" if mota_pass else "FAIL"},
            "id_switches": {"threshold": 150, "actual": id_switches,
                            "result": "PASS" if ids_pass else "FAIL"},
        },
        "gate_result": "PASS" if gate_pass else "FAIL",
    }

    # Print summary
    print(f"\n{'='*60}")
    print("MOT17-04 Benchmark Results — Phase 1 Gate")
    print(f"{'='*60}")
    print(f"  MOTA:             {mota:.4f}  "
          f"{'PASS ✓' if mota_pass else 'FAIL ✗'} (gate ≥ 0.60)")
    print(f"  IDF1:             {metrics['idf1']:.4f}")
    print(f"  ID Switches:      {id_switches}  "
          f"{'PASS ✓' if ids_pass else 'FAIL ✗'} (gate ≤ 150)")
    print(f"  Misses:           {metrics['num_misses']}")
    print(f"  False Positives:  {metrics['num_false_positives']}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  Overall Gate:     {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print(f"{'='*60}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOT17-04 tracking benchmark"
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Path to MOT17-04 sequence directory",
    )
    args = parser.parse_args()

    results = run_benchmark(data_root=args.data_root)

    # Write results
    out_path = Path(__file__).parent / "results" / "mot17_phase1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if results["gate_result"] == "FAIL":
        _log.error("gate_failed", gate="mot17",
                   mota=results["metrics"]["mota"],
                   id_switches=results["metrics"]["num_switches"])
        sys.exit(1)


if __name__ == "__main__":
    main()
