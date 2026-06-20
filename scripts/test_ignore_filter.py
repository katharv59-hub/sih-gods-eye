import os
from pathlib import Path
from benchmarks.benchmark_mot17 import (
    _parse_seqinfo,
    _run_pipeline_on_sequence,
    _compute_mot_metrics,
    _DEFAULT_DATA_ROOT
)

def _load_ground_truth_with_ignore(
    gt_path: Path,
) -> tuple[dict[int, list[tuple[int, float, float, float, float]]], dict[int, list[tuple[float, float, float, float]]]]:
    gt = {}
    ignore = {}
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

            # Class 1 is pedestrian
            if flag == 1 and class_id == 1 and visibility > 0.0:
                gt.setdefault(frame_id, []).append((track_id, x, y, x + w, y + h))
            else:
                ignore.setdefault(frame_id, []).append((x, y, x + w, y + h))
    return gt, ignore

def _filter_predictions_by_ignore(
    predictions: dict[int, list[tuple[str, float, float, float, float]]],
    ignore: dict[int, list[tuple[float, float, float, float]]],
    threshold: float = 0.5,
) -> dict[int, list[tuple[str, float, float, float, float]]]:
    filtered = {}
    for fid, preds in predictions.items():
        ignore_boxes = ignore.get(fid, [])
        if not ignore_boxes:
            filtered[fid] = preds
            continue
        
        filtered_preds = []
        for pid, px1, py1, px2, py2 in preds:
            p_area = (px2 - px1) * (py2 - py1)
            if p_area <= 0:
                continue
            
            overlap_found = False
            for ix1, iy1, ix2, iy2 in ignore_boxes:
                xx1 = max(px1, ix1)
                yy1 = max(py1, iy1)
                xx2 = min(px2, ix2)
                yy2 = min(py2, iy2)
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                
                i_area = (ix2 - ix1) * (iy2 - iy1)
                union = p_area + i_area - inter
                iou = inter / union if union > 0 else 0.0
                iop = inter / p_area
                
                # If prediction is heavily covered by an ignore target, ignore it
                if iou > threshold or iop > threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered_preds.append((pid, px1, py1, px2, py2))
        filtered[fid] = filtered_preds
    return filtered

def main() -> None:
    seq_info = _parse_seqinfo(_DEFAULT_DATA_ROOT)
    
    # Run yolov8s, imgsz=1440, conf=0.35 on full 1050 frames
    os.environ['GODS_EYE_DETECTION_MODEL'] = 'yolov8s.pt'
    os.environ['GODS_EYE_DETECTION_IMGSZ'] = '1440'
    os.environ['GODS_EYE_DETECTION_CONFIDENCE'] = '0.35'
    
    print("Running pipeline on full sequence...")
    preds = _run_pipeline_on_sequence(_DEFAULT_DATA_ROOT, seq_info)
    
    gt_path = _DEFAULT_DATA_ROOT / 'gt' / 'gt.txt'
    gt, ignore = _load_ground_truth_with_ignore(gt_path)
    
    # Evaluate BEFORE filtering
    print("\nMetrics BEFORE ignore filtering:")
    metrics_before = _compute_mot_metrics(gt, preds)
    print(metrics_before)
    
    # Evaluate AFTER filtering
    print("\nMetrics AFTER ignore filtering:")
    filtered_preds = _filter_predictions_by_ignore(preds, ignore, threshold=0.5)
    metrics_after = _compute_mot_metrics(gt, filtered_preds)
    print(metrics_after)

if __name__ == '__main__':
    main()
