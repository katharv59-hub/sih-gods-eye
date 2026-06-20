"""Sweep YOLOv8 model sizes to find gate-passing configuration."""
import os
from pathlib import Path
from benchmarks.benchmark_mot17 import (
    _parse_seqinfo,
    _load_ground_truth,
    _run_pipeline_on_sequence,
    _compute_mot_metrics,
    _DEFAULT_DATA_ROOT
)

def main() -> None:
    seq_info = _parse_seqinfo(_DEFAULT_DATA_ROOT)
    seq_info['seqlength'] = '200'

    print("Model size sweep on first 200 frames (conf=0.15)")
    print("-" * 90)

    for model in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']:
        os.environ['GODS_EYE_DETECTION_MODEL'] = model
        os.environ['GODS_EYE_DETECTION_CONFIDENCE'] = '0.15'
        preds = _run_pipeline_on_sequence(_DEFAULT_DATA_ROOT, seq_info)
        gt = _load_ground_truth(_DEFAULT_DATA_ROOT / 'gt' / 'gt.txt')
        gt_filtered = {k: v for k, v in gt.items() if k <= 200}
        metrics = _compute_mot_metrics(gt_filtered, preds)

        mota = metrics['mota']
        idf1 = metrics['idf1']
        id_sw = metrics['num_switches']
        misses = metrics['num_misses']
        fp = metrics['num_false_positives']
        recall = metrics['recall']
        precision = metrics['precision']

        status = "PASS" if mota >= 0.60 else "FAIL"
        print(f"Model: {model:14s} | MOTA: {mota:.4f} [{status}] | IDF1: {idf1:.4f} | "
              f"Recall: {recall:.4f} | Prec: {precision:.4f} | ID Sw: {id_sw} | Miss: {misses} | FP: {fp}")
    print("-" * 90)

if __name__ == '__main__':
    main()
