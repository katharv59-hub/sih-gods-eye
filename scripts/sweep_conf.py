import os
from pathlib import Path
from benchmarks.benchmark_mot17 import (
    _parse_seqinfo,
    _load_ground_truth,
    _run_pipeline_on_sequence,
    _compute_mot_metrics,
    _DEFAULT_DATA_ROOT
)

def main():
    seq_info = _parse_seqinfo(_DEFAULT_DATA_ROOT)
    seq_info['seqlength'] = '200'

    print("Running confidence threshold sweep on 200 frames...")
    print("-" * 80)
    for conf in [0.05, 0.10, 0.15, 0.20, 0.25]:
        os.environ['GODS_EYE_DETECTION_CONFIDENCE'] = str(conf)
        preds = _run_pipeline_on_sequence(_DEFAULT_DATA_ROOT, seq_info)
        gt = _load_ground_truth(_DEFAULT_DATA_ROOT / 'gt' / 'gt.txt')
        gt_filtered = {k: v for k, v in gt.items() if k <= 200}
        metrics = _compute_mot_metrics(gt_filtered, preds)
        
        mota = metrics['mota']
        idf1 = metrics['idf1']
        id_sw = metrics['num_switches']
        misses = metrics['num_misses']
        fps = metrics['num_false_positives']
        
        print(f"Conf: {conf:.2f} | MOTA: {mota:.4f} | IDF1: {idf1:.4f} | ID Sw: {id_sw} | Misses: {misses} | FP: {fps}")
    print("-" * 80)

if __name__ == '__main__':
    main()
