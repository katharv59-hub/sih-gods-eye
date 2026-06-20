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
    # Use 200 frames for quick sweep
    seq_info['seqlength'] = '200'

    print("Running combinations sweep on 200 frames...")
    print("-" * 110)
    print(f"{'Model':12s} | {'Imgsz':5s} | {'Conf':5s} | {'MOTA':6s} | {'IDF1':6s} | {'Recall':6s} | {'Prec':6s} | {'ID Sw':5s} | {'Miss':5s} | {'FP':5s}")
    print("-" * 110)

    combinations = [
        # yolov8n with larger input sizes
        ('yolov8n.pt', 1088, 0.15),
        ('yolov8n.pt', 1280, 0.15),
        ('yolov8n.pt', 1280, 0.20),
        # yolov8s with larger input sizes
        ('yolov8s.pt', 1088, 0.15),
        ('yolov8s.pt', 1280, 0.15),
        ('yolov8s.pt', 1280, 0.20),
    ]

    for model, imgsz, conf in combinations:
        os.environ['GODS_EYE_DETECTION_MODEL'] = model
        os.environ['GODS_EYE_DETECTION_IMGSZ'] = str(imgsz)
        os.environ['GODS_EYE_DETECTION_CONFIDENCE'] = str(conf)
        
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

        print(f"{model:12s} | {imgsz:5d} | {conf:5.2f} | {mota:6.4f} | {idf1:6.4f} | {recall:6.4f} | {precision:6.4f} | {id_sw:5d} | {misses:5d} | {fp:5d}")
    print("-" * 110)

if __name__ == '__main__':
    main()
