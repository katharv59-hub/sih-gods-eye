import os
import sys
import logging
from io import StringIO
from pathlib import Path

# Silence all loggers
logging.getLogger().setLevel(logging.CRITICAL)

try:
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
except ImportError:
    pass

from benchmarks.benchmark_mot17 import (
    _parse_seqinfo,
    _load_ground_truth,
    _run_pipeline_on_sequence,
    _compute_mot_metrics,
    _DEFAULT_DATA_ROOT
)

def main() -> None:
    seq_info = _parse_seqinfo(_DEFAULT_DATA_ROOT)
    # Run full sequence
    seq_length = int(seq_info.get("seqlength", "1050"))

    print("Running yolov8s full sequence sweep...")
    print("-" * 110)
    print(f"{'Model':12s} | {'Imgsz':5s} | {'Conf':5s} | {'MOTA':6s} | {'IDF1':6s} | {'Recall':6s} | {'Prec':6s} | {'ID Sw':5s} | {'Miss':5s} | {'FP':5s}")
    print("-" * 110)

    combinations = [
        ('yolov8s.pt', 1280, 0.35),
        ('yolov8s.pt', 1280, 0.40),
        ('yolov8s.pt', 1280, 0.45),
        ('yolov8s.pt', 1440, 0.35),
        ('yolov8s.pt', 1440, 0.40),
        ('yolov8s.pt', 1440, 0.45),
        ('yolov8s.pt', 1088, 0.30),
        ('yolov8s.pt', 1088, 0.35),
        ('yolov8s.pt', 1088, 0.40),
    ]

    for model, imgsz, conf in combinations:
        os.environ['GODS_EYE_DETECTION_MODEL'] = model
        os.environ['GODS_EYE_DETECTION_IMGSZ'] = str(imgsz)
        os.environ['GODS_EYE_DETECTION_CONFIDENCE'] = str(conf)
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            preds = _run_pipeline_on_sequence(_DEFAULT_DATA_ROOT, seq_info)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        gt = _load_ground_truth(_DEFAULT_DATA_ROOT / 'gt' / 'gt.txt')
        metrics = _compute_mot_metrics(gt, preds)

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
