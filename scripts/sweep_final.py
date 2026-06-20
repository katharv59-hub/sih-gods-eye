import os
import sys
import logging
from io import StringIO

logging.getLogger().setLevel(logging.CRITICAL)
try:
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
except ImportError:
    pass

from benchmarks.benchmark_mot17 import (
    _parse_seqinfo,
    _load_ground_truth,
    _load_ignore_regions,
    _run_pipeline_on_sequence,
    _compute_mot_metrics,
    _DEFAULT_DATA_ROOT
)

def main() -> None:
    seq_info = _parse_seqinfo(_DEFAULT_DATA_ROOT)
    gt_path = _DEFAULT_DATA_ROOT / 'gt' / 'gt.txt'

    print("Full 1050-frame sweep WITH ignore-region filtering")
    print("-" * 110)
    print(f"{'Model':12s} | {'Imgsz':5s} | {'Conf':5s} | {'MOTA':6s} | {'IDF1':6s} | {'Recall':6s} | {'Prec':6s} | {'ID Sw':5s} | {'Miss':5s} | {'FP':5s}")
    print("-" * 110)

    combinations = [
        ('yolov8s.pt', 1440, 0.20),
        ('yolov8s.pt', 1440, 0.25),
        ('yolov8s.pt', 1440, 0.30),
        ('yolov8s.pt', 1440, 0.35),
        ('yolov8s.pt', 1280, 0.25),
        ('yolov8s.pt', 1280, 0.30),
    ]

    for model, imgsz, conf in combinations:
        os.environ['GODS_EYE_DETECTION_MODEL'] = model
        os.environ['GODS_EYE_DETECTION_IMGSZ'] = str(imgsz)
        os.environ['GODS_EYE_DETECTION_CONFIDENCE'] = str(conf)
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()
        try:
            preds = _run_pipeline_on_sequence(_DEFAULT_DATA_ROOT, seq_info)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        
        gt = _load_ground_truth(gt_path)
        ignore = _load_ignore_regions(gt_path)
        metrics = _compute_mot_metrics(gt, preds, ignore=ignore)

        m, i, r, p = metrics['mota'], metrics['idf1'], metrics['recall'], metrics['precision']
        idsw, miss, fp = metrics['num_switches'], metrics['num_misses'], metrics['num_false_positives']
        status = "PASS" if m >= 0.60 else "FAIL"
        print(f"{model:12s} | {imgsz:5d} | {conf:5.2f} | {m:6.4f} [{status}] | {i:6.4f} | {r:6.4f} | {p:6.4f} | {idsw:5d} | {miss:5d} | {fp:5d}")
    print("-" * 110)

if __name__ == '__main__':
    main()
