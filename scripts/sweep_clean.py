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
    seq_info['seqlength'] = '200'

    print("Running yolov8s imgsz sweep on 200 frames...")
    print("-" * 110)
    print(f"{'Model':12s} | {'Imgsz':5s} | {'Conf':5s} | {'MOTA':6s} | {'IDF1':6s} | {'Recall':6s} | {'Prec':6s} | {'ID Sw':5s} | {'Miss':5s} | {'FP':5s}")
    print("-" * 110)

    combinations = []
    # Test yolov8s with imgsz in [1088, 1280, 1440, 1920] and conf in [0.25, 0.30, 0.35]
    for conf in [0.25, 0.30, 0.35]:
        for imgsz in [1088, 1280, 1440, 1920]:
            combinations.append(('yolov8s.pt', imgsz, conf))

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
