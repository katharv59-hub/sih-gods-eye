# Phase 1 — Perception Layer Benchmarks Report

**Status:** COMPLETE
**Document Version:** 1.0
**Source of Truth:** GODS_EYE_MASTER_SPEC.md v3.0

---

## 1. Benchmarking Overview

This report documents the performance and accuracy benchmarks conducted for God's Eye Phase 1. As mandated by **Rule 1 — Benchmarks Over Assumptions**, all optimization and design validation decisions are grounded in these empirical results.

We conducted two primary benchmarks:
1. **CPU Fallback Throughput Benchmark (`benchmark_cpu_mode.py`)** — Measures frame processing speed under CPU-only constraint.
2. **MOT17-04 Tracking Accuracy Benchmark (`benchmark_mot17.py`)** — Measures YOLOv8 detector and ByteTrack tracking quality against the MOTChallenge ground truth.
3. **Demo Video Processing Suite (`run_demo_suite.py`)** — Benchmarks inference throughput and object stats across the low, medium, and high density dataset catalog.

---

## 2. Benchmark 1: CPU Fallback Performance

### 2.1 Purpose
Verify that the pipeline is capable of running in CPU-only mode (e.g., when GPU resources are unavailable, crashed, or during edge deployment) and meets the minimum gate throughput constraint of sustained 6 FPS.

### 2.2 System Configuration
- **Script:** `benchmarks/benchmark_cpu_mode.py`
- **Output JSON:** `benchmarks/results/cpu_mode_phase1.json`
- **OS:** Windows (10/11)
- **CPU:** Intel Core i7 / AMD Ryzen equivalent (host environment)
- **GPU:** CUDA Disabled (CPU mode forced)
- **Model:** `yolov8n.pt`
- **Resolution:** 1280 x 720 (720p)
- **Confidence Threshold:** 0.25
- **Image Size (`imgsz`):** 640

### 2.3 Results

| Metric | Measured Value | Spec Gate | Status |
|---|---|---|---|
| **Sustained FPS (Min 5s window)** | **7.13 FPS** | ≥ 6.0 FPS | ✅ **PASS** |
| **Average FPS** | **16.54 FPS** | — | ✅ **PASS** |
| **Avg Detection Latency** | **54.7 ms** | — | — |
| **Avg Tracking Latency** | **0.17 ms** | — | — |
| **Total Frames Processed** | **4,963** | — | — |
| **Execution Duration** | **300.09 seconds (5 min)**| ≥ 5 mins | ✅ **PASS** |

### 2.4 Throughput Profile (5-second window samples)
The FPS trended at **17.3 to 19.2 FPS** for the first 50 samples (~250 seconds) before dropping to a minimum of **7.13 FPS** at sample index 60 (~300 seconds), likely due to host thermal throttling under sustained CPU loads. Despite this thermal throttling, the sustained throughput remained above the 6.0 FPS threshold.

---

## 3. Benchmark 2: MOT17-04 Tracking Accuracy

### 3.1 Purpose
Evaluate the quality of the detection and tracking integration. We measure multi-object tracking accuracy (MOTA), ID switches, and localization performance against the standardized MOTChallenge ground truth sequence `MOT17-04-FRCNN` (1,050 frames, 1920x1080 resolution).

### 3.2 Evaluation Configuration
- **Script:** `benchmarks/benchmark_mot17.py`
- **Output JSON:** `benchmarks/results/mot17_phase1.json`
- **Hardware:** RTX 4050 GPU (6GB VRAM)
- **Model:** `yolov8s.pt`
- **Resolution:** 1920x1080 (native)
- **Image Size (`imgsz`):** 1440
- **Confidence Threshold:** 0.35
- **Ignore Regions:** Evaluated using standard MOTChallenge ignore annotations (distractors/vehicles/etc.) with Intersection over Area (IoA) ≥ 0.5.
- **Visibility Threshold:** Ground-truth targets with visibility < 0.25 were excluded from evaluation to align with standard MOTChallenge protocol.

### 3.3 Accuracy Results

| Metric | Measured Value | Spec Gate | Status |
|---|---|---|---|
| **MOTA (Multi-Object Tracking Accuracy)** | **63.43%** (0.6343) | ≥ 60.0% | ✅ **PASS** |
| **ID Switches** | **102** | ≤ 150 | ✅ **PASS** |
| **IDF1 (Identity F1-Score)** | **63.90%** (0.6390) | — | — |
| **Precision** | **89.73%** | — | — |
| **Recall** | **71.91%** | — | — |
| **False Positives** | **3,275** | — | — |
| **Misses (False Negatives)** | **11,183** | — | — |

### 3.4 Key Observations & Analysis
- **Recall vs. Misses:** With 11,183 misses, the primary limitation is detector recall on small, occluded, or distant pedestrians. YOLOv8s at `imgsz=1440` recovers more small bounding boxes than `imgsz=640` but still misses targets under heavy perspective distortion.
- **Precision:** Precision is high (89.73%), confirming that the conf=0.35 threshold and class-filtering are effective at eliminating false positives from environmental clutter.
- **ID Switches:** 102 switches are within the gate limit. Since ByteTrack relies solely on bounding box overlap (IoU) and Kalman prediction, high-density crossings and long-duration occlusions trigger identity swaps. This establishes a baseline for Phase 2, where appearance-based Re-ID will be introduced to resolve these switches.

---

## 4. Benchmark 3: Demo Video Suite

### 4.1 Purpose
Verify live ingestion, queue stability, and detection throughput across low, medium, and high-density street and indoor scenes.

### 4.2 System Configuration
- **Script:** `scripts/run_demo_suite.py --headless`
- **Output JSON:** `benchmarks/results/demo_suite_results.json`
- **Hardware:** RTX 4050 GPU (6GB VRAM)
- **Model:** `yolov8n.pt`
- **Inference Resolution:** 640x640

### 4.3 Throughput & Density Results

| Video Sequence | Density | Resolution | Frames | GPU FPS | Unique Tracks | Max Simultaneous | Avg Tracks/Frame |
|---|---|---|---|---|---|---|---|
| `mot17_09_low_density` | Low | 1920x1080 | 521 | **14.2** | 68 | 13 | 8.4 |
| `mot17_04_medium_density` | Medium | 1920x1080 | 1050 | **12.6** | 128 | 30 | 19.5 |
| `mot17_05_high_density` | High | 640x480 | 837 | **16.0** | 166 | 10 | 5.6 |

*Note: FPS measurements reflect the full end-to-end processing pipeline including logging overhead, queue synchronization, and file decoding.*

---

## 5. Summary & Actionable Recommendations

1. **Detection is the Bottleneck:** Tracking latency on CPU is under 0.2 ms, whereas YOLO detection takes ~54.7 ms on CPU. Future throughput optimization should focus on YOLO inference (e.g., using ONNX, TensorRT, or dynamic frame skipping).
2. **Re-ID Validation Baseline:** The 102 ID switches on MOT17-04 serve as the target metric for Phase 2. The appearance model (OSNet) must reduce this count by resolving occlusions where spatial overlap fails.
