# Operations and System Verification Guide

This document provides procedures for demonstrating the **God's Eye** system, verifying system health, and recovering from failures.

---

## 1. System Demonstrations

### Webcam Demo
Demonstrates live integration of the detection, tracking, and annotation pipeline.
1. Connect a compatible USB webcam.
2. Run:
   ```bash
   python scripts/run_gods_eye.py demo webcam
   ```
3. Press **Q** in the GUI window to gracefully stop the demo.

### Recorded Pedestrian Demo
Demonstrates the tracking pipeline on standard multi-person tracking video files (MOT17 sequences).
1. Download video clips:
   ```bash
   python scripts/run_gods_eye.py demo
   ```
2. Run the pipeline over a specific video clip (e.g. `mot17_09` low density sequence):
   ```bash
   python scripts/run_gods_eye.py demo mot17_09
   ```
3. Controls:
   - **Q**: Skip current video / Exit.
   - **S**: Save a screenshot of the current annotated frame to `tests/data/screenshots/`.

### Benchmark Demo
Demonstrates performance profiling under different configurations (CPU-only, GPU/CUDA, and Re-ID evaluation).
```bash
# Run CPU-only mode profiling
python scripts/run_gods_eye.py benchmark cpu

# Run MOT17 accuracy evaluation
python scripts/run_gods_eye.py benchmark mot17
```

### Presentation Demo
To run the demo suite saving annotated output videos for offline review or presentation:
```bash
# Run the demo suite and save outputs
python scripts/run_gods_eye.py demo --save-output
```
Annotated videos will be generated under `tests/data/demo_outputs/`.

---

## 2. System Verification Procedures

### Verify PyTorch and CUDA
Ensure the environment detects the local GPU device for hardware-accelerated inference:
```bash
python scripts/run_gods_eye.py environment
```
Check that `CUDA Available: True` and verify the detected GPU card (e.g. `RTX 4050 Laptop GPU`).

### Verify Model Weights
Ensure detection and Re-ID model weights are downloaded and have correct integrity:
```bash
python scripts/run_gods_eye.py models
```
Ensure sizes and SHA256 checksums match the frozen weights configuration.

### Verify Datasets
Ensure tracking and validation datasets are present and unpacked in their correct directories:
```bash
python scripts/run_gods_eye.py datasets
```

### Verify Tests & Benchmarks
Ensure the codebase passes all assertion tests:
```bash
# Run unit tests
python scripts/run_gods_eye.py test

# Verify tracking accuracy metrics on MOT17
python scripts/run_gods_eye.py benchmark mot17
```

---

## 3. Failure Recovery Manual

### Issue: Missing Weights (`FileNotFoundError`)
* **Symptom**: Launcher reports missing weights or the execution fails with `FileNotFoundError` when initializing the extractor or detector.
* **Resolution**:
  - **YOLOv8n**: Delete any corrupted `yolov8n.pt` at the root and let the detector auto-download it on launch.
  - **OSNet Re-ID**: Verify the file `data/osnet_x1_0_market1501.pth` exists. If missing, download it from the official Google Drive link (`1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`) and place it in the `data/` folder. Verify its integrity:
    ```bash
    python scripts/run_gods_eye.py models
    ```

### Issue: CUDA Unavailable / CPU Fallback
* **Symptom**: `CUDA Available: False` in environment check. Pipeline FPS runs extremely low because YOLOv8/OSNet is executing on the CPU.
* **Resolution**:
  1. Verify the Nvidia GPU driver is installed and updated.
  2. Verify PyTorch is compiled with CUDA support:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```
  3. If false, reinstall PyTorch with the correct CUDA execution runtime:
     ```bash
     pip uninstall torch torchvision
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```

### Issue: Missing Dataset Directories
* **Symptom**: `datasets` check reports `MISSING` for demo videos, MOT17, or Market-1501.
* **Resolution**:
  - **Demo Videos**: Run `python scripts/download_demo_videos.py` to auto-fetch.
  - **MOT17**: Run `python scripts/download_mot17.py` to auto-fetch.
  - **Market-1501**: Download the dataset zip, place it at `data/market-1501.zip`, and extract it. The extractor folder must result in the path `data/market1501/Market-1501-v15.09.15/`.

### Issue: Dependency Conflict / Import Errors
* **Symptom**: `ImportError` or `ModuleNotFoundError` on packages like `torchreid` or `supervision`.
* **Resolution**:
  1. Ensure the virtual environment is activated before running scripts.
  2. Reinstall packages from `requirements.txt`:
     ```bash
     pip install --force-reinstall -r requirements.txt
     ```
  3. If `torchreid` still fails, verify compatibility with PyTorch: `torchreid` requires PyTorch versions < 2.6.
