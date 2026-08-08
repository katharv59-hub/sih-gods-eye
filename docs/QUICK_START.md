# Quick Start Guide

Welcome to **God's Eye**! This guide will walk you through setting up the workspace, downloading model weights and datasets, running demos, running tests and benchmarks, and validating pipeline phases.

---

## 1. Installation

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/katharv59-hub/gods_eye.git
cd gods_eye
```

### Environment Setup
We recommend using a Python 3.11 virtual environment.
```powershell
# Create environment
python -m venv venv

# Activate environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate environment (Linux/macOS)
source venv/bin/activate
```

### Dependencies
Install the required packages. Note that the project utilizes GPU-accelerated execution if compatible CUDA hardware is present.
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## 2. Model Downloads

God's Eye requires pretrained weights for person detection (YOLOv8n) and person Re-Identification (OSNet x1.0).

### Person Detection (YOLOv8n)
The system automatically downloads the `yolov8n.pt` model weights (6.2 MB) on first run, or you can retrieve it directly:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Person Re-ID (OSNet x1.0)
For Phase 2 appearance matching, you must download the Market-1501 fine-tuned weights:
1. Create a `data` folder at the root of the project:
   ```bash
   mkdir data
   ```
2. Download the model weights `osnet_x1_0_market1501.pth` (10.4 MB) from the official TorchReID Model Zoo:
   * **Source URL:** [Google Drive Download Link](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing)
   * Save the downloaded file directly to `data/osnet_x1_0_market1501.pth`.
3. Verify the weights file via the launcher status command:
   ```bash
   python scripts/run_gods_eye.py models
   ```

---

## 3. Dataset Setup

### Demo Videos
Download the short MOT17 high, medium, and low density demo video clips (used for quick testing):
```bash
python scripts/download_demo_videos.py
```

### MOT17 Dataset
Download the MOT17 sequence validation files (needed to run tracking accuracy benchmarks):
```bash
python scripts/download_mot17.py
```
This extracts the verification sequences directly into `tests/data/MOT17/`.

### Market-1501 Dataset
To run Re-ID model benchmarking:
1. Download the Market-1501 dataset zip file:
   * **Source URL:** [Market-1501 Download](http://128.84.21.199/ece/research/mcv/pdfs/Market-1501-v15.09.15.zip)
2. Save it to `data/market-1501.zip`.
3. Extract it so that the following directory structure is created:
   `data/market1501/Market-1501-v15.09.15/` containing the subfolders `query/` and `bounding_box_test/`.

Verify the dataset directories:
```bash
python scripts/run_gods_eye.py datasets
```

---

## 4. Run Demos

God's Eye has a unified project launcher: `scripts/run_gods_eye.py`.

### Run the Webcam Demo
Verify system-to-sensor integration live:
```bash
python scripts/run_gods_eye.py demo webcam
```

### Run the Demo Video Suite
Runs the detection and tracking pipeline over all downloaded demo videos:
```bash
# Run all demo videos
python scripts/run_gods_eye.py demo

# Run a specific demo video (e.g. mot17_09)
python scripts/run_gods_eye.py demo mot17_09
```

---

## 5. Run Tests

Verify code health and test suite assertions:
```bash
# Run all tests
python scripts/run_gods_eye.py test

# Run Re-ID / Similarity unit tests only
python scripts/run_gods_eye.py test reid

# Run tracking tests only
python scripts/run_gods_eye.py test tracking

# Run detection tests only
python scripts/run_gods_eye.py test detection
```

---

## 6. Run Benchmarks

Measure processing throughput and tracking accuracy:

```bash
# Run CPU-only mode throughput benchmark (Gate: >= 6 FPS)
python scripts/run_gods_eye.py benchmark cpu

# Run MOT17-04 Tracking Accuracy benchmark (Gate: MOTA >= 0.60, ID Switches <= 150)
python scripts/run_gods_eye.py benchmark mot17

# Run Re-ID Embedding Accuracy benchmark on Market-1501 (Gate: Rank-1 >= 0.88)
python scripts/run_gods_eye.py benchmark reid

# Run Demo Suite processing throughput benchmark (headless)
python scripts/run_gods_eye.py benchmark demo_suite
```

---

## 7. Validate a Phase

Run the full validation suite to gate release versions:

```bash
# Run Phase 1 validation (Ingestion, Detection, Tracking, Pipeline integration)
python scripts/run_gods_eye.py validate phase1

# Run Phase 2 validation (Re-ID unit tests and Market-1501 benchmarks)
python scripts/run_gods_eye.py validate phase2
```
