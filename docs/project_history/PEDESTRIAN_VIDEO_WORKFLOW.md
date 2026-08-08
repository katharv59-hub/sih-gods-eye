# Pedestrian Video Validation Workflow

This document records the design, implementation, and verification of the pedestrian video validation workflow introduced after the Phase 1 final release of **God's Eye**.

---

## 1. Motivation & Context

In Phase 1, primary verification of the real-time tracking pipeline was conducted using a laptop webcam and synthetic single-object test videos. While sufficient to prove the basic connectivity of ingestion, detection, tracking, and visualization threads, this setup failed to represent the production environments of an intelligent public safety or crowd analysis tracking system. 

Specifically, webcam validation suffered from the following shortcomings:
- **Low Target Density:** Test scenes typically contained only 1-2 cooperative individuals in an indoor office setting.
- **Homogeneous Scale & Motion:** Lack of complex trajectories, depth variations, scale transitions, and high-frequency occlusions.
- **Non-reproducible Inputs:** Live webcam inputs vary dynamically with environmental lighting, participant behavior, and frame rates, preventing consistent automated regression testing.
- **Stale Position Auditing Limitations:** Hard to validate tracker thresholding, eviction policies, and target loss behavior under simple paths.

To resolve these limits, we transitioned validation and demonstration tasks to a realistic pedestrian-scene workflow utilizing recorded benchmark datasets.

---

## 2. Dataset Selection

We chose sequences from the **MOT17 (Multi-Object Tracking 2017)** benchmark dataset. MOT17 is the industry-standard benchmark for evaluating multi-pedestrian tracking algorithms.

### Why MOT17?
1. **High-Precision Ground Truth:** Hand-annotated labels containing pedestrian coordinates, visibility ratios, and target classes (pedestrians, vehicles, occluded regions).
2. **Standardized Scenarios:** Fixed camera views, varying lighting, moving cameras, and different crowd densities.
3. **Established Baselines:** Direct comparison against standard multi-object tracking metrics (MOTA, IDF1, ID Switches).
4. **Permissive Non-Commercial Licensing:** Available under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license.

### Selection Process
We selected three sequences representing three distinct levels of pedestrian density and tracking complexity:
- **MOT17-09 (Low Density):** 1080p, static camera, outdoor corridor. Used as a baseline for detection recall and track lifecycle startup.
- **MOT17-04 (Medium Density):** 1080p, static camera, high vantage point, continuous pedestrian flows. Used as the recommended demonstration and validation default.
- **MOT17-05 (High Density):** 480p, moving camera, low vantage point, crowded market scene. Used to stress-test occlusion handling and tracker eviction.

---

## 3. Video Catalog & Benchmark Summary

All sequences were downloaded from the Hugging Face dataset mirror and converted from raw image series to highly compressed, high-fidelity MP4 video files using OpenCV (`mp4v` codec).

| Video File | Density | Resolution | Duration (s) | Unique Pedestrians (GT) | Track IDs (YOLO) | Verification | Status |
|---|---|---|---|---|---|---|---|
| `mot17_09_low_density.mp4` | Low | 1920x1080 | 17.4s | 26 | 68 | OpenCV Read-back | ✅ **PASS** |
| `mot17_04_medium_density.mp4` | Medium | 1920x1080 | 35.0s | 83 | 128 | OpenCV Read-back | ✅ **PASS** |
| `mot17_05_high_density.mp4` | High | 640x480 | 27.9s | 133 | 166 | OpenCV Read-back | ✅ **PASS** |

---

## 4. Verification & Demo Workflow

### 4.1 Automated Validation (CI/CD and Headless)
Automated testing and benchmark tracking are handled by `scripts/run_demo_suite.py`. This script runs YOLOv8 detection and ByteTrack tracking on each video sequentially without launching a GUI:

```bash
# Run the validation suite headless (no UI window)
python scripts/run_demo_suite.py --headless
```

Performance and detection stats are exported to a machine-readable JSON log for regressions:
- **Output file:** `benchmarks/results/demo_suite_results.json`

### 4.2 Presentation & Demo Mode
To demonstrate God's Eye performance visually, execute the visualization script targeting the recommended default video:

```bash
# Run visual demonstration with full overlay HUD on the medium density street scene
python run_demo.py tests/data/demo_videos/mot17_04_medium_density.mp4
```

### 4.3 Key Controls in Demo Mode
- `Q`: Quit pipeline and exit.
- `S`: Save current frame screenshot to `tests/data/screenshots/`.

---

## 5. Storage & Repository Hygiene

### 5.1 Storage Considerations
Raw MOT17 training sequences consist of uncompressed JPEGs and can exceed **5 GB** in size. To prevent repository bloating and allow for quick environment setups:
- Image directories under `tests/data/MOT17/` are utilized locally for tracking accuracy evaluation and are excluded from Git.
- Converted MP4 files are compressed into a compact suite totaling **107.2 MB** and are stored in `tests/data/demo_videos/`.
- Executing `python scripts/download_demo_videos.py` automatically reconstructs or downloads the required frames and outputs the MP4 files directly.

### 5.2 Git Exclusions (Hygiene Audit)
The following directories and patterns are configured in `.gitignore` to prevent large binary leaks:
- `tests/data/MOT17/` (Raw image files and ground truth datasets)
- `tests/data/MOT17-val/` (Validation split directories)
- `tests/data/demo_videos/*.mp4` (Highly compressed MP4s)
- `tests/data/demo_outputs/` (Rendered debug outputs)
- `*.zip` (Downloaded dataset zip archives)
- `*.pt` (YOLO model weights)

**Risk Assessment:** The repository is clean of binary bloat. Untracked metadata like `tests/data/demo_videos/catalog.json` should be tracked to maintain a record of the video schemas, which has been done.

---

## 6. Dataset Licensing

The MOT17 dataset is restricted by the **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)** license.
- **Research & Development:** Fully permitted.
- **Commercialization:** Commercial applications are strictly prohibited.
- **Derived Works:** The MP4 files created from the image frames are derivative works and are subject to the same license constraints.

---

## 7. Lessons Learned & Technical Recommendations

1. **Resolution Sensitivity:** YOLOv8n struggles with distant/small pedestrians when processing at its default `640px` resolution (noticeable in `mot17_05_high_density.mp4`). For high-vantage surveillance feeds, inference resolution should be dynamically scaled or tiled.
2. **Camera Motion Challenges:** In `mot17_05` (moving camera), the spatial displacement of targets frame-to-frame exceeds Kalman filter prediction bounds. Simple spatial-based association causes frequent ID switches. 
3. **Phase 2-6 Recommendation:** 
   - **Phase 2 (Re-ID & Identity Mapping):** Pedestrian videos must become the default validation source. Re-ID embeddings cannot be verified using static webcam feeds of a single person. Low and Medium density videos will act as a control set to tune embedding similarity thresholds.
   - **Phase 3 (Temporal Memory):** High-density videos will serve as the testing bed for database/in-memory state management, indexing speed, and lookup latency.
   - **Phases 4-6 (Reasoning, Dashboard, API):** The multi-pedestrian videos will simulate a live multi-camera feed to test dashboard rendering performance and LLM-driven query capabilities.
