# Pedestrian Video Setup Report

**Date:** 2026-06-22
**Scope:** Replace webcam-only validation with recorded pedestrian video workflow
**Architecture Impact:** None — no Phase 1 code modified

---

## 1. Videos Downloaded

Three MOT17 training sequences were downloaded from the Lekim89/MOT17 Hugging Face mirror and converted to MP4:

| Video | Source | Resolution | FPS | Frames | Duration | Size | Pedestrians |
|-------|--------|-----------|-----|--------|----------|------|-------------|
| `mot17_09_low_density.mp4` | MOT17-09-FRCNN | 1920x1080 | 30 | 521 | 17.4s | 43.4 MB | 26 |
| `mot17_04_medium_density.mp4` | MOT17-04-FRCNN | 1920x1080 | 30 | 1050 | 35.0s | 46.7 MB | 83 |
| `mot17_05_high_density.mp4` | MOT17-05-FRCNN | 640x480 | 30 | 837 | 27.9s | 17.1 MB | 133 |

**Total storage:** 107.2 MB

---

## 2. Storage Layout

```
tests/data/demo_videos/
    mot17_09_low_density.mp4       (43.4 MB)
    mot17_04_medium_density.mp4    (46.7 MB)  ← Recommended default
    mot17_05_high_density.mp4      (17.1 MB)
    catalog.json                   (metadata)
```

Source image sequences remain cached in `tests/data/MOT17/train/` for benchmark use.

---

## 3. Validation Status

All videos verified by:
1. **OpenCV read-back** — correct resolution, FPS, frame count
2. **Full pipeline execution** — detection + tracking successful on all 3 videos
3. **Demo suite** — `scripts/run_demo_suite.py --headless` completed without errors

### Pipeline Performance (yolov8n.pt, GPU)

| Video | Avg FPS | Detections/Frame | Unique Tracks | Max Simultaneous |
|-------|---------|-----------------|---------------|-----------------|
| Low density (MOT17-09) | 14.2 | 9.5 | 68 | 13 |
| Medium density (MOT17-04) | 12.6 | 20.8 | 128 | 30 |
| High density (MOT17-05) | 16.0 | 6.9 | 166 | 10 |

Note: MOT17-05 has lower detections/frame despite highest density because it is 640x480 resolution with a moving camera, making many pedestrians small and partially occluded.

---

## 4. Recommended Default Demo Video

**`mot17_04_medium_density.mp4`** is the recommended default for the following reasons:

- **1920x1080** — matches typical deployment resolution
- **35 seconds** — long enough for meaningful tracking analysis
- **83 unique pedestrians** — challenging but representative
- **Static camera** — matches the primary use case (fixed surveillance)
- **Already benchmarked** — MOTA 0.6343 validated against ground truth

Usage:
```bash
python run_demo.py tests/data/demo_videos/mot17_04_medium_density.mp4
```

---

## 5. Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/download_demo_videos.py` | Download MOT17 sequences from HF mirror and convert to MP4 |
| `scripts/run_demo_suite.py` | Run detection + tracking on all demo videos with overlay |

### Key Features

**`download_demo_videos.py`:**
- Parallel download (32 threads)
- Skip already-downloaded frames
- OpenCV-based image-to-video conversion (no ffmpeg required)
- Automatic verification after conversion
- JSON catalog output

**`run_demo_suite.py`:**
- Processes all videos sequentially
- `--headless` mode for CI/automated runs
- `--save-output` mode to produce annotated mp4 files
- `--video <name>` to run a single video
- Per-video statistics (FPS, tracks, detections)
- JSON results output to `benchmarks/results/demo_suite_results.json`

---

## 6. Files Created/Modified

| File | Action |
|------|--------|
| `scripts/download_demo_videos.py` | **Created** — download and convert script |
| `scripts/run_demo_suite.py` | **Created** — automated demo suite runner |
| `DATASET_CATALOG.md` | **Created** — full video catalog with license info |
| `.gitignore` | **Modified** — added demo video exclusions |
| `tests/data/demo_videos/catalog.json` | **Generated** — machine-readable catalog |
| `benchmarks/results/demo_suite_results.json` | **Generated** — pipeline performance results |

---

## 7. Existing Code — No Modifications

The following Phase 1 modules were used **without any changes**:

- `run_demo.py` — already supported `python run_demo.py <video.mp4>` ✅
- `gods_eye/detection/yolo_detector.py` — unchanged ✅
- `gods_eye/tracking/bytetrack_tracker.py` — unchanged ✅
- `gods_eye/ingestion/source.py` — `VideoFileSource` already existed ✅
- `gods_eye/pipeline.py` — unchanged ✅

**Test suite:** 43/43 tests passing (verified post-setup).
