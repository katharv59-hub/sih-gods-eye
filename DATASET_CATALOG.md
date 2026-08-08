# God's Eye — Demo Video Dataset Catalog

**Last Updated:** 2026-06-22
**Source:** MOT17 Multi-Object Tracking Benchmark (MOTChallenge)
**License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)
**Citation:** Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). MOT16: A Benchmark for Multi-Object Tracking.

---

## Overview

All demo videos are derived from the **MOT17** training set, converted from JPEG image sequences to MP4 using OpenCV (`mp4v` codec). They provide realistic multi-pedestrian scenes at varying crowd densities for detection, tracking, and future Re-ID testing.

**Download command:**
```bash
python scripts/download_demo_videos.py
```

**Run all demos:**
```bash
python scripts/run_demo_suite.py              # with GUI
python scripts/run_demo_suite.py --headless   # no GUI (CI mode)
python scripts/run_demo_suite.py --save-output # save annotated videos
```

**Run single video:**
```bash
python run_demo.py tests/data/demo_videos/mot17_04_medium_density.mp4
```

---

## Video Catalog

### 1. Low Density — `mot17_09_low_density.mp4`

| Property | Value |
|----------|-------|
| **Source Sequence** | MOT17-09-FRCNN |
| **Resolution** | 1920 x 1080 |
| **FPS** | 30 |
| **Frames** | 521 |
| **Duration** | 17.4 seconds |
| **File Size** | 43.4 MB |
| **Unique Pedestrians** | 26 (GT) / 68 (detected tracks) |
| **Scene** | Indoor/outdoor pedestrian area, static camera |
| **Density** | Low — typically 5-13 simultaneous persons |
| **Intended Use** | Baseline detection quality, single-person tracking validation, Re-ID gallery building |

### 2. Medium Density — `mot17_04_medium_density.mp4` ⭐ Recommended Default

| Property | Value |
|----------|-------|
| **Source Sequence** | MOT17-04-FRCNN |
| **Resolution** | 1920 x 1080 |
| **FPS** | 30 |
| **Frames** | 1050 |
| **Duration** | 35.0 seconds |
| **File Size** | 46.7 MB |
| **Unique Pedestrians** | 83 (GT) / 128 (detected tracks) |
| **Scene** | Urban street with continuous pedestrian flow, static camera |
| **Density** | Medium — typically 15-30 simultaneous persons |
| **Intended Use** | Primary benchmark video, MOT accuracy validation, pipeline stress testing |

### 3. High Density — `mot17_05_high_density.mp4`

| Property | Value |
|----------|-------|
| **Source Sequence** | MOT17-05-FRCNN |
| **Resolution** | 640 x 480 |
| **FPS** | 30 (14 native) |
| **Frames** | 837 |
| **Duration** | 27.9 seconds |
| **File Size** | 17.1 MB |
| **Unique Pedestrians** | 133 (GT) / 166 (detected tracks) |
| **Scene** | Busy pedestrian marketplace, moving camera |
| **Density** | High — frequent occlusions, camera motion, small targets |
| **Intended Use** | Occlusion handling, crowd analysis, ID switch stress testing |

---

## Storage Summary

| Video | Size |
|-------|------|
| mot17_09_low_density.mp4 | 43.4 MB |
| mot17_04_medium_density.mp4 | 46.7 MB |
| mot17_05_high_density.mp4 | 17.1 MB |
| **Total** | **107.2 MB** |

---

## License Compliance

The MOT17 dataset is released under **CC BY-NC-SA 3.0**:

- ✅ **Non-commercial research and development** — permitted
- ✅ **Attribution** — provided via citation above
- ✅ **Share-alike** — derivative works (our MP4 conversions) carry the same license
- ❌ **Commercial use** — not permitted under this license

These videos are used exclusively for internal development, benchmarking, and demo purposes within the God's Eye project.

---

## Verification Status

All videos verified via OpenCV read-back:

| Video | Format | Resolution | FPS | Frames | Readable | Status |
|-------|--------|-----------|-----|--------|----------|--------|
| mot17_09_low_density.mp4 | MP4 (mp4v) | 1920x1080 | 30.0 | 521 | ✅ | OK |
| mot17_04_medium_density.mp4 | MP4 (mp4v) | 1920x1080 | 30.0 | 1050 | ✅ | OK |
| mot17_05_high_density.mp4 | MP4 (mp4v) | 640x480 | 30.0 | 837 | ✅ | OK |
