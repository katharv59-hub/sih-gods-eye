# GOD'S EYE (SIH 26187)
## Autonomous Multi-Camera Spatiotemporal Surveillance & Forensic Intelligence System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch & YOLOv8](https://img.shields.io/badge/Computer%20Vision-YOLOv8%20%7C%20ByteTrack-ff69b4.svg)](https://ultralytics.com)
[![OpenCV YuNet + SFace](https://img.shields.io/badge/Face%20Biometrics-YuNet%20%2B%20SFace-green.svg)](https://opencv.org)
[![Tests](https://img.shields.io/badge/Tests-570%20passed%20%28100%25%29-brightgreen.svg)](tests/)

---

## 1. Executive Summary: What is God's Eye?

**God's Eye** is a production-grade, autonomous situational intelligence platform built for modern security installations (airports, critical infrastructure, defense perimeters, financial institutions, and smart cities).

In traditional surveillance setups, human security operators stare at dozens of CCTV monitors. Studies show that human operators **miss over 90% of suspicious activities after just 20 minutes** due to cognitive fatigue. Moreover, during investigations, officers must manually scrub through hundreds of hours of recorded footage across disconnected camera angles to reconstruct an incident.

**God's Eye solves both problems automatically:**
1. **Real-Time Automated Surveillance**: Continuously ingests multiple video streams, detects and tracks pedestrians and vehicles across non-overlapping camera views, recognizes enrolled faces, reads license plates (ANPR), and evaluates spatial rule violations (geofencing, loitering, off-hours night movement).
2. **Forensic Natural Language Intelligence (NLQ)**: Instead of watching recordings, investigators ask plain-English questions (e.g., *"Where was vehicle DL01AB1234 seen?"* or *"Show active hypotheses for vault intrusion"*). The deterministic reasoning engine queries an immutable, append-only event store and answers in milliseconds with cryptographic evidence chains.

---

## 2. Why is God's Eye Designed Like This? (Core Philosophy)

Every architectural decision in God's Eye was made to solve real-world operational and legal challenges:

### A. Deterministic Reasoning vs. LLM Hallucinations
* **The Problem**: Large Language Models (LLMs) frequently hallucinate facts, invent timestamps, or misread coordinates. In security, law enforcement, and courtrooms, fabricated evidence is unacceptable.
* **The God's Eye Solution**: The AI reasoning layer uses a **Strict 13-Tool Allowlist**. When an investigator asks a question, the system parses the intent into a deterministic query plan executed by mathematical modules (spatial ray-casting, statistical $Z$-scores, graph traversal, and SQL filters). It never fabricates data; if information is absent, it returns an explicit `not_found` verdict.

### B. Append-Only, Immutable Event Store (Legal Chain of Custody)
* **The Problem**: In conventional databases, rows can be overwritten or deleted, making evidence vulnerable to tampering or data corruption.
* **The God's Eye Solution**: Events are written to an append-only SQLite database operating in **Write-Ahead Logging (WAL) mode**. Events cannot be updated in-place. Every critical event (face detection, plate sighting, perimeter breach) stores:
  * Millisecond-level hardware timestamp (`timestamp_ns`)
  * Full-frame 1080p photographic evidence
  * Tight bounding-box crop
  * Cryptographic **SHA-256 hash** of the imagery to guarantee courtroom admissibility.

### C. Edge-First, Real-Time Architecture (Zero Cloud Subscriptions)
* **The Problem**: Sending dozens of 4K video feeds to the cloud introduces latency, incurs massive bandwidth costs, and exposes sensitive citizen footage to third-party cloud breaches.
* **The God's Eye Solution**: God's Eye runs 100% on-premises. Perception uses highly optimized neural models:
  * **YOLOv8 + ByteTrack**: Real-time object detection and Kalman-filter multi-object tracking.
  * **YuNet**: Ultralight (300KB) convolutional face detector running at 100+ FPS.
  * **SFace**: Efficient 128-dimensional biometric embedding model running in OpenCV DNN.
  * **MJPEG Low-Latency Streaming**: Camera viewports stream at 30 FPS with negligible CPU overhead (~9ms per frame encode time).

---

## 3. System Architecture & The 10 Engineering Phases

God's Eye was engineered systematically across 10 modular, strictly tested phases:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 SURVEILLANCE CAMERAS                   │
                  │  [CAM-01: Gate]  [CAM-02: Vault]  [CAM-03: Corridor]  │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ Video Streams (RTSP / MP4 / Webcams)
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1–3: INGESTION, TOPOLOGY & PERCEPTION ENGINE                                       │
│  ├─ Multi-Camera Ingestion (Timestamp sync, FPS throttle)                                │
│  ├─ YOLOv8 Object Detection (Persons, Vehicles, Classes)                                 │
│  ├─ ByteTrack Multi-Object Tracking (Trajectory smoothing, Track IDs)                   │
│  ├─ Multi-Camera Re-ID (HDBSCAN Clustering, Cross-Camera Handoffs)                       │
│  └─ Environmental Engine (Lux estimation, Night-IR transition, FOV overlaps)             │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │ Canonical Spatiotemporal Events
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 8–9: EXTENDED PERCEPTION & SPATIAL RULES                                           │
│  ├─ YuNet + SFace Facial Recognition (Known staff vs. Unknown strangers)                 │
│  ├─ Canny + Contour ANPR Engine (License Plate Extraction + EasyOCR floor >= 0.60)       │
│  ├─ Geofence Ray-Casting Engine (Point-in-polygon intrusion alerts)                      │
│  └─ Statistical Dwell Engine (Loitering anomaly detection: Dwell > 5.0σ)                 │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
┌──────────────────────────────────────────────┐ ┌─────────────────────────────────────────┐
│ PHASE 4 & 10: IMMUTABLE EVIDENCE VAULT       │ │ PHASE 5–7: REASONING & RISK ENGINE      │
│  ├─ SQLiteEventStore (Append-only, WAL mode) │ │  ├─ 13 Approved Deterministic Tools     │
│  ├─ EvidenceManager (SHA-256 Snapshots)      │ │  ├─ Hypothesis Forest (Anomaly trees)   │
│  ├─ AlertStore (Deduplication & Triage)      │ │  ├─ Trajectory Extrapolation (Markov)   │
│  └─ Auto-Retention & TTL Cleanup             │ │  └─ Situational Risk Evaluator (Z-score)│
└───────────────────────┬──────────────────────┘ └────────────────────┬────────────────────┘
                        │                                             │
                        └──────────────────────┬──────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 10: TACTICAL COMMAND & CONTROL (C2) & FASTAPI REST PLATFORM                        │
│  ├─ REST API (/cameras, /events, /alerts, /plates/{id}, /faces/enroll, /evidence/{id})   │
│  ├─ Real-Time MJPEG Multi-Camera Video Streaming Engine (/feeds/cam/{cam_id})            │
│  ├─ Tactical C2 Dashboard (Live camera grid, threat signal triage, NLQ console)          │
│  └─ Biometric Face Recognition Studio (Live device webcam scanner, enrollment, gallery)  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Phase Breakdown

| Phase | Subsystem | Responsibility |
|---|---|---|
| **Phase 1** | Ingestion Pipeline | Multi-camera RTSP/MP4 ingestion, timestamp normalization, FPS throttling, frame queue management. |
| **Phase 2** | Detection & Re-ID | YOLOv8 person/object detection, ByteTrack Kalman filtering, cross-camera global identity tracking with HDBSCAN. |
| **Phase 3** | Spatial Topology | Graph modeling of camera network (`node_cameras`), FOV overlaps, distance matrices, ambient lux calculation. |
| **Phase 4** | Immutable Event Store | Append-only `SQLiteEventStore` with strict schema validation, WAL mode, spatiotemporal indexing. |
| **Phase 5** | NLQ Reasoning Engine | Tools 1–8: spatial search, velocity calculation, timeline generation, cross-camera transition queries. |
| **Phase 6** | Trajectory Prediction | Markov transition matrices, dwell estimation, and predicting which camera a suspect will enter next. |
| **Phase 7** | Situational Risk Engine | Hypothesis Forest, statistical $Z$-score anomaly detection (`dwell > 5.0σ`), Tools 11 & 12. |
| **Phase 8** | Extended Perception | Vehicle tracking, ANPR plate OCR (`PlateOCR` with $\ge 0.60$ confidence floor), YuNet face detector + SFace recognizer. |
| **Phase 9** | Spatial Geofencing | Point-in-polygon ray casting (`FenceEngine`), loitering detection (`DwellEngine`), night lighting transition (`NightMovementEngine`). |
| **Phase 10** | Alert & Command Center | `AlertEngine` (30s incident deduplication), `EvidenceManager` (SHA-256 snapshots), FastAPI REST API, live tactical C2 web dashboard. |

---

## 4. Key Features & How to Use the System

### Feature 1: Tactical Command & Control (C2) Dashboard
Access via browser at `http://127.0.0.1:8000/dashboard`:
* **Live Surveillance Video Streams**: Real-time 30 FPS streams from all 4 viewports (North Gate, Vault Perimeter, ATM Corridor, and Night-IR Loading Dock).
* **Live Webcam Toggle on CAM-01**: Click `[ 📹 Live Webcam ]` to switch CAM-01 seamlessly between the MOT17 benchmark CCTV footage and your computer's actual webcam.
* **Threat Signals Panel**: Live incident cards (Critical, High Risk, Fences Monitored, Resolved) with one-click threat simulations (`⚡ Simulate Intrusion`, `⚡ Simulate Plate`).
* **Tactical Natural Language Investigation Console**: Type queries like *"Where was license plate DL01AB1234 seen?"* or *"Show active hypotheses for vault intrusion"* to dispatch queries directly to the deterministic Reasoning Engine (Tool 13).

### Feature 2: Face Biometrics & Live Scanner Studio
Navigate to the **Face Recognition & Biometrics** tab:
* **Live Biometric Webcam Scanner**: Prompts for browser camera permissions and scans faces in real-time.
* **HUD Bounding Boxes**:
  * 🟢 **GREEN BOX (`KNOWN: Name`)**: Recognized face matching an enrolled profile with cosine similarity score.
  * 🟠 **AMBER BOX (`UNKNOWN`)**: Unrecognized face detected in frame.
* **Instant Face Enrollment**:
  * Upload a reference photo OR click **`[ 📸 Snap & Enroll ]`** directly from the live webcam feed.
  * Extracted via SFace (128-dimensional embedding vector) and saved into the biometric gallery.
* **Enrollment Gallery Management**:
  * Inspect all enrolled staff members with reference photos, person IDs, and enrollment timestamps.
  * Delete individual profiles with one click (`🗑️`).
* **Audit Evidence Drawer**:
  * Click any live face detection in the stream to open the full Forensic Evidence Dossier showing full-frame 1080p snapshots, tight face crops, similarity scores, and SHA-256 verification hashes.

---

## 5. Quick Start Guide

### Prerequisites
* Windows 10/11, macOS, or Linux
* Python 3.11 or higher
* Webcam (optional, for live biometric testing)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/katharv59-hub/gods_eye.git
cd gods_eye

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
# (or pip install fastapi uvicorn opencv-python ultralytics pytest numpy)
```

### Running the Live Platform
Start the FastAPI server:
```bash
python -m uvicorn gods_eye.api.app:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000/dashboard`**

### Running Unit Tests
To verify all 570 unit tests across all 10 phases:
```bash
pytest
```
To run the API and biometrics suite specifically:
```bash
pytest tests/test_sih_api.py -v
```
*(All 41 API and biometric tests will pass with 100% success).*

---

## 6. REST API Reference

All endpoints support optional Bearer Token authentication via `GODS_EYE_API_TOKEN`:

| Endpoint | Method | Description |
|---|---|---|
| `/cameras` | `GET` | List all configured cameras, operational statuses, and stream URLs. |
| `/feeds/cam/{cam_id}` | `GET` | Live MJPEG video stream for a camera viewport (30 FPS). |
| `/events` | `GET` | Query canonical spatiotemporal events with filters (`limit`, `event_type`). |
| `/alerts` | `GET` | Fetch active security alerts filtered by severity (`CRITICAL`, `HIGH`, etc.). |
| `/alerts/{id}` | `GET` | Retrieve single alert details, explanations, and evidence references. |
| `/alerts/simulate` | `POST` | Inject a simulated threat incident (`intrusion` or `plate`) for testing. |
| `/plates/{plate}` | `GET` | Query vehicle sighting history and trajectory (Tool 13: `get_plate_history`). |
| `/faces/enrolled` | `GET` | List all enrolled subjects in the biometric gallery. |
| `/faces/enroll` | `POST` | Enroll a new identity (multipart form: `name` + reference image `file`). |
| `/faces/enrolled/{id}` | `DELETE` | Remove an enrolled identity from the database. |
| `/faces/recognize_frame` | `POST` | Real-time live frame analysis returning bounding boxes, status, and similarity. |
| `/evidence/file/{path}` | `GET` | Download tamper-proof snapshot frames and tight face crops. |

---

## 7. Performance Benchmarks

* **Detection Latency**: ~18ms per frame on standard CPU with YOLOv8.
* **Face Detection & Embedding**: ~12ms per face with YuNet + SFace.
* **Video Streaming Overhead**: ~9.7ms JPEG encode time per frame at 640x360 resolution (solid 30 FPS).
* **Investigation NLQ Dispatch**: < 15ms deterministic query execution across 50,000+ indexed events.
* **Test Suite**: 570 / 570 tests passing (0 regressions, 100% clean).

---

## 8. License & Attribution

Developed for the **Smart India Hackathon (SIH 26187)**.
Architecture, algorithms, and implementation are designed for mission-critical situational awareness, defense, and public safety.
