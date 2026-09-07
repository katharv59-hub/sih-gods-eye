# Sub-Phase 7.5 — Real-World Feasibility & Validation Review

## Executive Summary

- **Review Target**: Sub-Phase 7.5 — Real-World Feasibility, Camera Integration & Empirical Validation
- **Repository Baseline Commit**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Current Test Count**: **515 / 515 passing tests**
- **Sub-Phase Status**: **READ-ONLY FEASIBILITY REVIEW ONLY. Zero runtime code, zero schema changes, zero test edits.**
- **Hardware Constraint Baseline**: Single authorized college CCTV camera feed OR exported video recordings.
- **Review Verdict**: **FEASIBLE FOR SINGLE-CAMERA PHYSICAL VALIDATION + SYNTHETIC MULTI-CAMERA REPLAY**

---

## 1. Single-Camera Real Hardware vs. Multi-Camera Replay Boundaries

To maintain rigorous scientific standards without claiming unverified multi-camera coverage, Phase 7.5 establishes a strict operational boundary:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PATH A: REAL SINGLE-CAMERA PHYSICAL VALIDATION                              │
│ Source: 1 Authorized College CCTV Camera (RTSP / Video File)                 │
│ Validated Layers: Perception (YOLO), Tracking (ByteTrack), Identity (OSNet), │
│                   Events, Temporal Memory, Single-Camera Risk & Hypotheses │
├─────────────────────────────────────────────────────────────────────────────┤
│ PATH B: SYNTHETIC / REPLAY MULTI-CAMERA BENCHMARK                           │
│ Source: Multi-camera synthetic replay streams & GraphStore simulation       │
│ Validated Layers: Cross-Camera ReID, Markov Transition Forecasts (Tool 9), │
│                   Trajectory Clustering (Tool 10), Multi-Camera Graph Edges │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> Synthetic multi-camera evaluation MUST NOT be presented as equivalent to multi-camera physical CCTV validation. Multi-camera physical deployment remains deferred until additional physical camera streams are authorized.

---

## 2. Authorized Camera Acquisition Options

The repository already provides a complete ingestion abstraction layer in `gods_eye/ingestion/source.py`:

| Ingestion Method | Class (`FrameSource`) | Protocol / Input | Feasibility Status |
|---|---|---|---|
| RTSP Live Stream | `RTSPSource` | `rtsp://user:pass@ip:port/stream` | **READY** (Native support) |
| Exported CCTV Recording | `VideoFileSource` | `.mp4`, `.mkv`, `.avi` files | **READY** (Native support) |
| Local USB/Webcam Stream | `WebcamSource` | `/dev/video0` or device index `0` | **READY** (Native support) |
| IP Camera ONVIF Feed | `RTSPSource` | H.264/H.265 RTSP sub-stream | **READY** (Native support) |

---

## 3. Required Permissions & Operational Constraints

1. **Authorization & Access**: Institutional authorization from college security administration for RTSP stream URL or exported CCTV files.
2. **Privacy Boundary**:
   - Automatic pseudonymous token assignment (`subject_ref`).
   - Zero raw `global_id` UUID strings or identity gallery images exposed in public reasoning tools.
   - Raw video frame data stored strictly in local memory/temporary cache; automatic purging after 48 hours.
3. **Network Isolation**: Zero external cloud or network data egress. All perception, tracking, ReID, memory, and reasoning run 100% locally.

---

## 4. Single-Camera Real-World Validation Capabilities

A single physical college CCTV camera feed can empirically validate:

1. **Person Detection**: YOLO / Supervision bounding box precision and recall under real lighting.
2. **Single-Camera Tracking**: ByteTrack track initiation, tracklet continuity, and velocity calculation.
3. **Identity Persistence**: OSNet ReID embedding extraction and cosine similarity thresholding across temporary occlusions.
4. **Event Generation**: Temporal event creation for zone entries, exits, and dwell duration.
5. **Temporal Memory**: `TimelineReconstructionEngine` visit segment aggregation.
6. **Situational Risk**: `SituationalRiskEvaluator` dwell anomaly and zone risk evaluation (Tool 11).
7. **Hypothesis Generation**: `HypothesisGenerator` single-camera temporal/dwell hypothesis tree construction (Tool 12).

---

## 5. Multi-Camera Capabilities Bounded to Replay Validation

The following Phase 6/7 capabilities fundamentally require $\ge 2$ physical camera views and MUST remain synthetic/replay-validated during Phase 7.5:

- Cross-camera identity re-association ($C_1 \to C_2 \to C_3$).
- Cross-camera Markov destination probability forecasting (`MarkovBehavioralPredictor`).
- Multi-camera trajectory cluster extraction (`TrajectoryClusteringEngine`).
- `SQLiteGraphStore` transition edge formation (`add_transition_edge`).

---

## 6. Dataset Design & Recording Duration

- **Dataset A (Real Single-Camera Footage)**: 2 hours of continuous footage from a high-traffic college location (e.g. main entrance or library corridor), capturing $\sim 144,000$ frames at 20 FPS.
- **Dataset B (Synthetic Multi-Camera Simulation)**: 4-camera synthetic transition dataset (500 transitions) generated for Phase 6.7 prediction benchmarking.

---

## 7. Annotation & Ground Truth Protocol

To evaluate empirical accuracy, a 15-minute subset of Dataset A ($\sim 18,000$ frames) will be annotated with:
- Bounding box ground truth for MOTA/IDF1 tracking evaluation.
- Timestamped ground truth log for zone entry, exit, and dwell events.

---

## 8. Empirical Evaluation Metrics

| Subsystem | Metric | Target Threshold | Validation Dataset |
|---|---|---|---|
| Person Detection | Precision / Recall / mAP50 | $\ge 85.0\%$ mAP50 | Dataset A (Annotated) |
| Single-Camera Tracking | IDF1 / MOTA / ID Switches | IDF1 $\ge 75.0\%$, ID Sw $< 50$ | Dataset A (Annotated) |
| Identity Persistence | Top-1 ReID Rank Accuracy | $\ge 80.0\%$ | Dataset A (Annotated) |
| Event Correctness | Event Precision / Recall | $\ge 90.0\%$ | Dataset A (Annotated) |
| Behavioral Prediction | Top-1 / Top-3 Accuracy | Top-1 $\ge 80\%$, Top-3 $= 100\%$ | Dataset B (Synthetic) |
| Situational Risk (Tool 11) | Latency / False Positive Suppression | Latency $< 5\text{ ms}$, $100\%$ mode suppression | Dataset A & B |
| Hypothesis Engine (Tool 12) | Latency / Tree Depth Bounds | Latency $< 5\text{ ms}$, Depth $\le 3$, Nodes $\le 25$ | Dataset A & B |

---

## 9. Real CCTV Failure Modes & Mitigation

1. **Severe Occlusion**: People overlapping in crowded corridors $\to$ *Mitigated by ByteTrack Kalman prediction & ReID embedding re-association.*
2. **Lighting Variations**: Fluorescent lighting changes & dusk glare $\to$ *Mitigated by OSNet normalization.*
3. **Compression Artifacts**: H.264 blockiness $\to$ *Mitigated by minimum detection confidence thresholding ($0.50$).*
4. **ID Switching**: Brief identity swaps during crossovers $\to$ *Mitigated by temporal visit segment smoothing.*

---

## 10. Source Code & Interface Impact Assessment

- **Required Ingestion Code Changes**: **ZERO**.
- `FrameSource` ABC and concrete classes (`RTSPSource`, `VideoFileSource`, `WebcamSource`) in `gods_eye/ingestion/source.py` already provide full RTSP and video file support.
- **Required Perception Pipeline Changes**: **ZERO**.
- **Required Memory & Reasoning Engine Changes**: **ZERO**.

---

## 11. Hardware Performance Requirements

Target development machine specifications for real-time single-camera processing:
- **CPU**: 8-core $x86\_64$ CPU.
- **GPU**: NVIDIA GPU (GTX 1660 / RTX 2060 or higher) with CUDA support.
- **RAM**: 16 GB system memory.
- **Storage**: 50 GB local SSD storage for 2-hour footage cache.

---

## 12. Operational Separation Matrix

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ SOFTWARE CORRECTNESS: 515 / 515 Unit & Integration Tests Passing (VERIFIED)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ SYNTHETIC BENCHMARK: Top-1 80%, Top-3 100%, Tool 11/12 Latency < 0.2 ms    │
├─────────────────────────────────────────────────────────────────────────────┤
│ REAL-WORLD EMPIRICAL ACCURACY: To be evaluated in Phase 7.5 physical test   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Proposed Phase 7.5 Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SUB-PHASE 7.5.1: Real Camera Feed / Video File Dataset Acquisition          │
│ - Acquire 2-hour authorized college CCTV video file or configure RTSP feed   │
│ - Annotate 15-minute ground truth subset for detection, tracking, & events  │
├─────────────────────────────────────────────────────────────────────────────┤
│ SUB-PHASE 7.5.2: Physical Single-Camera Execution & Metric Reporting        │
│ - Run pipeline against real feed using VideoFileSource / RTSPSource         │
│ - Compute empirical Precision, Recall, IDF1, ReID accuracy, and Tool 11/12   │
│ - Generate docs/audits/PHASE_7_5_EMPIRICAL_VALIDATION_REPORT.md             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Conclusion & Status

Phase 7.5 feasibility review is **COMPLETE**. The existing repository architecture is fully capable of real-world single-camera validation without source code modifications.
