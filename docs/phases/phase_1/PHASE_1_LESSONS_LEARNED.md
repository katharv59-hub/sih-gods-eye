# Phase 1 — Perception Layer Lessons Learned

**Status:** COMPLETE
**Document Version:** 1.0
**Source of Truth:** GODS_EYE_MASTER_SPEC.md v3.0

---

## 1. Technical Challenges & Solutions

### 1.1 Resolution Sensitivity in Object Detection
- **Challenge:** Detecting pedestrians at far distances or under perspective distortions on 1080p camera feeds. YOLOv8n at its default `imgsz=640` downscaled native 1920x1080 streams, leading to low target recall (50–60%) and missed tracks.
- **Solution:** Switched to YOLOv8s at native scale (`imgsz=1440` during accuracy benchmarking). Native scale preserved pedestrian features, improving recall to 71.91%.
- **Action for Future Phases:** Keep the detection resolution configurable. In multi-camera (Phase 3), resolution settings should be optimized dynamically per-stream (e.g., lower resolution for close-up indoor portals, higher resolution for broad outdoor spaces).

### 1.2 Distractor Regions inflating False Positives
- **Challenge:** In real-world urban footage (such as MOT17-04), non-pedestrian structures (parked vehicles, bicycle racks, window advertisements) triggered spurious YOLO bounding boxes that accumulated into thousands of false positives over long sequences.
- **Solution:** Implemented ignore-region filtering (IoA ≥ 0.5) using MOT17 standard polygon regions. Spurious detections in non-evaluation areas were successfully filtered out, raising precision to 89.73%.
- **Action for Future Phases:** The environmental intelligence layer (Phase 3.5) must support configurable exclusion zones and static mask regions to prevent processing clutter.

### 1.3 ByteTrack Deprecation Risk
- **Challenge:** During validation, `supervision` issued warnings that `ByteTrack` was deprecated in v0.28.0 and scheduled for removal in v0.30.0.
- **Solution:** Pinned the dependency to `supervision>=0.28.0,<0.30.0` in `pyproject.toml` (ADR-005).
- **Action for Future Phases:** Establish a roadmap to migrate the tracking engine before supervision upgrades to 0.30+. The tracking module is decoupled via the `Tracker` ABC interface, so swapping the implementation will not affect downstream modules.

---

## 2. Iterations and Failures

### 2.1 The Lost Tracks False Positive Spike
- **Initial Implementation:** Both `ACTIVE` and `LOST` states from the tracker were output as final detections.
- **Result:** Over 26,000 false positives on MOT17-04 because lost tracks retained static last-known positions rather than predicting movement.
- **Iteration:** Filtered the output to emit only `ACTIVE` state tracks for final prediction.
- **Outcome:** False positives dropped from 26,000 to 3,275.

### 2.2 CPU Fallback Thermal Throttling
- **Initial Implementation:** CPU fallback benchmark ran at full utilization.
- **Result:** Frame throughput dropped from ~18 FPS to ~7 FPS after 4 minutes.
- **Iteration/Learning:** Sustained CPU workload causes thermal throttling on laptop hardware.
- **Outcome:** The pipeline must support thread-sleep backpressure limits to prevent hardware exhaustion in CPU-degraded mode.

---

## 3. Actionable Guidelines for Future Phases

### Lesson 1: Benchmark configuration must match evaluation protocols.
- **Evidence:** Evaluating MOTA on MOTChallenge sequences without ignore-region masks leads to artificially low scores (~40%) due to unannotated distractor detections.
- **Action:** Future benchmarks (Re-ID on Market-1501, Anomaly Detection on VIRAT) must use standard evaluation protocols from day one.

### Lesson 2: Spatial-only tracking fails during crossings and camera switches.
- **Evidence:** 102 ID switches occurred on MOT17-04 where pedestrians crossed paths and IoU overlap confused the Kalman predictions.
- **Action:** Phase 2 must introduce appearance-based embedding matching (OSNet) to resolve identity conflicts when bounding boxes intersect.

### Lesson 3: Structured logging is critical for multi-threaded debugging.
- **Evidence:** JSON logging with clear subsystem tags allowed tracing frame life cycles through four concurrent threads without print statement clutter.
- **Action:** Maintain strict structured logging patterns across all future modules (Memory, Reasoning, Multi-Camera).
