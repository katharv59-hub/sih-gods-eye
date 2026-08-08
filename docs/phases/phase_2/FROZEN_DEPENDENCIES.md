# Phase 2 Re-ID Stack (Frozen Configuration)

This document establishes the immutable, validated baseline for the Person Re-Identification (Re-ID) subsystem in Phase 2 of the **God's Eye** project. Any subsequent development, scaling, or optimization must maintain reproducibility against this configuration.

---

## 1. Model Configuration

* **Backbone Model**: `osnet_x1_0`
* **Weight File Name**: `osnet_x1_0_market1501.pth`
* **Weight File Size**: `10,399,605 bytes` (~10.4 MB)
* **SHA256 Checksum**: `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`
* **Weight Source URL**: [Kaiyang Zhou TorchReID Model Zoo Checkpoint](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing)
* **Weight Source Identifier**: Google Drive ID `1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`
* **Training Dataset**: Market-1501 (fine-tuned representation learning)

---

## 2. Embedding Configuration

* **Embedding Dimension**: `512`
* **Precision Mode**: `FP32` (Single-precision floating-point)
* **Similarity Metric**: Cosine Similarity
* **Match Threshold**: `0.75`
* **Batch Size**: Dynamic (typically 1–32 crops, matching the count of bounding box detections per frame)
* **Output Normalization Strategy**: L2-normalization (all extracted feature vectors are unit-normalized to scale factor $\|\mathbf{v}\|_2 = 1.0$)

---

## 3. Runtime Environment

* **Python Version**: `3.11.9`
* **PyTorch Version**: `2.5.1+cu121`
* **CUDA Version**: `12.1` (driver runtime matching CUDA Toolkit 12.1)
* **TorchReID Version**: `0.2.5`
* **GPU Model**: `NVIDIA GeForce RTX 4050 Laptop GPU`
* **Available VRAM**: `6 GB` (Runtime allocation footprint: ~38.1 MB detector, ~100 MB model load overhead)
* **Operating System**: `Windows (win32)`

---

## 4. Validation Results

* **Validation Dataset**: Market-1501 (standard cross-camera matching protocol)
* **Rank-1 Accuracy**: `94.33%` (Gate check: $\ge 88.0\%$ — **PASS**)
* **Rank-5 Accuracy**: `97.80%`
* **Rank-10 Accuracy**: `98.60%`
* **mAP**: `83.57%`
* **Recommended Threshold**: `0.75` (obtained from maximum F1-score optimization)
* **Benchmark Verdict**: **PASS**

---

## 5. Engineering Decisions

### Why FP32 was selected over FP16
During Task 0 (TorchReID Feasibility Validation), benchmarking verified that executing OSNet depthwise separable convolutions under FP16 on target RTX 4050 hardware yields slower throughput than FP32. This is due to extra casting overhead and a lack of optimized Tensor Core kernels for shallow depthwise convolutions. Thus, FP32 was locked to achieve optimal extraction latency (~39.24ms per crop) and prevent numerical underflow during L2-normalization.

### Why Market1501-trained weights were selected over ImageNet weights
An initial benchmark run utilizing ImageNet-trained backbone weights resulted in a failing **Rank-1 accuracy of 14.46%** (mAP = 4.45%). ImageNet pretraining learns categorical boundaries (distinguishing a person from a vehicle), whereas Re-ID requires fine-grained, intra-class metric representations to differentiate individual human identities under diverse illumination, occlusion, and pose changes. The Market-1501 fine-tuned weights successfully satisfy the accuracy gate.

### Why Cosine Similarity was selected
As locked by ADR-002, cosine similarity is optimal for L2-normalized vectors because it isolates feature orientation from vector magnitude. Magnitude varies with bounding box crop scaling and lighting, whereas direction represents stable color/texture semantic signatures.

### Why threshold 0.75 was frozen
A threshold sweep on 300 test queries proved that a cosine similarity threshold of **0.75** achieves **100.00% Precision** (0 False Positives) and **99.67% Recall** (1 False Negative, F1 = 0.9983). For the Identity Persistence Layer, matching two different people under a single ID (False Positive) is highly destructive to tracking continuity. Freezing the threshold at `0.75` minimizes false merges while retaining high target re-linking rates.

---

## 6. Reproducibility Guarantee

### Replication Protocol
To reproduce the Phase 2 benchmark scores, execute:
```bash
python scripts/run_gods_eye.py benchmark reid
```
Ensure that the extracted Market-1501 dataset is placed at `data/market1501/Market-1501-v15.09.15/` and the frozen weights reside in `data/osnet_x1_0_market1501.pth`.

### Invalidation Criteria
The comparability of subsequent benchmarks is invalidated if:
1. Crop dimensions are modified from the $256 \times 128$ height-by-width aspect ratio.
2. Image preprocessing norms (such as division by 255.0 or ImageNet channels normalization) are altered.
3. Feature post-processing L2-normalization is bypassed.
4. The evaluation matching filter (which discards matches from the same camera and sequence) is disabled.

### Frozen Components
The following architectural elements are frozen for the duration of Phase 2:
* Backbone structure (`osnet_x1_0`) and model checkpoint.
* Normalization and Cosine Similarity metric.
* Feature dimension size (512).

---

## 7. Approval Status

* **Current Phase**: `2 — Identity Persistence Layer`
* **Approval Date**: `2026-06-28`
* **Validation Status**: `PASS`
* **Approval Verdict**: `APPROVED`

---

Unless future benchmark evidence demonstrates superior performance, this configuration remains the authoritative Re-ID baseline for the entirety of Phase 2.
