"""Task 3: Market-1501 Re-ID Benchmark.

Evaluates OSNet embedding quality on the standard Market-1501 benchmark.
Downloads dataset automatically via torchreid if not present.

Metrics computed:
- CMC Rank-1 / Rank-5 / Rank-10
- mAP (mean Average Precision)
- Intra-class / inter-class similarity distributions
- Threshold sweep (precision / recall / F1)
- Inference latency (p50 / p95 / p99)

Gate: Rank-1 >= 0.88 (Phase 2 implementation plan)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import configure_logging, get_logger
from gods_eye.reid.osnet_extractor import OSNetExtractor
from gods_eye.reid.similarity import cosine_similarity_matrix
from gods_eye.schemas.detection import BoundingBox

configure_logging("INFO")
log = get_logger("benchmark.market1501")

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_ROOT = str(PROJECT_ROOT / "data")
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_FILE = RESULTS_DIR / "market1501_results.json"
BATCH_SIZE = 32
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# ─── Dataset Loading ────────────────────────────────────────────────────────


def load_market1501() -> (
    tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]
):
    """Load Market-1501 dataset.

    Downloads via gdown if not present. Parses the standard directory
    structure to extract (image_path, person_id, camera_id) tuples.

    Market-1501 filename format: PPPP_CCSS_FFFFFF_DD.jpg
    - PPPP: person ID (4 digits, -1 for junk, 0000 for distractor)
    - CC: camera ID (1-6)
    - SS: sequence within camera
    - FFFFFF: frame number
    - DD: detection index

    Returns:
        (query_data, gallery_data) where each item is (path, pid, camid).
    """
    import gdown
    import zipfile

    dataset_dir = Path(DATA_ROOT) / "market1501" / "Market-1501-v15.09.15"

    # Download if not present
    if not dataset_dir.exists():
        zip_path = Path(DATA_ROOT) / "market1501" / "Market-1501-v15.09.15.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        if not zip_path.exists():
            log.info("dataset_downloading", dataset="Market-1501")
            # Known Google Drive ID for Market-1501
            gdown.download(
                id="0B8-rUzbwVRk0c054eNBVV0pIcUU",
                output=str(zip_path),
                quiet=False,
            )

        log.info("dataset_extracting", zip_path=str(zip_path))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(zip_path.parent))
        log.info("dataset_extracted")

    query_dir = dataset_dir / "query"
    gallery_dir = dataset_dir / "bounding_box_test"

    if not query_dir.exists() or not gallery_dir.exists():
        raise FileNotFoundError(
            f"Market-1501 not found at {dataset_dir}. "
            f"Expected 'query/' and 'bounding_box_test/' subdirectories."
        )

    def parse_split(split_dir: Path) -> list[tuple[str, int, int]]:
        data: list[tuple[str, int, int]] = []
        for img_path in sorted(split_dir.glob("*.jpg")):
            name = img_path.stem
            parts = name.split("_")
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            camid = int(parts[1][1])  # 'c1s1' → 1
            data.append((str(img_path), pid, camid))
        return data

    query_data = parse_split(query_dir)
    gallery_data = parse_split(gallery_dir)

    # Filter out junk (-1) from query but keep in gallery for protocol
    query_data = [(p, pid, c) for p, pid, c in query_data if pid != -1]

    query_pids = set(pid for _, pid, _ in query_data)
    gallery_pids = set(pid for _, pid, _ in gallery_data if pid != -1)

    log.info(
        "dataset_loaded",
        query_images=len(query_data),
        gallery_images=len(gallery_data),
        query_identities=len(query_pids),
        gallery_identities=len(gallery_pids),
    )

    return query_data, gallery_data


# ─── Embedding Extraction ───────────────────────────────────────────────────


def extract_embeddings(
    extractor: OSNetExtractor,
    data: list[tuple[str, int, int]],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings for all images in a dataset split.

    Args:
        extractor: OSNet embedding extractor.
        data: List of (image_path, person_id, camera_id).
        label: Label for logging ("query" or "gallery").

    Returns:
        (embeddings, pids, camids) as numpy arrays.
    """
    log.info("embedding_extraction_start", split=label, images=len(data))
    t0 = time.perf_counter()

    all_embeddings: list[np.ndarray] = []
    all_pids: list[int] = []
    all_camids: list[int] = []
    latencies: list[float] = []

    for batch_start in range(0, len(data), BATCH_SIZE):
        batch = data[batch_start : batch_start + BATCH_SIZE]

        for img_path, pid, camid in batch:
            img = cv2.imread(img_path)
            if img is None:
                log.warning("image_load_failed", path=img_path)
                all_embeddings.append(
                    np.zeros(extractor.embedding_dim, dtype=np.float32)
                )
                all_pids.append(pid)
                all_camids.append(camid)
                continue

            h, w = img.shape[:2]
            bbox = BoundingBox(x1=0, y1=0, x2=float(w), y2=float(h))

            t_start = time.perf_counter()
            embs = extractor.extract(img, [bbox])
            latencies.append((time.perf_counter() - t_start) * 1000)

            all_embeddings.append(embs[0])
            all_pids.append(pid)
            all_camids.append(camid)

        # Progress logging every 1000 images
        done = min(batch_start + BATCH_SIZE, len(data))
        if done % 1000 < BATCH_SIZE or done == len(data):
            log.info(
                "embedding_extraction_progress",
                split=label,
                done=done,
                total=len(data),
                pct=round(100 * done / len(data), 1),
            )

    elapsed = time.perf_counter() - t0
    log.info(
        "embedding_extraction_complete",
        split=label,
        images=len(data),
        elapsed_s=round(elapsed, 2),
        fps=round(len(data) / elapsed, 1),
    )

    return (
        np.array(all_embeddings, dtype=np.float32),
        np.array(all_pids, dtype=np.int64),
        np.array(all_camids, dtype=np.int64),
        latencies,
    )  # type: ignore[return-value]


# ─── CMC and mAP Evaluation ─────────────────────────────────────────────────


def evaluate_cmc_map(
    query_embs: np.ndarray,
    query_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 10,
) -> tuple[np.ndarray, float]:
    """Compute CMC curve and mAP using standard Market-1501 evaluation.

    Protocol:
    - For each query, rank all gallery images by cosine similarity
    - Exclude gallery images from same person AND same camera
    - Exclude junk images (pid == -1)
    - Compute CMC and AP for each query
    """
    log.info(
        "evaluation_start",
        queries=len(query_embs),
        gallery=len(gallery_embs),
    )

    num_query = len(query_embs)
    cmc = np.zeros(max_rank, dtype=np.float64)
    all_ap: list[float] = []

    for qi in range(num_query):
        # Compute similarity of this query against all gallery images
        scores = cosine_similarity_matrix(query_embs[qi], gallery_embs)

        # Build valid mask: exclude same person+camera, exclude junk
        q_pid = query_pids[qi]
        q_camid = query_camids[qi]

        # Remove same person + same camera (standard protocol)
        same_person_same_cam = (gallery_pids == q_pid) & (
            gallery_camids == q_camid
        )
        # Remove junk images
        junk = gallery_pids == -1

        valid_mask = ~(same_person_same_cam | junk)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        valid_scores = scores[valid_indices]
        valid_pids = gallery_pids[valid_indices]

        # Rank by similarity (descending)
        ranked_order = np.argsort(-valid_scores)
        ranked_pids = valid_pids[ranked_order]

        # CMC: check if correct match appears in top-k
        matches = ranked_pids == q_pid
        for k in range(max_rank):
            if np.any(matches[: k + 1]):
                cmc[k] += 1

        # AP: average precision for this query
        ap = _compute_ap(matches)
        all_ap.append(ap)

        if (qi + 1) % 500 == 0:
            log.info(
                "evaluation_progress",
                done=qi + 1,
                total=num_query,
            )

    cmc = cmc / num_query
    mAP = float(np.mean(all_ap)) if all_ap else 0.0

    log.info(
        "evaluation_complete",
        rank1=round(float(cmc[0]), 4),
        rank5=round(float(cmc[4]), 4),
        rank10=round(float(cmc[9]), 4),
        mAP=round(mAP, 4),
    )

    return cmc, mAP


def _compute_ap(matches: np.ndarray) -> float:
    """Compute average precision for a single query."""
    num_relevant = np.sum(matches)
    if num_relevant == 0:
        return 0.0

    cumsum = np.cumsum(matches).astype(np.float64)
    precision_at_k = cumsum / np.arange(1, len(matches) + 1, dtype=np.float64)
    ap = float(np.sum(precision_at_k * matches) / num_relevant)
    return ap


# ─── Similarity Distribution Analysis ───────────────────────────────────────


def analyze_similarity_distribution(
    query_embs: np.ndarray,
    query_pids: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_pids: np.ndarray,
    sample_queries: int = 200,
) -> dict[str, dict[str, float]]:
    """Compute intra-class and inter-class similarity distributions.

    Samples a subset of queries for tractability.
    """
    log.info("similarity_analysis_start", sample_queries=sample_queries)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(query_embs), size=min(sample_queries, len(query_embs)), replace=False)

    intra_scores: list[float] = []
    inter_scores: list[float] = []

    for qi in indices:
        scores = cosine_similarity_matrix(query_embs[qi], gallery_embs)
        q_pid = query_pids[qi]

        same_pid_mask = gallery_pids == q_pid
        diff_pid_mask = (gallery_pids != q_pid) & (gallery_pids != -1)

        if np.any(same_pid_mask):
            intra_scores.extend(scores[same_pid_mask].tolist())
        if np.any(diff_pid_mask):
            # Sample inter-class to keep tractable
            diff_indices = np.where(diff_pid_mask)[0]
            sampled = rng.choice(
                diff_indices, size=min(100, len(diff_indices)), replace=False
            )
            inter_scores.extend(scores[sampled].tolist())

    intra = np.array(intra_scores, dtype=np.float64)
    inter = np.array(inter_scores, dtype=np.float64)

    result = {
        "intra_class": {
            "count": len(intra),
            "mean": round(float(np.mean(intra)), 4),
            "median": round(float(np.median(intra)), 4),
            "std": round(float(np.std(intra)), 4),
            "p5": round(float(np.percentile(intra, 5)), 4),
            "p95": round(float(np.percentile(intra, 95)), 4),
            "p99": round(float(np.percentile(intra, 99)), 4),
        },
        "inter_class": {
            "count": len(inter),
            "mean": round(float(np.mean(inter)), 4),
            "median": round(float(np.median(inter)), 4),
            "std": round(float(np.std(inter)), 4),
            "p5": round(float(np.percentile(inter, 5)), 4),
            "p95": round(float(np.percentile(inter, 95)), 4),
            "p99": round(float(np.percentile(inter, 99)), 4),
        },
        "separation": {
            "mean_gap": round(
                float(np.mean(intra)) - float(np.mean(inter)), 4
            ),
        },
    }

    log.info(
        "similarity_analysis_complete",
        intra_mean=result["intra_class"]["mean"],
        inter_mean=result["inter_class"]["mean"],
        gap=result["separation"]["mean_gap"],
    )

    return result


# ─── Threshold Sweep ─────────────────────────────────────────────────────────


def threshold_sweep(
    query_embs: np.ndarray,
    query_pids: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_pids: np.ndarray,
    thresholds: list[float],
    sample_queries: int = 300,
) -> list[dict[str, float]]:
    """Evaluate precision/recall/F1 across a range of thresholds.

    For each threshold, checks if the top-1 gallery match exceeds the
    threshold and whether it's the correct identity.
    """
    log.info("threshold_sweep_start", thresholds=thresholds)

    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(query_embs), size=min(sample_queries, len(query_embs)), replace=False
    )

    # Pre-compute top-1 match and score for each sampled query
    top1_data: list[tuple[float, bool]] = []  # (score, is_correct)

    for qi in indices:
        scores = cosine_similarity_matrix(query_embs[qi], gallery_embs)
        q_pid = query_pids[qi]

        # Exclude junk
        valid_mask = gallery_pids != -1
        valid_scores = scores.copy()
        valid_scores[~valid_mask] = -2.0  # Below minimum

        best_idx = int(np.argmax(valid_scores))
        best_score = float(valid_scores[best_idx])
        is_correct = gallery_pids[best_idx] == q_pid
        top1_data.append((best_score, is_correct))

    # Sweep thresholds
    results: list[dict[str, float]] = []

    for threshold in thresholds:
        tp = fp = fn = 0
        for score, is_correct in top1_data:
            if score >= threshold:
                if is_correct:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_correct:
                    fn += 1
                # True negatives (score < threshold, wrong person) not counted

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append({
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        log.info(
            "threshold_result",
            threshold=threshold,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        )

    return results


# ─── Latency Benchmark ──────────────────────────────────────────────────────


def compute_latency_stats(latencies: list[float]) -> dict[str, float]:
    """Compute latency percentiles from per-image timings."""
    arr = np.array(latencies, dtype=np.float64)
    return {
        "count": len(arr),
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def benchmark_search_latency(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    num_trials: int = 100,
) -> dict[str, float]:
    """Measure query-to-gallery search latency."""
    log.info("search_latency_start", trials=num_trials, gallery_size=len(gallery_embs))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(query_embs), size=min(num_trials, len(query_embs)), replace=False)

    latencies: list[float] = []
    for qi in indices:
        t0 = time.perf_counter()
        _ = cosine_similarity_matrix(query_embs[qi], gallery_embs)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = compute_latency_stats(latencies)
    log.info("search_latency_complete", **stats)
    return stats


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the complete Market-1501 benchmark."""
    log.info("benchmark_start", dataset="Market-1501")
    t_total = time.perf_counter()

    # 1. Load dataset
    query_data, gallery_data = load_market1501()

    # 2. Initialize extractor
    import dataclasses
    settings = Settings.from_env()
    
    # Auto-fallback to local downloaded Market-1501 weights if they exist and no weights_path is set
    local_weights = PROJECT_ROOT / "data" / "osnet_x1_0_market1501.pth"
    if settings.reid_weights_path is None and local_weights.exists():
        settings = dataclasses.replace(settings, reid_weights_path=str(local_weights))
        log.info("using_local_reid_weights", path=settings.reid_weights_path)

    extractor = OSNetExtractor(settings)
    extractor.warmup()

    # 3. Extract embeddings
    query_embs, query_pids, query_camids, query_latencies = extract_embeddings(  # type: ignore[misc]
        extractor, query_data, "query"
    )
    gallery_embs, gallery_pids, gallery_camids, gallery_latencies = extract_embeddings(  # type: ignore[misc]
        extractor, gallery_data, "gallery"
    )

    # 4. CMC and mAP evaluation
    cmc, mAP = evaluate_cmc_map(
        query_embs, query_pids, query_camids,
        gallery_embs, gallery_pids, gallery_camids,
    )

    # 5. Similarity distribution analysis
    similarity_dist = analyze_similarity_distribution(
        query_embs, query_pids, gallery_embs, gallery_pids,
    )

    # 6. Threshold sweep
    threshold_results = threshold_sweep(
        query_embs, query_pids, gallery_embs, gallery_pids, THRESHOLDS,
    )

    # 7. Latency benchmarks
    embedding_latency = compute_latency_stats(query_latencies + gallery_latencies)
    search_latency = benchmark_search_latency(query_embs, gallery_embs)

    # 8. Gate evaluation
    rank1 = float(cmc[0])
    gate_pass = rank1 >= 0.88

    # 9. Best F1 threshold
    best_threshold_entry = max(threshold_results, key=lambda x: x["f1"])

    # 10. Assemble results
    total_time = time.perf_counter() - t_total
    results = {
        "benchmark": "Market-1501",
        "model": settings.reid_model,
        "device": extractor.device,
        "dataset": {
            "query_images": len(query_data),
            "gallery_images": len(gallery_data),
            "query_identities": len(set(int(p) for _, p, _ in query_data)),
            "gallery_identities": len(set(int(p) for _, p, _ in gallery_data)),
        },
        "accuracy": {
            "rank1": round(rank1, 4),
            "rank5": round(float(cmc[4]), 4),
            "rank10": round(float(cmc[9]), 4),
            "mAP": round(mAP, 4),
        },
        "similarity_distribution": similarity_dist,
        "threshold_sweep": threshold_results,
        "recommended_threshold": best_threshold_entry["threshold"],
        "latency": {
            "embedding_extraction": embedding_latency,
            "gallery_search": search_latency,
        },
        "gate": {
            "criterion": "Rank-1 >= 0.88",
            "rank1": round(rank1, 4),
            "target": 0.88,
            "verdict": "PASS" if gate_pass else "FAIL",
        },
        "total_time_s": round(total_time, 1),
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log.info(
        "benchmark_complete",
        rank1=round(rank1, 4),
        mAP=round(mAP, 4),
        gate="PASS" if gate_pass else "FAIL",
        total_time_s=round(total_time, 1),
        results_file=str(RESULTS_FILE),
    )

    # Print summary for human consumption
    print(f"\n{'=' * 60}")
    print(f"Market-1501 Benchmark Results")
    print(f"{'=' * 60}")
    print(f"  Model:     {settings.reid_model}")
    print(f"  Device:    {extractor.device}")
    print(f"  Rank-1:    {rank1:.4f} ({rank1*100:.2f}%)")
    print(f"  Rank-5:    {cmc[4]:.4f} ({cmc[4]*100:.2f}%)")
    print(f"  Rank-10:   {cmc[9]:.4f} ({cmc[9]*100:.2f}%)")
    print(f"  mAP:       {mAP:.4f} ({mAP*100:.2f}%)")
    print(f"  Gate (>=0.88): {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Best threshold: {best_threshold_entry['threshold']} (F1={best_threshold_entry['f1']:.4f})")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Results:   {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
