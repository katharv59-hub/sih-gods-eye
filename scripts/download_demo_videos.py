"""Download MOT17 sequences and convert to demo video files.

Downloads image frames from the Lekim89/MOT17 Hugging Face mirror,
then encodes them into MP4 videos using OpenCV for use as demo inputs.

Produces three videos in tests/data/demo_videos/:
  - mot17_09_low_density.mp4    (525 frames, ~17.5s at 30fps)
  - mot17_04_medium_density.mp4 (1050 frames, 35s at 30fps)
  - mot17_05_high_density.mp4   (837 frames, ~28s at 30fps)

Usage:
  python scripts/download_demo_videos.py
  python scripts/download_demo_videos.py --sequences MOT17-09  # single sequence
  python scripts/download_demo_videos.py --skip-download        # convert only (if images exist)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

# ─── Sequence definitions ────────────────────────────────────────────────────

_BASE_URL = "https://huggingface.co/datasets/Lekim89/MOT17/resolve/main"

DEMO_VIDEO_DIR = Path("tests/data/demo_videos")


@dataclass
class SequenceSpec:
    """Defines a MOT17 sequence to download and convert."""
    sequence_id: str          # e.g. "MOT17-09"
    detector: str             # e.g. "FRCNN"
    num_frames: int           # total frames
    fps: int                  # framerate
    resolution: tuple[int, int]  # (width, height)
    density_label: str        # "low" | "medium" | "high"
    description: str          # human-readable scene description
    output_filename: str      # output mp4 filename


SEQUENCES: list[SequenceSpec] = [
    SequenceSpec(
        sequence_id="MOT17-09",
        detector="FRCNN",
        num_frames=525,
        fps=30,
        resolution=(1920, 1080),
        density_label="low",
        description="Indoor/outdoor transition, sparse pedestrians, static camera",
        output_filename="mot17_09_low_density.mp4",
    ),
    SequenceSpec(
        sequence_id="MOT17-04",
        detector="FRCNN",
        num_frames=1050,
        fps=30,
        resolution=(1920, 1080),
        density_label="medium",
        description="Urban street scene, moderate pedestrian flow, static camera",
        output_filename="mot17_04_medium_density.mp4",
    ),
    SequenceSpec(
        sequence_id="MOT17-05",
        detector="FRCNN",
        num_frames=837,
        fps=30,
        resolution=(1920, 1080),
        density_label="high",
        description="Busy pedestrian area, high density, moving camera",
        output_filename="mot17_05_high_density.mp4",
    ),
]


# ─── Download helpers ─────────────────────────────────────────────────────────


def _download_file(url: str, dest_path: Path) -> None:
    """Download a single file. Skip if already exists and non-empty."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as f:
        f.write(response.read())


def download_sequence(spec: SequenceSpec, workers: int = 32) -> Path:
    """Download all frames for one MOT17 sequence. Returns image dir path."""
    seq_dir_name = f"{spec.sequence_id}-{spec.detector}"
    img_dir = Path(f"tests/data/MOT17/train/{seq_dir_name}/img1")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Check if already fully downloaded
    existing = list(img_dir.glob("*.jpg"))
    if len(existing) >= spec.num_frames:
        print(f"  [SKIP] {seq_dir_name}: {len(existing)} frames already present")
        return img_dir

    print(f"  Downloading {spec.num_frames} frames for {seq_dir_name}...")

    # Also download metadata files
    meta_dir = Path(f"tests/data/MOT17/train/{seq_dir_name}")
    meta_files = [
        (f"train/{seq_dir_name}/seqinfo.ini", meta_dir / "seqinfo.ini"),
        (f"train/{seq_dir_name}/gt/gt.txt", meta_dir / "gt" / "gt.txt"),
    ]
    for rel_path, dest_path in meta_files:
        url = f"{_BASE_URL}/{rel_path}"
        try:
            _download_file(url, dest_path)
        except Exception as e:
            print(f"    [WARN] Could not download {dest_path.name}: {e}")

    # Download frames in parallel
    frame_tasks = []
    for i in range(1, spec.num_frames + 1):
        filename = f"{i:06d}.jpg"
        url = f"{_BASE_URL}/train/{seq_dir_name}/img1/{filename}"
        dest_path = img_dir / filename
        frame_tasks.append((url, dest_path))

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_file, url, path): (url, path)
            for url, path in frame_tasks
        }
        for future in as_completed(futures):
            _, path = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"    [FAIL] {path.name}: {e}")

            if completed % 100 == 0:
                print(f"    Progress: {completed}/{spec.num_frames}", end="\r")

    print(f"    Downloaded: {completed}/{spec.num_frames} frames"
          f"{f' ({failed} failed)' if failed else ''}          ")

    if failed > spec.num_frames * 0.1:
        print(f"    [ERROR] Too many failures for {seq_dir_name}. Skipping conversion.")
        raise RuntimeError(f"Download failed for {seq_dir_name}")

    return img_dir


# ─── Video conversion ─────────────────────────────────────────────────────────


def convert_to_video(
    img_dir: Path, output_path: Path, fps: int, expected_frames: int,
) -> dict[str, object]:
    """Convert a directory of numbered JPEGs to an MP4 video.

    Returns metadata dict for the catalog.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect and sort frames
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {img_dir}")

    # Read first frame for dimensions
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise ValueError(f"Could not read {frame_paths[0]}")
    h, w = sample.shape[:2]

    # Use mp4v codec (universally supported by OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    frame_count = 0
    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is not None:
            writer.write(frame)
            frame_count += 1

    writer.release()

    duration_s = frame_count / fps if fps > 0 else 0
    file_size = output_path.stat().st_size

    metadata = {
        "file": output_path.name,
        "path": str(output_path),
        "resolution": f"{w}x{h}",
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": round(duration_s, 1),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 1),
    }

    return metadata


# ─── Verification ──────────────────────────────────────────────────────────────


def verify_video(path: Path) -> dict[str, object]:
    """Open a video with OpenCV and verify its properties."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "resolution": f"{w}x{h}",
        "fps": round(fps, 1),
        "frame_count": frame_count,
        "duration_s": round(frame_count / fps, 1) if fps > 0 else 0,
        "status": "OK",
    }


# ─── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MOT17 sequences and convert to demo videos."
    )
    parser.add_argument(
        "--sequences", nargs="*", default=None,
        help="Sequence IDs to process (default: all). E.g. MOT17-09 MOT17-04",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; convert existing images only.",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Number of parallel download threads (default: 32).",
    )
    args = parser.parse_args()

    # Filter sequences
    if args.sequences:
        seqs = [s for s in SEQUENCES if s.sequence_id in args.sequences]
        if not seqs:
            print(f"[ERROR] No matching sequences for: {args.sequences}")
            sys.exit(1)
    else:
        seqs = SEQUENCES

    print("=" * 64)
    print("God's Eye — Demo Video Downloader")
    print("=" * 64)
    print(f"Sequences: {[s.sequence_id for s in seqs]}")
    print(f"Output:    {DEMO_VIDEO_DIR.resolve()}")
    print()

    catalog: list[dict[str, object]] = []

    for spec in seqs:
        print(f"\n{'─' * 64}")
        print(f"Processing: {spec.sequence_id} ({spec.density_label} density)")
        print(f"  Description: {spec.description}")
        print(f"  Expected: {spec.num_frames} frames, {spec.fps} FPS, "
              f"{spec.resolution[0]}x{spec.resolution[1]}")

        # Step 1: Download
        seq_dir_name = f"{spec.sequence_id}-{spec.detector}"
        img_dir = Path(f"tests/data/MOT17/train/{seq_dir_name}/img1")

        if not args.skip_download:
            try:
                img_dir = download_sequence(spec, workers=args.workers)
            except RuntimeError as e:
                print(f"  [ERROR] {e}")
                continue
        else:
            if not img_dir.exists():
                print(f"  [SKIP] Image directory not found: {img_dir}")
                continue

        # Step 2: Convert to MP4
        output_path = DEMO_VIDEO_DIR / spec.output_filename
        print(f"  Converting to {output_path}...")

        try:
            meta = convert_to_video(img_dir, output_path, spec.fps, spec.num_frames)
            print(f"  [OK] {meta['frame_count']} frames → "
                  f"{meta['file_size_mb']} MB "
                  f"({meta['duration_s']}s)")
        except Exception as e:
            print(f"  [ERROR] Conversion failed: {e}")
            continue

        # Step 3: Verify
        try:
            verification = verify_video(output_path)
            print(f"  Verification: {verification['status']} "
                  f"({verification['resolution']}, "
                  f"{verification['fps']} FPS, "
                  f"{verification['frame_count']} frames)")
        except Exception as e:
            print(f"  [WARN] Verification failed: {e}")
            verification = {"status": "UNVERIFIED"}

        catalog.append({
            "sequence_id": spec.sequence_id,
            "density": spec.density_label,
            "description": spec.description,
            **meta,
            "verification": verification,
        })

    # Save catalog as JSON
    DEMO_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    catalog_path = DEMO_VIDEO_DIR / "catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"\n{'=' * 64}")
    print(f"Catalog saved: {catalog_path}")

    # Summary
    print(f"\n{'=' * 64}")
    print("SUMMARY")
    print(f"{'=' * 64}")
    total_size = 0
    for entry in catalog:
        size_mb = entry.get("file_size_mb", 0)
        total_size += size_mb  # type: ignore[arg-type]
        status = entry.get("verification", {})
        v_status = status.get("status", "?") if isinstance(status, dict) else "?"
        print(f"  {entry['density']:>8s}  {entry['file']}  "
              f"{entry['frame_count']} frames  "
              f"{entry['duration_s']}s  "
              f"{size_mb} MB  [{v_status}]")
    print(f"\n  Total storage: {total_size:.1f} MB")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
