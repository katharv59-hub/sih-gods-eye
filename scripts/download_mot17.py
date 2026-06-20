"""Helper script to download MOT17-04-FRCNN sequence from Lekim89/MOT17 on Hugging Face.

Saves download time and avoids visiting motchallenge.net if it is blocked or slow.
Uses ThreadPoolExecutor for fast parallel downloading.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Base Hugging Face resolve URL
_BASE_URL = "https://huggingface.co/datasets/Lekim89/MOT17/resolve/main"


def download_file(url: str, dest_path: Path) -> None:
    """Download a single file to a destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid re-downloading if file already exists and is non-empty
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return

    # Use a custom User-Agent to avoid issues
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MOT17-04 sequence")
    parser.add_argument(
        "--max-frames", type=int, default=1050,
        help="Maximum frames to download (default: 1050 = full sequence)",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Number of parallel download threads (default: 32)",
    )
    args = parser.parse_args()

    dest_dir = Path("tests/data/MOT17/train/MOT17-04-FRCNN")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MOT17-04-FRCNN Downloader (from Lekim89/MOT17 HF mirror)")
    print("=" * 60)
    print(f"Destination: {dest_dir.resolve()}")
    print(f"Max workers: {args.workers}")
    print(f"Max frames:  {args.max_frames}")

    # 1. Download metadata files
    meta_files = [
        ("train/MOT17-04-FRCNN/seqinfo.ini", dest_dir / "seqinfo.ini"),
        ("train/MOT17-04-FRCNN/gt/gt.txt", dest_dir / "gt" / "gt.txt"),
        ("train/MOT17-04-FRCNN/det/det.txt", dest_dir / "det" / "det.txt"),
    ]

    print("\nDownloading metadata files...")
    for rel_path, dest_path in meta_files:
        url = f"{_BASE_URL}/{rel_path}"
        try:
            download_file(url, dest_path)
            print(f"  [OK] {dest_path.name}")
        except Exception as e:
            print(f"  [FAIL] Failed to download {dest_path.name}: {e}")
            return

    # 2. Build list of frame URLs
    frame_tasks = []
    for i in range(1, args.max_frames + 1):
        filename = f"{i:06d}.jpg"
        url = f"{_BASE_URL}/train/MOT17-04-FRCNN/img1/{filename}"
        dest_path = dest_dir / "img1" / filename
        frame_tasks.append((url, dest_path))

    print(f"\nDownloading {len(frame_tasks)} frames in parallel...")

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, url, path): (url, path)
            for url, path in frame_tasks
        }
        for future in as_completed(futures):
            url, path = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                failed += 1
                print(f"\n  [FAIL] Failed to download {path.name}: {e}")

            if completed % 50 == 0 or completed == len(frame_tasks):
                print(f"  Progress: {completed}/{len(frame_tasks)} frames downloaded...", end="\r")

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Successfully downloaded: {completed} frames")
    if failed > 0:
        print(f"  Failed downloads:        {failed} frames")
    print(f"  Metadata files verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
