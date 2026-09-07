#!/usr/bin/env python3
"""God's Eye Unified Project Launcher and Operational Tool.

Acts as a single entry point for running demos, benchmarks, tests, validations,
and inspecting environment/dataset status.

Usage:
  python scripts/run_gods_eye.py <mode> [submode] [extra_args...]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> bool:
    """Run a subprocess command and stream output. Returns True if exit code is 0."""
    print(f"\n[RUNNING] {' '.join(cmd)}")
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    pythonpath = current_env.get("PYTHONPATH", "")
    current_env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
    try:
        result = subprocess.run(cmd, env=current_env, cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            print(f"[SUCCESS] Command completed successfully.")
            return True
        else:
            print(f"[FAILED] Command failed with exit code: {result.returncode}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}", file=sys.stderr)
        return False


def get_sha256(path: Path) -> str:
    """Calculate SHA256 of a file."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def handle_demo(submode: str | None, extra: list[str]) -> None:
    """Handle demo execution modes."""
    interpreter = sys.executable

    if not submode:
        # Default runs the full demo suite
        cmd = [interpreter, "scripts/run_demo_suite.py"] + extra
        run_command(cmd)
    elif submode == "webcam":
        cmd = [interpreter, "scripts/live_demo.py"] + extra
        run_command(cmd)
    else:
        # Match specific video name
        cmd = [interpreter, "scripts/run_demo_suite.py", "--video", submode] + extra
        run_command(cmd)


def handle_benchmark(submode: str | None, extra: list[str]) -> None:
    """Handle benchmark execution modes."""
    interpreter = sys.executable

    if not submode:
        print("Please specify a benchmark type: cpu, mot17, reid, gallery, demo_suite")
        sys.exit(1)

    if submode == "cpu":
        # Default cpu benchmark to 30s in unified runner for faster validation, but override if specified
        duration = ["--duration-s", "30"] if not any(arg in extra for arg in ["--duration-s", "-d"]) else []
        cmd = [interpreter, "benchmarks/benchmark_cpu_mode.py"] + duration + extra
        # CPU benchmark requires setting env variable
        run_command(cmd, env={"GODS_EYE_CPU_ONLY": "1"})
    elif submode == "mot17":
        cmd = [interpreter, "benchmarks/benchmark_mot17.py"] + extra
        run_command(cmd)
    elif submode == "reid":
        cmd = [interpreter, "benchmarks/benchmark_market1501.py"] + extra
        run_command(cmd)
    elif submode == "gallery":
        cmd = [interpreter, "-m", "benchmarks.benchmark_gallery_latency"] + extra
        run_command(cmd)
    elif submode == "demo_suite":
        cmd = [interpreter, "scripts/run_demo_suite.py", "--headless"] + extra
        run_command(cmd)
    else:
        print(f"Unknown benchmark mode: {submode}")
        sys.exit(1)


def handle_test(submode: str | None, extra: list[str]) -> None:
    """Handle pytest execution modes."""
    cmd = ["pytest"]

    if submode == "reid":
        cmd.extend(["tests/test_reid.py", "tests/test_similarity.py"])
    elif submode == "identity":
        cmd.extend(["tests/test_identity_mapper.py", "tests/test_identity.py",
                    "tests/test_gallery.py", "tests/test_lifecycle.py"])
    elif submode == "tracking":
        cmd.append("tests/test_tracking.py")
    elif submode == "detection":
        cmd.append("tests/test_detection.py")
    elif submode:
        # Fallback to direct path or name matching
        cmd.append(f"tests/test_{submode}.py")

    cmd.extend(extra)
    run_command(cmd)


def handle_validate(submode: str | None, extra: list[str]) -> None:
    """Handle Phase 1 and Phase 2 validation runs."""
    interpreter = sys.executable

    if not submode:
        print("Please specify a validation phase: phase1, phase2")
        sys.exit(1)

    if submode == "phase1":
        print("\n=== RUNNING PHASE 1 VALIDATION SUITE ===")
        scripts = [
            "scripts/validate_ingestion.py",
            "scripts/validate_detection.py",
            "scripts/validate_tracking.py",
            "scripts/validate_pipeline.py"
        ]
        success = True
        for script in scripts:
            success &= run_command([interpreter, script])
        if success:
            print("\n[VALIDATION PASS] Phase 1 Validation completed successfully!")
        else:
            print("\n[VALIDATION FAIL] Phase 1 Validation has failed steps.", file=sys.stderr)
            sys.exit(1)

    elif submode == "phase2":
        print("\n=== RUNNING PHASE 2 VALIDATION SUITE ===")
        # Phase 2 Validation runs unit tests and Market-1501 Re-ID benchmark
        tests_passed = run_command(["pytest", "tests/test_reid.py", "tests/test_similarity.py"])
        benchmark_passed = run_command([interpreter, "benchmarks/benchmark_market1501.py"])
        if tests_passed and benchmark_passed:
            print("\n[VALIDATION PASS] Phase 2 Validation completed successfully!")
        else:
            print("\n[VALIDATION FAIL] Phase 2 Validation failed.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown validation mode: {submode}")
        sys.exit(1)


def handle_status() -> None:
    """Display overall project health and Git status."""
    print("=" * 64)
    print("GOD'S EYE PROJECT STATUS")
    print("=" * 64)
    
    # 1. Git branch/commit info
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        commit = subprocess.check_output(["git", "log", "-1", "--format=%h - %s (%ad)"]).decode().strip()
        print(f"Git Branch:    {branch}")
        print(f"Last Commit:   {commit}")
    except Exception:
        print("Git info:      Not available")

    # 2. File counts
    reid_files = list((PROJECT_ROOT / "gods_eye" / "reid").glob("*.py"))
    tracking_files = list((PROJECT_ROOT / "gods_eye" / "tracking").glob("*.py"))
    detection_files = list((PROJECT_ROOT / "gods_eye" / "detection").glob("*.py"))
    test_files = list((PROJECT_ROOT / "tests").glob("test_*.py"))

    print(f"Re-ID Files:   {len(reid_files)}")
    print(f"Tracking:      {len(tracking_files)}")
    print(f"Detection:     {len(detection_files)}")
    print(f"Unit Tests:    {len(test_files)}")

    # 3. Model Weights presence
    yolo_pt = PROJECT_ROOT / "yolov8n.pt"
    reid_pt = PROJECT_ROOT / "data" / "osnet_x1_0_market1501.pth"
    print(f"YOLOv8n:       {'FOUND' if yolo_pt.exists() else 'MISSING'}")
    print(f"OSNet Re-ID:   {'FOUND' if reid_pt.exists() else 'MISSING'}")

    print("=" * 64)


def handle_environment() -> None:
    """Display Python, OS, and PyTorch/CUDA info."""
    print("=" * 64)
    print("ENVIRONMENT SUMMARY")
    print("=" * 64)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"OS Platform:    {sys.platform}")

    # Check torch/cuda
    try:
        import torch
        print(f"PyTorch Ver:    {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device:    {torch.cuda.get_device_name(0)}")
            print(f"Device Count:   {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch:        Not Installed")

    # Check opencv
    try:
        import cv2
        print(f"OpenCV Ver:     {cv2.__version__}")
    except ImportError:
        print("OpenCV:         Not Installed")

    # Check numpy
    try:
        import numpy as np
        print(f"NumPy Ver:      {np.__version__}")
    except ImportError:
        print("NumPy:          Not Installed")

    print("=" * 64)


def handle_models() -> None:
    """Display models metadata (size and checksum)."""
    print("=" * 64)
    print("MODEL WEIGHTS METADATA")
    print("=" * 64)

    models_list = [
        ("YOLOv8n", PROJECT_ROOT / "yolov8n.pt"),
        ("OSNet x1.0 Re-ID", PROJECT_ROOT / "data" / "osnet_x1_0_market1501.pth")
    ]

    for name, path in models_list:
        print(f"Model: {name}")
        print(f"  Path:     {path}")
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  Size:     {size_mb:.2f} MB")
            print(f"  SHA256:   {get_sha256(path)}")
        else:
            print(f"  Status:   MISSING")
        print()
    print("=" * 64)


def handle_datasets() -> None:
    """Inspect dataset folders and contents."""
    print("=" * 64)
    print("DATASET INVENTORY")
    print("=" * 64)

    # Resolve Market-1501 path candidate
    market_path = PROJECT_ROOT / "data" / "Market-1501-v15.09.15"
    if not market_path.exists():
        market_path = PROJECT_ROOT / "data" / "market1501" / "Market-1501-v15.09.15"
    if not market_path.exists():
        market_path = PROJECT_ROOT / "data" / "market1501"

    datasets = [
        ("Demo Videos", PROJECT_ROOT / "tests" / "data" / "demo_videos", "*.mp4"),
        ("MOT17", PROJECT_ROOT / "tests" / "data" / "MOT17", None),
        ("Market-1501", market_path, None)
    ]

    for name, path, pattern in datasets:
        print(f"Dataset: {name}")
        print(f"  Path:   {path}")
        if path.exists():
            if pattern:
                files = list(path.glob(pattern))
                print(f"  Status: FOUND ({len(files)} files)")
                for f in sorted(files)[:5]:
                    print(f"    - {f.name}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
            else:
                # Count files recursively
                count = sum(1 for _ in path.rglob("*") if _.is_file())
                print(f"  Status: FOUND ({count} total files)")
        else:
            print(f"  Status: MISSING")
        print()
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified God's Eye project management launcher.",
        usage="python scripts/run_gods_eye.py <mode> [submode] [extra_args...]"
    )
    parser.add_argument(
        "mode",
        choices=["demo", "benchmark", "test", "validate", "status", "environment", "models", "datasets"],
        help="Command execution mode."
    )
    parser.add_argument(
        "submode",
        nargs="?",
        default=None,
        help="Command sub-mode or target argument."
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra options passed directly to the underlying script."
    )

    args = parser.parse_args()

    if args.mode == "demo":
        handle_demo(args.submode, args.extra)
    elif args.mode == "benchmark":
        handle_benchmark(args.submode, args.extra)
    elif args.mode == "test":
        handle_test(args.submode, args.extra)
    elif args.mode == "validate":
        handle_validate(args.submode, args.extra)
    elif args.mode == "status":
        handle_status()
    elif args.mode == "environment":
        handle_environment()
    elif args.mode == "models":
        handle_models()
    elif args.mode == "datasets":
        handle_datasets()


if __name__ == "__main__":
    main()
