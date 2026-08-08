# Project Commands Reference

This document serves as the canonical command reference for development, execution, testing, version control, and dataset preparation within the **God's Eye** repository.

---

## 1. Unified Launcher Interface

Most operational commands can be executed using the unified project launcher `scripts/run_gods_eye.py`.

```bash
python scripts/run_gods_eye.py <mode> [submode] [extra_args...]
```

| Mode | Submode | Description |
|------|---------|-------------|
| `demo` | *None* | Runs the visual demo suite on all demo videos |
| `demo` | `webcam` | Runs the live webcam tracking demo |
| `demo` | `<name>` | Runs the visual demo suite on a specific video (e.g. `mot17_04`) |
| `benchmark` | `cpu` | Runs the CPU-only FPS throughput benchmark |
| `benchmark` | `mot17` | Runs tracking accuracy evaluation on MOT17-04 |
| `benchmark` | `reid` | Runs appearance embedding validation on Market-1501 |
| `benchmark` | `demo_suite` | Runs headless demo suite throughput profiling |
| `test` | *None* | Runs the complete unit and integration test suite |
| `test` | `reid` | Runs appearance embedding and similarity unit tests |
| `test` | `tracking` | Runs ByteTrack tracker-specific unit tests |
| `test` | `detection` | Runs YOLOv8 detector-specific unit tests |
| `validate` | `phase1` | Sequentially executes all Phase 1 validation scripts |
| `validate` | `phase2` | Sequentially executes all Phase 2 validation scripts |
| `status` | *None* | Prints repository metrics, branch, and model file availability |
| `environment` | *None* | Prints OS, Python, PyTorch, CUDA, and library versions |
| `models` | *None* | Verifies size and SHA256 checksums of model weights files |
| `datasets` | *None* | Counts files and validates dataset directory structures |

---

## 2. Development Commands

### Environment Initialization
```bash
# Create Python 3.11 virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Unix/macOS)
source venv/bin/activate
```

### Dependency Management
```bash
# Upgrade package manager
python -m pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt

# Freeze current dependencies
pip freeze > requirements.txt
```

---

## 3. Testing & Static Analysis

### Unit & Integration Testing
```bash
# Run pytest with warning output
pytest -v

# Run with coverage report
pytest --cov=gods_eye --cov-report=term-missing
```

### Static Type Verification
```bash
# Strict type check the entire gods_eye package
mypy --strict gods_eye/
```

### Code Formatting and Linting
```bash
# Format code
black gods_eye/ tests/ benchmarks/ scripts/

# Lint verify
flake8 gods_eye/ tests/ benchmarks/ scripts/
```

---

## 4. Git Workflows

We adhere to a structured git workflow matching the Phase progression.

### Commit Workflow
Commits must be meaningful and include subsystem scopes:
```bash
# Example commit format
git commit -m "feat(reid): implement cosine similarity matrix and top-k indices"
```

### Phase Tagging
Tagging marks official Phase boundaries:
```bash
# Tag Phase 1 completion
git tag -a phase1-final -m "Phase 1: Single-Camera Person Tracking Complete"
git push origin phase1-final

# Tag Phase 2 completion
git tag -a phase2-final -m "Phase 2: Identity Persistence Layer Complete"
git push origin phase2-final
```

### Release Versioning
```bash
# Create a release branch
git checkout -b release/v2.0

# Merge release branch to master
git checkout master
git merge release/v2.0
```

---

## 5. Dataset Operations

### Demo Videos
```bash
# Download sample video clips
python scripts/download_demo_videos.py
```

### MOT17 Dataset
```bash
# Download and unpack MOT17-04 sequence
python scripts/download_mot17.py
```

### Market-1501 Dataset
```bash
# Run Re-ID Market-1501 benchmarking using the launcher
python scripts/run_gods_eye.py benchmark reid
```

---

## 6. Re-ID Subsystem Commands

### Weights Checksum Verification
Verify that local weights match the frozen signature for Phase 2:
```bash
python scripts/run_gods_eye.py models
```
The frozen model weights parameters must read:
* **Model:** `osnet_x1_0`
* **Weights File:** `osnet_x1_0_market1501.pth`
* **Size:** `10,399,605 bytes`
* **SHA256:** `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`
