"""Task 0: TorchReID Feasibility Validation Script.

Validates that OSNet can load and run on the target hardware.
This is a throwaway validation script, NOT production code.
"""

import json
import sys
import time
import traceback

results = {
    "environment": {},
    "torchreid_import": {"status": "PENDING"},
    "model_load": {"status": "PENDING"},
    "cuda_execution": {"status": "PENDING"},
    "inference_timing": {"status": "PENDING"},
    "issues": [],
}

# ─── Step 1: Environment ─────────────────────────────────────────────────────
print("=" * 60)
print("TASK 0: TorchReID Feasibility Validation")
print("=" * 60)

try:
    import torch
    results["environment"]["python"] = sys.version
    results["environment"]["pytorch"] = torch.__version__
    results["environment"]["cuda_available"] = torch.cuda.is_available()
    results["environment"]["cuda_version"] = str(torch.version.cuda)
    if torch.cuda.is_available():
        results["environment"]["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        results["environment"]["gpu_vram_mb"] = round(vram / 1024 / 1024)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()} ({torch.version.cuda})")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {results['environment']['gpu_vram_mb']} MB")
except Exception as e:
    results["issues"].append(f"Environment check failed: {e}")
    print(f"  [ERROR] {e}")

# ─── Step 2: Import torchreid ────────────────────────────────────────────────
print("\n[1/4] Importing torchreid...")
try:
    import torchreid
    results["torchreid_import"]["status"] = "PASS"
    results["torchreid_import"]["version"] = torchreid.__version__
    print(f"  torchreid version: {torchreid.__version__}")
    print("  Status: PASS")
except Exception as e:
    results["torchreid_import"]["status"] = "FAIL"
    results["torchreid_import"]["error"] = str(e)
    results["issues"].append(f"torchreid import failed: {e}")
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ─── Step 3: Load OSNet pretrained model ─────────────────────────────────────
print("\n[2/4] Loading OSNet model (osnet_x1_0)...")
model = None
try:
    t0 = time.perf_counter()
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1,  # Dummy; we only extract features
        pretrained=True,
    )
    load_time = time.perf_counter() - t0
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024 / 1024

    results["model_load"]["status"] = "PASS"
    results["model_load"]["model_name"] = "osnet_x1_0"
    results["model_load"]["load_time_s"] = round(load_time, 3)
    results["model_load"]["total_params"] = total_params
    results["model_load"]["model_size_mb"] = round(model_size_mb, 2)

    print(f"  Model loaded in {load_time:.3f}s")
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print("  Status: PASS")
except Exception as e:
    results["model_load"]["status"] = "FAIL"
    results["model_load"]["error"] = str(e)
    results["issues"].append(f"Model load failed: {e}")
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ─── Step 4: CUDA execution ─────────────────────────────────────────────────
print("\n[3/4] Testing CUDA execution...")
if model is not None and torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        model = model.to(device)

        # Create a dummy person crop: 256x128 (HxW), standard Re-ID input size
        dummy_input = torch.randn(1, 3, 256, 128, device=device)

        # Warmup
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()

        # Measure VRAM after model load
        vram_allocated_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
        vram_reserved_mb = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)

        results["cuda_execution"]["status"] = "PASS"
        results["cuda_execution"]["device"] = str(device)
        results["cuda_execution"]["vram_allocated_mb"] = vram_allocated_mb
        results["cuda_execution"]["vram_reserved_mb"] = vram_reserved_mb

        print(f"  CUDA execution: OK")
        print(f"  VRAM allocated: {vram_allocated_mb} MB")
        print(f"  VRAM reserved:  {vram_reserved_mb} MB")
        print("  Status: PASS")
    except Exception as e:
        results["cuda_execution"]["status"] = "FAIL"
        results["cuda_execution"]["error"] = str(e)
        results["issues"].append(f"CUDA execution failed: {e}")
        print(f"  [FAIL] {e}")
        traceback.print_exc()
elif model is None:
    results["cuda_execution"]["status"] = "SKIP"
    results["cuda_execution"]["reason"] = "Model failed to load"
    print("  [SKIP] Model not loaded")
else:
    results["cuda_execution"]["status"] = "SKIP"
    results["cuda_execution"]["reason"] = "CUDA not available"
    print("  [SKIP] CUDA not available")

# ─── Step 5: Inference timing ───────────────────────────────────────────────
print("\n[4/4] Measuring inference latency...")
if model is not None and torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        model = model.to(device)
        model.eval()

        # Test with various batch sizes
        batch_sizes = [1, 5, 10, 20, 30]
        timing_results = {}

        for bs in batch_sizes:
            dummy = torch.randn(bs, 3, 256, 128, device=device)

            # Warmup (3 runs)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy)
            torch.cuda.synchronize()

            # Measure (20 runs)
            latencies = []
            with torch.no_grad():
                for _ in range(20):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    output = model(dummy)
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)

            import numpy as np
            latencies_arr = np.array(latencies)
            embedding_dim = output.shape[1] if len(output.shape) == 2 else output.shape[-1]

            timing_results[f"batch_{bs}"] = {
                "batch_size": bs,
                "mean_ms": round(float(latencies_arr.mean()), 2),
                "std_ms": round(float(latencies_arr.std()), 2),
                "min_ms": round(float(latencies_arr.min()), 2),
                "max_ms": round(float(latencies_arr.max()), 2),
                "p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
                "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
                "p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
                "per_image_ms": round(float(latencies_arr.mean() / bs), 2),
            }

            print(f"  Batch {bs:>2d}: {latencies_arr.mean():.2f}ms "
                  f"(per-image: {latencies_arr.mean()/bs:.2f}ms) "
                  f"p95={np.percentile(latencies_arr, 95):.2f}ms")

        results["inference_timing"]["status"] = "PASS"
        results["inference_timing"]["embedding_dim"] = int(embedding_dim)
        results["inference_timing"]["input_size"] = "3x256x128"
        results["inference_timing"]["timings"] = timing_results

        # Also test FP16
        print("\n  FP16 (half precision) test:")
        model_half = model.half()
        dummy_half = torch.randn(10, 3, 256, 128, device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(3):
                _ = model_half(dummy_half)
        torch.cuda.synchronize()
        fp16_latencies = []
        with torch.no_grad():
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model_half(dummy_half)
                torch.cuda.synchronize()
                fp16_latencies.append((time.perf_counter() - t0) * 1000)
        fp16_arr = np.array(fp16_latencies)
        results["inference_timing"]["fp16_batch10_mean_ms"] = round(float(fp16_arr.mean()), 2)
        results["inference_timing"]["fp16_batch10_per_image_ms"] = round(float(fp16_arr.mean() / 10), 2)
        print(f"  FP16 batch 10: {fp16_arr.mean():.2f}ms "
              f"(per-image: {fp16_arr.mean()/10:.2f}ms)")

        print("\n  Status: PASS")
    except Exception as e:
        results["inference_timing"]["status"] = "FAIL"
        results["inference_timing"]["error"] = str(e)
        results["issues"].append(f"Inference timing failed: {e}")
        print(f"  [FAIL] {e}")
        traceback.print_exc()
else:
    results["inference_timing"]["status"] = "SKIP"
    print("  [SKIP] Model or CUDA not available")

# ─── Final verdict ───────────────────────────────────────────────────────────
all_pass = all(
    results[k]["status"] == "PASS"
    for k in ["torchreid_import", "model_load", "cuda_execution", "inference_timing"]
)
results["verdict"] = "PASS" if all_pass else "FAIL"

print(f"\n{'=' * 60}")
print(f"VERDICT: {results['verdict']}")
if results["issues"]:
    print(f"Issues: {len(results['issues'])}")
    for issue in results["issues"]:
        print(f"  - {issue}")
print(f"{'=' * 60}")

# Save JSON results
output_path = "docs/phases/phase_2/task_0_results.json"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {output_path}")
