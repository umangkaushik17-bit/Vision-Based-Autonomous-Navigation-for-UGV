"""Phase 13.26A: Intel GPU/XPU Perception Acceleration Benchmark & Parity Verification.

Validates:
1. Numerical Parity between CPU and Intel XPU for LightweightTraversabilityNet
2. Safety parity across 10 representative scenarios (real & synthetic)
3. Granular latency breakdown:
   - Preprocessing latency
   - Model inference latency (with hardware synchronization)
   - Postprocessing latency
   - Total perception stage latency
4. End-to-end integration with C++ DWA:
   - CPU perception + C++ DWA vs XPU perception + C++ DWA
5. Generates machine-readable report: reports/perception_intel_gpu_validation.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.costmap import BEVTerrainCostmap
from ai.planning.backends import CPPDWABackend, DWABackendType, UnifiedDWAPlanner
from ai.planning.base import GoalState, PlanningConfig, RobotState
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.dense_predictor import DenseTraversabilityPredictor
from scripts.benchmark_dwa_dynamic import create_world_map, make_obstacle


def create_synthetic_scenarios() -> List[Tuple[str, np.ndarray]]:
    """Generates synthetic scenarios complementing real image files."""
    scenarios = []

    # 1. Open flat terrain
    img_open = np.full((400, 640, 3), 120, dtype=np.uint8)
    img_open[200:, :] = [160, 150, 130]  # Smooth dirt trail
    scenarios.append(("Synthetic: Open Traversable Trail", img_open))

    # 2. Central obstacle
    img_obs = img_open.copy()
    cv2.circle(img_obs, (320, 280), 45, (40, 40, 40), -1)  # Dark boulder
    scenarios.append(("Synthetic: Central Obstacle", img_obs))

    # 3. Non-traversable water/mud pit
    img_mud = img_open.copy()
    cv2.ellipse(img_mud, (320, 320), (160, 50), 0, 0, 360, (25, 45, 60), -1)
    scenarios.append(("Synthetic: Non-Traversable Water/Mud", img_mud))

    # 4. Low-contrast twilight scene
    img_twilight = (img_open.astype(np.float32) * 0.35).astype(np.uint8)
    scenarios.append(("Synthetic: Low-Contrast Twilight Scene", img_twilight))

    return scenarios


def load_test_scenarios() -> List[Tuple[str, np.ndarray]]:
    """Loads 10 representative test scenarios (real + synthetic)."""
    scenarios = []

    # Real benchmark images
    real_paths = [
        ("Real: Dirt Trail 0000", "data/benchmark_test/own_real/rgb/own_seq_01_dirt_0000.png"),
        ("Real: Dirt Trail 0001", "data/benchmark_test/own_real/rgb/own_seq_01_dirt_0001.png"),
        ("Real: Dirt Trail 0002", "data/benchmark_test/own_real/rgb/own_seq_01_dirt_0002.png"),
        ("Real: Dirt Trail 0003", "data/benchmark_test/own_real/rgb/own_seq_01_dirt_0003.png"),
        ("Real: Test Sample", "data/test/sample.jpg"),
        ("Real: Processed Sample", "data/processed/sample_out.jpg"),
    ]

    for label, rel_path in real_paths:
        if os.path.exists(rel_path):
            bgr = cv2.imread(rel_path)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                scenarios.append((label, rgb))

    # Add synthetic scenarios
    scenarios.extend(create_synthetic_scenarios())
    return scenarios


def compute_stage_latencies(
    predictor: DenseTraversabilityPredictor,
    image_rgb: np.ndarray,
    is_xpu: bool,
    n_iterations: int = 100,
) -> Dict[str, List[float]]:
    """Measures preprocessing, model inference, postprocessing, and total latency."""
    t_pre_list = []
    t_inf_list = []
    t_post_list = []
    t_tot_list = []

    w_tgt, h_tgt = predictor.target_wh
    orig_h, orig_w = image_rgb.shape[:2]
    mean_val = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_val = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    device = predictor._torch_device

    for _ in range(n_iterations):
        # 1. Total wall clock start
        t_start = time.perf_counter()

        # Preprocessing
        t0 = time.perf_counter()
        resized = cv2.resize(image_rgb, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        normalized = ((resized.astype(np.float32) / 255.0) - mean_val) / std_val
        tensor_in = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        if is_xpu:
            torch.xpu.synchronize(device)
        t_pre = (time.perf_counter() - t0) * 1000.0

        # Model Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_tensor = predictor.model(tensor_in)
            probs_tensor = F.softmax(logits_tensor, dim=1).squeeze(0)
            if is_xpu:
                torch.xpu.synchronize(device)
        t_inf = (time.perf_counter() - t0) * 1000.0

        # Postprocessing
        t0 = time.perf_counter()
        probs_np = probs_tensor.cpu().numpy()
        logits_np = logits_tensor.squeeze(0).cpu().numpy()
        prob_safe = probs_np[0]
        prob_unsafe = probs_np[2]
        drivability_small = prob_safe * (1.0 - (prob_unsafe * 0.9))
        drivability_small = np.clip(drivability_small, 0.0, 1.0)
        drivability_map = cv2.resize(drivability_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        drivability_map = np.clip(drivability_map, 0.0, 1.0)
        _ = cv2.resize(logits_np[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        t_post = (time.perf_counter() - t0) * 1000.0

        t_tot = (time.perf_counter() - t_start) * 1000.0

        t_pre_list.append(t_pre)
        t_inf_list.append(t_inf)
        t_post_list.append(t_post)
        t_tot_list.append(t_tot)

    return {
        "preprocessing": t_pre_list,
        "inference": t_inf_list,
        "postprocessing": t_post_list,
        "total": t_tot_list,
    }


def compute_statistics(samples: List[float]) -> Dict[str, float]:
    arr = np.array(samples)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "fps": round(1000.0 / max(0.001, float(np.mean(arr))), 1),
    }


def verify_numerical_parity(
    cpu_pred: DenseTraversabilityPredictor,
    xpu_pred: DenseTraversabilityPredictor,
    scenarios: List[Tuple[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    parity_results = []

    for name, img in scenarios:
        res_cpu = cpu_pred.predict(img)
        res_xpu = xpu_pred.predict(img)

        # Check for NaN / Inf
        has_nan_cpu = np.isnan(res_cpu.mask).any() or np.isnan(res_cpu.logits).any()
        has_nan_xpu = np.isnan(res_xpu.mask).any() or np.isnan(res_xpu.logits).any()
        has_inf_cpu = np.isinf(res_cpu.mask).any() or np.isinf(res_cpu.logits).any()
        has_inf_xpu = np.isinf(res_xpu.mask).any() or np.isinf(res_xpu.logits).any()

        # Logits difference
        abs_diff_logits = np.abs(res_cpu.logits - res_xpu.logits)
        max_abs_logits = float(np.max(abs_diff_logits))
        mean_abs_logits = float(np.mean(abs_diff_logits))

        # Traversability mask difference
        abs_diff_mask = np.abs(res_cpu.mask - res_xpu.mask)
        max_abs_mask = float(np.max(abs_diff_mask))
        mean_abs_mask = float(np.mean(abs_diff_mask))

        # Classification agreement
        class_cpu = res_cpu.mask >= 0.50
        class_xpu = res_xpu.mask >= 0.50
        agreement_pct = float(np.mean(class_cpu == class_xpu) * 100.0)

        # False safe check: CPU says unsafe (< 0.3) but XPU says safe (> 0.7)
        false_safe_pixels = int(np.sum((res_cpu.mask < 0.30) & (res_xpu.mask > 0.70)))

        parity_results.append({
            "scenario": name,
            "dimensions": list(img.shape),
            "no_nan": not (has_nan_cpu or has_nan_xpu),
            "no_inf": not (has_inf_cpu or has_inf_xpu),
            "max_abs_diff_logits": round(max_abs_logits, 5),
            "mean_abs_diff_logits": round(mean_abs_logits, 6),
            "max_abs_diff_mask": round(max_abs_mask, 5),
            "mean_abs_diff_mask": round(mean_abs_mask, 6),
            "class_agreement_pct": round(agreement_pct, 2),
            "false_safe_divergence_count": false_safe_pixels,
            "parity_passed": (max_abs_mask < 0.05) and (agreement_pct >= 98.0) and (false_safe_pixels == 0),
        })

    return parity_results


def benchmark_end_to_end_comparison(
    cpu_pred: DenseTraversabilityPredictor,
    xpu_pred: DenseTraversabilityPredictor,
    test_image: np.ndarray,
    n_cycles: int = 15,
) -> Dict[str, Any]:
    """Runs synchronous End-to-End comparison: CPU perception + C++ DWA vs XPU perception + C++ DWA."""
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.25, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))
    wmap = create_world_map()
    obs = make_obstacle(track_id=1, start_pos=(2.0, 0.5), velocity=(0.0, -0.4))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    planner = UnifiedDWAPlanner(backend=DWABackendType.CPP_CPU)

    def run_e2e(pred: DenseTraversabilityPredictor, is_xpu: bool):
        perc_times = []
        dwa_times = []
        tot_times = []

        # Warmup
        _ = pred.predict(test_image)
        _ = planner.compute_trajectory(state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs])

        for _ in range(n_cycles):
            t_start = time.perf_counter()

            # Perception
            t0 = time.perf_counter()
            _ = pred.predict(test_image)
            if is_xpu:
                torch.xpu.synchronize(0)
            t_perc = (time.perf_counter() - t0) * 1000.0

            # DWA Trajectory computation
            t0 = time.perf_counter()
            _ = planner.compute_trajectory(state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs])
            t_dwa = (time.perf_counter() - t0) * 1000.0

            t_tot = (time.perf_counter() - t_start) * 1000.0

            perc_times.append(t_perc)
            dwa_times.append(t_dwa)
            tot_times.append(t_tot)

        return {
            "perception_mean_ms": round(float(np.mean(perc_times)), 2),
            "dwa_mean_ms": round(float(np.mean(dwa_times)), 2),
            "total_mean_ms": round(float(np.mean(tot_times)), 2),
            "effective_fps": round(1000.0 / max(0.001, float(np.mean(tot_times))), 1),
        }

    res_cpu_e2e = run_e2e(cpu_pred, is_xpu=False)
    res_xpu_e2e = run_e2e(xpu_pred, is_xpu=True)

    return {
        "with_cpu_perception_cpp_dwa": res_cpu_e2e,
        "with_xpu_perception_cpp_dwa": res_xpu_e2e,
        "e2e_total_speedup": round(res_cpu_e2e["total_mean_ms"] / max(0.001, res_xpu_e2e["total_mean_ms"]), 2),
    }


def main() -> None:
    print("=" * 95)
    print("PHASE 13.26A: INTEL GPU/XPU PERCEPTION ACCELERATION BENCHMARK & AUDIT")
    print("=" * 95)

    ckpt_path = "outputs/rellis3d/checkpoints/best.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at: {ckpt_path}"
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "PyTorch XPU runtime unavailable!"

    print(f"[*] PyTorch Version:     {torch.__version__}")
    print(f"[*] XPU Device:          {torch.xpu.get_device_name(0)}")
    print(f"[*] Model Checkpoint:    {ckpt_path}")
    print(f"[*] C++ DWA Available:   {CPPDWABackend.is_available()}")

    # 1. Initialize Predictors
    print("\n[*] Initializing CPU and Intel XPU Predictors...")
    cpu_pred = DenseTraversabilityPredictor(checkpoint=ckpt_path, device="cpu")
    xpu_pred = DenseTraversabilityPredictor(checkpoint=ckpt_path, device="xpu:0")
    print(f"    CPU Predictor Device: {cpu_pred.device}")
    print(f"    XPU Predictor Device: {xpu_pred.device}")

    # 2. Warmup
    print("\n[*] Warming up CPU and Intel XPU computation graphs (15 passes)...")
    cpu_pred.warmup(15)
    xpu_pred.warmup(15)

    # 3. Load Scenarios
    scenarios = load_test_scenarios()
    print(f"[*] Loaded {len(scenarios)} evaluation scenarios (Real + Synthetic).")

    # 4. Numerical Parity Audit
    print("\n" + "=" * 95)
    print("PHASE 8 & 9: NUMERICAL PARITY & SAFETY AUDIT (CPU vs Intel XPU)")
    print("=" * 95)
    parity_records = verify_numerical_parity(cpu_pred, xpu_pred, scenarios)

    for p in parity_records:
        status_str = "PASS" if p["parity_passed"] else "FAIL"
        print(f"  [{status_str}] {p['scenario']:<38} | MaxAbsMask: {p['max_abs_diff_mask']:.4f} | ClassAgreement: {p['class_agreement_pct']:5.1f}% | FalseSafe: {p['false_safe_divergence_count']}")

    all_parity_passed = all(p["parity_passed"] for p in parity_records)
    print(f"\n[*] Overall Numerical & Safety Parity Status: {'PASS (Validated)' if all_parity_passed else 'FAIL'}")

    # 5. Performance Benchmark (100 measured iterations per device)
    print("\n" + "=" * 95)
    print("PHASE 10: GRANULAR LATENCY BENCHMARK (100 measured iterations)")
    print("=" * 95)

    bench_img = scenarios[0][1]
    print(f"[*] Benchmark Input: {scenarios[0][0]} (Shape: {bench_img.shape})")

    print("[*] Benchmarking CPU predictor...")
    cpu_latencies = compute_stage_latencies(cpu_pred, bench_img, is_xpu=False, n_iterations=100)
    cpu_stats = {stage: compute_statistics(samples) for stage, samples in cpu_latencies.items()}

    print("[*] Benchmarking Intel XPU predictor...")
    xpu_latencies = compute_stage_latencies(xpu_pred, bench_img, is_xpu=True, n_iterations=100)
    xpu_stats = {stage: compute_statistics(samples) for stage, samples in xpu_latencies.items()}

    # Print summary table
    print("\n" + "-" * 95)
    header = f"{'Pipeline Stage':<22} | {'CPU Mean':<12} | {'CPU P95':<11} | {'XPU Mean':<12} | {'XPU P95':<11} | {'Speedup':<9}"
    print(header)
    print("-" * 95)
    for stg in ["preprocessing", "inference", "postprocessing", "total"]:
        c_m = cpu_stats[stg]["mean_ms"]
        c_p = cpu_stats[stg]["p95_ms"]
        x_m = xpu_stats[stg]["mean_ms"]
        x_p = xpu_stats[stg]["p95_ms"]
        sp = round(c_m / max(0.001, x_m), 2)
        print(f"{stg.capitalize():<22} | {c_m:8.2f} ms | {c_p:7.2f} ms | {x_m:8.2f} ms | {x_p:7.2f} ms | {sp:7.2f}x")
    print("-" * 95)

    inf_speedup = round(cpu_stats["inference"]["mean_ms"] / max(0.001, xpu_stats["inference"]["mean_ms"]), 2)
    tot_speedup = round(cpu_stats["total"]["mean_ms"] / max(0.001, xpu_stats["total"]["mean_ms"]), 2)
    print(f"[*] Pure Model Inference Acceleration:  {inf_speedup}x ({cpu_stats['inference']['mean_ms']} ms -> {xpu_stats['inference']['mean_ms']} ms)")
    print(f"[*] Total Perception Stage Acceleration: {tot_speedup}x ({cpu_stats['total']['mean_ms']} ms -> {xpu_stats['total']['mean_ms']} ms)")

    # 6. End-to-End Navigation Comparison
    print("\n" + "=" * 95)
    print("PHASE 13: END-TO-END LATENCY TEST (Perception + C++ DWA)")
    print("=" * 95)
    e2e_res = benchmark_end_to_end_comparison(cpu_pred, xpu_pred, bench_img, n_cycles=15)
    c_e2e = e2e_res["with_cpu_perception_cpp_dwa"]
    x_e2e = e2e_res["with_xpu_perception_cpp_dwa"]
    print(f"[*] CPU Perception + C++ DWA: Total = {c_e2e['total_mean_ms']} ms (Perc = {c_e2e['perception_mean_ms']} ms, DWA = {c_e2e['dwa_mean_ms']} ms, Rate = {c_e2e['effective_fps']} Hz)")
    print(f"[*] XPU Perception + C++ DWA: Total = {x_e2e['total_mean_ms']} ms (Perc = {x_e2e['perception_mean_ms']} ms, DWA = {x_e2e['dwa_mean_ms']} ms, Rate = {x_e2e['effective_fps']} Hz)")
    print(f"[*] End-to-End Navigation Cycle Speedup: {e2e_res['e2e_total_speedup']}x")

    # 7. Save JSON Artifact
    report_dict = {
        "metadata": {
            "title": "Phase 13.26A Intel GPU/XPU Perception Acceleration & Parity Report",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": {
                "cpu": "Intel(R) Core(TM) Ultra 7 155H",
                "gpu": torch.xpu.get_device_name(0),
                "device": "xpu:0",
            },
            "pytorch_version": torch.__version__,
            "model": "LightweightTraversabilityNet (RELLIS-3D Frozen)",
            "checkpoint": ckpt_path,
        },
        "numerical_parity": {
            "all_passed": all_parity_passed,
            "scenarios_evaluated": len(parity_records),
            "records": parity_records,
        },
        "stage_benchmarks": {
            "cpu": cpu_stats,
            "xpu": xpu_stats,
            "speedups": {
                "inference_speedup": inf_speedup,
                "total_perception_speedup": tot_speedup,
            },
        },
        "end_to_end_navigation": e2e_res,
    }

    out_file = "reports/perception_intel_gpu_validation.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[*] Validation artifact persisted to: {out_file}")


if __name__ == "__main__":
    main()
