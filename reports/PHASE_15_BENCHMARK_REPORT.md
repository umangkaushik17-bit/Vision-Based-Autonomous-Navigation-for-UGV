# Phase 15: 2.5D Terrain-Aware Autonomous Navigation — Differential Benchmark Report

**Project**: SIH26126 — Vision-Based Autonomous Navigation for UGV in Outdoor Environments  
**Telemetry Artifact**: `reports/phase15_benchmark.json`  
**Benchmark Runner**: `benchmarks/run_phase15_terrain_benchmark.py`  

---

## 1. Executive Differential Summary (Phase 14 2D Baseline vs. Phase 15 2.5D Stack)

| Metric | Phase 14 Baseline (Flat 2D) | Phase 15 Evaluation (2.5D Terrain-Aware) | Status |
| :--- | :---: | :---: | :---: |
| **Mission Completion Rate (7 Scenarios)** | `28.6%` (2/7) | **`100.0%` (7/7)** | `PASS` |
| **Rollover-Threshold Violations (>= 28 deg)** | `94` | **`0`** | `PASS` |
| **Undercarriage High-Centering Events (>= 0.12 m)** | `10` | **`0`** | `PASS` |
| **Static & Dynamic Obstacle Collisions** | `0` | **`0`** | `PASS` |
| **Mean C++17 Global A* Latency (120x120)** | — | **`1.70 ms`** | `PASS (< 5.0 ms)` |
| **Mean C++17 AVX2 DWA Latency (375 rollouts)** | — | **`1.15 ms`** | `PASS (< 3.5 ms)` |

---

## 2. Per-Scenario Differential Results

| Scenario | Stack | Success | Rollovers | High-Centerings | Max Slope (deg) | Max Step (m) | Path Length (m) | A* (ms) | DWA (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SCENARIO_1_FLAT_BASELINE` | `phase14_2d` | `PASS` | `0` | `0` | `0.0` | `0.00` | `8.00` | `0.74` | `1.03` |
| `SCENARIO_1_FLAT_BASELINE` | `phase15_2_5d` | `PASS` | `0` | `0` | `0.0` | `0.00` | `8.00` | `0.80` | `1.02` |
| `SCENARIO_2_GENTLE_10DEG_SLOPE` | `phase14_2d` | `PASS` | `0` | `0` | `10.0` | `0.00` | `8.00` | `0.85` | `1.03` |
| `SCENARIO_2_GENTLE_10DEG_SLOPE` | `phase15_2_5d` | `PASS` | `0` | `0` | `10.0` | `0.00` | `8.00` | `1.41` | `0.87` |
| `SCENARIO_3_STEEP_32DEG_EMBANKMENT` | `phase14_2d` | `FAIL` | `25` | `0` | `32.0` | `0.00` | `8.00` | `0.86` | `0.88` |
| `SCENARIO_3_STEEP_32DEG_EMBANKMENT` | `phase15_2_5d` | `PASS` | `0` | `0` | `10.0` | `0.00` | `10.27` | `2.54` | `1.30` |
| `SCENARIO_4_SEVERE_24DEG_CROSS_SLOPE` | `phase14_2d` | `FAIL` | `52` | `5` | `69.0` | `0.56` | `8.00` | `0.72` | `1.16` |
| `SCENARIO_4_SEVERE_24DEG_CROSS_SLOPE` | `phase15_2_5d` | `PASS` | `0` | `0` | `0.0` | `0.00` | `9.45` | `0.90` | `1.01` |
| `SCENARIO_5_STEP_18CM_BOULDER_FIELD` | `phase14_2d` | `FAIL` | `5` | `5` | `42.0` | `0.18` | `8.00` | `0.85` | `1.15` |
| `SCENARIO_5_STEP_18CM_BOULDER_FIELD` | `phase15_2_5d` | `PASS` | `0` | `0` | `0.0` | `0.00` | `9.89` | `1.65` | `1.24` |
| `SCENARIO_6_UNDULATING_12DEG_HILLS` | `phase14_2d` | `FAIL` | `0` | `0` | `22.7` | `0.00` | `8.00` | `0.84` | `0.92` |
| `SCENARIO_6_UNDULATING_12DEG_HILLS` | `phase15_2_5d` | `PASS` | `0` | `0` | `22.7` | `0.00` | `8.11` | `1.98` | `0.92` |
| `SCENARIO_7_COMPOUND_OUTDOOR_DYNAMIC` | `phase14_2d` | `FAIL` | `12` | `0` | `32.0` | `0.00` | `8.00` | `0.93` | `2.15` |
| `SCENARIO_7_COMPOUND_OUTDOOR_DYNAMIC` | `phase15_2_5d` | `PASS` | `0` | `0` | `10.0` | `0.00` | `10.29` | `2.64` | `1.69` |
