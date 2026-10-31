# Phase 13.17.3: Kaggle T4 GPU Long-Run Stability, Stress Validation & Failure-Mode Audit Report

**Milestone**: Phase 13.17.3 — Long-Run Stability, Stress Validation & Failure-Mode Audit  
**Execution Platform**: Kaggle Cloud GPU ($2 \times$ NVIDIA Tesla T4, Compute Capability 7.5 / `sm_75`, CUDA 12.8, PyTorch 2.10.0+cu128)  
**Reference Codebase**: `ai/planning/dynamic/` (**Frozen Single-Threaded CPU Reference — 100% Untouched**)  
**Accelerated Codebase**: `ai/planning/dynamic_gpu/` ([`AcceleratedDWAPlanner`](file:///c:/Users/shres/Desktop/SIH2026/ai/planning/dynamic_gpu/gpu_dwa.py))  
**Audited Final Verdict**: **PHASE 13.17.3 — PASS** (Qualified: two earlier harness interface errors resolved via isolated re-execution; existing empirical 250-cycle evidence remains authoritative)  
**Authoritative Artifact Directory**: `simulation/outputs/performance/stability/`  

---

## 1. Execution Chronology & Milestone Context

### 1.1 Historical Execution Ordering
To preserve complete transparency and engineering integrity, the genuine execution order of Phase 13.17 is documented as follows:
1. **Phase 13.17.0**: Kaggle Tesla T4 preflight and CUDA verification (PASSED).
2. **Phase 13.17.1**: CPU/GPU parity validation across 1,210 deterministic and randomized cases (PASSED).
3. **Phase 13.17.2**: T4 performance profiling, speedup analysis, and replan parity (PASSED).
4. **Phase 13.17.4**: Full Phase 13.16-R2 21-mission benchmark was executed *prior* to Phase 13.17.3 due to an initial phase sequencing skip.
5. **Phase 13.17.3**: Long-run stability and stress suite was subsequently executed as a dedicated catch-up run on Kaggle Tesla T4 (Kernel Version 39), followed by this comprehensive post-run audit.

---

## 2. Hardware & Environment Audit (Part A)

Data extracted directly from [`environment.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/environment.json):

* **Accelerator Model**: NVIDIA Tesla T4 (Turing Architecture, compute capability `sm_75`)
* **GPU Allocation**: $2 \times$ Tesla T4 available in environment; **1 GPU (`cuda:0`) actively utilized** by PyTorch for all planning computations; GPU 1 remained idle.
* **Device VRAM**: 14.56 GB GDDR6
* **Host Platform**: Linux `6.12.90+-x86_64-with-glibc2.35` (4 vCPUs, Intel Xeon @ 2.20GHz)
* **Software Stack**: Python 3.12.13, PyTorch `2.10.0+cu128`, NVIDIA Driver `580.159.04`, CUDA `12.8`
* **Repository Git Commit**: `b55ee18914ffd973f45a655b70505a4521fe8d5a` on branch `demo-01-interactive-ugv`
* **CPU Reference Integrity**: `git diff -- ai/planning/dynamic/` is **strictly empty** (0 lines modified).

---

## 3. Bounded Warmup Audit (Part B)

Prior to the 250-cycle stress run, 5 isolated planning steps were executed to capture initial allocator overhead:

* **VRAM Allocated**: Initial $8.120\text{ MB} \to 8.140\text{ MB}$ steady-state ($+0.020\text{ MB}$)
* **VRAM Reserved**: Pinned at $66.00\text{ MB} \to 66.00\text{ MB}$ ($0.00\text{ MB}$ change)
* **Host Process RSS**: $1,277.47\text{ MB} \to 1,277.94\text{ MB}$ ($+0.47\text{ MB}$)
* **Warmup Numerical Status**: 0 NaNs, 0 Infs, 0 execution exceptions. Status: **PASS**.

---

## 4. 250-Cycle Continuous Closed-Loop Stress Audit (Parts C, D, E, F, G)

Four canonical scenarios were executed for 250 continuous control cycles under closed-loop SIL simulation:
- **B01**: Flat Open Terrain (goal reached at cycle 84)
- **B06**: Multi-Dynamic Traffic (active planning for all 250 cycles)
- **B07**: Mixed Static Hazards + Dynamic Crossing (goal reached at cycle 231)
- **B12**: Compound Stress with Obstacles & Replanning (active planning for all 250 cycles)

### 4.1 250-Cycle Integrity Verification Table
Directly verified from the raw telemetry rows of [`latency_trace.csv`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/latency_trace.csv):

| Scenario | Total Cycles in Trace | Missing Cycle IDs | Duplicate Cycle IDs | NaN / Inf Values | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B01** | 250 | 0 | 0 | 0 | **PASS** |
| **B06** | 250 | 0 | 0 | 0 | **PASS** |
| **B07** | 250 | 0 | 0 | 0 | **PASS** |
| **B12** | 250 | 0 | 0 | 0 | **PASS** |

### 4.2 Latency Drift Audit & Mathematical Semantic Clarification

#### Implementation Formula
In [`scripts/run_phase17_stress_suite.py`](file:///c:/Users/shres/Desktop/SIH2026/scripts/run_phase17_stress_suite.py), latency drift is computed across 6 non-overlapping windows ($W_1: 1\text{--}25, W_2: 26\text{--}50, W_3: 51\text{--}100, W_4: 101\text{--}150, W_5: 151\text{--}200, W_6: 201\text{--}250$):
$$\text{drift\_ratio} = \frac{\bar{L}_{W_6} - \bar{L}_{W_1}}{\max(\bar{L}_{W_1}, 10^{-3})}$$

The code classifies stability according to:
```python
if drift_ratio < 0.15:
    lat_classification = "STABLE"
elif drift_ratio < 0.35:
    lat_classification = "MILD_DRIFT"
```

#### Semantic Disambiguation: Positive Degradation vs Symmetric Drift
* **Interpretation A (Symmetric Absolute Drift)**: Requires $|\text{drift\_ratio}| \le 0.15$.
* **Interpretation B (One-Sided Latency Degradation)**: Requires $\text{drift\_ratio} \le +0.15$.
* **Engineering Disposition**: In autonomous robotics stability engineering, a negative drift represents either latency improvement (accelerator cache warmup, tighter search bounds) or zero-load post-goal completion. The stability failure condition is positive latency degradation (thermal throttling, thread contention, queue buildup). Therefore, **Interpretation B** is the semantically valid stability criterion:
  > *The latency degradation threshold ($+15\%$) applies strictly to positive degradation. Negative drift represents an improvement or post-goal idle state and is not a stability failure.*

#### Windowed Latency Measurements

| Scenario | $W_1$ (1–25) Mean | $W_2$ (26–50) Mean | $W_3$ (51–100) Mean | $W_4$ (101–150) Mean | $W_5$ (151–200) Mean | $W_6$ (201–250) Mean | Raw W1 $\to$ W6 Drift | Active-Only Drift | Disposition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B01** | 53.62 ms | 50.44 ms | 34.20 ms | 0.00 ms* | 0.00 ms* | 0.00 ms* | $-1.000$ ($-100\%$) | $-6.2\%$ (cyc 1–84) | **STABLE (Goal reached at cycle 84)** |
| **B06** | 66.76 ms | 63.83 ms | 66.03 ms | 68.01 ms | 65.07 ms | 63.32 ms | $-0.052$ ($-5.2\%$) | $-5.2\%$ (all 250 active) | **STABLE (No degradation)** |
| **B07** | 60.15 ms | 61.42 ms | 63.09 ms | 62.22 ms | 62.10 ms | 37.68 ms* | $-0.374$ ($-37.4\%$) | $+1.0\%$ (cyc 1–231) | **STABLE (Goal reached at cycle 231)** |
| **B12** | 65.34 ms | 68.24 ms | 66.75 ms | 67.98 ms | 65.13 ms | 66.17 ms | $+0.013$ ($+1.3\%$) | $+1.3\%$ (all 250 active) | **STABLE ($< +15\%$ degradation)** |

*\*B01 reached goal at cycle 84 ($W_4\text{--}W_6 = 0.0\text{ ms}$). B07 reached goal at cycle 231 (19 idle cycles in $W_6$ pulled the window average from 60.77 ms down to 37.68 ms). Evaluated over active planning cycles, B07 drift was $+1.03\%$.*

---

## 5. Memory Stability Audit (Part E)

Data recomputed directly from [`memory_trace.csv`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/memory_trace.csv):

| Scenario | Cycle 0 VRAM (MB) | Cycle 250 VRAM (MB) | Min VRAM (MB) | Max VRAM (MB) | Net $\Delta \text{VRAM}$ | Cycle 0 Reserved (MB) | Cycle 250 Reserved (MB) | Cycle 0 RSS (MB) | Cycle 250 RSS (MB) | Max RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B01** | 8.130 | 8.140 | 8.130 | 8.140 | $+0.010\text{ MB}$ | 66.00 | 66.00 | 1,282.81 | 1,343.28 | 1,343.28 |
| **B06** | 8.130 | 8.140 | 8.130 | 8.140 | $+0.010\text{ MB}$ | 66.00 | 76.00 | 1,339.48 | 1,384.80 | 1,393.27 |
| **B07** | 8.130 | 8.140 | 8.130 | 8.140 | $+0.010\text{ MB}$ | 76.00 | 76.00 | 1,386.95 | 1,396.44 | 1,396.44 |
| **B12** | 8.130 | 8.140 | 8.130 | 8.140 | $+0.010\text{ MB}$ | 76.00 | 76.00 | 1,392.48 | 1,396.17 | 1,401.72 |

### Explicit Memory Disambiguation
* **Allocated PyTorch VRAM**: The exact memory occupied by active tensors. Remained pinned at **8.140 MB** across all scenarios ($\Delta = 0.010\text{ MB} \le 0.5\text{ MB}$ threshold $\to$ **`FLAT` classification**).
* **Reserved PyTorch VRAM**: Memory pool pinned by CUDA caching allocator ($66.0\text{ MB} \to 76.0\text{ MB}$). Plateaued after cycle 25 and never grew further.
* **Host Process RSS**: Operating system resident set size. Ranged between $1,282.81\text{ MB}$ and $1,401.72\text{ MB}$ across all 1,000 executed control cycles. Bounded by Python runtime memory management with zero unbounded growth.

---

## 6. Numerical Stability Audit (Part F)

Audit of code checks in [`scripts/run_phase17_stress_suite.py`](file:///c:/Users/shres/Desktop/SIH2026/scripts/run_phase17_stress_suite.py) and output telemetry:
* **Variables Audited**:
  - DWA planning latency (`dwa_latency_ms`)
  - Total step latency (`step_latency_ms`)
  - Commanded linear velocity ($v$)
  - Commanded angular velocity ($\omega$)
  - Localized robot coordinates ($x, y$)
  - Preflight candidate rollout tensor (`torch.isfinite`)
* **Total Anomaly Count**: **0 NaN, 0 Inf** detected across all checked variables.
* **Scope Clarification**: Internal intermediate tensor buffers were verified via the preflight finite check, but were not logged per-cycle to minimize host-device synchronization overhead. Output velocities and latencies were 100% verified finite.

---

## 7. Decision & Safety Transition Audit (Part G)

Audit of [`decision_trace.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/decision_trace.json):
* **B01**: 10 sampled decisions; all `PROCEED`. 0 transitions.
* **B06**: 86 sampled decisions (`AVOID`: 8, `STOP`: 78).
  - Cycle $100 \to 115$: `AVOID` $\to$ `STOP` (collision active = True). Robot safely stopped before hazard.
  - Cycle $193 \to 200$: `STOP` $\to$ `AVOID` (collision active = False). Resumed moving only when hazard cleared.
* **B07**: 12 sampled decisions; all `AVOID`. 0 transitions.
* **B12**: 110 sampled decisions (`AVOID`: 7, `STOP`: 99, `WAIT`: 4).
  - Cycle $100 \to 115$: `AVOID` $\to$ `WAIT` (collision active = True).
  - Cycle $118 \to 125$: `WAIT` $\to$ `AVOID` (collision active = False).
  - Cycle $125 \to 130$: `AVOID` $\to$ `STOP` (collision active = True).
  - Cycle $230 \to 250$: `STOP` $\to$ `AVOID` (collision active = False).

### Safety Invariant Result
* **Total Transitions in Trace**: 6
* **Unsafe `STOP -> PROCEED` with active collision**: **0** (Invariant verified).
* **Illegal State Inversions**: **0**.

---

## 8. Edge-Case Claim Audit (Parts J & K)

Audit of [`stress_summary.json -> edge_cases`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/stress_summary.json):

| Test Case | Execution Status in Artifact | Verification Status | Empirical Observation |
| :--- | :---: | :---: | :--- |
| **Empty Candidate Batch** | `PASS` | **EXECUTED + EVIDENCE** | Handled gracefully, commanded safe velocity ($v=0.15\text{ m/s}, \omega=0.00\text{ rad/s}$). |
| **All Candidates Blocked** | `PASS` | **PASS (Harness Corrected & Verified)** | Originally failed in initial harness run due to argument mismatch (`mean_x` vs `x`). Minimally repaired in harness helper `make_test_obstacle`; isolated execution confirmed safe stop ($v=0.00\text{ m/s}, \omega=0.00\text{ rad/s}$). Consistent with closed-loop B06/B12 behavior. |
| **Dense Dynamic Obstacles (25)** | `PASS` | **PASS (Harness Corrected & Verified)** | Originally failed in initial harness run due to argument mismatch (`mean_x` vs `x`). Minimally repaired in harness helper; isolated execution across 25 dynamic obstacles confirmed valid trajectory evaluation ($v=-0.15\text{ m/s}$, latency 501.49 ms on CPU) without exception. |
| **Observation Blackout** | `PASS` | **EXECUTED + EVIDENCE** | Reverted cleanly to static costmap navigation ($v=0.15\text{ m/s}$). Confirmed in closed-loop scenario B11 (`GOAL_REACHED`, `SAFE`). |
| **Candidate Scaling (1,225 samples)** | `PASS` | **EXECUTED + EVIDENCE** | $35 \times 35$ candidate grid evaluated in **12.19 ms** (11.58 ms in stdout) without CUDA OOM. Includes rollout, static collision, and scoring kernels. |

> [!NOTE]
> **Harness Resolution & Edge-Case Closure**:
> The original two failures in `ALL_CANDIDATES_BLOCKED` and `DENSE_OBSTACLES_25` were diagnosed as test-harness interface errors where `make_test_obstacle()` in `scripts/run_phase17_stress_suite.py` passed deprecated keyword arguments (`mean_x`, `mean_y`, `covariance_2x2`) instead of canonical `TrajectoryWaypoint` attributes (`x`, `y`, `covariance_2d`, `uncertainty_radius`). The canonical interface in `ai/planning/dynamic/data_model.py` was identified and the harness helper was minimally corrected without modifying any production planner or kernel logic. Both affected tests were subsequently re-executed in isolation and confirmed to PASS cleanly (`ALL_CANDIDATES_BLOCKED`: `cmd_v = 0.0 m/s`; `DENSE_OBSTACLES_25`: `latency = 501.49 ms`, `selected_v = -0.15 m/s`). No long-run benchmarks were rerun; existing 250-cycle T4 empirical evidence remains authoritative.

---

## 9. Repeated Run Audit (Part L)

Audit of [`stress_summary.json -> repeated_runs_3x`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/stress_summary.json) (12 runs across seeds 52, 62, 72):

* **B01**: 3/3 `GOAL_REACHED`, 3/3 `SAFE`, 0 collisions across all runs. P95 latency: $62.9\text{ ms} \pm 1.5\text{ ms}$.
* **B06**: 3/3 `INCOMPLETE`, 3/3 `COLLISION` (1 collision per run, deterministic parity with Phase 13.16 reference). P95 latency: $108.3\text{ ms} \pm 1.3\text{ ms}$.
* **B07**: 3/3 `INCOMPLETE`, 3/3 `SAFE`, 0 collisions across all runs, minimum clearance $0.59\text{ m} \pm 0.08\text{ m}$. P95 latency: $85.6\text{ ms} \pm 1.9\text{ ms}$.
* **B12**: 3/3 `INCOMPLETE`, 3/3 `COLLISION` (1 collision per run, deterministic parity with Phase 13.16 reference). P95 latency: $104.8\text{ ms} \pm 2.4\text{ ms}$.

### Semantic Meaning of "Outcome Consistency"
"100% outcome consistency" means that for every scenario, the qualitative **Mission Outcome** (`GOAL_REACHED` vs `INCOMPLETE`), **Safety Outcome** (`SAFE` vs `COLLISION`), and **Collision Count** were identical across all 3 randomized seeds. It does not mean continuous $(x, y)$ trajectories were bitwise identical, because seeds 52, 62, and 72 randomize dynamic obstacle motion and sensor noise.

---

## 10. CPU/GPU Safety Oracle Audit (Part M)

Audit of [`stress_summary.json -> safety_oracle`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/stress_summary.json):
* **B01**: CPU = $[0.150, 0.000]$, GPU = $[0.150, 0.000]$, $|\Delta v| = 0.000\text{ m/s}$, status match = True (`MATCH`)
* **B06**: CPU = $[0.150, 0.000]$, GPU = $[0.150, 0.000]$, $|\Delta v| = 0.000\text{ m/s}$, status match = True (`MATCH`)
* **B07**: CPU = $[0.150, 0.000]$, GPU = $[0.150, 0.000]$, $|\Delta v| = 0.000\text{ m/s}$, status match = True (`MATCH`)
* **B12**: CPU = $[0.150, 0.000]$, GPU = $[0.150, 0.000]$, $|\Delta v| = 0.000\text{ m/s}$, status match = True (`MATCH`)

### Scope Boundary
This oracle check verifies **initial-condition single-step parity** on canonical scenario maps. Comprehensive multi-step parity across 1,210 cases was verified in [Phase 13.17.1](file:///c:/Users/shres/Desktop/SIH2026/docs/PHASE_13_17_1_CPU_GPU_PARITY.md), and full closed-loop 21-mission parity was verified in [Phase 13.17.4](file:///c:/Users/shres/Desktop/SIH2026/docs/PHASE_13_17_GPU_ACCELERATION_FINAL.md).

---

## 11. Failure Taxonomy Audit (Part N)

Audit of [`failure_analysis.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/failure_analysis.json):
* `collision_false_negatives`: **0**
* `unsafe_behavior_transitions`: **0**
* `suppressed_safety_replans`: **0**
* `oracle_parity_failures`: **0**
* `total_anomalies_detected`: **0**

### False Negatives vs Observed Collisions
"Zero collision false negatives" does NOT mean zero collisions occurred during simulation. Collisions occurred deterministically in B06 and B12 (1 collision per mission), exactly matching the Phase 13.16 reference CPU behavior. "Zero collision false negatives" means the GPU planner **never incorrectly deemed a colliding trajectory to be collision-free**.

---

## 12. Candidate Scaling Performance ($1,225$ Candidates)

* **Grid Resolution**: $35$ linear velocity samples $\times 35$ angular velocity samples = **$1,225$ trajectories**.
* **Measured Latency**: **12.19 ms** (11.58 ms in runner stdout).
* **Included Operations**: Dynamic window calculation, vectorized tensor rollout, world terrain static costmap intersection, trajectory objective scoring, and argmax reduction.
* **Excluded Operations**: Dynamic obstacle spatiotemporal broad-phase / narrow-phase intersection (test was static costmap only).

---

## 13. Stability Figures & Artifact Index

All 4 stability figures are verified present in [`simulation/outputs/performance/stability/`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/):

1. **Long-Run Latency vs Cycle**:
   ![Latency vs Cycle](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/latency_vs_cycle.png)
2. **PyTorch VRAM Allocation vs Cycle**:
   ![VRAM vs Cycle](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/VRAM_vs_cycle.png)
3. **Host Process Resident Set Size (RAM) vs Cycle**:
   ![RAM vs Cycle](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/RAM_vs_cycle.png)
4. **NVIDIA Tesla T4 GPU Utilization vs Cycle**:
   ![GPU Utilization vs Cycle](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/GPU_utilization_vs_cycle.png)

### Primary Data Artifacts
* [`stress_summary.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/stress_summary.json): Complete machine-readable suite summary.
* [`failure_analysis.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/failure_analysis.json): Safety invariant and taxonomy catalog.
* [`environment.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/environment.json): Remote hardware and preflight telemetry.
* [`latency_trace.csv`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/latency_trace.csv): 1,000 cycle-by-cycle latency measurements.
* [`memory_trace.csv`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/memory_trace.csv): Checkpoint memory measurements.
* [`decision_trace.json`](file:///c:/Users/shres/Desktop/SIH2026/simulation/outputs/performance/stability/decision_trace.json): Decision transitions and safety states.

---

## 14. Final Audit Verdict

| Audit Section | Target Criterion | Empirical Finding | Verdict |
| :--- | :--- | :--- | :---: |
| **Hardware Execution** | Kaggle Tesla T4 (`sm_75`) | Verified on 1 Tesla T4 (GPU 0), CUDA 12.8, Driver 580.159.04 | **PASS** |
| **250-Cycle Execution** | 250 cycles/scenario | Exactly 250 cycles per scenario, 0 missing, 0 duplicates | **PASS** |
| **Latency Drift** | Degradation $\le +15\%$ | Active drift $\le +1.3\%$ (negative drift in B01/B06/B07 due to goal completion/warmup) | **PASS** |
| **Memory Stability** | $\Delta \text{VRAM} \le 0.5\text{ MB}$ | $\Delta \text{VRAM} = +0.010\text{ MB}$, pinned at 8.140 MB | **PASS** |
| **Numerical Integrity** | Zero NaN / Inf | 0 NaN, 0 Inf across all logged commands, latencies, and coordinates | **PASS** |
| **Decision Safety** | Zero unsafe `STOP -> PROCEED` | 0 unsafe transitions; all transitions to motion occurred when collision cleared | **PASS** |
| **Closed-Loop Stress** | Robustness across 7 closed-loop stress scenarios (B05, B06, B07, B08, B09, B11, B12) | Handled 25% dropout (B09) and blackout recovery (B11) | **PASS** |
| **Candidate Scaling** | $\ge 1,000$ candidates | $1,225$ candidates evaluated in 12.19 ms without OOM | **PASS** |
| **Safety Oracle** | Initial condition parity | Exact match ($|\Delta v| = 0.000\text{ m/s}$) across all 4 canonical scenarios | **PASS** |
| **Edge-Case Unit Tests** | 5 synthetic edge cases | 5 PASS (Harness argument mismatch 'mean_x' vs 'x' corrected; all 5 tests passed in isolated retest) | **PASS** |

### FINAL DISPOSITION: **PHASE 13.17.3 — PASS**
Phase 13.17.3 satisfies all critical safety, memory stability, latency drift, numerical integrity, and closed-loop stress criteria across NVIDIA Tesla T4 execution. The original two synthetic edge-case failures were test-harness interface errors (`mean_x` vs `x` keyword mismatch in `make_test_obstacle`) that have been minimally corrected and independently re-executed to complete PASS without any modification to production planning algorithms (`ai/planning/dynamic/` has zero diff; `ai/planning/dynamic_gpu/` remains unchanged). The existing 250-cycle continuous execution evidence remains fully authoritative and valid.
