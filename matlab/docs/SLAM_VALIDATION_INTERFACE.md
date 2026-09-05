# SLAM Validation Interface & Data Contract (Phase 7 -> Phase 8 Handoff)

---

## 1. Purpose
This document establishes the official data exchange contract between the future **Phase 8 Visual Odometry / SLAM pipeline** and the **Phase 7 MATLAB Evaluation Framework** (`matlab/slamming/`).

Any visual localization, odometry, or SLAM module developed in Phase 8 must emit trajectories adhering to this specification to enable automated ATE (Absolute Trajectory Error) and RPE (Relative Pose Error) benchmarking.

---

## 2. Required Trajectory Format

### Coordinate Conventions:
- **World / Map Coordinate Frame:**
  - $+X$: East / Forward (meters)
  - $+Y$: North / Left (meters)
  - $+Z$: Up / Elevation (meters)
- **Robot Body Frame (`base_link`):**
  - $+X$: Forward along rover longitudinal axis
  - $+Y$: Left along rover lateral axis
  - $+Z$: Up orthogonal to wheel contact plane
- **Camera Optical Frame (`camera_link`):**
  - $+X$: Right in image plane
  - $+Y$: Down in image plane
  - $+Z$: Forward along optical axis

### Data Representation (Matrix or CSV):
Each trajectory must be represented as an $N \times 8$ array or CSV file with header:

```csv
timestamp,x,y,z,qx,qy,qz,qw
```

Where:
- `timestamp`: Monotonic Unix timestamp in seconds (float64, e.g. `1725350400.123456`)
- `x, y, z`: Estimated or ground-truth 3D position in meters (float64)
- `qx, qy, qz, qw`: Unit quaternion orientation (Hamilton convention: $q_w + q_x i + q_y j + q_z k$)

For 2D / planar ground operations, the minimal format is $N \times 4$:
```csv
timestamp,x,y,yaw
```
where `yaw` is in radians $[-\pi, \pi]$.

---

## 3. Ground Truth Trajectory Format
Reference / ground truth trajectories (from RTK-GPS, motion capture, or high-fidelity simulation odometry) must use the identical format.
- If timestamps do not match exactly, the MATLAB interface will perform temporal linear interpolation matching the higher-frequency stream to the reference timestamps with a maximum matching tolerance $\Delta t \le 0.05\text{ s}$.

---

## 4. Trajectory Alignment Procedure
The MATLAB evaluation suite uses the **rigid Umeyama SE(3) / SE(2) algorithm** (`matlab/slamming/align_trajectories.m`):
1. Computes centroids $\boldsymbol{\mu}_{\text{gt}}$ and $\boldsymbol{\mu}_{\text{est}}$.
2. Computes cross-covariance $H = \frac{1}{N} \sum (\mathbf{p}_{\text{est}, i} - \boldsymbol{\mu}_{\text{est}})(\mathbf{p}_{\text{gt}, i} - \boldsymbol{\mu}_{\text{gt}})^T$.
3. Extracts optimal rotation $R$ via SVD.
4. Scale $s \equiv 1.0$ is **strictly preserved** (no arbitrary scale factor alignment for metric visual odometry).

---

## 5. Output Evaluation Metrics

When fed reference and estimated trajectories, `matlab/slamming/` produces:

### Absolute Trajectory Error (ATE):
$$\mathbf{E}_i = \mathbf{P}_{\text{gt}, i} - (R \mathbf{P}_{\text{est}, i} + \mathbf{t})$$
- $\text{ATE}_{\text{RMSE}}$ (meters): Primary benchmark metric
- $\text{ATE}_{\text{mean}}$ (meters)
- $\text{ATE}_{\text{median}}$ (meters)
- $\text{ATE}_{\text{std}}$ (meters)
- $\text{ATE}_{\max}$ (meters)

### Relative Pose Error (RPE):
Evaluates local drift over fixed temporal window $\Delta t = 1.0\text{ s}$ ($10$ steps at $10\text{ Hz}$):
- $\text{RPE}_{\text{trans\_RMSE}}$ ($\text{m/s}$ or $\text{m/m}$)
- $\text{RPE}_{\text{rot\_RMSE}}$ ($\text{deg/s}$ or $\text{deg/m}$)

---

## 6. Verification Status in Phase 7
- **Mathematical Evaluation Engines:** Fully implemented and validated on synthetic perturbed trajectories in `matlab/slamming/test_synthetic_slam_metrics.m`.
- **Real Visual SLAM Data:** Labeled **`PENDING PHASE 8`** (awaiting Phase 8 visual odometry implementation).
