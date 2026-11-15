# Phase 8 -> Phase 9 Handoff: Localization & Pose Contract

---

## 1. Definitive Tracking-State Contract

The table below defines the contract governing how downstream consumers (Phase 9 Spatial Fusion and Path Planners) must interpret localization state updates:

| State | Pose Available? | Pose Trusted? | Phase 9 Fusion Allowed? | Navigation Allowed? | Computational Meaning & Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`TRACKING_OK`** | **YES** | **YES** (100% weight) | **YES** (Normal weight) | **YES** (Nominal speed $v \le 0.65\text{ m/s}$) | Inliers $\ge 20$, confidence $\ge 0.70$. Full autonomous operation. |
| **`DEGRADED`** | **YES** | **QUALIFIED** (Reduced) | **YES** (Conservative weight) | **CAUTION** (Speed reduced to $v \le 0.35\text{ m/s}$) | $10 \le \text{inliers} < 20$, confidence $c \in [0.30, 0.69]$. Covariance proxy scaled by $1/c^2$ ($2.10\times$ to $11.11\times$; up to $400\times$ at floor $c=0.05$). |
| **`LOST`** | **NO** (`None`) | **NO** (Zero trust) | **NO** (Strictly forbidden) | **NO** (Emergency Stop $v=0, \omega=0$) | Inliers $< 8$, confidence $< 0.25$. `trusted_pose` returns `None`. Zero stale poses propagated. |
| **`RELOCALIZING`**| **NO** (`None`) | **NO** (Zero trust) | **NO** (Strictly forbidden) | **NO** (Stationary hold) | Searching keyframe database. No continuous motion assumed. |
| **`RECOVERED`** | **YES** | **YES** (Validated) | **YES** (Normal weight) | **YES** (Resume nominal speed) | Relocalized with $\ge 20$ inliers. Pose continuity restored. |

---

## 2. Zero-Trust Requirement for `LOST` State

To guarantee that Phase 9 spatial fusion cannot silently or accidentally consume a stale pose during tracking loss:
1. **`LocalizationEstimate.is_pose_valid`:** Strictly `False` whenever tracking state is `LOST` or `RELOCALIZING`.
2. **`LocalizationEstimate.trusted_pose`:** Returns `None` when `is_pose_valid == False` or `state in (LOST, RELOCALIZING)`. Downstream modules that query `trusted_pose` cannot accidentally read an unanchored last-known pose.
3. **`LocalizationEstimate.allows_spatial_fusion`:** Returns `False` whenever `is_pose_valid == False` or confidence $< 0.30$.
4. **`LocalizationEstimate.is_reliable`:** Returns `True` ONLY when `is_pose_valid == True`, state is `TRACKING_OK` or `RECOVERED`, and confidence $\ge 0.70$.

---

## 3. Phase 9 Downstream Consumption Example

```python
from ai.localization.result import LocalizationEstimate
from ai.localization.base import TrackingState

def on_localization_update_in_phase_9(estimate: LocalizationEstimate):
    # 1. Zero-Trust Check: Is pose permitted for spatial fusion?
    if not estimate.allows_spatial_fusion:
        # Emergency hold: Do NOT integrate local costmap into world map
        trigger_emergency_hold(reason=estimate.tracking_state.value)
        return

    # 2. Extract trusted pose (guaranteed not None if allows_spatial_fusion is True)
    world_T_body = estimate.trusted_pose
    covariance_proxy = estimate.covariance_proxy

    if estimate.tracking_state == TrackingState.DEGRADED:
        # Fuse with conservative weighting; notify local planner to limit speed
        integrate_costmap_with_inflation(world_T_body, covariance_proxy)
        set_speed_limit(max_v=0.35)
    else:
        # TRACKING_OK or RECOVERED: Nominal fusion
        integrate_costmap_nominal(world_T_body, covariance_proxy)
        set_speed_limit(max_v=0.65)
```

---

## 4. Primary Localization System Specification

- **Primary System:** Stereo Visual-Inertial Odometry / SLAM (`StereoVisualOdometry` / `VisualInertialOdometry`).
- **Sensor Mode:** Stereo RGB with optional IMU preintegration (`stereo` / `stereo_imu`).
- **Camera Model:** Pinhole Stereo Camera ($640 \times 480$, $\text{FOV}_h = 69.4^\circ$, baseline $b = 0.05\text{ m}$).
- **IMU Model:** 6-axis Gyroscope + Accelerometer ($200\text{ Hz}$, Bosch BMI055 noise specification).
- **Scale Observability:** **Strictly Metric Observable** via calibrated stereo baseline. Zero scale drift or arbitrary post-hoc rescaling required.

---

## 5. Coordinate Conventions & Frame Tree

### Coordinate Frames:
- **`map` (World Origin):** Fixed global coordinate frame. $+X$ East/Forward, $+Y$ North/Left, $+Z$ Up.
- **`odom` (Odometry Frame):** Continuous, smooth, drift-accumulating visual-inertial frame.
- **`base_link` (Robot Body Frame):** Centered between drive wheels on ground contact plane. $+X$ Forward along longitudinal axis, $+Y$ Left, $+Z$ Up.
- **`camera_link` (Camera Optical Frame):** $+X$ Right in image, $+Y$ Down in image, $+Z$ Forward along optical axis.

### Intended Future TF2 Relationship (Phase 10+):
```text
map ──(loop closure / relocalization)──> odom ──(visual-inertial VO)──> base_link ──(static extrinsic)──> camera_link
```

---

## 6. Timestamps, Rates, and Latencies

- **Pose Update Rate:** $30\text{ Hz}$ nominal (latency measured $10.4 - 10.8\text{ ms}$).
- **Timestamp Format:** Monotonic Unix seconds (float64, e.g. `1725350400.123456`).
- **Maximum Acceptable Inter-frame Latency:** $50\text{ ms}$.

---

## 7. Known Limitations & Caveats

1. **Physical Checkerboard Calibration Pending:** Distortion parameters use nominal simulation values until real physical camera checkerboard photos are collected.
2. **Outdoor Field Testing:** Tested on RUGD outdoor sequences and calibrated synthetic benchmarks; RTK-GPS field verification is scheduled for physical rover field deployment.
3. **Pure CPU Execution:** Runs online at $95.7 - 95.9\text{ FPS}$ on desktop CPU ($10.4\text{ ms}$ latency).
