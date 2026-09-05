# SLAM Configuration Reference Guide

This document details the configuration parameters available in [`configs/slam.yaml`](file:///c:/Users/shres/Desktop/SIH2026/configs/slam.yaml) for the SIH26126 GPS-denied visual localization subsystem.

---

## 1. System Parameters (`slam.system`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `name` | string | `"sih26126_visual_slam"` | Node / component identifier. |
| `primary_mode` | string | `"stereo"` | Primary localization mode: `monocular`, `stereo`, `stereo_imu`, or `rgbd`. |
| `publish_rate_hz` | float | `30.0` | Target localization update frequency in Hertz. |
| `enable_loop_closure` | bool | `true` | Enables DBoW2-style visual place recognition and trajectory relaxation. |
| `enable_relocalization`| bool | `true` | Enables automatic tracking recovery after entering `LOST` state. |
| `diagnostic_logging` | bool | `true` | Exports detailed per-frame tracking states, inliers, and latencies. |

---

## 2. Camera Parameters (`slam.camera`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `width` / `height` | int | `640` / `480` | Image frame resolution in pixels. |
| `fps` | int | `30` | Camera capture frame rate. |
| `horizontal_fov_deg` | float | `69.4` | Camera horizontal field of view. |
| `intrinsics.fx`, `fy` | float | `462.14` | Focal lengths in pixels ($f_x = f_y$). |
| `intrinsics.cx`, `cy` | float | `320.0`, `240.0` | Principal point in pixels. |
| `stereo.baseline_m` | float | `0.05` | Distance between left and right optical centers ($5\text{ cm}$ for D435i, $12\text{ cm}$ for ZED 2). |
| `extrinsics_base_to_camera.position_m` | list | `[0.15, 0.00, 0.40]` | Camera optical center relative to rover `base_link` ($X$-fwd, $Y$-left, $Z$-up). |
| `extrinsics_base_to_camera.pitch_deg` | float | `15.0` | Downward camera tilt angle in degrees. |

---

## 3. IMU Parameters (`slam.imu`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `enabled` | bool | `true` | Enables visual-inertial sensor fusion. |
| `rate_hz` | float | `200.0` | IMU sampling rate (Bosch BMI055 specification). |
| `gravity_m_s2` | float | `9.80665` | Gravitational acceleration magnitude. |
| `gyro_noise_density` | float | `0.004` | Gyroscope white noise in $\text{rad/s}/\sqrt{\text{Hz}}$. |
| `accel_noise_density` | float | `0.04` | Accelerometer white noise in $\text{m/s}^2/\sqrt{\text{Hz}}$. |
| `gyro_bias_walk` | float | `0.00004` | Gyroscope random walk bias instability. |
| `accel_bias_walk` | float | `0.004` | Accelerometer random walk bias instability. |

---

## 4. Frontend & Keyframing (`slam.frontend` & `slam.keyframing`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `feature_type` | string | `"ORB"` | Oriented FAST and Rotated BRIEF detector. |
| `max_features` | int | `1500` | Maximum candidate keypoints per frame. |
| `pyramid_levels` | int | `8` | Number of image scale pyramid octaves. |
| `scale_factor` | float | `1.2` | Scale reduction factor between pyramid levels. |
| `keyframing.min_translation_m` | float | `0.15` | Minimum displacement to trigger keyframe creation ($15\text{ cm}$). |
| `keyframing.min_rotation_deg` | float | `5.0` | Minimum angular rotation to trigger keyframe creation ($5^\circ$). |

---

## 5. Tracking State Thresholds (`slam.tracking_state`)

| State | Condition | Meaning |
| :--- | :--- | :--- |
| **`TRACKING_OK`** | $\ge 20$ inliers | High confidence, nominal autonomous path tracking enabled. |
| **`DEGRADED`** | $10 \le \text{inliers} < 20$ | Marginal feature density; rely equally on IMU dead-reckoning. |
| **`LOST`** | $< 8$ inliers | Visual tracking lost; dead-reckon on IMU, alert path planner. |
| **`RELOCALIZING`** | Database query active | Searching keyframe database for place recognition. |
| **`RECOVERED`** | $\ge 25$ loop inliers | Re-acquired pose, trajectory drift corrected. |
