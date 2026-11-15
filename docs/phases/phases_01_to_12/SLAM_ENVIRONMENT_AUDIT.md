# Phase 8: SLAM Environment & Repository Audit

---

## 1. Host Execution Environment
- **Operating System:** Windows 10 Pro (x86_64)
- **CPU:** Intel Core Family (8 physical cores / 16 threads)
- **RAM:** 15.47 GB total
- **Python Version:** 3.10.11 (64-bit)
- **OpenCV Version:** 5.0.0 (Supports `cv2.ORB_create`, `cv2.findEssentialMat`, `cv2.recoverPose`, `cv2.triangulatePoints`, `cv2.StereoSGBM`, `cv2.solvePnPRansac`)
- **NumPy Version:** 2.2.6
- **SciPy Version:** 1.15.2
- **PyTorch Version:** 2.6.0+cpu (CUDA Available: False, pure CPU execution)
- **C++ Compiler in PATH:** None detected in active environment.
- **Architectural Consequence:** Visual odometry, visual-inertial fusion, loop closure, and evaluation are implemented as an ultra-fast, optimized, pure-Python/OpenCV engine within `ai/localization/`, drawing directly on algorithmic designs from `references/visual-slam-ros2` and ORB-SLAM3.

---

## 2. Available Reference Repositories
1. **`references/visual-slam-ros2`**:
   - High-quality, modern VIO/SLAM repository designed for ROS2 Humble.
   - Core components studied:
     - ORB feature frontend with grid-based uniform feature distribution ($30 \times 40$ cells).
     - FAST thresholding (`ini_th_fast=20`, `min_th_fast=7`), scale pyramid ($8$ levels, scale factor $1.2$).
     - Tightly-coupled IMU preintegration (accel/gyro noise models based on RealSense D435i BMI055).
     - DBoW2-style visual place recognition for loop closure and relocalization.
     - Configuration files for RealSense D435i ($b = 0.05\text{ m}$), ZED 2 ($b = 0.12\text{ m}$), and EuRoC datasets.
2. **`references/3d_slam_and_object_detection`**:
   - LiDAR/Visual SLAM integration reference.

---

## 3. Available Datasets & Sequences
1. **RUGD Outdoor Sequences (`data/processed/sequences/rugd/`)**:
   - 18 sequences available: `creek`, `park-1`, `park-2`, `park-8`, `trail`, `trail-10`, `trail-11`, `trail-12`, `trail-13`, `trail-14`, `trail-15`, `trail-3`, `trail-4`, `trail-5`, `trail-6`, `trail-7`, `trail-9`, `village`.
   - Real outdoor unstructured terrain (dirt trails, vegetation, water crossings, shadows, sunny patches).
   - Video frames available at $640 \times 480$ / $1920 \times 1080$.
2. **Synthetic Outdoor Sequences (`data/processed/sequences/synthetic/`)**:
   - 7 sequences available: `syn_seq_01_trail_sun`, `syn_seq_02_trail_shadows`, `syn_seq_03_rocks_sun`, `syn_seq_04_grass_overcast`, `syn_seq_05_rocks_shadows`, `syn_seq_06_mixed_unseen_01`, `syn_seq_07_ditch_unseen_02`.
   - Controlled lighting, surface roughness, and obstacles.
3. **Local Recorded Sequences (`data/processed/sequences/own_data/`)**:
   - 3 sequences: `own_seq_01_dirt`, `own_seq_02_gravel`, `own_seq_03_rocks`.
4. **Synthetic Reference Trajectories (`matlab/data/synthetic_slam_trajectories.mat`)**:
   - 30-second ground truth figure-8, translation drift, Gaussian noise, and loop-closure trajectories.

---

## 4. Missing Requirements & Honest Data Caveats
1. **Physical Camera Calibration Target Photos:**
   - No multi-angle checkerboard photos exist in the repository.
   - Status: **`PENDING PHYSICAL DATA`**. We utilize the canonical camera model established in Phase 7 ($640 \times 480$, $f_x=f_y=462.14\text{ px}$, $c_x=320, c_y=240$).
2. **Outdoor RTK-GPS Ground Truth Poses:**
   - The RUGD dataset contains high-resolution video and semantic segmentations, but does not provide sub-centimeter 6-DoF RTK-GPS pose streams.
   - Status: Ground truth ATE/RPE evaluation will be performed against synthetic/calibrated sequences and validated reference trajectories. Real outdoor UGV field evaluation is documented as **`PENDING PHYSICAL ROVER FIELD TESTS`**.
3. **Hardware IMU Stream:**
   - Live hardware IMU data is not present on host. We utilize simulated IMU streams synthesized with realistic noise and bias parameters modeled after the Bosch BMI055 on the Intel RealSense D435i.

---

## 5. Recommended Sensor Modes & Experiments
1. **Monocular RGB:** Evaluates feature tracking, 5-point essential matrix estimation, and explicitly analyzes scale unobservability.
2. **Stereo RGB:** Evaluates metric 3D triangulation and PnP RANSAC with known baseline ($b = 0.05\text{ m}$ / $0.12\text{ m}$).
3. **Stereo + IMU (VIO):** Evaluates visual-inertial EKF fusion, gravity alignment, and tracking stability across sharp turns and texture loss.
4. **Failure Injection:** Evaluates system transitions between `TRACKING_OK`, `DEGRADED`, `LOST`, `RELOCALIZING`, `RECOVERED`.
5. **Phase 7 MATLAB Cross-Validation:** Direct export and evaluation using `matlab/slamming/` ATE/RPE engines.
