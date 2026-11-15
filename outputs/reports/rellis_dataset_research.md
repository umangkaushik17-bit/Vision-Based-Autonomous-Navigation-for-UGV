# RELLIS-3D Dataset Research & Technical Specification Report
**Project:** SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Document:** `reports/rellis_dataset_research.md`  
**Authoritative Reference:** Jiang et al., *"RELLIS-3D Dataset: Data, Multi-modal Sensor Fusion, and Benchmarks"* (arXiv:2011.12954, 2021)  
**Dataset URL:** [https://github.com/unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)  
**License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  

---

## 1. Dataset Overview

The **RELLIS-3D** dataset is a multi-modal outdoor off-road dataset collected at the Texas A&M University RELLIS campus. It was captured using a customized Clearpath Robotics Warthog all-terrain unmanned ground vehicle (UGV) navigating unstructured off-road environments, including:
- Natural trails and dirt roads.
- Dense brush, tall grass, and overgrown vegetation.
- Muddy ruts, puddles, and deep water bodies.
- Concrete walkways, fences, obstacles, and steep embankments.

### Sensor Modalities:
- **RGB Camera:** Basler acA1920-50gc (pylon camera), capturing $1920 \times 1200$ resolution color images at $10\,\text{Hz}$.
- **LiDAR:** Ouster OS1-64 3D LiDAR (64 rings, 1024 points/ring @ $10\,\text{Hz}$).
- **Stereo Camera:** Stereolabs ZED stereo camera ($1280 \times 720$).
- **IMU/GPS:** LORD MicroStrain 3DM-GX5-25 AHRS / GNSS receiver.

---

## 2. Directory Structure & Image Conventions

In the official dataset distribution, the RGB camera recordings and corresponding 2D semantic annotations are structured by sequence identifier (e.g., `00000`, `00001`, `00002`, `00003`, `00004`):

```text
RELLIS-3D/
├── 00000/
│   ├── pylon_camera_node/                # Raw RGB images (*.jpg, 1920x1200)
│   │   ├── frame000000.jpg
│   │   └── ...
│   ├── pylon_camera_node_label_id/       # Single-channel class ID masks (*.png, uint8)
│   │   ├── frame000000.png
│   │   └── ...
│   └── pylon_camera_node_label_color/    # 3-channel RGB visual palette overlays (*.png)
├── 00001/
│   └── ...
├── train.txt                             # Sequence / frame lists for training split
├── val.txt                               # Sequence / frame lists for validation split
└── test.txt                              # Sequence / frame lists for test split
```

---

## 3. Class Definitions & Labeling Schemes

RELLIS-3D features an off-road semantic ontology distinguishing fine-grained outdoor classes:

### Two Class ID Schemes Documented in Literature:

1. **Original 19-Class Semantic Split (Paper Definition):**
   - 1: Sky
   - 2: Grass
   - 3: Tree
   - 4: Bush
   - 5: Concrete
   - 6: Mud
   - 7: Person
   - 8: Puddle
   - 9: Rubble
   - 10: Barrier
   - 11: Log
   - 12: Fence
   - 13: Vehicle
   - 14: Object
   - 15: Pole
   - 16: Water
   - 17: Asphalt
   - 18: Building
   - 19 / 0: Void (Ignore)

2. **RUGD-Harmonized Benchmark Split (e.g., Open3D-ML / mmsegmentation):**
   - 0: Void / Ignore
   - 1: Dirt
   - 3: Grass
   - 4: Tree
   - 5: Pole
   - 6: Water
   - 7: Sky
   - 8: Vehicle
   - 9: Object
   - 10: Asphalt
   - 12: Building
   - 15: Log
   - 17: Person
   - 18: Fence
   - 19: Bush
   - 23: Concrete
   - 27: Barrier
   - 31: Puddle
   - 33: Mud
   - 34: Rubble

---

## 4. Key Cross-Domain Differences: RUGD vs. RELLIS-3D

| Dimension | RUGD Dataset | RELLIS-3D Dataset | Engineering Implication |
| :--- | :--- | :--- | :--- |
| **Camera Sensor** | Point Grey Bumblebee2 | Basler acA1920-50gc (Pylon) | Different focal lengths, lens distortion, chromatic response |
| **Resolution** | $688 \times 550$ | $1920 \times 1200$ | Requires downsampling to canonical $480 \times 640$ |
| **Puddles & Mud** | Grouped into water/dirt | Explicitly separated: `puddle` (31), `mud` (33) | Crucial for safety: mud may cause wheel slip; puddles may conceal deep ruts |
| **Debris & Barriers** | Rock-bed / Container | Explicit `rubble` (34) and `barrier` (27) | High-consequence collision hazards |
| **Ego-Vehicle** | Foot patrol / Polaris GEM | Clearpath Warthog 4x4 UGV | Camera mount height and forward blind spot vary |

---

## 5. Licensing & Permissible Use

- **License:** Creative Commons Attribution-NonCommercial 4.0 International (`CC BY-NC 4.0`).
- **Permissible Scope:** Free for educational, research, and non-commercial hackathon prototypes.
- **Commercial Restrictions:** Commercial exploitation requires a direct commercial license from Texas A&M University.
- **Citation Requirement:** All publications and engineering reports referencing RELLIS-3D must formally cite Jiang et al. (2021).
