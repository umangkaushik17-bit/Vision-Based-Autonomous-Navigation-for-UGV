# RELLIS-3D Authoritative Dataset Research Report

**Repository:** `unmannedlab/RELLIS-3D` (Texas A&M University)  
**Publication:** *RELLIS-3D: A Multi-modal Dataset for Off-road Robotics* (Jiang et al., IEEE/RSJ IROS 2021)  
**Task Target:** RGB Dense Traversability & Semantic Terrain Segmentation for Outdoor UGV Navigation  

---

## Section A: Facts Verified from Downloaded Files

1. **Split Files (`Rellis_3D_image_split.zip`):**
   - The archive contains three text split files: `train.lst`, `val.lst`, and `test.lst`.
   - Each line consists of a relative image path and its corresponding label mask path separated by whitespace:
     ```
     00000/pylon_camera_node/frame000000-0000.jpg 00000/pylon_camera_node_label_id/frame000000-0000.png
     ```
   - Exact sample counts:
     - `train.lst`: **3,302** image-mask pairs
     - `val.lst`: **983** image-mask pairs
     - `test.lst`: **1,672** image-mask pairs
     - Total: **5,957** paired frames across 5 sequences (`00000`, `00001`, `00002`, `00003`, `00004`).

2. **Archive Packaging & File Layout:**
   - Full RGB images archive: `Rellis_3D_pylon_camera_node.zip` (10.81 GB).
   - Label masks archive: `Rellis_3D_pylon_camera_node_label_id.zip` (94.27 MB).
   - Split archive: `Rellis_3D_image_split.zip` (44.6 KB).
   - Naming convention: Sequences are numbered `00000` to `00004`. Image files are `.jpg`, annotation masks are single-channel `.png` representing class IDs directly (not RGB color palettes).

3. **Label ID Mask Encoding:**
   - Annotation masks in `pylon_camera_node_label_id/` store raw integer class IDs directly in 8-bit or 16-bit single-channel PNG values.
   - Background/unannotated pixels are marked with ID `0` (void).

---

## Section B: Facts Verified from Official Online Documentation

1. **Camera Sensor & Image Specifications:**
   - **Camera:** Basler acA1920-50gc industrial gigabit Ethernet camera.
   - **Lens:** Edmund Optics 16mm / F1.8 high-resolution lens.
   - **Native Resolution:** 1920 x 1200 pixels.
   - **Aspect Ratio:** 16:10 (1.6:1).
   - **ROS Topic:** `/pylon_camera_node/image_raw`.

2. **Official Class Taxonomy (`rellis.yaml`):**
   The dataset specifies 20 semantic classes with non-contiguous class IDs:
   - `0: void` (unlabeled / background)
   - `1: dirt` (unpaved trail / flat soil)
   - `3: grass` (short to medium green vegetation)
   - `4: tree` (vertical trunk / canopy)
   - `5: pole` (vertical post / signpost)
   - `6: water` (deep open water body)
   - `7: sky` (overhead atmosphere)
   - `8: vehicle` (car / truck / ATV)
   - `9: object` (general rigid obstacle)
   - `10: asphalt` (paved road)
   - `12: building` (architectural structure)
   - `15: log` (fallen tree trunk)
   - `17: person` (pedestrian)
   - `18: fence` (boundary barrier)
   - `19: bush` (dense foliage / shrub)
   - `23: concrete` (walkway / slab)
   - `27: barrier` (roadside crash barrier)
   - `31: puddle` (shallow standing water)
   - `33: mud` (saturated slippery soil)
   - `34: rubble` (loose stones / rock debris)

3. **Licensing & Usage:**
   - RELLIS-3D is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
   - Intended for academic, research, and benchmarking purposes in autonomous robotics.

---

## Section C: Engineering Assumptions and Inferences

1. **Resolution Selection:**
   - *Original:* 1920x1200 (1.6:1 aspect ratio).
   - *Resizing Choice:* Downsampling to **640x400** maintains the exact 1.6:1 aspect ratio without distorting perspective geometry or vertical obstacle features.
   - *Model Compatibility:* `LightweightTraversabilityNet` has 4 downsampling stages (total downsample factor of 16). 640 and 400 are both evenly divisible by 16 (`640 / 16 = 40`, `400 / 16 = 25`), preserving spatial alignment across all skip connections.

2. **Interpolation Semantics:**
   - RGB imagery: Bilinear interpolation (`cv2.INTER_LINEAR` or PIL `BILINEAR`).
   - Semantic annotation masks: **Strict Nearest-Neighbor Interpolation (`cv2.INTER_NEAREST`)**. Continuous interpolation (e.g. bilinear/bicubic) creates non-existent intermediate class IDs at object boundaries and is strictly forbidden.

3. **Traversability Harmonization Scheme (3-Class Space):**
   - In alignment with the existing UGV system contract, the 20 RELLIS-3D classes are harmonized into:
     - `0: SAFE / TRAVERSABLE` (dirt, grass, asphalt, concrete)
     - `1: UNKNOWN / RISKY` (bush, puddle, mud)
     - `2: UNSAFE / NON_TRAVERSABLE` (tree, pole, water, vehicle, object, building, log, person, fence, barrier, rubble)
     - `255: VOID` (void, sky - excluded from loss computation and costmap updates).
