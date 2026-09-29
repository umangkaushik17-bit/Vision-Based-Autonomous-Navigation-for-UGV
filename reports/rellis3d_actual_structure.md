# RELLIS-3D ACTUAL DATASET STRUCTURE & INVENTORY REPORT

- **Dataset Root Path:** `/tmp/rellis3d/extracted`
- **Total Sequences Discovered:** 5
- **Total RGB Images Found:** 13556
- **Total Annotation Masks Found:** 6234
- **Valid Aligned Pairs:** 6234
- **Missing Masks:** 7322
- **Corrupt Files:** 0

## 1. Sequence Breakdown
| Sequence | Total Images | Total Masks | Valid Pairs | Missing Masks |
| :--- | :---: | :---: | :---: | :---: |
| `00000` | 2847 | 1200 | 1200 | 1647 |
| `00001` | 2319 | 1161 | 1161 | 1158 |
| `00002` | 4147 | 2074 | 2074 | 2073 |
| `00003` | 2184 | 900 | 900 | 1284 |
| `00004` | 2059 | 899 | 899 | 1160 |

## 2. Official Splits Specification
| Split | Pair Count | Sample Mapping Entry |
| :--- | :---: | :--- |
| **train** | 3302 | `00000/pylon_camera_node/frame000308-1581624683_550.jpg -> 00000/pylon_camera_node_label_id/frame000308-1581624683_550.png` |
| **val** | 983 | `00000/pylon_camera_node/frame001707-1581624823_450.jpg -> 00000/pylon_camera_node_label_id/frame001707-1581624823_450.png` |
| **test** | 1672 | `00000/pylon_camera_node/frame000000-1581624652_750.jpg -> 00000/pylon_camera_node_label_id/frame000000-1581624652_750.png` |

## 3. Image & Mask Geometry
- **RGB Image Formats:** `['JPEG']`
- **RGB Native Resolution (W x H):** `[(1920, 1200)]`
- **Mask Formats:** `['PNG']`
- **Mask Native Resolution (W x H):** `[(1920, 1200)]`
- **Native Aspect Ratio:** `1.6` (16:10 standard)
- **Spatial Alignment:** `IDENTICAL`

## 4. Discovered Semantic Class IDs in Annotations
| Class ID | Class Name | Observed in Mask Sample |
| :---: | :--- | :---: |
| 0 | void | YES |
| 3 | grass | YES |
| 4 | tree | YES |
| 5 | pole | YES |
| 6 | water | YES |
| 7 | sky | YES |
| 8 | vehicle | YES |
| 9 | object | YES |
| 10 | asphalt | YES |
| 12 | building | YES |
| 15 | log | YES |
| 17 | person | YES |
| 18 | fence | YES |
| 19 | bush | YES |
| 23 | concrete | YES |
| 27 | barrier | YES |
| 31 | puddle | YES |
| 33 | mud | YES |
| 34 | rubble | YES |

## 5. Architectural Compatibility & Resizing Strategy
- **Aspect Ratio Preservation:** Resizing from `1920x1200` (1.6:1) to **`640x400`** preserves exact 1.6:1 ratio.
- **Interpolation Rules:**
  - RGB: Bilinear (`INTER_LINEAR`)
  - Mask: **Nearest Neighbor (`INTER_NEAREST`) ONLY** to prevent invalid class mixing.
