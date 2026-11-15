# SIH26126 — Phase 2: Dataset and Perception Pipeline

**Document Version:** 1.0.0  
**Phase Status:** **COMPLETE & SATISFIED**  
**Core Governance Rule:** `TRAINING SEQUENCES ≠ VALIDATION SEQUENCES ≠ FINAL TEST SEQUENCES`

---

## 1. Objective

Phase 2 establishes the data ingestion, annotation normalization, sequence-level partitioning, and high-throughput PyTorch streaming pipeline for the SIH26126 autonomous UGV. It strictly enforces the **Corrected Hybrid Dataset Strategy**, uniting:
1. **Synthetic Data:** Controlled hazard geometry, domain-randomized lighting, and pixel-perfect ground truth.
2. **Public Real-World Data:** RUGD and RELLIS-3D benchmark adapters with standardized 3-class traversability label remapping.
3. **Our Own Real-World Data:** Ingestion of our own outdoor recordings across dirt trails, gravel, rocks, and mixed foliage.

---

## 2. Tri-Source Data Pipeline Architecture

```
 ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
 │    SYNTHETIC DATA    │  │   PUBLIC BENCHMARKS  │  │    OWN REAL DATA     │
 │ (Controlled Hazards) │  │  (RUGD & RELLIS-3D)  │  │ (Outdoor Recordings) │
 └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
            │                         │                         │
            ▼                         ▼                         ▼
      SyntheticSequence        PublicAdapter            OwnDataImporter
      (Domain Random)        (Label Remapping)        (Stride Extraction)
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   SplitManager            │
                        │   • Sequence Separation   │
                        │   • Zero-Leakage Proof    │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  dataset_manifest.json    │
                        │  (100 Samples Indexed)    │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      [TRAIN SPLIT]             [VAL SPLIT]            [FINAL TEST SPLIT]
      • 4 Sequences             • 3 Sequences          • 3 Sequences
      • 48 Samples              • 24 Samples           • 28 Samples
      • Synthetic + Own         • Synthetic + Own      • STRICTLY HELD-OUT
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  PyTorch UGVDataset       │
                        │  • Outdoor Augmentations  │
                        │  • Fast Batch Streaming   │
                        └───────────────────────────┘
```

---

## 3. Strict Data Leakage Prevention

- **Problem:** Adjacent video frames in continuous recording share $>98\%$ visual similarity. Frame-level random splitting causes severe data leakage, resulting in artificial high accuracy that fails catastrophically in the field.
- **Solution:** Partitioning is performed exclusively at the **sequence / video level**:
  $$\text{Sequences}(\text{Train}) \cap \text{Sequences}(\text{Val}) = \emptyset$$
  $$\text{Sequences}(\text{Train}) \cap \text{Sequences}(\text{Test}) = \emptyset$$
  $$\text{Sequences}(\text{Val}) \cap \text{Sequences}(\text{Test}) = \emptyset$$
- **Programmatic Enforcement:** `SplitManager.validate_zero_leakage()` performs set intersection assertions across all sequences. If a sequence is detected in multiple splits, an immediate `ValueError` is raised.

---

## 4. Label Remapping Standardization

Disparate outdoor benchmark classes are normalized into the project's standard 3-tier traversability taxonomy:

| Target Class ID | Traversability Level | Ground Surface Types | Loss Weight Strategy |
| :---: | :--- | :--- | :--- |
| **0** | **SAFE** | Worn dirt trail, smooth soil, flat gravel, dry mowed grass, asphalt | Nominal loss weight |
| **1** | **UNKNOWN** | Tall unmapped brush, deep canopy shadows, transition margins | Moderated loss weight |
| **2** | **UNSAFE** | Boulders, tree trunks, logs, ditches, puddles, steep drops | **Elevated loss weight** (Minimize False-Safe Errors) |
| **255** | **VOID** | Sky horizon, vehicle chassis, unannotated borders | **Zero loss weight** (Ignored in CrossEntropy) |

---

## 5. Statistical Health & Gap Analysis Audit

Generated via `python scripts/audit_dataset.py --manifest data/manifests/dataset_manifest.json`:

```
=================== UGV DATASET AUDIT REPORT ===================
Total Annotated Samples : 100
Split Breakdown         : Train=48 | Val=24 | Test=28
Unique Sequences        : 10 (Train=4, Val=3, Test=3)
Source Breakdown        : {'synthetic': 72, 'own_real': 28}
Environments Represented: ['trail', 'rocks', 'grass', 'mixed', 'dirt_trail', 'gravel']
Lighting Distributions  : {'sunlight': 36, 'shadows': 38, 'overcast': 16, 'glare': 10}
----------------- Traversability Class Coverage -----------------
SAFE    (Class 0) : 16.78%
UNKNOWN (Class 1) : 47.46%
UNSAFE  (Class 2) :  0.76%
VOID    (Sky/Ego) : 35.00%
------------------ Data Leakage Verification -------------------
Sequence-Disjointness Check: PASS (Zero Leakage)
------------------- Identified Data Gaps -----------------------
 [!] Under-representation of UNSAFE hazards (< 5% of pixels). Collect/synthesize more boulders/ditches.
================================================================
```

### Gap Identification & Action Plan
- **Gap:** UNSAFE hazard pixels represent only $0.76\%$ in initial baseline trails.
- **Remediation for Phase 3:** Use class-weighted focal loss ($\alpha_{\text{unsafe}} = 3.0$) and inject dense boulder/rock sequences during training to prevent hazard neglect.

---

## 6. CLI Tools & Usage

### 6.1 Ingest Own Real-World Video Sequence
```powershell
# Import outdoor recording with stride 4 to remove near-duplicate frame redundancy
python scripts/import_own_data.py --input data/test/sample.mp4 --sequence-id own_seq_01_dirt --split train --env dirt_trail --stride 4
```

### 6.2 Assemble Unified Manifest
```powershell
# Scan all synthetic, public, and own sequences and enforce zero leakage
python scripts/build_dataset_manifest.py --output data/manifests/dataset_manifest.json
```

### 6.3 Run Dataset Health Audit
```powershell
# Audit class distributions and confirm zero leakage
python scripts/audit_dataset.py --manifest data/manifests/dataset_manifest.json
```

---

## 7. Automated Testing

Run the full automated test suite:
```powershell
pytest -v tests/
```
**Results: 41 passed in 15.55s (100% pass rate)**
- `tests/test_dataset_base.py` (4 tests): Traversability classes, sample serialization, RUGD/RELLIS remapping.
- `tests/test_split_manager.py` (2 tests): Sequence assignment, leakage detection, set disjointness.
- `tests/test_ugv_dataset.py` (2 tests): Outdoor augmentations, PyTorch batch streaming.
- `tests/test_dataset_audit.py` (1 test): Statistical health auditor.
- Phase 0 & Phase 1 regressions (32 tests): All passing.

---

## 8. Readiness for Phase 3

The dataset ingestion, sequence-level governance, outdoor augmentations, and PyTorch DataLoaders are verified and ready to power **Phase 3: Traversability Model Training & Evaluation**.
