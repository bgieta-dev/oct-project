# Plan: OCT Seg (Maks)

## Status: Stage III (16.05 - 15.06.2025)

### Stage II (Done)
- [x] Shared data pipeline (RETOUCH).
- [x] Baseline SegFormer-B0 (mIoU: 0.7427).
- [x] 2D aug (Flip, Bright/Contrast).
- [x] Multi-modal input (Orig + Denoised + Edge).

### Stage III: Transformers & Clinical Validation (16.05 - 15.06)
Goal: Optimize transformers. Systemic clinical comparison.

#### 1. Model Upgrade & Stability
- [x] Centralize config in `config.py`.
- [x] Upgrade to **SegFormer-B2** (`nvidia/mit-b2`).
- [x] **VRAM Optimization (RTX 4060/3060):**
    - Dynamic `batch_size` based on model size.
    - **Gradient Accumulation** (Effective batch size = 32) for stability.
- [x] Move to **SegFormer-B3** for final comparison (Test 7).
- [ ] Implement **Attention Maps** visualization.

#### 2. Advanced Preprocessing & Data
- [x] **Intensity Normalization**: 1-99 percentile clipping + [0, 1] scaling.
- [x] **Augmentation Expansion**: `Rotate(10)`, `GaussNoise`, `RandomResizedCrop`.
- [x] **Stratified Split**: 80/10/10 split stratified by OCT device (Cirrus/Spectralis/Topcon).
- [x] **Clinical Unification**: Standardized Class 3 as **PED** (Pigment Epithelium Detachment).
- [ ] Optional: Cross-dataset test on **AROI**.

#### 3. Metrics & Benchmarking
- [x] **Surface Metrics**: Added **HD95** (Hausdorff Distance) and **ASD** (Average Surface Distance).
- [x] **Fixed Evaluation**: `eval.py` now picks best class-specific examples for static reporting.
- [x] **Test Set**: Evaluation now runs on dedicated 10% test set (previously 20% val).
- [ ] Run **nnUNet 2D baseline** (joint task with Eryk).

#### 4. Thesis & Documentation (Report due 15.06)
- [ ] Write partial report (min. 15 pages).
- [ ] Theoretical background: CNN vs Transformers in OCT.
- [ ] Qualitative analysis: Error cases for IRF vs SRF.

---

## Technical TODOs

### Active Tasks
- [ ] **SwinUNETR**: Implement model in MONAI framework.
- [ ] **Attention maps**: Script to extract transformer weights for IRF/SRF focus.
- [ ] **B3 Training**: Run long training (50+ epochs) on B3.

### Done
- [x] Add log
- [x] Test B2 model
- [x] Add HD95/ASD to main.py logging.

---

## Log / Diary

### 22.05.2025: Test 8 Implementation (B2 Base)
- **Objective**: Improve PED boundary precision and IRF recall using Tversky Loss and CLAHE.
- **Changes**:
    - Switched model to **nvidia/mit-b2** to isolate impact.
    - Switched Dice Loss to **Tversky Loss** (alpha=0.3, beta=0.7) to penalize False Negatives.
    - Added **CLAHE** to the preprocessing pipeline for edge salience.
    - Strengthened **ElasticTransform** to fight overfitting.
    - Increased morphological cleaning to **100px**.
- **Status**: Ready to run.
