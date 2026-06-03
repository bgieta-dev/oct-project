# Plan: OCT Seg 

## Status: Stage III (16.05 - 15.06.2025)

### Stage II (Done)
- [x] Shared data pipeline (RETOUCH).
- [x] Baseline SegFormer-B0 (mIoU: 0.7427).
- [x] 2D aug (Flip, Bright/Contrast).
- [x] Multi-modal input (Orig + Denoised + Edge).

### Stage III: Transformers & Clinical Validation (16.05 - 15.06)
Goal: Optimize transformers. Systemic clinical comparison (CNN vs Transformer).

#### 1. Model Upgrade & Stability
- [x] Centralize config in `config.py`.
- [x] Upgrade to **SegFormer-B2** (`nvidia/mit-b2`) - **Current High Score (0.7409)**.
- [x] Test **SegFormer-B3** - Done (Struggles with overfitting on small dataset).
- [x] Implement **SwinUNETR** (MONAI) - **Validated in 2.5D**.
- [ ] **SwinUNETR Optimization**: Address the test set collapse (Recall vs Generalization).
- [ ] **Attention Maps Visualization**: Script to extract weights for IRF/SRF focus (Thesis Chapter 4 requirement).

#### 2. Advanced Preprocessing & Data
- [x] **Intensity Normalization**: 1-99 percentile clipping + [0, 1] scaling.
- [x] **Augmentation Expansion**: `Rotate(10)`, `GaussNoise`, `RandomResizedCrop`.
- [x] **Stratified Split**: 80/10/10 split stratified by OCT device (Cirrus/Spectralis/Topcon).
- [x] **Clinical Unification**: Standardized Class 3 as **PED** (Pigment Epithelium Detachment).
- [x] **2.5D Context**: Implemented motion-compensated stacking ($t-1, t, t+1$).

#### 3. Metrics & Benchmarking
- [x] **Surface Metrics**: Added **HD95** (Hausdorff Distance) and **ASD** (Average Surface Distance).
- [x] **Boundary Precision**: Anderson et al. (2023) methodology implemented.
- [x] **Fixed Evaluation**: `eval.py` now picks best class-specific examples for static reporting.
- [ ] Run **nnUNet 2D baseline** (Joint task with Eryk).

#### 4. Thesis & Documentation (Partial Report due 15.06)
- [ ] Write partial report (min. 15 pages).
- [ ] **Chapter 3**: Transformer architectures for OCT (Theory).
- [ ] **Chapter 4**: Attention Analysis (Visualizing focus on IRF/SRF).
- [ ] **Chapter 5**: Clinical comparison (CNN vs Transformer).

---

## Log / Diary

### 29.05.2025: SwinUNETR 3D Feasibility Analysis
- **Analysis**: 12GB VRAM is insufficient for full 3D SwinUNETR without extreme patch-based downsampling (which loses context). 56 volumes are insufficient for 3D transformer training from scratch.
- **Decision**: Focus on **Golden 2.5D Model (test 11)** as the primary research output. 3D remains as "Future Work".

### 28.05.2025: SwinUNETR 2.5D (test 10)
- **Results**: Val mIoU: 0.7432 (Record), Test mIoU: 0.6489 (Overfitting).
- **Conclusion**: Swin's window attention is powerful but extremely sensitive to noise without ImageNet pre-training.

### 22.05.2025: Test 8 Implementation (B2 Base)
- **Objective**: Best result so far (0.7409 mIoU) using Tversky Loss and CLAHE.
