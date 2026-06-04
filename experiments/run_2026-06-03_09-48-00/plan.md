# Plan: OCT Segmentation (Maksymilian - Student 2)

## Status: Stage III (16.05 - 15.06.2025)

### Stage II (Done)
- [x] Shared data pipeline (RETOUCH).
- [x] Baseline SegFormer-B0 (mIoU: 0.7427).
- [x] 2D augmentation (Flip, Bright/Contrast).
- [x] Multi-modal input (Original + Denoised + Edge map).

### Stage III: Transformer Experiments & Partial Report (16.05 - 15.06.2025)
**Goal:** SegFormer results on RETOUCH + comparison with nnUNet baseline. Report due June 15th.

#### 1. Model Experiments
- [x] **Model 1 – SegFormer-B2**: Fine-tuning pretrained encoder (MiT-B2, trained on ImageNet/ADE20K)
  - [x] OCT Data Adaptation: `in_channels=3` (2.5D context), `num_labels=4`
  - [x] Model Source: HuggingFace – `nvidia/mit-b2`
- [x] **Model 2 – SegFormer-B3 (Larger Variant)**: Investigating the impact of encoder scale on performance.
  - *Conclusion: B3 backbone suffers from severe overfitting on the 56-volume training set. B2 remains the optimal scale.*
- [x] **Training Protocol**: AdamW, `lr=6e-5`, Hybrid Warmup (15 ep) + Polynomial Decay scheduler, effective batch size 32 (8 x 4 accum), 80+ epochs.

#### 2. Attention Maps Analysis
- [ ] Develop script for attention weights extraction and visualization.
- [ ] Identify which B-scan regions the model treats as diagnostically significant (IRF/SRF focus).

#### 3. Baseline Comparison
- [ ] Comparative Table: **nnUNet 2D baseline vs SegFormer-B2 vs SegFormer-B3** (DSC per class, HD95).
  - *Status: B2 and B3 results captured. Awaiting nnUNet baseline from Student 1 (Eryk).*
- [ ] Prediction Visualizations: Qualitative analysis of where Transformers outperform or fail against CNNs.

#### 4. Partial Report (15.06.2025)
- [ ] Minimum 15 pages + tables + figures.
- [ ] Architecture description (SegFormer) and fine-tuning methodology.
- [ ] Results on RETOUCH test/validation sets.
- [ ] nnUNet baseline comparison.
- [ ] Initial Attention Maps analysis.

---

## Log / Diary

### 29.05.2025: SwinUNETR 3D Feasibility Analysis
- **Analysis**: 12GB VRAM is insufficient for full 3D SwinUNETR without extreme patch-based downsampling. 56 volumes are insufficient for 3D transformer training from scratch.
- **Decision**: Focus on **Golden 2.5D Model (test 11)** as the primary research output. 3D remains as "Future Work".

### 28.05.2025: SwinUNETR 2.5D (test 10)
- **Results**: Val mIoU: 0.7432 (Record), Test mIoU: 0.6489 (Overfitting).
- **Conclusion**: Swin's window attention is powerful but extremely sensitive to noise without ImageNet pre-training.

### 22.05.2025: Test 8 Implementation (B2 Base)
- **Objective**: Best result so far (0.7409 mIoU) using Tversky Loss and CLAHE.
