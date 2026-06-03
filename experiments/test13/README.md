# OCT Fluid Segmentation (Student 2: Maksymilian)

Automated segmentation of retinal biomarkers in OCT images using Transformer-based architectures.

## Clinical Context
Task: Semantic segmentation of fluid spaces in DME (Diabetic Macular Edema).
- **IRF**: Intraretinal Fluid (cysts within layers).
- **SRF**: Subretinal Fluid (detachment under retina).
- **PED**: Pigment Epithelium Detachment (RPE lift).
## Key Features
- **Architecture**: SegFormer (**B0-B3** backbones) with MiT encoders.
- **Preprocessing**: 
  - Clinical Intensity Normalization (1-99 percentile clipping).
  - Heavy Augmentation: Rotate, GaussNoise, RandomResizedCrop (Albumentations 2.0).
- **Input Strategy**: **2.5D Context** (3-channel stack of adjacent slices: $t-1, t, t+1$).
- **Data Volume Analysis**:
  - The dataset contains volumes with **49 slices** (Spectralis) and **128 slices** (Cirrus/Topcon).
  - **Hardware Constraint**: Full volumetric 3D modeling (SwinUNETR 3D) for 128-slice stacks is not feasible on **12GB VRAM** without extreme downsampling.
  - **Strategy**: 2.5D stacking is used as the optimal compromise between vertical context and memory efficiency.
- **Optimization**:
  - Hybrid Loss: $0.5 \cdot FocalLoss + 0.5 \cdot DiceLoss$.
  - **Dynamic Class Weighting**: Automatically balances fluid vs. background classes.
  - Gradient Accumulation: Simulated batch size 16 for stability.
- **Evaluation** (Enhanced per Teacher Guidelines):
  - **Boundary Precision**: Intensity difference between outer and inner boundaries (Anderson et al. 2023).
  - **Cyst/Region Analysis**: Connected-component counting and pixel area tracking.
  - **Morphological Cleaning**: Automated removal of noisy regions < 50px.
  - Stratified 80/10/10 Split (by device: Cirrus, Spectralis, Topcon).
  - Metrics: Dice, IoU, HD95, ASD, BP, Fluid Area.

## Project Structure
- `config.py`: Central source of truth (Environment support via `.env`).
- `train.py`: Training loop with 2.5D support and mixed precision.
- `eval.py`: Quantitative (Dice, IoU, BP) and qualitative (Failure Analysis) evaluation with prediction cleaning.
- `main.py`: Pipeline entry point (Train -> Eval -> Archive).
- `dataset.py`: 2.5D Patient-aware OCT loader with percentile normalization.

---

## Usage
1. Configure settings in `config.py` (e.g., set `MODEL_NAME = "nvidia/mit-b2"`).
2. Run pipeline:
```bash
python main.py
```
3. Check `experiments/run_TIMESTAMP/` for logs, `best_model.pth`, and `predictions.png`.

## Stage III Findings / Thesis Conclusions

Based on the official thesis requirements, the systematic comparison between SegFormer-B2 and SegFormer-B3 yielded the following conclusions:

1. **Impact of Encoder Scale:** 
   - Scaling from B2 (24M parameters) to B3 (44M parameters) resulted in **severe overfitting** due to the small size of the RETOUCH dataset (56 training volumes).
   - **B2 (Golden Model)** proved to be the optimal scale, effectively generalizing and capturing both large (SRF) and small (IRF) fluid regions when supported by 2.5D context and heavy regularization.

2. **Benchmark Results (B2 vs B3):**
   - **SegFormer-B2 (Test 11):** DSC = **0.8521**, HD95 = **67.30**
   - **SegFormer-B3 (Test 9):** DSC = 0.8014, HD95 = 80.37
   - *Note: Waiting for nnUNet 2D baseline results from Student 1 (Eryk) for the final comparative table.*

---

### 2026-06-02 (test13 - THE GRAND SYNTHESIS - PLANNED)
**Model:** SegFormer (**nvidia/mit-b2**)
**Objective:** Surpass the 0.75 mIoU record while maintaining the 43.43 HD95 edge precision through Loss Annealing and hard-example mining.
**Setup:** `LR=6e-5`, **Dropout (0.2)**, **Gated Boundary Loss (Annealed)**, **Focal Boost (Gamma 3.0)**, **Hybrid Scheduler**, **Patience (15)**, Multi-scale TTA.

**Key Upgrades:**
- **Synchronized Data Pipeline:** Fully aligned CLAHE and Resize transforms across training, validation, and final evaluation to eliminate domain shift during testing.
- **Boundary Annealing:** Gradually introduce boundary penalties (weight: 0.01 -> 0.1) to allow the model to prioritize detection (mIoU) in the early phase and anatomical precision (HD95) in the late phase.
- **Focal Gamma Boost (3.0):** Increased focus on the hardest clinical pixels to maximize IRF recall.
- **Optimized TTA Scales:** Adjusted to `[0.75, 1.0, 1.25, 1.5]` to capture both wide contextual detachments and microscopic cysts.

---

### 2026-06-01 (test12 - THE CLINICAL FINAL - COMPLETED)
**Model:** SegFormer (**nvidia/mit-b2**) - 27.35M parameters.
**Status:** **Best Clinical Balance.**
**Objective:** Achieve definitive clinical-grade segmentation by optimizing the HD95 metric and boosting the recall of small fluid structures (IRF).
**Setup:** `LR=6e-5` (Official Compliance), **Dropout (0.2)**, **Gated Boundary Loss**, **Hybrid Warmup-Polynomial Scheduler**, **2.5D Motion-Compensated Input**, Multi-scale TTA.

**Final Metrics:**
- **mIoU:** 0.7341
- **mDice:** 0.8382
- **mHD95:** 67.25 (Significant outlier reduction)
- **IRF (Class 1) HD95:** **43.43** (Record precision for small structures)
- **Boundary Precision:** 4.41 - 8.22 (Teacher's metric: High clinical fidelity)

**Technical Highlights:**
- **Gated Boundary-Aware Loss:** Successfully penalized anatomical outliers without sacrificing internal region coverage.
- **Anatomical Retina Masking:** Eliminated False Positives in the vitreous/sclera regions, leading to a cleaner clinical profile.
- **Hybrid Scheduling:** 15-epoch linear warmup provided maximum stability for B2 backbone under heavy spatial distortion.
- **Selective Smoothing:** Maintained the sharp, "spiky" boundaries of PED/SRF while ensuring rounded, realistic IRF cysts.

---

- **Attention Map Visualization:** Integrated a script to extract and visualize spatial attention heatmaps for clinical interpretability (Thesis Chapter 4).

### 2026-05-29 (test11 - THE GOLDEN MODEL)
**Model:** SegFormer (**nvidia/mit-b2**)
**Results:** Final Test mIoU: **0.7529** (Record), Final mDice: **0.8521**, mHD95: 67.30
**Setup:** `LR=5e-5`, **Dropout (0.2)**, **Focal-Tversky Loss**, **2.5D Motion-Compensated Input**, Multi-scale TTA.
**Observations:** 
- **New State-of-the-Art:** This run successfully broke the 0.75 mIoU barrier, confirming that the B2 backbone combined with heavy regularization and 2.5D context is the optimal configuration for the RETOUCH dataset.
- **PED Breakthrough:** Class 3 (PED) IoU reached **0.6977**, a significant jump from previous experiments, validating the use of Focal-Tversky loss to focus on difficult anatomical boundaries.
- **Generalization:** The 0.2 Dropout rate effectively closed the validation-test gap that plagued the B3 and SwinUNETR runs.
**Conclusion:** The project terminates here with a highly robust, clinically-relevant segmentation model.

### 2026-05-28 (test10 - SwinUNETR Architectural Shift)
**Model:** MONAI SwinUNETR (2D variant)
**Results:** Final Test mIoU: **0.6489**, Best Val mIoU: **0.7432**, Final mDice: 0.7714.
**Setup:** `LR=5e-5`, **Focal-Tversky Loss**, **2.5D Context**, Multi-scale TTA.
**Observations:** 
- **Validation vs. Test Gap:** The model hit a record high validation score (0.7432) but collapsed on the test set (0.6489). This indicates extreme overfitting, likely due to the lack of ImageNet pre-trained weights for the SwinUNETR backbone in this implementation.
- **Local Feature Strength:** IRF IoU (0.57) was competitive with SegFormer, showing Swin's window-based attention is effective for small cysts.
- **Convergence:** Significantly slower than SegFormer (Early stopping at epoch 60 vs 30).
**Conclusion:** Architectural complexity (Swin) does not outweigh the benefits of robust pre-training (SegFormer) for this small dataset size.

### 2026-05-23 (test9 - Planned/In Progress)
**Model:** SegFormer (**nvidia/mit-b3**)
**Objective:** Final attempt to tame the B3 backbone using the most advanced regularization and loss strategies. If B3 cannot beat B2's score (0.7409) here, we pivot to Swin.
**Pipeline Upgrades:**
- **Local Deformation:** Introduce `GridDistortion` alongside `ElasticTransform`.
- **Aggressive Dropout:** Increased to `0.2` to stop B3 from "memorizing" the 56-volume training set.
- **Loss Tuning (Focal-Tversky):** Combined Focal (background penalty) + Tversky (recall boost).
- **Learning Rate Schedule:** 15-epoch warmup for stability.

### 2026-05-26 (test9 - SegFormer B3 Regression)
**Model:** SegFormer (**nvidia/mit-b3**)
**Results:** Final mIoU: 0.6844, Final mDice: 0.8014, mHD95: 80.38, mASD: 25.79
**Setup:** `LR=5e-5`, Tversky Loss, CLAHE, ElasticTransform.
**Observations:** **Significant performance drop.** 
- **Backbone Overshoot:** B3 backbone is consistently underperforming compared to B2 (Test 8: 0.7409). 
- **Severe SRF Issues:** Class 2 (SRF) HD95 exploded to **114.96**, indicating the model is predicting many small, disconnected "island" pixels far from the true pathology.
- **PED Regression:** IoU for PED dropped back to **0.52**, suggesting B3 cannot maintain the boundary precision achieved by B2 even with CLAHE.
**Conclusion:** SegFormer B3 is officially deemed **too complex/unstable** for this specific dataset and task. It memorizes training noise and produces fragmented segmentations. We will **abandon SegFormer expansion** and pivot entirely to **SwinUNETR** for the final comparison.

### 2026-05-22 (test8 - Recall Optimization & Edge Salience)
**Model:** SegFormer (**nvidia/mit-b2**)
**Setup:** `LR=5e-5`, **Tversky Loss** ($\alpha=0.3, \beta=0.7$), **CLAHE (p=0.5)**, **ElasticTransform (p=0.3)**.
**Objective:** Ablation study on loss and preprocessing. Improve PED boundary precision and IRF recall.
**Key Changes:**
- **Loss Refinement:** Replaced Dice with Tversky Loss to prioritize catching small/thin fluid regions (minimizing False Negatives).
- **Contrast Enhancement:** Integrated CLAHE into the input pipeline to sharpen faint clinical boundaries (RPE line).
- **Robustness:** Strengthened ElasticTransform to force the model to learn anatomical structures rather than pixel noise.
- **Model Choice:** Reverted to B2 to isolate the impact of these algorithmic changes from the B3 backbone complexity.

### 2026-05-21 (test7)
**Model:** SegFormer (**nvidia/mit-b3**)
**Results:** Final mIoU: 0.7080, Final mDice: 0.8189, mHD95: 73.97, mASD: 22.05
**Setup:** `LR=5e-5`, **Dropout (0.1)**, **Focal Gamma (3.0)**, **Weight Decay (5e-2)**.
**Observations:** Modest improvement over test6. **PED (Class 3)** IoU increased to **0.5561**, validating the aggressive Focal Gamma. However, the B3 backbone still underperforms compared to **B2** (test4, mIoU 0.7322).
**Visual Failure Analysis (from `failures/` folder):**
- **Disconnected IRF:** Small intraretinal cysts are frequently missed or fragmented, indicating the model needs higher sensitivity for small structures.
- **PED-SRF Confusion:** In complex detachments, the model struggles to distinguish the RPE line (PED) from the overlying subretinal fluid (SRF) due to overlapping intensity profiles.
- **Noisy Outliers:** High HD95 is driven by "island" pixels—randomly predicted clusters far from the true anatomy.
**Conclusion:** B3 is "memorizing" noise (overfitting). Next steps require non-rigid augmentation (Elastic) and a loss function prioritizing recall (Tversky).

### 2026-05-21 (test6)
**Model:** SegFormer (**nvidia/mit-b3**)
**Results:** Final mIoU: 0.6850, Final mDice: 0.7987, mHD95: 71.66
**Setup:** **2.5D Input Stacking**, `LR=5e-5`, **Morphological Cleaning (50px)**.
**Observations:** Mixed results. **SRF (Class 2)** reached its highest IoU yet (**0.7123**), proving the value of 2.5D context. However, **PED (Class 3)** continues to struggle (0.46), and the model exhibits significant overfitting. The 2.5D logic improved vertical consistency but didn't solve the B3 backbone's tendency to memorize noise.

### 2026-05-20 (test5)
**Model:** SegFormer (**nvidia/mit-b3**)
**Results:** Final mIoU: 0.6897, Final mDice: 0.8047, mHD95: 74.25
**Setup:** `LR=5e-5`, Warmup: 10, Batch: 8, Accum: 4.
**Observations:** Paradoxical results. Training mIoU reached **0.7880** (best ever), but validation/test mIoU dropped significantly compared to test4 (B2). This indicates **heavy overfitting** or a mismatch in how the B3 model handles the stratified test set. PED IoU dropped to 0.51, suggesting B3 might be too sensitive to noise or requires stronger regularization.
**New Metrics:** First run with **Boundary Precision (BP)** and **Fluid Area** metrics. Average BP was ~0.04 across classes.

### 2026-05-20 (test4)
**Model:** SegFormer (nvidia/mit-b2)
**Results:** Final mIoU: 0.7322, Final mDice: 0.8378, mHD95: 72.45, mASD: 22.62
**Setup:** Same as test3 (`LR=5e-5`, Warmup: 10) but with **Dynamic Class Weighting** enabled.
**Observations:** Best results so far. Significant IoU improvement in SRF (0.69) and PED (0.66) compared to test3. IRF remains stable at 0.60 IoU. The model is now much more robust against the background class dominance.

### 2026-05-19 (test3)
**Model:** SegFormer (nvidia/mit-b2)
**Results:** Final mIoU: 0.6879, Final mDice: 0.8047, mHD95: 70.47, mASD: 21.77
**Setup:** `LR=5e-5`, Warmup: 10 epochs, Early Stopping Patience: 10.
**Rationale:** Previous runs failed due to instability. Lowering LR and extending warmup to 10 epochs (Cosine schedule) stabilized training. Early stopping triggered at epoch 29.
**Observations:** Significantly better stability. IRF (Class 1) remains the hardest class to segment precisely (IoU: 0.60).

### 2026-05-18 (test2 Failed Run)
**Model:** SegFormer (nvidia/mit-b2)
**Setup:** `LR=1e-4`, Warmup: 5 epochs.
**Observations:** Training collapsed at epoch 27 (mIoU dropped to 0.24). 5-epoch warmup was insufficient for the `1e-4` learning rate.

### 2026-05-17 (test1 Failed Run)
**Model:** SegFormer (nvidia/mit-b2)
**Results:** Final mIoU: 0.7125, Final mDice: 0.8220
**Observations:** Training was highly unstable. Around epoch 24, the model collapsed to predicting only background, driving mIoU down to ~0.24. It eventually recovered at epoch 68 as the `CosineAnnealingLR` decayed the learning rate.
**Conclusions:** The `mit-b2` Transformer requires a learning rate warmup phase (e.g. 5 epochs) when trained with AdamW at `LR=1e-4` to prevent gradient spikes and local minimum collapse early in training.

---

## References
- Ronneberger (2015) U-Net
- Xie (2021) SegFormer
- Tang (2022) SwinUNETR
- Isensee (2021) nnU-Net
