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

---

## Experiments

### 2026-05-21 (test7 - Regularization & Multi-Scale)
**Model:** SegFormer (**nvidia/mit-b3**)
**Setup:** `LR=5e-5`, **Dropout (0.1)**, **Focal Gamma (3.0)**, **Weight Decay (5e-2)**.
**Objective:** Solve the B3 overfitting crisis. 
**Pipeline Upgrades:**
- **Spatial Regularization:** Added hidden/attention/classifier dropout.
- **Advanced TTA:** Implemented Multi-Scale (0.8x, 1.0x, 1.2x) + Flip inference.
- **Loss Tuning:** Increased Focal Gamma to force focus on difficult PED structures.

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
