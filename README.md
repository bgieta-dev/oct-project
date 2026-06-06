# OCT Fluid Segmentation (Student 2: Maksymilian)

Automated segmentation of retinal biomarkers in OCT images using Transformer-based architectures.

## Clinical Context
Task: Semantic segmentation of fluid spaces in DME (Diabetic Macular Edema).
- **IRF**: Intraretinal Fluid (cysts within layers).
- **SRF**: Subretinal Fluid (detachment under retina).
- **PED**: Pigment Epithelium Detachment (RPE lift).

## Key Features
- **Architecture**: SegFormer (**MiT-B2**) with ImageNet-21k pre-training.
- **Input Strategy**: **2.5D Volumetric Context** (3-slice stack: $t-1, t, t+1$).
- **Normalization**: Clinical 1-99 percentile clipping (robust to hardware noise).
- **Optimization Strategy**:
  - **Hybrid Loss**: $0.5 \cdot FocalLoss + 0.5 \cdot TverskyLoss$.
  - **Clinical Post-Processing**: Anatomical retina masking + selective morphological smoothing.
  - **Bilateral Refinement**: Soft-CRF logic to snap boundaries to OCT intensity interfaces.

---

## Experimental History

### 2026-06-03 (test16 - THE FINAL STABILITY RUN - ACTIVE)
**Model:** SegFormer (**MiT-B2**)
**Status:** **Current Production Candidate.**
**Strategy (The "Stability Fusion"):**
- **Loss Strategy:** Focal-Tversky ($0.5/0.5$) with fixed weights `[0.5, 5.0, 2.0, 2.0]`. 
- **Critical Change:** Disabled **Boundary Loss**. Past regressions (Test 13) proved that SDF-based boundary penalties destabilize training on the small RETOUCH dataset (56 patients).
- **Post-Processing:** Argmax-based assignment + selective morphological cleaning (MIN_REGION=10).
- **Interpretability:** Full integration of **Attention Map** extraction for Thesis Chapter 4.

### 2026-06-03 (test15 - Anatomical Masking Regression)
**Results:** mIoU: **0.7013** (Performance Drop).
**Analysis:** Attempting to mask the loss during training blinded the model to anatomy-pathology boundaries, leading to disconnected predictions.

### 2026-06-03 (test14 - Clinical Refinement)
**Results:** mIoU: **0.7405**, mDice: **0.8431**, mHD95: **67.97**.
**Key Success:** Integrated **Soft-CRF** (Bilateral smoothing) and **Attention Maps**. Proved that B2 is the "sweet spot" for 12GB VRAM.

### 2026-05-29 (test11 - The Golden Model)
**Results:** mIoU: **0.7529** (Record), mDice: **0.8521**, mHD95: **67.30**.
**Conclusion:** Confirmed that 2.5D context + heavy regularization (Dropout 0.2) is the optimal setup for Transformer-based OCT segmentation.

---

## Project Structure
- `config.py`: Central hyperparameter management (English documented).
- `train.py`: Training pipeline with 2.5D support and hybrid losses.
- `eval.py`: Quantitative (HD95, Dice) and clinical boundary analysis.
- `dataset.py`: 2.5D Patient-aware loader with percentile normalization.
- `main.py`: Full research pipeline (Train -> Eval -> Attention -> Archive).
