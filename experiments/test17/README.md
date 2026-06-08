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

### 2026-06-03 (test17 - THE FINAL SCALE-UP - ACTIVE)
**Model:** SegFormer (**MiT-B3**)
**Status:** **Active Final Benchmark.**
**Strategy (Scaling for Precision):**
- **Architecture:** Scaling to MiT-B3 (~44M parameters) to capture high-frequency clinical details.
- **Regularization:** Increased **Dropout to 0.3** to prevent overfitting on the small RETOUCH cohort.
- **True Relaxed Post-Processing:** 
    - **Removed all morphological operations** (`MORPH_OPEN`, `MORPH_CLOSE`).
    - Reduced **MIN_REGION to 5px** to allow B3's sharper features to persist.
    - This allows the model's raw high-resolution output to drive the final metrics without artificial distortion.
- **Consistency:** Maintaining the successful Test 16 pipeline (Fixed weights `[0.5, 5.0, 2.0, 2.0]`, Argmax, Focal-Tversky).

### 2026-06-03 (test16 - THE FINAL STABILITY RUN - COMPLETED)
**Model:** SegFormer (**MiT-B2**)
**Results:** mIoU: **0.7621** (Epoch 37 Record), Final Eval mIoU: **0.7069**.
**Analysis:** Confirmed that MiT-B2 can reach record intelligence, but final metrics were suppressed by over-aggressive post-processing (lessons applied to Test 17).


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
