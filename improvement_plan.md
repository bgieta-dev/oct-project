# OCT Segmentation Improvement Plan (Maksymilian - Student 2)

## Current Status (Post-test15 Regression Analysis)
- **Best Record (Test 11):** mIoU: **0.7529**, mDice: **0.8521**
- **Lessons from Failure (Test 13, 15):** 
    - **Boundary Loss (SDF):** Destabilized training. The high-frequency noise in OCT scans conflicts with SDF gradients on small datasets.
    - **Anatomical Loss Masking:** Corrupted global spatial awareness during training.

---

## Final Phase: Stability & Interpretability (Test 16)

### 1. Robustness & Stability (Test 16 Implementation)
- [x] **Loss:** Focal-Tversky (Fixed). Replaces standard Dice to handle class imbalance (Fluid vs BG) and prioritize Recall for IRF.
- [x] **Post-Processing:** Soft-CRF (Bilateral Filtering) + Retina Masking. 
- [x] **Critical Adjustment:** Disabled MORPH_OPEN and reduced MIN_REGION to 10px. This preserves small intraretinal cysts (IRF) which are key biomarkers.

### 2. Clinical Interpretability (Thesis Chapter 4)
- [x] **Attention Visualization:** Automated extraction of multi-head attention weights.
- [x] **Goal:** Prove that Transformers "look" at anatomically correct areas (e.g., the RPE layer when detecting PED).

### 3. Future Generalization (Post-Thesis)
- **Cross-Device Calibration:** Fine-tuning on AROI or Spectralis-only sub-sets to investigate hardware bias.
- **MedSAM-2 Integration:** Speculative research on zero-shot segmentation for rare OCT pathologies.

---

## Final Benchmarking Goals
1. **Stabilize HD95:** Aim for < 68.0 consistently.
2. **Maximize IRF Recall:** Ensure Dice > 0.73 for class 1.
3. **Hardware Efficiency:** Confirm inference time < 500ms per B-scan on CPU.
