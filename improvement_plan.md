# OCT Segmentation Improvement Plan (Maksymilian - Student 2)

## Current Status (Post-test11: The Golden Model)
- **Best Model:** SegFormer (nvidia/mit-b2)
- **Results:** mIoU: **0.7529**, DSC: **0.8521**, HD95: **67.30**
- **Analysis:** We have achieved excellent pixel overlap, but the **HD95 (Hausdorff Distance)** remains our biggest bottleneck. High HD95 values (especially in SRF: 86.48) indicate that the model's boundaries are either "jagged" or there are small outlier pixels (islands) far from the true pathology.

---

## Phase 3: HD95 & Boundary Optimization (The Clinical Refinement)

### 1. Boundary-Aware Loss (Boundary Loss)
- **Action:** Integrate a **Boundary Loss** term into the hybrid criterion.
- **Why:** Current losses (Focal-Tversky) focus on the region volume. Boundary Loss specifically penalizes the distance between the predicted edge and the ground truth edge. This forces the model to prioritize the "clinical correctness" of the fluid boundaries during the backpropagation step.

### 2. Advanced Morphological Post-Processing
- **Action:** Implement a two-step refinement: `MORPH_CLOSE` followed by `MORPH_OPEN`.
- **Why:** 
    - **Closing:** Fills small internal holes in segmented cysts, making them solid.
    - **Opening:** Removes small "spiky" protrusions and isolated noise pixels that drastically inflate the HD95 metric.
    - **Result:** Smoother, more anatomically plausible shapes that align better with the smooth retina layers.

### 3. CRF (Conditional Random Fields)
- **Action:** Implement a **Dense CRF** refinement step after inference.
- **Why:** CRF looks at the original image intensities. It "snaps" the segmentation mask to the high-contrast boundaries of the OCT (e.g., the RPE line). This corrects the "bleeding" effect where the model slightly overshoots the fluid boundary into solid tissue.

### 4. Attention-Guided Refinement (Chapter 4 Thesis Goal)
- **Action:** Use the extracted **Attention Maps** to identify where the model is "uncertain".
- **Why:** If we see the attention is scattered in areas with high HD95, we can apply localized smoothing only in those uncertain regions, preserving the sharp details of the high-confidence segments.

---

## Phase 4: Final Benchmarking & Thesis Closure
1. **nnUNet Baseline:** Obtain Eryk's results to complete the comparison table.
2. **Qualitative Analysis:** Select 5 "Golden Examples" where the Transformer's attention mechanism clearly outperforms CNN-based local pooling.
3. **Inference Profiling:** Measure millisecond-per-slice speed for real-time viability discussion in Chapter 5.
