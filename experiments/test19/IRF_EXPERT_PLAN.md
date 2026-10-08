# Plan: Binary IRF Expert Implementation

## 1. Objective
Improve Intraretinal Fluid (IRF) segmentation performance. Current metrics show the multi-class model misses ~50% of IRF regions (GT: 8.9 vs Pred: 4.3). We will train a specialized `mit-b0` expert model to act as a high-sensitivity "sniper" for IRF cysts.

## 2. Architecture: `mit-b0` Expert
*   **Model**: `nvidia/mit-b0` (3.7M parameters).
*   **Input**: 3-channel (2.5D context: t-1, t, t+1).
*   **Output**: Binary (Background vs. IRF).
*   **Rationale**: Smaller footprint allows for concurrent inference with the base model within 12GB VRAM.

## 3. Training Strategy
### A. Binary Dataset Mode
*   Modify `dataset.py` to support `binary_class_id`.
*   If `binary_class_id=1`, the dataset will map:
    *   Class 1 (IRF) -> 1
    *   Class 2 (SRF) -> 0
    *   Class 3 (PED) -> 0
    *   Class 0 (BG)  -> 0

### B. Loss Configuration
*   **Tversky Loss**: Set `alpha=0.05`, `beta=0.95`. This aggressively penalizes False Negatives (missed cysts).
*   **Focal Loss**: `gamma=3.0` to focus on hard, diffuse pixels.

### C. Augmentations
*   Focus on **ElasticTransform** and **RandomResizedCrop** to help the model recognize cysts of varying sizes and distortions.

## 4. Inference & Merging (Ensemble)
We will implement a `hybrid_inference.py` script:
1.  **Base Run**: Execute the `mit-b2` multi-class model for SRF and PED.
2.  **Expert Run**: Execute the `mit-b0` binary model for IRF.
3.  **Merge Logic**:
    *   **Priority 1**: Keep SRF and PED predictions from `mit-b2`.
    *   **Priority 2**: Replace `mit-b2`'s IRF predictions with `mit-b0`'s expert mask.
    *   **Priority 3 (Refinement)**: Use a logical OR for IRF if `mit-b0` confidence is > 0.7 to recover missed small regions.

## 5. Success Metrics
*   **Primary**: Increase IRF IoU from ~0.51 to >0.65.
*   **Secondary**: Increase "Regions Predicted" for IRF to match GT (closer to 8.9).
*   **Constraint**: Total inference time must remain under 1s per slice on 3060/4060 GPUs.
