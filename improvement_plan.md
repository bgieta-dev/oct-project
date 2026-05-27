# Improvement Plan - Post Test 9

## Current State Analysis (Test 9 Failure)
- **Model**: SegFormer (nvidia/mit-b3)
- **Status**: **Total Failure to Generalize.** mIoU dropped to 0.6844.
- **Key Issues Identified**:
    1. **Structural Fragmentation**: SRF HD95 reached **114.96**. The model is hallucinating small fluid clusters (islands) all over the image.
    2. **PED Collapse**: PED IoU fell back to 0.52. B3 fails to leverage CLAHE and Tversky loss effectively.
    3. **Complexity Mismatch**: The 44M parameter B3 backbone is too heavy for the 56-volume training set, leading to "memorization" of noise rather than features.

---

## Pivotal Shift: SwinUNETR Migration

### 1. High-Priority Action: Abandon SegFormer-B3
- **Action**: Halt all SegFormer-B3 experiments.
- **Rationale**: Continuous testing proves that increasing backbone size in SegFormer does not yield clinical improvements for this dataset.

### 2. Implementation: SwinUNETR (MONAI)
- **Action**: Implement **SwinUNETR** with Shifted-Window Attention.
- **Rationale**: Swin Transformers use a hierarchical approach that captures local texture (IRF) and global context (PED) simultaneously without the fragmentation seen in B3.
- **Goal**: Regain mIoU > 0.74 and stabilize clinical boundaries.

### 3. Debugging Tool: Attention Map Visualization
- **Action**: Script to export **attention maps** for the failed B3 run and the upcoming SwinUNETR.
- **Rationale**: We need to see *where* B3 was looking when it hallucinated the SRF islands. This is critical for the "Analysis of Failure" section of the diploma thesis.

### 4. Loss Function Stability
- **Action**: Re-tune **Tversky Loss** parameters ($\alpha=0.5, \beta=0.5$) for initial SwinUNETR runs.
- **Rationale**: The aggressive recall prioritization ($\beta=0.7$) might have amplified the noise in B3. We will return to a balanced Dice-like Tversky to stabilize the new architecture.

---

## Immediate Next Step (Test 10 Plan)
1. **Model**: MONAI SwinUNETR (2D or 3D).
2. **Preprocessing**: Maintain CLAHE and 2.5D context.
3. **Task**: Initial "Swin-Baseline" run to compare against SegFormer-B2 (0.74).
