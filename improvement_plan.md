# Improvement Plan - Post Test 7

## Current State Analysis (Test 7)
- **Model**: SegFormer (nvidia/mit-b3)
- **Status**: Improved over Test 6 but still underperforming compared to smaller B2 model (Test 4).
- **Key Issues**:
    1. **Overfitting**: Large gap between training and validation metrics in `metrics.png`.
    2. **PED Underperformance**: IoU 0.56. Boundaries of RPE lift are blurry or missing thin segments.
    3. **High HD95 (73.97)**: Visualization shows noisy "island" pixels far from true anatomy.

## Visual Failure Analysis (failures/ folder)
Detailed review of indices 52, 128, 131, 150, 162 reveals:
- **Small IRF Misses**: Tiny intraretinal cysts are often ignored or treated as background.
- **PED-SRF Confusion**: Model struggles to distinguish the RPE line (PED) from the fluid immediately above it (SRF) in complex detachments.
- **Disconnected Predictions**: Fluid regions are fragmented, lacking the anatomical smoothness seen in GT.

---

## Proposed Actions

### 1. Hybrid Loss Refinement (Tversky)
- **Action**: Switch from Dice + Focal to **Focal + Tversky Loss** ($\alpha=0.3, \beta=0.7$).
- **Rationale**: Failures show high False Negatives (missed small cysts/thin PED). Tversky allows prioritizing recall.
- **Target**: Ensure model captures thin/small fluid structures.

### 2. Aggressive Regularization
- **Action**: Add **ElasticTransform** and increase **Weight Decay** (1e-1).
- **Rationale**: `metrics.png` proves B3 is memorizing noise. Non-rigid deformations will break this pattern.
- **Target**: Close the Train-Val gap.

### 3. Edge Salience (CLAHE + Edge Weighting)
- **Action**: Apply **CLAHE** and potentially double the weight of the "Edge Map" channel in the input stack.
- **Rationale**: PED-SRF confusion happens because the RPE boundary is faint. CLAHE will sharpen this clinical marker.
- **Target**: Improve PED IoU and boundary definition.

### 4. Hierarchical Attention (SwinUNETR)
- **Action**: Implement **SwinUNETR**.
- **Rationale**: Window-based attention might handle the "layered" nature of retina better than SegFormer's global maps.
- **Target**: Benchmark hierarchical vs uniform attention.

### 5. HD95 Mitigation (Morphological Opening + CRF)
- **Action**: Increase morphological cleaning to **100px** and test a **CRF** layer.
- **Rationale**: "Island" pixels in `failures/` are the main cause of high HD95. CRF will pull these outliers back to the main mass.
- **Target**: Lower HD95 to <50.

---

## Immediate Next Step (Test 8 Plan)
1. Implement **Focal + Tversky Loss**.
2. Add **ElasticTransform** to `train.py`.
3. Test **CLAHE** on the input stack.
4. Use **nvidia/mit-b2** to isolate impact of loss/aug changes.
