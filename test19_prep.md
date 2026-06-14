# Test 19 Preparation: IRF Expert Ensemble

## Overview
Test 19 transitions from a single multi-class model to a **Hybrid Expert Ensemble**. This strategy addresses the performance bottleneck in Intraretinal Fluid (IRF) detection.

## 🛠 Architectural Changes
### 1. The Models
*   **Base Model (mit-b2)**: Handles structural segmentation (SRF, PED).
*   **Expert Model (mit-b0)**: Specialized "Sniper" model for IRF. 
    *   Binary classification (IRF vs. Background).
    *   Lightweight architecture allows parallel inference without VRAM overflow.

### 2. High-Recall Optimization
The Expert model is trained with **Extreme Tversky Loss**:
*   **Alpha (FP penalization)**: 0.05
*   **Beta (FN penalization)**: 0.95
*   **Goal**: Force the model to detect tiny, diffuse cysts even at the cost of slight over-segmentation (which is safer for clinical high-recall requirements).

## 📂 New Pipeline Files
| File | Purpose |
| :--- | :--- |
| `config_irf_expert.py` | Specialized hyperparameters for IRF binary training. |
| `train_expert.py` | Training script using the modular `train_model` logic. |
| `hybrid_inference.py` | Real-time ensemble engine. Merges Base and Expert masks with priority logic. |
| `eval_hybrid.py` | Validation script for computing IoU, Dice, and HD95 on the ensemble. |

## 🧬 Dataset Enhancements
*   `OCTDataset` now supports a `target_class` parameter.
*   When `target_class=1`, all non-IRF pixels are automatically remapped to Background, creating a perfect binary training signal for the expert.

## 📈 Expected Outcomes for Test 19
1.  **IRF Detection**: Increase in predicted region count (approaching GT 8.9).
2.  **IRF IoU**: Targeted improvement from 0.51 to >0.60.
3.  **VRAM Stability**: Concurrent execution within 12GB limits.
4.  **Ensemble Robustness**: Expert IRF mask will override Base IRF/BG but respect Base SRF/PED boundaries to maintain anatomical integrity.

## 🚀 Execution Steps
1. Run `python3 train_expert.py`.
2. Run `python3 eval_hybrid.py`.
3. Analyze `hybrid_eval_results/` visualizations.
