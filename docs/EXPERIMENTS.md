# Experiments

Frozen history of the SegFormer runs lives in `experiments/` (`test2` … `test18`, plus `legacy_root/` with
old root-level leftovers). Those folders are evidence: their code snapshots use the old flat imports and are
**never edited or refactored**. They are excluded from ruff/pytest (`pyproject.toml`).

## Index

Final numbers come from the `experiment.log` of each run (the first log line keeps the original
`run_<timestamp>` name and shows the runs were executed on the Windows lab machine).

| dir | original run | backbone | final mIoU / mDice |
|---|---|---|---|
| test2 | run_2026-05-17_09-35-47 | mit-b2 | 0.7125 / 0.8220 |
| test3 | run_2026-05-18_11-08-54 | mit-b2 | 0.7149 / 0.8247 |
| test4 | run_2026-05-19_09-35-43 | mit-b2 | 0.6879 / 0.8047 |
| test5 | run_2026-05-19_22-53-23 | mit-b2 | 0.7322 / 0.8378 |
| test6 | run_2026-05-20_09-10-48 | mit-b3 | 0.6897 / 0.8047 |
| test7 | run_2026-05-20_19-17-35 | mit-b3 | 0.6850 / 0.7987 |
| test8 | run_2026-05-21_09-00-18 | mit-b3 | 0.7080 / 0.8189 |
| test9 | run_2026-05-22_09-53-35 | mit-b2 | 0.7409 / 0.8434 |
| test10 | run_2026-05-26_08-11-36 | mit-b3 | 0.6844 / 0.8014 |
| test11-gold | run_2026-05-29_13-59-49 | mit-b2 | 0.7529 / 0.8521 |
| test12 | run_2026-06-01_19-28-20 | — (only log + metrics.png) | — |
| test13 | run_2026-06-02_02-38-14 | mit-b2 | 0.7186 / 0.8273 |
| test14 | run_2026-06-02_22-51-11 | mit-b2 | 0.7405 / 0.8431 |
| test15 | run_2026-06-03_09-48-00 | mit-b2 | 0.7013 / 0.8147 |
| test16 | run_2026-06-06_07-58-47 | mit-b2 | 0.7037 / 0.8169 (first "Final" line) |
| test17 | run_2026-06-06_21-26-47 | mit-b3 | — (no "Final mIoU" line) |
| test18 | run_2026-06-08_10-31-03 | mit-b2 | 0.6990 / 0.8118 |

No SegFormer checkpoint (`*.pth`) exists in this repository (they are gitignored). test11-gold and test18 can
only be re-evaluated on the lab machine that holds them.

## New runs and promotion to history

New runs are written to `outputs/runs/<timestamp>_<name>/` (gitignored) by `python -m octseg.pipeline` /
`python -m octseg.train_expert`. Each run directory contains `config.yaml` (resolved config), `git.txt`
(commit + `git status --porcelain`), `src/` (code snapshot), `requirements.txt`, `splits/`, `experiment.log`,
`metrics.yaml`, `predictions.png`, `metrics.png`, `attention_maps/` and `best_model.pth`.

To promote a run to history (no script, by hand):

```bash
rsync -a --exclude '*.pth' outputs/runs/<run>/ experiments/test19-<slug>/
```

then add a row to the index table above. Keep the checkpoint outside git (lab disk) and note its path here.

## Experimental history (moved verbatim from the old README)

## Experimental History

### 2026-06-08 (test18 - Transformer Overdrive / Clinical High-Recall)
**Model:** SegFormer (**MiT-B2**)
**Status:** **Completed (Golden Benchmark).**
**Results:** mIoU: **0.6990** | mDice: **0.8118** | mHD95: **74.84** | mASD: **26.16**
**Class Performance:**
- **IRF (Class 1):** Dice: 0.6800 | HD95: 49.12
- **SRF (Class 2):** Dice: 0.8187 | HD95: 94.03
- **PED (Class 3):** Dice: 0.7634 | HD95: 81.38
**Strategy (Matching CNN baselines & Clinical Reality):**
- **Architecture Downgrade:** Returned to MiT-B2 (from B3) due to severe noise overfitting on the small RETOUCH cohort.
- **Clinical High-Recall (Loss & Thresholds):** 
    - In medicine, False Negatives (missing a cyst) are more dangerous than False Positives.
    - **Tversky Tuning:** Shifted balance to strongly penalize False Negatives (`Alpha=0.1`, `Beta=0.9`).
    - **Thresholding:** Replaced strict `argmax` with probability thresholds (IRF triggers at just `0.30` confidence).
- **Morphological Tuning (Addressing Bilinear Blurring):**
    - **IRF Separation:** Added Morphological Opening specifically for IRF to break thin probability bridges, preventing distinct cysts from merging.
    - **PED Sharpening:** SegFormer's 4x upsampling smooths boundaries. Added a 2D Sharpening Filter (Unsharp Mask) to the PED probability map to restore the clinically correct "spiky" RPE lifts.
- **Advanced TTA:** Reintroduced full Test-Time Augmentation (Scales: 0.8, 1.0, 1.2 + Flips) to smooth boundary artifacts and boost DSC.
- **Visual Clarity:** Upgraded attention visualization to Stage 2 (64x64 grid) with Gamma correction for dramatically sharper heatmaps.

### 📊 Comparative Analysis: Test 11-Gold vs. Test 18
| Metric | **Test 11-Gold** (Balanced) | **Test 18** (High-Recall) | Difference |
| :--- | :---: | :---: | :---: |
| **mIoU** | **0.7529** | 0.6990 | -0.0539 |
| **mDice** | **0.8521** | 0.8118 | -0.0403 |
| **mHD95** (lower is better) | **67.31** | 74.84 | +7.53 |
| **mASD** (lower is better) | **22.32** | 26.16 | +3.84 |
| **IRF Dice** (Class 1) | **0.7483** | 0.6800 | -0.0683 |
| **SRF Dice** (Class 2) | **0.8489** | 0.8187 | -0.0302 |
| **PED Dice** (Class 3) | **0.8220** | 0.7634 | -0.0586 |

**Thesis Discussion (The High-Recall Trade-off):**
Test 11-Gold remains the absolute leader in terms of pure spatial overlap and boundary smoothness. However, **Test 18 represents the "Clinical Safe-Mode" version of the project.** By shifting Tversky to 0.1/0.9 and lowering thresholds to 0.30, we intentionally allowed more False Positives to ensure that no pathology (especially IRF) is missed. The drop in metrics is a mathematical artifact of this aggressive detection strategy—a "thicker" predicted boundary or a low-confidence detection of a tiny cyst slightly reduces the Dice denominator, but drastically increases clinical utility for a radiologist who would rather verify a false alarm than miss a treatable lesion.

**Empirical Results & Clinical Analysis (Ablation on Morphological Tuning):**
- **Thesis Conclusion:** Test 18 achieved the highest clinical sensitivity. While mIoU stayed around 0.70, the boundary precision (BP) and separation of IRF cysts significantly improved. The model now prioritizes detecting every potential cyst, even with low confidence (0.30 threshold), making it a much more valuable clinical assistant than raw CNN models that prioritize clean backgrounds over pathology recall.

### 2026-06-08 (test17 - THE FINAL SCALE-UP - COMPLETED/FAILED)
**Model:** SegFormer (**MiT-B3**)
**Status:** **Completed (Architecture Dropped).**
**Strategy (Scaling for Precision):**
- **Architecture:** Scaling to MiT-B3 (~44M parameters) to capture high-frequency clinical details.
- **Regularization:** Increased **Dropout to 0.3** to prevent overfitting on the small RETOUCH cohort.
- **True Relaxed Post-Processing:** 
    - **Removed all morphological operations** (`MORPH_OPEN`, `MORPH_CLOSE`).
    - Reduced **MIN_REGION to 5px** to allow B3's sharper features to persist.
    - This allows the model's raw high-resolution output to drive the final metrics without artificial distortion.
- **Consistency:** Maintaining the successful Test 16 pipeline (Fixed weights `[0.5, 5.0, 2.0, 2.0]`, Argmax, Focal-Tversky).
**Analysis:** Scaling to MiT-B3 caused severe overfitting on the small 56-patient training set, even with high dropout. The model memorized noise, leading to suppressed metrics in validation compared to MiT-B2. Strategy abandoned in favor of Test 18 (MiT-B2 + advanced TTA).

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
