# Plan: OCT Segmentation for DME

## 1. Data Prep
- [x] Explore dataset (`data_folder/`).
- [x] Data split: Train/Val. Patient integrity.
- [x] Preprocessing: SegFormer norm, 512x512, augment (Flip, Bright/Contrast).
- [ ] Deep class analysis: IRF, SRF, HRF (HRF hardest).

## 2. Architecture
- [x] SegFormer (B0) (transformers).
- [ ] Test SegFormer-B2 for precision.
- [x] Multi-channel: `edge_map_images` + `denoised_images` (R: Orig, G: Denoised, B: Edge).

## 3. Loss & Optimization
- [x] Hybrid Loss: $Loss = 0.5 \cdot CE + 0.5 \cdot DiceLoss$.
- [x] AdamW, LR 1e-4.
- [x] Weighted Focal Loss. Weight 2.0 for IRF.
- [x] Speed: `num_workers=4`, `pin_memory=True`.

## 4. Metrics
- [x] Metrics script.
- [x] Validation mIoU in train loop.
- Baseline Results (SegFormer-B0):
- mIoU: 0.6483, Dice: 0.7709
- Class 1 (IRF) IoU: 0.4925
- Class 2 (SRF) IoU: 0.5746
- Class 3 (HRF) IoU: 0.5380
- [ ] HD95 (Hausdorff Distance).

## 5. Train & Validation
- [x] Baseline train loop.
- [x] Focal Loss with class weights.
- [x] Auto-save `best_model.pth`.
- [x] Train 50 epochs with Cosine LR Scheduler.

## 6. Eval & Interpretation
- [x] Visualize (`eval_results.png`).
- [ ] Error analysis: HRF, edges.
- [ ] Compare `denoised_images`.

## 7. Docs & Report
- [ ] Metrics table.
- [ ] Stability conclusions.
