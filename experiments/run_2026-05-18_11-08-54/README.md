# OCT Fluid Segmentation (Student 2: Maksymilian)

Automated segmentation of retinal biomarkers in OCT images using Transformer-based architectures.

## Clinical Context
Task: Semantic segmentation of fluid spaces in DME (Diabetic Macular Edema).
- **IRF**: Intraretinal Fluid (cysts within layers).
- **SRF**: Subretinal Fluid (detachment under retina).
- **PED**: Pigment Epithelium Detachment (RPE lift).

## Key Features
- **Architecture**: SegFormer (B0-B3 backbones) with MiT encoders.
- **Preprocessing**: 
  - Clinical Intensity Normalization (1-99 percentile clipping).
  - Heavy Augmentation: Rotate, GaussNoise, RandomResizedCrop (Albumentations 2.0).
- **Input Strategy**: Multi-modal stack (3 channels: Original + Denoised + Edge map).
- **Optimization**:
  - Hybrid Loss: $0.5 \cdot FocalLoss + 0.5 \cdot DiceLoss$.
  - Gradient Accumulation: Simulated batch size 16 for RTX 3060/4060 stability.
- **Evaluation**: 
  - Stratified 80/10/10 Split (by OCT device: Cirrus, Spectralis, Topcon).
  - Metrics: Dice, IoU, **HD95** (Hausdorff Distance), **ASD** (Average Surface Distance).
  - Class-specific fixed visualization for static reporting.

## Project Structure
- `config.py`: Central source of truth for all parameters.
- `train.py`: Training loop with gradient accumulation and weighted loss.
- `eval.py`: Quantitative and qualitative evaluation on the test set.
- `main.py`: Pipeline entry point (Train -> Eval -> Archive).
- `dataset.py`: Patient-aware OCT loader with percentile normalization.

## Requirements
```bash
pip install torch transformers albumentations scikit-learn matplotlib tqdm medpy SimpleITK
```

## Usage
1. Configure settings in `config.py` (e.g., set `MODEL_NAME = "nvidia/mit-b2"`).
2. Run pipeline:
```bash
python main.py
```
3. Check `experiments/run_TIMESTAMP/` for logs, `best_model.pth`, and `predictions.png`.

---

## Experiments

### 2026-05-18 (Current Run)
**Model:** SegFormer (nvidia/mit-b2)
**Setup:** `LR=1e-4` with 5-epoch warmup (`get_cosine_schedule_with_warmup`), Early Stopping (15 epochs patience). Evaluation includes `Avg Regions GT/Pred` counting via `scipy.ndimage`. 
**Status:** Starting training to verify stability and performance improvements.

### 2026-05-17
**Model:** SegFormer (nvidia/mit-b2)
**Results:** Final mIoU: 0.7125, Final mDice: 0.8220
**Observations:** Training was highly unstable. Around epoch 24, the model collapsed to predicting only background, driving mIoU down to ~0.24. It eventually recovered at epoch 68 as the `CosineAnnealingLR` decayed the learning rate.
**Conclusions:** The `mit-b2` Transformer requires a learning rate warmup phase (e.g. 5 epochs) when trained with AdamW at `LR=1e-4` to prevent gradient spikes and local minimum collapse early in training.

---

## References
- Ronneberger (2015) U-Net
- Xie (2021) SegFormer
- Tang (2022) SwinUNETR
- Isensee (2021) nnU-Net
