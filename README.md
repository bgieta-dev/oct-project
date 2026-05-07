# Scans
https://www.kaggle.com/datasets/saivikassingamsetty/retouch?resource=download-directory&select=retouch_processed

# Project
Task: Neural net for OCT segmentation in DME diagnosis.

DME (Diabetic Macular Edema): Vision loss from retinal vessel damage.

IRF: Black spaces in retina, cysts. Intraretinal. Mostly Henle’s layer.
SRF: Black space between retina and RPE. Retinal detachment. Subretinal.
HRF: Small bright spots. All layers. Predict inflammation/lipids.

# Models
https://github.com/qubvel-org/segmentation_models.pytorch

CNN: UNet++, MA-Net
Transformers: DPT, SegFormer

Refs:
- Oktay (2018) Attention U-Net: https://arxiv.org/abs/1804.03999
- Zhou (2018) UNet++: https://arxiv.org/abs/1807.10165
- Lee (2022) MDPI Sensors: https://www.mdpi.com/1424-8220/22/8/3055
- Vaswani (2017) Attention: https://arxiv.org/abs/1706.03762
- Dosovitskiy (2020) ViT: https://arxiv.org/abs/2010.11929
- Xie (2021) SegFormer: https://arxiv.org/abs/2105.15203
- Tang (2022) SwinUNETR: https://arxiv.org/abs/2201.01266

# Methods
Hybrid Loss: $Loss = a \cdot DiceLoss + b \cdot FocalLoss$

Focal Loss: Handle imbalance. Focus on hard pixels.
GDL: Weight by inverse size. Balance small HRF.

Metrics:
- Precision/Recall: Sensitivity + false alarms.
- IoU (Jaccard): Strict, pixel-sensitive.
- HD95: Contour distance. Shape precision.

Focal Loss detects HRF. HD95 evaluates morphology.

