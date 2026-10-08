"""Volumetric evaluation and sliding-window inference for 3D SwinUNETR."""
import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from octseg.config import Config
from octseg.metrics import SegmentationMetrics, save_metrics_yaml, slice_mean_iou
from octseg.model import load_swin_unetr
from octseg.postprocess import apply_thresholds, clean_regions, sharpen_ped
from octseg.splits import split_files
from octseg.viz import save_failure_cases, save_predictions_grid, select_vis_indices
from octseg.volume_data import group_volumes, load_volume

log = logging.getLogger(__name__)


def predict_volume(model, image_u8: np.ndarray, cfg: Config, tta: bool = False) -> np.ndarray:
    """Infers on a whole volume using sliding window and chunked interpolation.

    Args:
        model: SwinUNETR model in eval mode.
        image_u8: uint8 array [Z, H, W] resampled at SWIN_VOLUME_SIZE.
        cfg: config providing SWIN_ROI, SWIN_SW_BATCH, SWIN_SW_OVERLAP, AUG_SIZE, DEVICE, USE_AMP.
        tta: whether to perform lateral-flip test-time augmentation.

    Returns:
        Softmax probabilities float32 [Z, C, *AUG_SIZE] on CPU.
    """
    from monai.inferers import sliding_window_inference

    model.eval()
    x = torch.from_numpy(image_u8).float().div(255.0)[None, None].to(cfg.DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
            logits = sliding_window_inference(
                x,
                roi_size=cfg.SWIN_ROI,
                sw_batch_size=cfg.SWIN_SW_BATCH,
                predictor=model,
                overlap=cfg.SWIN_SW_OVERLAP,
                mode="gaussian",
            )
            if tta:
                x_flip = torch.flip(x, dims=[4])
                logits_flip = sliding_window_inference(
                    x_flip,
                    roi_size=cfg.SWIN_ROI,
                    sw_batch_size=cfg.SWIN_SW_BATCH,
                    predictor=model,
                    overlap=cfg.SWIN_SW_OVERLAP,
                    mode="gaussian",
                )
                logits = (logits + torch.flip(logits_flip, dims=[4])) * 0.5

        # logits shape: [1, C, Z, H, W] -> permute to [Z, C, H, W]
        perm = logits[0].permute(1, 0, 2, 3)
        probs_chunks = []
        for start in range(0, perm.shape[0], 16):
            chunk = perm[start:start + 16]
            interp = F.interpolate(chunk.float(), size=cfg.AUG_SIZE, mode="bilinear", align_corners=False)
            p = torch.softmax(interp, dim=1).cpu().numpy()
            probs_chunks.append(p)
        probs = np.concatenate(probs_chunks, axis=0)

    return probs


def evaluate_model_3d(model_path: str, output_dir: str, cfg: Config) -> dict:
    """Evaluates a 3D SwinUNETR checkpoint on the test split.

    Writes predictions.png (3-column grid), failure cases, and metrics.yaml.
    Matches the 2D SegFormer post-processing and metric computation resolution (512x512).

    Args:
        model_path: path to trained checkpoint (.pth).
        output_dir: output directory for evaluation artifacts.
        cfg: configuration object.

    Returns:
        Metrics dictionary with 'params' key included.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_imgs, test_masks = split_files(cfg, "test")
    records = group_volumes(test_imgs, test_masks)

    all_mask_paths = [p for r in records for p in r.mask_paths]
    target_classes = list(range(1, cfg.NUM_LABELS))
    log.info("Scanning raw masks for visualisation slice selection...")
    vis_indices, class_max_counts = select_vis_indices(all_mask_paths, target_classes, cfg.TARGET_CLASS)
    log.info(f"Dynamic vis indices: {vis_indices}")

    log.info(f"Loading weights from {model_path}")
    model = load_swin_unetr(cfg, model_path)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    metrics = SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg)
    thresholds = cfg.CLASS_THRESHOLDS
    vis_data = {}
    failure_records = []

    log.info(f"Evaluating 3D model on {len(records)} volumes ({len(all_mask_paths)} slices)...")
    global_idx = 0
    for record in tqdm(records, desc="Evaluating volumes"):
        input_vol = load_volume(record, cfg.SWIN_VOLUME_SIZE)[0]
        gt_imgs, gt_masks = load_volume(record, cfg.AUG_SIZE)

        probs = predict_volume(model, input_vol, cfg, tta=cfg.USE_TTA)

        for z in range(len(probs)):
            p_slice = probs[z]
            sharpen_ped(p_slice, factor=cfg.PED_SHARPEN_FACTOR)
            pred = apply_thresholds(p_slice, thresholds, irf_override=True)
            if cfg.MIN_REGION_SIZE > 0:
                pred = clean_regions(pred, cfg.MIN_REGION_SIZE, irf_open=True)

            gt = gt_masks[z]
            img2d = gt_imgs[z]
            img_rgb = np.stack([img2d] * 3, axis=-1)

            for c_key, t_idx in vis_indices.items():
                if global_idx == t_idx:
                    vis_data[c_key] = (img_rgb, gt, pred, None)

            metrics.update(gt, pred, img2d)

            slice_miou = slice_mean_iou(gt, pred, cfg.NUM_LABELS)
            if slice_miou is not None:
                failure_records.append((slice_miou, global_idx, img_rgb, gt, pred))

            global_idx += 1

    log.info("Generating predictions.png...")
    save_predictions_grid(vis_data, class_max_counts, cfg, os.path.join(output_dir, "predictions.png"))
    if getattr(cfg, "SAVE_FAILURES", True) and failure_records:
        log.info("Saving failure cases...")
        save_failure_cases(failure_records, output_dir, cfg, top_k=getattr(cfg, "TOP_K_FAILURES", 5))

    result = metrics.compute()
    result["params"] = total_params
    save_metrics_yaml(result, os.path.join(output_dir, "metrics.yaml"))
    return result
