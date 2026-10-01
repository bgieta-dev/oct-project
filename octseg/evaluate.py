import argparse
import logging
import os
import sys
import traceback
from datetime import datetime

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerImageProcessor

from octseg.config import Config, load_config, save_config, ROOT
from octseg.dataset import OCTDataset
from octseg.metrics import SegmentationMetrics, save_metrics_yaml
from octseg.model import load_segformer
from octseg.postprocess import apply_thresholds, clean_regions, predict_logits, sharpen_ped
from octseg.runs import setup_logging
from octseg.splits import load_splits, patient_of
from octseg.viz import save_predictions_grid, select_vis_indices

log = logging.getLogger(__name__)


def evaluate_model(model_path, output_dir, cfg: Config):
    """Evaluates a checkpoint on the frozen test patients and writes predictions.png + metrics.yaml.

    Raises FileNotFoundError if `model_path` does not exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    all_files = sorted(os.listdir(cfg.IMG_DIR))
    test_patients = set(load_splits(cfg.SPLIT_DIR)[2])
    test_imgs = [os.path.join(cfg.IMG_DIR, f) for f in all_files if patient_of(f) in test_patients]
    test_masks = [os.path.join(cfg.MASK_DIR, f) for f in all_files if patient_of(f) in test_patients]

    target_classes = list(range(1, cfg.NUM_LABELS))
    log.info("Scanning raw masks for visualisation slice selection...")
    vis_indices, class_max_counts = select_vis_indices(test_masks, target_classes, cfg.TARGET_CLASS)
    log.info(f"Dynamic vis indices: {vis_indices}")

    # 2. Model Init
    processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)
    log.info(f"Loading weights from {model_path}")
    model = load_segformer(cfg, model_path, output_attentions=True)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. Setup Dataset
    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])
    dataset = OCTDataset(test_imgs, test_masks, processor, transform=val_transform,
                         cfg=cfg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    metrics = SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg)
    thresholds = cfg.CLASS_THRESHOLDS
    vis_data = {}

    # 4. Evaluation Loop
    log.info(f"Evaluating on {len(dataset)} slices...")
    global_idx = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(cfg.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()

        with torch.no_grad():
            # Encoder block 5 attention for the visualisation grid
            outputs = model(pixel_values=pixel_values)
            avg_att = torch.mean(outputs.attentions[5], dim=1)
            spatial_att = torch.mean(avg_att, dim=1)
            grid_size = int(np.sqrt(spatial_att.shape[1]))
            att_maps = spatial_att.view(-1, grid_size, grid_size).cpu().numpy()

            logits = predict_logits(model, pixel_values, cfg.AUG_SIZE, cfg)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()

        sharpen_ped(probs_batch)
        preds_batch = apply_thresholds(probs_batch, thresholds, irf_override=True)
        if cfg.MIN_REGION_SIZE > 0:
            preds_batch = np.array([clean_regions(p, cfg.MIN_REGION_SIZE, irf_open=True) for p in preds_batch])

        for b_idx in range(len(preds_batch)):
            labels, pred = labels_batch[b_idx], preds_batch[b_idx]
            orig_img = orig_img_batch[b_idx]
            for c_key, t_idx in vis_indices.items():
                if global_idx == t_idx:
                    vis_data[c_key] = (orig_img, labels, pred, att_maps[b_idx])
            metrics.update(labels, pred, orig_img)
            global_idx += 1

    log.info("Generating predictions.png...")
    save_predictions_grid(vis_data, class_max_counts, cfg, os.path.join(output_dir, "predictions.png"))

    result = metrics.compute()
    result["params"] = total_params
    save_metrics_yaml(result, os.path.join(output_dir, "metrics.yaml"))
    return result


def log_metrics(metrics, cfg):
    """Logs the aggregate and per-class metrics (the lines the thesis tables are built from)."""
    log.info(f"Final mIoU: {metrics['mIoU']:.4f} | Final mDice: {metrics['mDice']:.4f}")
    log.info(f"Final mHD95: {metrics['mHD95']:.4f} | Final mASD: {metrics['mASD']:.4f}")
    if "params" in metrics:
        log.info(f"Model Parameters: {metrics['params']:.2f}M")
    for c in range(1, cfg.NUM_LABELS):
        log.info(f"Class {c} ({cfg.CLASS_NAMES[c]}) | IoU: {metrics['class_ious'][c]:.4f} | Dice: {metrics['class_dices'][c]:.4f} | HD95: {metrics['class_hd95'][c]:.2f} | ASD: {metrics['class_asd'][c]:.2f}")
        log.info(f"  Regions GT/Pred: {metrics['class_avg_regions_gt'][c]:.1f}/{metrics['class_avg_regions_pred'][c]:.1f} | BP: {metrics['class_boundary_precision'][c]:.4f} | Avg Area: {metrics['class_avg_pixel_area'][c]:.1f} px")


def main():
    parser = argparse.ArgumentParser(description="Standalone Clinical Evaluation Pipeline for OCT Segmentation")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml; use configs/irf_expert.yaml to evaluate the expert)")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--output", type=str, help="Output directory (default: outputs/eval/<timestamp>)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg = load_config(args.config)
    eval_dir = os.path.abspath(args.output or ROOT / "outputs" / "eval" / timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    save_config(cfg, os.path.join(eval_dir, "config.yaml"))
    setup_logging(eval_dir, "eval_standalone.log")

    log.info(f"Starting Standalone Evaluation. Results will be saved to: {eval_dir}")
    try:
        metrics = evaluate_model(model_path=args.model, output_dir=eval_dir, cfg=cfg)
        log.info("--- GLOBAL EVALUATION RESULTS ---")
        log_metrics(metrics, cfg)
        log.info(f"Evaluation complete. All clinical artifacts saved in: {eval_dir}")
    except Exception:
        log.error("CRITICAL ERROR encountered during standalone evaluation:")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
