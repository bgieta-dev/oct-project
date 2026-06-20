import os
import torch
import numpy as np
import cv2
import logging
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from medpy.metric.binary import hd95, asd
from hybrid_inference import HybridInference
import config
from utils import get_stratified_splits
from dataset import OCTDataset
from eval import BoundaryPrecisionAnalyzer

def evaluate_hybrid(base_weights="best_model.pth", expert_weights="irf_expert_best.pth", output_dir="hybrid_eval_results", ensemble_mode="soft", expert_weight=0.4, blend_strategy="linear", irf_threshold=None):
    """
    Evaluates the ensemble of Base (mit-b2) and Expert (mit-b0) models.
    Computes class-specific metrics to verify IRF improvement.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"--- STARTING HYBRID EVALUATION ---")
    logging.info(f"Ensemble Mode: {ensemble_mode} | Expert Weight: {expert_weight} | Blend Strategy: {blend_strategy} | IRF Threshold: {irf_threshold}")
    
    if not os.path.exists(base_weights) or not os.path.exists(expert_weights):
        logging.error("Missing weight files. Ensure both base and expert models are trained.")
        return

    # 1. Initialize Hybrid Engine
    engine = HybridInference(base_weights, expert_weights, ensemble_mode=ensemble_mode, expert_weight=expert_weight, blend_strategy=blend_strategy, irf_threshold=irf_threshold)
    
    # 2. Setup Test Data
    all_files = sorted(os.listdir(config.IMG_DIR))
    _, _, test_patients = get_stratified_splits(all_files)
    
    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    
    logging.info(f"Testing on {len(test_imgs)} images from {len(test_patients)} patients.")

    # 3. Setup Test Dataset with correct transforms and 2.5D config
    val_transform = None
    if hasattr(config, "AUG_SIZE") and config.AUG_SIZE is not None:
        import albumentations as A
        val_transform = A.Compose([A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1])])
        
    ds = OCTDataset(
        image_paths=test_imgs,
        mask_paths=test_masks,
        processor=engine.processor,
        transform=val_transform,
        cfg=config
    )
    
    logging.info(f"Testing on {len(ds)} images from {len(test_patients)} patients using OCTDataset workflow.")

    # 4. Metrics Accumulators
    num_classes = config.NUM_LABELS
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_hd95 = {c: [] for c in range(1, num_classes)}
    class_asd = {c: [] for c in range(1, num_classes)}
    class_regions_gt = {c: [] for c in range(1, num_classes)}
    class_regions_pred = {c: [] for c in range(1, num_classes)}
    class_boundary_diffs = {c: [] for c in range(1, num_classes)}
    class_pixel_areas_gt = {c: [] for c in range(1, num_classes)}

    # 5. Evaluation Loop
    for idx in tqdm(range(len(ds)), desc="Evaluating Hybrid"):
        batch = ds[idx]
        image_np = batch["orig_img"] # Contains proper 2.5D context [H, W, 3] or Multimodal
        gt_mask = batch["labels"].numpy().astype(np.uint8)
        
        # Predict using Hybrid Engine
        pred_mask = engine.segment(image_np)
        
        # Update Confusion Matrix
        mask_valid = (gt_mask >= 0) & (gt_mask < num_classes)
        total_cm += np.bincount(
            num_classes * gt_mask[mask_valid].astype(np.int64) + pred_mask[mask_valid].astype(np.int64), 
            minlength=num_classes**2
        ).reshape(num_classes, num_classes)
        
        # Region and Shape Metrics
        for c in range(1, num_classes):
            gt_c = (gt_mask == c).astype(np.uint8)
            pred_c = (pred_mask == c).astype(np.uint8)
            
            if np.any(gt_c):
                _, gt_num = ndimage.label(gt_c)
                class_regions_gt[c].append(gt_num)
                class_boundary_diffs[c].append(BoundaryPrecisionAnalyzer(cfg=config).get_boundary_contrast(image_np, gt_c))
                class_pixel_areas_gt[c].append(np.sum(gt_c))
            if np.any(pred_c):
                _, pred_num = ndimage.label(pred_c)
                class_regions_pred[c].append(pred_num)
            if np.any(gt_c) and np.any(pred_c):
                class_hd95[c].append(hd95(pred_c, gt_c))
                class_asd[c].append(asd(pred_c, gt_c))
        
        # Optional: Save visual comparison for first 10 images
        if idx < 10:
            save_viz(image_np[:, :, 1] if len(image_np.shape) == 3 else image_np, gt_mask, pred_mask, os.path.join(output_dir, f"viz_{idx}.png"))

    # 6. Final Metrics Calculation
    ious = np.diag(total_cm) / (total_cm.sum(axis=0) + total_cm.sum(axis=1) - np.diag(total_cm) + 1e-6)
    dices = (2 * np.diag(total_cm)) / (total_cm.sum(axis=0) + total_cm.sum(axis=1) + 1e-6)
    
    def safe_mean(lst): return np.mean(lst) if lst else 0.0
    
    mhd95 = safe_mean([safe_mean(class_hd95[c]) for c in range(1, num_classes) if class_hd95[c]])
    masd = safe_mean([safe_mean(class_asd[c]) for c in range(1, num_classes) if class_asd[c]])
    
    logging.info("\n" + "="*30)
    logging.info("FINAL HYBRID METRICS")
    logging.info("="*30)
    logging.info(f"mIoU: {np.mean(ious):.4f} | mDice: {np.mean(dices):.4f}")
    logging.info(f"mHD95: {mhd95:.4f} | mASD: {masd:.4f}")
    
    for c in range(1, num_classes):
        name = config.CLASS_NAMES[c]
        avg_hd = safe_mean(class_hd95[c])
        avg_asd = safe_mean(class_asd[c])
        avg_reg_gt = safe_mean(class_regions_gt[c])
        avg_reg_pred = safe_mean(class_regions_pred[c])
        avg_bp = safe_mean(class_boundary_diffs[c])
        avg_area = safe_mean(class_pixel_areas_gt[c])
        
        logging.info(f"Class {c} ({name}) | IoU: {ious[c]:.4f} | Dice: {dices[c]:.4f} | HD95: {avg_hd:.2f} | ASD: {avg_asd:.2f}")
        logging.info(f"  Regions GT/Pred: {avg_reg_gt:.1f}/{avg_reg_pred:.1f} | BP: {avg_bp:.4f} | Avg Area: {avg_area:.1f} px")
    
    return ious

def save_viz(img, gt, pred, path):
    """Helper to save a side-by-side comparison"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("OCT Image")
    axes[1].imshow(gt, cmap='nipy_spectral', vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap='nipy_spectral', vmin=0, vmax=3)
    axes[2].set_title("Hybrid Prediction")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clinical Hybrid Ensemble Evaluation Pipeline")
    parser.add_argument("--base-weights", type=str, default="best_model.pth", help="Path to base model weights")
    parser.add_argument("--expert-weights", type=str, default="irf_expert_best.pth", help="Path to expert model weights")
    parser.add_argument("--output", type=str, default="hybrid_eval_results", help="Output directory")
    parser.add_argument("--ensemble-mode", type=str, default="soft", choices=["soft", "hard"], help="Ensemble mode: soft (probability blend) or hard (mask override)")
    parser.add_argument("--expert-weight", type=float, default=0.4, help="Weight of expert predictions in soft ensemble (0.0 to 1.0)")
    parser.add_argument("--blend-strategy", type=str, default="linear", choices=["linear", "geometric", "harmonic", "max", "min", "confidence"], help="Blending strategy for soft ensembling")
    parser.add_argument("--irf-threshold", type=float, default=None, help="Custom decision threshold for IRF (e.g. 0.20 or 0.15) to control sensitivity/recall")
    args = parser.parse_args()
    
    evaluate_hybrid(
        base_weights=args.base_weights,
        expert_weights=args.expert_weights,
        output_dir=args.output,
        ensemble_mode=args.ensemble_mode,
        expert_weight=args.expert_weight,
        blend_strategy=args.blend_strategy,
        irf_threshold=args.irf_threshold
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
