import os
import torch
import numpy as np
import cv2
import logging
from tqdm import tqdm
from PIL import Image
from medpy.metric.binary import hd95, asd
from hybrid_inference import HybridInference
import config
from utils import get_stratified_splits
from dataset import OCTDataset
from transformers import SegformerImageProcessor

def evaluate_hybrid(base_weights="best_model.pth", expert_weights="irf_expert_best.pth", output_dir="hybrid_eval_results"):
    """
    Evaluates the ensemble of Base (mit-b2) and Expert (mit-b0) models.
    Computes class-specific metrics to verify IRF improvement.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"--- STARTING HYBRID EVALUATION ---")
    
    if not os.path.exists(base_weights) or not os.path.exists(expert_weights):
        logging.error("Missing weight files. Ensure both base and expert models are trained.")
        return

    # 1. Initialize Hybrid Engine
    engine = HybridInference(base_weights, expert_weights)
    
    # 2. Setup Test Data
    all_files = sorted(os.listdir(config.IMG_DIR))
    _, _, test_patients = get_stratified_splits(all_files)
    
    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    
    logging.info(f"Testing on {len(test_imgs)} images from {len(test_patients)} patients.")

    # 3. Metrics Accumulators
    num_classes = config.NUM_LABELS
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_hd95 = {c: [] for c in range(1, num_classes)}
    class_regions_gt = {c: [] for c in range(1, num_classes)}
    class_regions_pred = {c: [] for c in range(1, num_classes)}

    # 4. Evaluation Loop
    for idx in tqdm(range(len(test_imgs)), desc="Evaluating Hybrid"):
        # Load Image
        img_pil = Image.open(test_imgs[idx]).convert("RGB")
        img_np = np.array(img_pil)
        
        # Load GT Mask
        gt_mask = np.array(Image.open(test_masks[idx])).astype(np.uint8)
        
        # Predict using Hybrid Engine
        pred_mask = engine.segment(img_np)
        
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
            
            # Connected Components (Regions)
            n_gt, _, _, _ = cv2.connectedComponentsWithStats(gt_c, connectivity=8)
            n_pred, _, _, _ = cv2.connectedComponentsWithStats(pred_c, connectivity=8)
            class_regions_gt[c].append(n_gt - 1)
            class_regions_pred[c].append(n_pred - 1)
            
            # HD95 (Only if both exist)
            if np.any(gt_c) and np.any(pred_c):
                class_hd95[c].append(hd95(pred_c, gt_c))
        
        # Optional: Save visual comparison for first 10 images
        if idx < 10:
            save_viz(img_np, gt_mask, pred_mask, os.path.join(output_dir, f"viz_{idx}.png"))

    # 5. Final Metrics Calculation
    ious = np.diag(total_cm) / (total_cm.sum(axis=0) + total_cm.sum(axis=1) - np.diag(total_cm) + 1e-6)
    dices = (2 * np.diag(total_cm)) / (total_cm.sum(axis=0) + total_cm.sum(axis=1) + 1e-6)
    
    logging.info("\n" + "="*30)
    logging.info("FINAL HYBRID METRICS")
    logging.info("="*30)
    logging.info(f"mIoU: {np.mean(ious):.4f} | mDice: {np.mean(dices):.4f}")
    
    for c in range(1, num_classes):
        name = config.CLASS_NAMES[c]
        avg_hd = np.mean(class_hd95[c]) if class_hd95[c] else 0.0
        avg_reg_gt = np.mean(class_regions_gt[c])
        avg_reg_pred = np.mean(class_regions_pred[c])
        
        logging.info(f"Class {c} ({name}):")
        logging.info(f"  IoU: {ious[c]:.4f} | Dice: {dices[c]:.4f} | HD95: {avg_hd:.2f}")
        logging.info(f"  Avg Regions (GT/Pred): {avg_reg_gt:.1f} / {avg_reg_pred:.1f}")
    
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    evaluate_hybrid()
