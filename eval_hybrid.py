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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_hybrid(base_weights="best_model.pth", expert_weights="irf_expert_best.pth", output_dir="hybrid_eval_results", ensemble_mode="soft", expert_weight=0.4, blend_strategy="linear", irf_threshold=0.25, irf_min_region_size=12, irf_override=False):
    """
    Evaluates the ensemble of Base (mit-b2) and Expert (mit-b0) models.
    Computes class-specific metrics to verify IRF improvement.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"--- STARTING HYBRID EVALUATION ---")
    logging.info(f"Ensemble Mode: {ensemble_mode} | Expert Weight: {expert_weight} | Blend Strategy: {blend_strategy} | IRF Threshold: {irf_threshold} | IRF Min Region Size: {irf_min_region_size} | IRF Override: {irf_override}")
    
    if not os.path.exists(base_weights) or not os.path.exists(expert_weights):
        logging.error("Missing weight files. Ensure both base and expert models are trained.")
        return

    # 1. Initialize Hybrid Engine
    engine = HybridInference(
        base_weights, 
        expert_weights, 
        ensemble_mode=ensemble_mode, 
        expert_weight=expert_weight, 
        blend_strategy=blend_strategy, 
        irf_threshold=irf_threshold, 
        irf_min_region_size=irf_min_region_size,
        irf_override=irf_override
    )
    
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

    # [REPLICATION] Scan raw masks for vis indices (Dynamically sized to target classes)
    logging.info("Scanning raw masks for visualization slice selection...")
    target_classes = list(range(1, config.NUM_LABELS))
    class_max_counts = {c: 0 for c in target_classes}
    vis_indices = {c: -1 for c in target_classes}
    for i, m_path in enumerate(test_masks):
        m = np.array(Image.open(m_path))
        for c in target_classes:
            cnt = np.sum(m == c)
            if cnt > class_max_counts[c]:
                class_max_counts[c] = cnt
                vis_indices[c] = i
    logging.info(f"Dynamic vis indices: {vis_indices}")

    # 4. Metrics Accumulators
    num_classes = config.NUM_LABELS
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_hd95 = {c: [] for c in range(1, num_classes)}
    class_asd = {c: [] for c in range(1, num_classes)}
    class_regions_gt = {c: [] for c in range(1, num_classes)}
    class_regions_pred = {c: [] for c in range(1, num_classes)}
    class_boundary_diffs = {c: [] for c in range(1, num_classes)}
    class_pixel_areas_gt = {c: [] for c in range(1, num_classes)}
    vis_data = {}

    # 5. Evaluation Loop
    for idx in tqdm(range(len(ds)), desc="Evaluating Hybrid"):
        batch = ds[idx]
        image_np = batch["orig_img"] # Contains proper 2.5D context [H, W, 3] or Multimodal
        gt_mask = batch["labels"].numpy().astype(np.uint8)
        
        # Predict using Hybrid Engine (extracting mask and base attention map)
        pred_mask, att_map = engine.segment_with_attention(image_np)
        
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
        
        # Collect visualization data for predictions.png
        for c_key, t_idx in vis_indices.items():
            if idx == t_idx and t_idx != -1:
                vis_data[c_key] = (image_np, gt_mask, pred_mask, att_map)

    # 6. Grid Visualization Generator (replicating eval.py predictions.png)
    logging.info("Generating predictions.png...")
    num_plots = len(vis_data)
    if num_plots > 0:
        plt.figure(figsize=(22, 5 * num_plots))
        sorted_c_ids = [c for c in target_classes if c in vis_data]
        for i, c_id in enumerate(sorted_c_ids):
            img, gt, prd, att = vis_data[c_id]
            
            ax1 = plt.subplot(num_plots, 4, i*4+1)
            ax1.imshow(img) 
            ax1.set_title(f"OCT: {config.CLASS_NAMES.get(c_id, f'Class {c_id}')}", fontsize=14, fontweight='bold')
            ax1.axis("off")
            
            ax2 = plt.subplot(num_plots, 4, i*4+2)
            ax2.imshow(gt, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1)
            ax2.set_title(f"GT (px: {class_max_counts[c_id]})", fontsize=12)
            ax2.axis("off")
            
            tp = np.sum((gt == c_id) & (prd == c_id))
            fp = np.sum((gt != c_id) & (prd == c_id))
            fn = np.sum((gt == c_id) & (prd != c_id))
            c_iou = tp / (tp + fp + fn + 1e-6)
            
            ax3 = plt.subplot(num_plots, 4, i*4+3)
            ax3.imshow(prd, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1)
            ax3.set_title(f"Prediction (IoU: {c_iou:.2f})", fontsize=12)
            ax3.axis("off")
            
            ax4 = plt.subplot(num_plots, 4, i*4+4)
            att_resized = cv2.resize(att, (img.shape[1], img.shape[0]))
            att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
            att_norm = np.power(att_norm, getattr(config, "ATTENTION_CONTRAST", 0.6))
            
            gray_bg = img[:, :, 1] if img.ndim == 3 else img
            ax4.imshow(gray_bg, cmap="gray")
            ax4.imshow(att_norm, cmap="jet", alpha=0.5)
            ax4.set_title("Self-Attention (Stage 2)", fontsize=12)
            ax4.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=200, bbox_inches='tight')
        plt.close()

    # 7. Final Metrics Calculation
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clinical Hybrid Ensemble Evaluation Pipeline")
    parser.add_argument("--base-weights", type=str, default="best_model.pth", help="Path to base model weights")
    parser.add_argument("--expert-weights", type=str, default="irf_expert_best.pth", help="Path to expert model weights")
    parser.add_argument("--output", type=str, default="hybrid_eval_results", help="Output directory")
    parser.add_argument("--ensemble-mode", type=str, default="soft", choices=["soft", "hard"], help="Ensemble mode: soft (probability blend) or hard (mask override)")
    parser.add_argument("--expert-weight", type=float, default=0.4, help="Weight of expert predictions in soft ensemble (0.0 to 1.0)")
    parser.add_argument("--blend-strategy", type=str, default="linear", choices=["linear", "geometric", "harmonic", "max", "min", "confidence"], help="Blending strategy for soft ensembling")
    parser.add_argument("--irf-threshold", type=float, default=0.25, help="Custom decision threshold for IRF (default: 0.25) to control sensitivity/recall")
    parser.add_argument("--irf-min-region-size", type=int, default=12, help="Minimum region size for IRF cysts (default: 12) to preserve small detections")
    parser.add_argument("--irf-override", action="store_true", help="Allow IRF to override/fragment SRF and PED (not recommended due to fragmentation)")
    args = parser.parse_args()
    
    evaluate_hybrid(
        base_weights=args.base_weights,
        expert_weights=args.expert_weights,
        output_dir=args.output,
        ensemble_mode=args.ensemble_mode,
        expert_weight=args.expert_weight,
        blend_strategy=args.blend_strategy,
        irf_threshold=args.irf_threshold,
        irf_min_region_size=args.irf_min_region_size,
        irf_override=args.irf_override
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
