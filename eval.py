import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from dataset import OCTDataset
from tqdm import tqdm
import config as global_config
from medpy.metric.binary import hd95, asd
from utils import get_stratified_splits
import logging
from scipy import ndimage
import cv2
from PIL import Image

# --- CLINICAL METRIC ANALYZERS ---

class BoundaryPrecisionAnalyzer:
    def __init__(self, kernel_size=3, cfg=global_config):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.cfg = cfg

    def get_boundary_contrast(self, image, mask):
        if not np.any(mask): return 0.0
        # Use centralized central slice parameter
        if image.ndim == 3 and image.shape[2] == 3:
            slice_idx = getattr(self.cfg, "CENTRAL_SLICE_IDX", 1)
            image = image[:, :, slice_idx]
        dilated = cv2.dilate(mask.astype(np.uint8), self.kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), self.kernel, iterations=1)
        outer_edge = dilated - mask.astype(np.uint8)
        inner_edge = mask.astype(np.uint8) - eroded
        outer_vals = image[outer_edge > 0]
        inner_vals = image[inner_edge > 0]
        if len(outer_vals) == 0 or len(inner_vals) == 0: return 0.0
        return np.abs(np.mean(inner_vals) - np.mean(outer_vals))

# --- MODEL EVALUATION PIPELINE ---

def evaluate_model(model_path="best_model.pth", output_dir=".", cfg=global_config):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    all_files = sorted(os.listdir(cfg.IMG_DIR))
    if os.path.exists("test_patients.txt"):
        with open("test_patients.txt", "r") as f: test_patients = f.read().splitlines()
    else:
        _, _, test_patients = get_stratified_splits(all_files)
    
    test_imgs = [os.path.join(cfg.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(cfg.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]

    # Dynamic target classes (skipping background 0)
    target_classes = list(range(1, cfg.NUM_LABELS))

    # [REPLICATION] Scan raw masks for vis indices (Dynamically sized to target classes)
    logging.info("Replicating slice selection (scanning raw masks)...")
    class_max_counts = {c: 0 for c in target_classes}
    vis_indices = {c: -1 for c in target_classes}
    for i, m_path in enumerate(test_masks):
        m = np.array(Image.open(m_path))
        
        # Adapt for Binary Expert models where labels are 0 and 1
        if getattr(cfg, "TARGET_CLASS", None) is not None:
            # Map mask to binary matching expert behavior
            binary_m = np.zeros_like(m)
            binary_m[m == cfg.TARGET_CLASS] = 1
            m = binary_m

        for c in target_classes:
            cnt = np.sum(m == c)
            if cnt > class_max_counts[c]:
                class_max_counts[c] = cnt
                vis_indices[c] = i
    logging.info(f"Dynamic vis indices: {vis_indices}")

    # 2. Model Init
    processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS, ignore_mismatched_sizes=True, output_attentions=True)
    
    if os.path.exists(model_path):
        logging.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(cfg.DEVICE).eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. Setup Dataset
    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])
    target_cls = getattr(cfg, "TARGET_CLASS", None)
    dataset = OCTDataset(test_imgs, test_masks, processor, transform=val_transform, use_multimodal=cfg.USE_MULTIMODAL, target_class=target_cls, cfg=cfg)
    
    # Use 2 workers for faster disk I/O throughput during evaluation
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 4. State
    total_cm = np.zeros((cfg.NUM_LABELS, cfg.NUM_LABELS), dtype=np.int64)
    class_hd95_vals = {c: [] for c in target_classes}
    class_asd_vals = {c: [] for c in target_classes}
    class_region_counts_gt = {c: [] for c in target_classes}
    class_region_counts_pred = {c: [] for c in target_classes}
    class_boundary_diffs = {c: [] for c in target_classes}
    class_pixel_areas_gt = {c: [] for c in target_classes}
    
    vis_data = {} 

    # 5. Evaluation Loop
    logging.info(f"Evaluating on {len(dataset)} slices...")
    global_idx = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(cfg.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(outputs.logits, size=cfg.AUG_SIZE, mode="bilinear", align_corners=False)
            
            # Extract Stage 2 attention (64x64 resolution) for rich context visualization
            att_stage = outputs.attentions[1] 
            avg_att = torch.mean(att_stage, dim=1) 
            spatial_att = torch.mean(avg_att, dim=1) 
            grid_size = int(np.sqrt(spatial_att.shape[1]))
            att_maps = spatial_att.view(-1, grid_size, grid_size).cpu().numpy()
            
            if getattr(cfg, "USE_TTA", False):
                scales = getattr(cfg, "TTA_SCALES", [1.0])
                all_logits = []
                
                for s in scales:
                    if s != 1.0:
                        scaled_size = (int(cfg.AUG_SIZE[0] * s), int(cfg.AUG_SIZE[1] * s))
                        scaled_pixels = torch.nn.functional.interpolate(
                            pixel_values, size=scaled_size, mode="bilinear", align_corners=False
                        )
                    else:
                        scaled_pixels = pixel_values
                    
                    s_outputs = model(pixel_values=scaled_pixels)
                    s_logits = torch.nn.functional.interpolate(
                        s_outputs.logits, size=cfg.AUG_SIZE, mode="bilinear", align_corners=False
                    )
                    all_logits.append(s_logits)
                    
                    f_pixels = torch.flip(scaled_pixels, [3])
                    f_outputs = model(pixel_values=f_pixels)
                    f_logits = torch.nn.functional.interpolate(
                        f_outputs.logits, size=cfg.AUG_SIZE, mode="bilinear", align_corners=False
                    )
                    uf_logits = torch.flip(f_logits, [3])
                    all_logits.append(uf_logits)
                
                logits = torch.mean(torch.stack(all_logits), dim=0)
            
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            
            # --- CLINICAL HEURISTIC: MORPHOLOGICAL SHARPENING FOR PED ---
            # Compensates for bilinear blurring of peaks in SegFormer. Applied only if PED exists (Class 3).
            if cfg.NUM_LABELS > 3:
                sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                for b_idx in range(probs_batch.shape[0]):
                    ped_prob = probs_batch[b_idx, 3, :, :]
                    sharp_ped = cv2.filter2D(ped_prob, -1, sharpen_kernel)
                    probs_batch[b_idx, 3, :, :] = np.clip(sharp_ped, 0, 1)

            # --- CLINICAL HEURISTIC: REVERSE-PRIORITY THRESHOLDING ---
            # Maximize sensitivity for high-risk findings (e.g. fluid/cysts) by overriding background argmax.
            preds_batch = np.zeros(probs_batch.shape[0:1] + probs_batch.shape[2:], dtype=np.uint8)
            thresholds = getattr(cfg, "CLASS_THRESHOLDS", {c: 0.5 for c in target_classes})
            
            # Loop backwards through target classes so high-priority categories (like class 1) get laid down last
            for c in reversed(target_classes):
                thresh = thresholds.get(c, 0.5)
                preds_batch[probs_batch[:, c] > thresh] = c
            
            if getattr(cfg, "MIN_REGION_SIZE", 0) > 0:
                cleaned_preds = []
                kernel_3x3 = np.ones((3, 3), np.uint8)
                for b_idx in range(len(preds_batch)):
                    p = preds_batch[b_idx]
                    new_p = np.zeros_like(p)
                    for c in target_classes:
                        c_mask = (p == c).astype(np.uint8)
                        
                        if np.any(c_mask):
                            # Morphological separation for IRF (Class 1) to break thin false-positive bridges
                            if c == 1 and cfg.NUM_LABELS > 1:
                                c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel_3x3, iterations=1)
                                
                            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                            for label in range(1, num_labels):
                                if stats[label, cv2.CC_STAT_AREA] >= cfg.MIN_REGION_SIZE:
                                    new_p[labels_im == label] = c
                    cleaned_preds.append(new_p)
                preds_batch = np.array(cleaned_preds)

        for b_idx in range(len(preds_batch)):
            labels, pred, att_map = labels_batch[b_idx], preds_batch[b_idx], att_maps[b_idx]
            orig_img = orig_img_batch[b_idx]
            
            for c_key, t_idx in vis_indices.items():
                if global_idx == t_idx and t_idx != -1:
                    vis_data[c_key] = (orig_img, labels, pred, att_map)

            mask = (labels >= 0) & (labels < cfg.NUM_LABELS)
            total_cm += np.bincount(cfg.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=cfg.NUM_LABELS**2).reshape(cfg.NUM_LABELS, cfg.NUM_LABELS)

            for c in target_classes:
                gt_c, pred_c = (labels == c), (pred == c)
                if np.any(gt_c):
                    _, gt_num = ndimage.label(gt_c)
                    class_region_counts_gt[c].append(gt_num)
                    class_boundary_diffs[c].append(BoundaryPrecisionAnalyzer(cfg=cfg).get_boundary_contrast(orig_img, gt_c))
                    class_pixel_areas_gt[c].append(np.sum(gt_c))
                if np.any(pred_c):
                    _, pred_num = ndimage.label(pred_c)
                    class_region_counts_pred[c].append(pred_num)
                if np.any(gt_c) and np.any(pred_c):
                    class_hd95_vals[c].append(hd95(pred_c, gt_c))
                    class_asd_vals[c].append(asd(pred_c, gt_c))
            global_idx += 1

    # 6. Grid Visualization Generator
    logging.info("Generating predictions.png...")
    num_plots = len(vis_data)
    if num_plots > 0:
        plt.figure(figsize=(22, 5 * num_plots))
        # Ensure rows are ordered logically by class ID (1=IRF, 2=SRF, 3=PED) instead of random insertion sequence
        sorted_c_ids = [c for c in target_classes if c in vis_data]
        for i, c_id in enumerate(sorted_c_ids):
            img, gt, prd, att = vis_data[c_id]
            
            ax1 = plt.subplot(num_plots, 4, i*4+1)
            ax1.imshow(img) 
            ax1.set_title(f"OCT: {cfg.CLASS_NAMES.get(c_id, f'Class {c_id}')}", fontsize=14, fontweight='bold')
            ax1.axis("off")
            
            ax2 = plt.subplot(num_plots, 4, i*4+2)
            ax2.imshow(gt, cmap="jet", vmin=0, vmax=cfg.NUM_LABELS-1)
            ax2.set_title(f"GT (px: {class_max_counts[c_id]})", fontsize=12)
            ax2.axis("off")
            
            tp = np.sum((gt == c_id) & (prd == c_id))
            fp = np.sum((gt != c_id) & (prd == c_id))
            fn = np.sum((gt == c_id) & (prd != c_id))
            c_iou = tp / (tp + fp + fn + 1e-6)
            
            ax3 = plt.subplot(num_plots, 4, i*4+3)
            ax3.imshow(prd, cmap="jet", vmin=0, vmax=cfg.NUM_LABELS-1)
            ax3.set_title(f"Prediction (IoU: {c_iou:.2f})", fontsize=12)
            ax3.axis("off")
            
            ax4 = plt.subplot(num_plots, 4, i*4+4)
            att_resized = cv2.resize(att, (img.shape[1], img.shape[0]))
            att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
            att_norm = np.power(att_norm, getattr(cfg, "ATTENTION_CONTRAST", 0.6))
            
            gray_bg = img[:, :, 1] if img.ndim == 3 else img
            ax4.imshow(gray_bg, cmap="gray")
            ax4.imshow(att_norm, cmap="jet", alpha=0.5)
            ax4.set_title("Self-Attention (Stage 2)", fontsize=12)
            ax4.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    # 7. Aggregate Metrics
    tp = np.diag(total_cm)
    fp, fn = total_cm.sum(axis=0) - tp, total_cm.sum(axis=1) - tp
    ious, dices = tp/(tp+fp+fn+1e-6), (2*tp)/(2*tp+fp+fn+1e-6)
    
    def safe_mean(lst): return np.mean(lst) if lst else 0.0

    return {
        'params': total_params, 'mIoU': np.mean(ious), 'mDice': np.mean(dices),
        'class_ious': {c: ious[c] for c in range(cfg.NUM_LABELS)}, 'class_dices': {c: dices[c] for c in range(cfg.NUM_LABELS)},
        'class_hd95': {c: safe_mean(class_hd95_vals[c]) for c in target_classes},
        'class_asd': {c: safe_mean(class_asd_vals[c]) for c in target_classes},
        'class_avg_regions_gt': {c: safe_mean(class_region_counts_gt[c]) for c in target_classes},
        'class_avg_regions_pred': {c: safe_mean(class_region_counts_pred[c]) for c in target_classes},
        'class_boundary_precision': {c: safe_mean(class_boundary_diffs[c]) for c in target_classes},
        'class_avg_pixel_area': {c: safe_mean(class_pixel_areas_gt[c]) for c in target_classes},
        'mHD95': safe_mean([safe_mean(class_hd95_vals[c]) for c in target_classes if class_hd95_vals[c]]),
        'mASD': safe_mean([safe_mean(class_asd_vals[c]) for c in target_classes if class_asd_vals[c]])
    }

def setup_logging(log_file: str):
    """Sets up an isolated logger instance to avoid global logging side-effects."""
    import sys
    logger = logging.getLogger("eval_standalone")
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Clear any existing handlers
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def main():
    import argparse
    from datetime import datetime
    import traceback
    import sys
    
    parser = argparse.ArgumentParser(description="Standalone Clinical Evaluation Pipeline for OCT Segmentation")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--output", type=str, help="Override output directory for metrics and visual graphs")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_dir = args.output or os.path.abspath(f"eval_results_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    log_file = os.path.join(eval_dir, "eval_standalone.log")
    logger = setup_logging(log_file)
    
    logger.info(f"Starting Standalone Evaluation. Results will be saved to: {eval_dir}")
    if not os.path.exists(args.model):
        logger.error(f"Model checkpoint NOT FOUND at: {args.model}. Please supply a correct path via --model.")
        sys.exit(1)
        
    try:
        metrics = evaluate_model(model_path=args.model, output_dir=eval_dir, cfg=global_config)
        logger.info("--- GLOBAL EVALUATION RESULTS ---")
        logger.info(f"Final mIoU: {metrics['mIoU']:.4f} | Final mDice: {metrics['mDice']:.4f}")
        logger.info(f"Final mHD95: {metrics['mHD95']:.4f} | Final mASD: {metrics['mASD']:.4f}")
        logger.info("--- CLASS-SPECIFIC FINDINGS ---")
        for c in range(1, global_config.NUM_LABELS):
            name = global_config.CLASS_NAMES[c]
            logger.info(f"Class {c} ({name}) | IoU: {metrics['class_ious'][c]:.4f} | Dice: {metrics['class_dices'][c]:.4f} | HD95: {metrics['class_hd95'][c]:.2f}")
        logger.info(f"Evaluation complete. All clinical artifacts saved in: {eval_dir}")
    except Exception:
        logger.error("CRITICAL ERROR encountered during standalone evaluation:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
