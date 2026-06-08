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
import config
from medpy.metric.binary import hd95, asd
from utils import get_stratified_splits
import logging
from scipy import ndimage
import cv2
from PIL import Image

# --- CLINICAL METRIC ANALYZERS ---

class BoundaryPrecisionAnalyzer:
    def __init__(self, kernel_size=3):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get_boundary_contrast(self, image, mask):
        if not np.any(mask): return 0.0
        # For 2.5D image, use the central slice for boundary analysis
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, 1]
        dilated = cv2.dilate(mask.astype(np.uint8), self.kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), self.kernel, iterations=1)
        outer_edge = dilated - mask.astype(np.uint8)
        inner_edge = mask.astype(np.uint8) - eroded
        outer_vals = image[outer_edge > 0]
        inner_vals = image[inner_edge > 0]
        if len(outer_vals) == 0 or len(inner_vals) == 0: return 0.0
        return np.abs(np.mean(inner_vals) - np.mean(outer_vals))

def get_retina_mask(img):
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, 1] # Use central slice for mask
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0: img_f /= 255.0
    blurred = cv2.GaussianBlur(img_f, (31, 31), 0)
    row_means = np.mean(blurred, axis=1)
    mask_rows = row_means > 0.03
    retina_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if np.any(mask_rows):
        rows = np.where(mask_rows)[0]
        retina_mask[max(0, rows[0]-30):min(img.shape[0], rows[-1]+50), :] = 1
    else:
        retina_mask.fill(1)
    return retina_mask

# --- MODEL EVALUATION PIPELINE ---

def evaluate_model(model_path="best_model.pth", output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    all_files = sorted(os.listdir(config.IMG_DIR))
    if os.path.exists("test_patients.txt"):
        with open("test_patients.txt", "r") as f: test_patients = f.read().splitlines()
    else:
        _, _, test_patients = get_stratified_splits(all_files)
    
    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]

    # [REPLICATION] Scan raw masks for vis indices (Exactly like Test 15)
    logging.info("Replicating Test 15 slice selection (scanning raw masks)...")
    class_max_counts = {1: 0, 2: 0, 3: 0}
    vis_indices = {1: -1, 2: -1, 3: -1}
    for i, m_path in enumerate(test_masks):
        m = np.array(Image.open(m_path))
        for c in [1, 2, 3]:
            cnt = np.sum(m == c)
            if cnt > class_max_counts[c]:
                class_max_counts[c] = cnt
                vis_indices[c] = i
    logging.info(f"Fixed vis indices to match Test 15: {vis_indices}")

    # 2. Model Init
    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True, output_attentions=True)
    
    if os.path.exists(model_path):
        logging.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(config.DEVICE).eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. Setup Dataset
    val_transform = A.Compose([A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1])])
    dataset = OCTDataset(test_imgs, test_masks, processor, transform=val_transform, use_multimodal=config.USE_MULTIMODAL)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. State
    total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
    class_hd95_vals = {1: [], 2: [], 3: []}
    class_asd_vals = {1: [], 2: [], 3: []}
    class_region_counts_gt = {1: [], 2: [], 3: []}
    class_region_counts_pred = {1: [], 2: [], 3: []}
    class_boundary_diffs = {1: [], 2: [], 3: []}
    class_pixel_areas_gt = {1: [], 2: [] , 3: []}
    
    # Store visualization data separately for target indices
    vis_data = {} 

    # 5. Evaluation Loop
    logging.info(f"Evaluating on {len(dataset)} slices...")
    global_idx = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(outputs.logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False)
            
            # [FIX] Better Attention Map: Use Stage 2 (64x64) for better detail than Stage 4 (16x16)
            # attentions is a list of 4 stages. Stage 1 is index 1.
            att_stage = outputs.attentions[1] 
            avg_att = torch.mean(att_stage, dim=1) # Mean over heads
            spatial_att = torch.mean(avg_att, dim=1) # Mean over queries
            grid_size = int(np.sqrt(spatial_att.shape[1]))
            att_maps = spatial_att.view(-1, grid_size, grid_size).cpu().numpy()
            
            if getattr(config, "USE_TTA", False):
                scales = getattr(config, "TTA_SCALES", [1.0])
                all_logits = []
                
                for s in scales:
                    if s != 1.0:
                        scaled_size = (int(config.AUG_SIZE[0] * s), int(config.AUG_SIZE[1] * s))
                        scaled_pixels = torch.nn.functional.interpolate(
                            pixel_values, size=scaled_size, mode="bilinear", align_corners=False
                        )
                    else:
                        scaled_pixels = pixel_values
                    
                    s_outputs = model(pixel_values=scaled_pixels)
                    s_logits = torch.nn.functional.interpolate(
                        s_outputs.logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                    )
                    all_logits.append(s_logits)
                    
                    f_pixels = torch.flip(scaled_pixels, [3])
                    f_outputs = model(pixel_values=f_pixels)
                    f_logits = torch.nn.functional.interpolate(
                        f_outputs.logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                    )
                    uf_logits = torch.flip(f_logits, [3])
                    all_logits.append(uf_logits)
                
                logits = torch.mean(torch.stack(all_logits), dim=0)
            
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            
            # --- MORPHOLOGICAL SHARPENING FOR PED (Class 3) ---
            # SegFormer's bilinear upsampling smooths out sharp peaks. We apply a sharpening 
            # filter to the PED probability map to restore the "spiky" clinical appearance.
            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            for b_idx in range(probs_batch.shape[0]):
                ped_prob = probs_batch[b_idx, 3, :, :]
                sharp_ped = cv2.filter2D(ped_prob, -1, sharpen_kernel)
                probs_batch[b_idx, 3, :, :] = np.clip(sharp_ped, 0, 1)

            # [CLINICAL UPDATE] Threshold-based prediction to maximize recall instead of simple argmax
            preds_batch = np.zeros(probs_batch.shape[0:1] + probs_batch.shape[2:], dtype=np.uint8)
            thresholds = getattr(config, "CLASS_THRESHOLDS", {1: 0.5, 2: 0.5, 3: 0.5})
            
            # Apply in reverse priority. IRF (1) is applied last so it overwrites background/others
            for c in [3, 2, 1]:
                thresh = thresholds.get(c, 0.5)
                preds_batch[probs_batch[:, c] > thresh] = c
            
            if getattr(config, "MIN_REGION_SIZE", 0) > 0:
                cleaned_preds = []
                kernel_3x3 = np.ones((3, 3), np.uint8)
                for b_idx in range(len(preds_batch)):
                    p = preds_batch[b_idx]
                    new_p = np.zeros_like(p)
                    for c in range(1, config.NUM_LABELS):
                        c_mask = (p == c).astype(np.uint8)
                        
                        # --- MORPHOLOGICAL SEPARATION FOR IRF (Class 1) ---
                        # Break thin bridges that cause distinct cysts to merge into one blob.
                        if c == 1:
                            c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel_3x3, iterations=1)
                            
                        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                        for label in range(1, num_labels):
                            if stats[label, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                                new_p[labels_im == label] = c
                    cleaned_preds.append(new_p)
                preds_batch = np.array(cleaned_preds)

        for b_idx in range(len(preds_batch)):
            labels, pred, att_map = labels_batch[b_idx], preds_batch[b_idx], att_maps[b_idx]
            orig_img = orig_img_batch[b_idx]
            
            # Check if this is a target index for visualization
            for c_key, t_idx in vis_indices.items():
                if global_idx == t_idx:
                    vis_data[c_key] = (orig_img, labels, pred, att_map)

            mask = (labels >= 0) & (labels < config.NUM_LABELS)
            total_cm += np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)

            for c in [1, 2, 3]:
                gt_c, pred_c = (labels == c), (pred == c)
                if np.any(gt_c):
                    _, gt_num = ndimage.label(gt_c)
                    class_region_counts_gt[c].append(gt_num)
                    class_boundary_diffs[c].append(BoundaryPrecisionAnalyzer().get_boundary_contrast(orig_img, gt_c))
                    class_pixel_areas_gt[c].append(np.sum(gt_c))
                if np.any(pred_c):
                    _, pred_num = ndimage.label(pred_c)
                    class_region_counts_pred[c].append(pred_num)
                if np.any(gt_c) and np.any(pred_c):
                    class_hd95_vals[c].append(hd95(pred_c, gt_c))
                    class_asd_vals[c].append(asd(pred_c, gt_c))
            global_idx += 1

    # 6. Replicated Visualization Grid
    logging.info("Generating predictions.png (Exact Test 15 indices)...")
    plt.figure(figsize=(22, 16))
    for i, c_id in enumerate([1, 2, 3]):
        if c_id not in vis_data: continue
        img, gt, prd, att = vis_data[c_id]
        
        # Display OCT - Restore colored 2.5D view if applicable
        ax1 = plt.subplot(3, 4, i*4+1)
        ax1.imshow(img) # Restore RGB/2.5D view to match Test 15
        ax1.set_title(f"OCT: {config.CLASS_NAMES[c_id]}", fontsize=14, fontweight='bold')
        ax1.axis("off")
        
        # Display Ground Truth
        ax2 = plt.subplot(3, 4, i*4+2)
        ax2.imshow(gt, cmap="jet", vmin=0, vmax=3)
        ax2.set_title(f"GT (px: {class_max_counts[c_id]})", fontsize=12)
        ax2.axis("off")
        
        # Display Prediction
        tp = np.sum((gt == c_id) & (prd == c_id))
        fp = np.sum((gt != c_id) & (prd == c_id))
        fn = np.sum((gt == c_id) & (prd != c_id))
        c_iou = tp / (tp + fp + fn + 1e-6)
        
        ax3 = plt.subplot(3, 4, i*4+3)
        ax3.imshow(prd, cmap="jet", vmin=0, vmax=3)
        ax3.set_title(f"Prediction (IoU: {c_iou:.2f})", fontsize=12)
        ax3.axis("off")
        
        # Display Shaper Attention Map (Stage 2)
        ax4 = plt.subplot(3, 4, i*4+4)
        att_resized = cv2.resize(att, (img.shape[1], img.shape[0]))
        att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
        att_norm = np.power(att_norm, 0.6) # Even higher contrast for Stage 2
        
        # Use grayscale for attention background for clarity
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
        'class_ious': {c: ious[c] for c in range(4)}, 'class_dices': {c: dices[c] for c in range(4)},
        'class_hd95': {c: safe_mean(class_hd95_vals[c]) for c in [1, 2, 3]},
        'class_asd': {c: safe_mean(class_asd_vals[c]) for c in [1, 2, 3]},
        'class_avg_regions_gt': {c: safe_mean(class_region_counts_gt[c]) for c in [1, 2, 3]},
        'class_avg_regions_pred': {c: safe_mean(class_region_counts_pred[c]) for c in [1, 2, 3]},
        'class_boundary_precision': {c: safe_mean(class_boundary_diffs[c]) for c in [1, 2, 3]},
        'class_avg_pixel_area': {c: safe_mean(class_pixel_areas_gt[c]) for c in [1, 2, 3]},
        'mHD95': safe_mean([safe_mean(class_hd95_vals[c]) for c in [1, 2, 3] if class_hd95_vals[c]]),
        'mASD': safe_mean([safe_mean(class_asd_vals[c]) for c in [1, 2, 3] if class_asd_vals[c]])
    }

if __name__ == "__main__":
    import sys
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_dir = os.path.abspath(f"eval_results_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    log_file = os.path.join(eval_dir, "eval_standalone.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"Starting Standalone Evaluation. Results will be in: {eval_dir}")
    target_model = "best_model.pth"
    if not os.path.exists(target_model):
        logging.error(f"Model {target_model} not found! Please place best_model.pth in the project root."); sys.exit(1)
        
    try:
        metrics = evaluate_model(model_path=target_model, output_dir=eval_dir)
        logging.info("--- EVALUATION RESULTS ---")
        logging.info(f"Final mIoU: {metrics['mIoU']:.4f} | Final mDice: {metrics['mDice']:.4f}")
        logging.info(f"Final mHD95: {metrics['mHD95']:.4f} | Final mASD: {metrics['mASD']:.4f}")
        logging.info("--- CLASS-SPECIFIC FINDINGS ---")
        for c in [1, 2, 3]:
            name = config.CLASS_NAMES[c]
            logging.info(f"Class {c} ({name}) | IoU: {metrics['class_ious'][c]:.4f} | Dice: {metrics['class_dices'][c]:.4f} | HD95: {metrics['class_hd95'][c]:.2f}")
        logging.info(f"Evaluation complete. Artifacts saved in: {eval_dir}")
    except Exception as e:
        import traceback
        logging.error("CRITICAL ERROR in standalone evaluation:")
        logging.error(traceback.format_exc())
