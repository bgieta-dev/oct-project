import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend for server/CLI usage
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

# --- CLINICAL METRIC ANALYZERS ---

class BoundaryPrecisionAnalyzer:
    def __init__(self, kernel_size=3):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get_boundary_contrast(self, image, mask):
        if not np.any(mask): return 0.0
        dilated = cv2.dilate(mask.astype(np.uint8), self.kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), self.kernel, iterations=1)
        outer_edge = dilated - mask.astype(np.uint8)
        inner_edge = mask.astype(np.uint8) - eroded
        outer_vals = image[outer_edge > 0]
        inner_vals = image[inner_edge > 0]
        if len(outer_vals) == 0 or len(inner_vals) == 0: return 0.0
        return np.abs(np.mean(inner_vals) - np.mean(outer_vals))

def get_retina_mask(img):
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
    image_metrics = [] # Stores: (iou, idx, img, gt, pred, attention)

    # 5. Evaluation Loop
    logging.info(f"Evaluating on {len(dataset)} slices...")
    global_idx = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(outputs.logits, size=config.AUG_SIZE, mode="bilinear")
            attentions = outputs.attentions[-1]
            
            # Extract spatial attention
            avg_att = torch.mean(attentions, dim=1)
            spatial_att = torch.mean(avg_att, dim=1)
            grid_size = int(np.sqrt(spatial_att.shape[1]))
            att_maps = spatial_att.view(-1, grid_size, grid_size).cpu().numpy()
            
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            preds_batch = np.argmax(probs_batch, axis=1).astype(np.uint8)
            
            cleaned_preds = []
            for b_idx in range(len(preds_batch)):
                p = preds_batch[b_idx]
                ret_mask = get_retina_mask(orig_img_batch[b_idx])
                new_p = np.zeros_like(p)
                for c in range(1, config.NUM_LABELS):
                    c_mask = ((p == c).astype(np.uint8) * ret_mask)
                    if config.MIN_REGION_SIZE > 0:
                        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                        for label in range(1, num_labels):
                            if stats[label, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                                new_p[labels_im == label] = c
                    else:
                        new_p[c_mask > 0] = c
                cleaned_preds.append(new_p)
            preds_batch = np.array(cleaned_preds)

        for b_idx in range(len(preds_batch)):
            labels, pred, att_map = labels_batch[b_idx], preds_batch[b_idx], att_maps[b_idx]
            orig_img = orig_img_batch[b_idx]
            central_img = orig_img[:, :, 1] if config.USE_25D else orig_img[:, :, 0]
            
            mask = (labels >= 0) & (labels < config.NUM_LABELS)
            total_cm += np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)
            
            # Slice-level metric for ranking
            img_counts = np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)
            img_tp = np.diag(img_counts)
            img_fp = img_counts.sum(axis=0) - img_tp
            img_fn = img_counts.sum(axis=1) - img_tp
            img_iou = np.mean(img_tp[1:] / (img_tp[1:] + img_fp[1:] + img_fn[1:] + 1e-6))
            
            # Store data for visualization (Keep memory lean: only 1 in 5 or specific ones if needed)
            # For small datasets, we store all to pick best.
            image_metrics.append((img_iou, global_idx, orig_img, labels, pred, att_map))

            for c in [1, 2, 3]:
                gt_c, pred_c = (labels == c), (pred == c)
                if np.any(gt_c):
                    _, gt_num = ndimage.label(gt_c)
                    class_region_counts_gt[c].append(gt_num)
                    class_boundary_diffs[c].append(BoundaryPrecisionAnalyzer().get_boundary_contrast(central_img, gt_c))
                    class_pixel_areas_gt[c].append(np.sum(gt_c))
                if np.any(pred_c):
                    _, pred_num = ndimage.label(pred_c)
                    class_region_counts_pred[c].append(pred_num)
                if np.any(gt_c) and np.any(pred_c):
                    class_hd95_vals[c].append(hd95(pred_c, gt_c))
                    class_asd_vals[c].append(asd(pred_c, gt_c))
            global_idx += 1

    # 6. Visualization - Pick BEST examples from the FULL test set
    logging.info("Generating visualizations...")
    vis_indices = {}
    for c in [1, 2, 3]:
        # Find index with MOST pixels for class c to show clear pathology
        best_pos = -1
        max_px = 0
        for i, m in enumerate(image_metrics):
            px_count = np.sum(m[3] == c)
            if px_count > max_px:
                max_px = px_count
                best_pos = i
        if best_pos != -1:
            vis_indices[c] = best_pos

    if not vis_indices:
        logging.warning("No pathology found in test set! Picking first 3 images for predictions.png")
        for i in range(min(3, len(image_metrics))): vis_indices[i+1] = i

    plt.figure(figsize=(16, 12))
    for i, (c_id, list_pos) in enumerate(vis_indices.items()):
        iou, _, img, gt, prd, att = image_metrics[list_pos]
        vis_img = img[:, :, 1] if config.USE_25D else img[:, :, 0]
        
        plt.subplot(3, 4, i*4+1); plt.imshow(vis_img, cmap="gray"); plt.axis("off"); plt.title(f"OCT: {config.CLASS_NAMES[c_id]}")
        plt.subplot(3, 4, i*4+2); plt.imshow(gt, cmap="jet", vmin=0, vmax=3); plt.axis("off"); plt.title("Ground Truth")
        plt.subplot(3, 4, i*4+3); plt.imshow(prd, cmap="jet", vmin=0, vmax=3); plt.axis("off"); plt.title(f"Pred (IoU: {iou:.2f})")
        
        # Overlay Attention
        att_norm = (cv2.resize(att, (vis_img.shape[1], vis_img.shape[0])) - att.min()) / (att.max() - att.min() + 1e-8)
        plt.subplot(3, 4, i*4+4); plt.imshow(vis_img, cmap="gray"); plt.imshow(att_norm, cmap="jet", alpha=0.5); plt.axis("off"); plt.title("Attention Map")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150)
    plt.close()

    # 7. Failure Analysis (Top 5 worst)
    fail_dir = os.path.join(output_dir, "failures"); os.makedirs(fail_dir, exist_ok=True)
    image_metrics.sort(key=lambda x: x[0]) # Sort by IoU ascending
    for i, (iou_val, _, img, gt, prd, _) in enumerate(image_metrics[:5]):
        plt.figure(figsize=(12, 4))
        vis_img = img[:, :, 1] if config.USE_25D else img[:, :, 0]
        plt.subplot(1,3,1); plt.imshow(vis_img, cmap="gray"); plt.axis("off"); plt.title("OCT")
        plt.subplot(1,3,2); plt.imshow(gt, cmap="jet", vmin=0, vmax=3); plt.axis("off"); plt.title("Ground Truth")
        plt.subplot(1,3,3); plt.imshow(prd, cmap="jet", vmin=0, vmax=3); plt.axis("off"); plt.title(f"Pred (IoU: {iou_val:.2f})")
        plt.savefig(os.path.join(fail_dir, f"failure_{i+1}_iou_{iou_val:.2f}.png")); plt.close()

    # 8. Aggregate Metrics
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
