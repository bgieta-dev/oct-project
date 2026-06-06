import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    """
    Analytical tool for measuring boundary clinical correctness (Anderson et al. 2023).
    Evaluates the intensity contrast across the predicted segmentation boundary.
    """
    def __init__(self, kernel_size=3):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get_boundary_contrast(self, image, mask):
        """
        Measures average intensity difference between inner and outer boundary.
        High contrast indicates better alignment with physical tissue interfaces.
        """
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
    """
    Anatomical Prior: Generates a heuristic mask identifying the retina zone.
    Used to filter out non-anatomical noise in vitreous and sclera regions.
    """
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= 255.0
    
    blurred = cv2.GaussianBlur(img_f, (31, 31), 0)
    row_means = np.mean(blurred, axis=1)
    
    mask_rows = row_means > 0.03
    
    retina_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if np.any(mask_rows):
        rows = np.where(mask_rows)[0]
        start_row = max(0, rows[0] - 30) 
        end_row = min(img.shape[0], rows[-1] + 50) 
        retina_mask[start_row:end_row, :] = 1
    else:
        retina_mask.fill(1)
    return retina_mask

# --- MODEL EVALUATION PIPELINE ---

def evaluate_model(model_path="best_model.pth", output_dir="."):
    """
    Comprehensive evaluation pipeline.
    Calculates Dice, IoU, HD95, ASD and clinical boundary metrics.
    Supports multi-scale TTA, edge-aware bilateral smoothing, and Attention visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = config.DEVICE

    # 1. Load Data Splits
    all_files = sorted(os.listdir(config.IMG_DIR))
    if os.path.exists("test_patients.txt"):
        with open("test_patients.txt", "r") as f:
            test_patients = f.read().splitlines()
        logging.info("Loaded test set from test_patients.txt")
    else:
        _, _, test_patients = get_stratified_splits(all_files)

    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]

    # 2. Initialize Model and Processor (with Attention support)
    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS, 
        ignore_mismatched_sizes=True,
        output_attentions=True # Required for Chapter 4 visualization
    )
    
    state_dict = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. Setup Evaluation Pipeline
    import albumentations as A
    eval_aug_list = []
    if getattr(config, "USE_CLAHE", False):
        eval_aug_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    eval_aug_list.append(A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1]))
    eval_transform = A.Compose(eval_aug_list)

    dataset = OCTDataset(test_imgs, test_masks, processor, transform=eval_transform, use_multimodal=config.USE_MULTIMODAL)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. Metric Accumulators & Visualization State
    total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
    class_hd95_vals = {1: [], 2: [], 3: []}
    class_asd_vals = {1: [], 2: [], 3: []}
    class_region_counts_gt = {1: [], 2: [], 3: []}
    class_region_counts_pred = {1: [], 2: [], 3: []}
    class_boundary_diffs = {1: [], 2: [], 3: []}
    class_pixel_areas_gt = {1: [], 2: [], 3: []}
    
    boundary_analyzer = BoundaryPrecisionAnalyzer()
    image_metrics = [] # (fluid_miou, g_idx, img, labels, pred, attention_map)

    # Pre-scan for best visualization targets
    logging.info("Scanning for best pathology examples...")
    vis_indices = {}
    class_max_counts = {1: 0, 2: 0, 3: 0}
    for idx in range(len(dataset)):
        m = dataset.get_raw_mask(idx)
        for c in [1, 2, 3]:
            cnt = np.sum(m == c)
            if cnt > class_max_counts[c]:
                class_max_counts[c] = cnt
                vis_indices[c] = idx
    target_indices = set(vis_indices.values())

    # 5. Evaluation Loop
    logging.info(f"Evaluating on {len(loader)} batches with TTA={config.USE_TTA}...")
    global_idx = 0
    
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            if config.USE_TTA:
                scales = getattr(config, "TTA_SCALES", [1.0])
                all_logits = []
                all_attentions = []
                for s in scales:
                    scaled_pixels = torch.nn.functional.interpolate(pixel_values, scale_factor=s, mode="bilinear") if s != 1.0 else pixel_values
                    s_outputs = model(pixel_values=scaled_pixels)
                    s_logits = torch.nn.functional.interpolate(s_outputs.logits, size=config.AUG_SIZE, mode="bilinear")
                    all_logits.append(s_logits)
                    all_attentions.append(s_outputs.attentions[-1]) # Use last stage attention
                    
                    f_outputs = model(pixel_values=torch.flip(scaled_pixels, [3]))
                    f_logits = torch.nn.functional.interpolate(f_outputs.logits, size=config.AUG_SIZE, mode="bilinear")
                    all_logits.append(torch.flip(f_logits, [3]))
                logits = torch.mean(torch.stack(all_logits), dim=0)
                attentions = all_attentions[0] # Take first scale attention for simplicity
            else:
                outputs = model(pixel_values=pixel_values)
                logits = torch.nn.functional.interpolate(outputs.logits, size=config.AUG_SIZE, mode="bilinear")
                attentions = outputs.attentions[-1]

            # Extract spatial attention map (Average over heads and keys)
            # attentions shape: (batch, heads, seq_len, seq_len)
            avg_att = torch.mean(attentions, dim=1) # (batch, seq_len, seq_len)
            spatial_att = torch.mean(avg_att, dim=1) # (batch, seq_len)
            grid_size = int(np.sqrt(spatial_att.shape[1]))
            att_maps = spatial_att.view(-1, grid_size, grid_size).cpu().numpy()

            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            
            # 6. Post-Processing
            if getattr(config, "USE_SOFT_CRF", True):
                for b_idx in range(len(probs_batch)):
                    guide_img = (orig_img_batch[b_idx, :, :, 1] if config.USE_25D else orig_img_batch[b_idx, :, :, 0])
                    guide_uint8 = (guide_img * 255).astype(np.uint8)
                    for c in [0, 1]:
                        prob_uint8 = (probs_batch[b_idx, c] * 255).astype(np.uint8)
                        probs_batch[b_idx, c] = cv2.bilateralFilter(prob_uint8, 9, 75, 75).astype(np.float32) / 255.0

            preds_batch = np.argmax(probs_batch, axis=1).astype(np.uint8)
            cleaned_preds = []
            for b_idx in range(len(preds_batch)):
                p = preds_batch[b_idx]
                ret_mask = get_retina_mask(orig_img_batch[b_idx])
                new_p = np.zeros_like(p)
                for c in range(1, config.NUM_LABELS):
                    c_mask = ((p == c).astype(np.uint8) * ret_mask)
                    
                    # [RELAXED] Removing morphology to let B3's high-frequency details shine.
                    # No MORPH_CLOSE, No MORPH_OPEN.
                    
                    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                    for label in range(1, num_labels):
                        if stats[label, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                            new_p[labels_im == label] = c
                cleaned_preds.append(new_p)
            preds_batch = np.array(cleaned_preds)

        # 7. Metric Accumulation & Tracking
        for b_idx in range(len(preds_batch)):
            labels = labels_batch[b_idx]
            pred = preds_batch[b_idx]
            orig_img = orig_img_batch[b_idx]
            att_map = att_maps[b_idx]
            
            mask = (labels >= 0) & (labels < config.NUM_LABELS)
            total_cm += np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)
            
            # Fluid mIoU calculation for failure/vis tracking
            img_tp = np.diag(np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS))
            img_fp = np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS).sum(axis=0) - img_tp
            img_fn = np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS).sum(axis=1) - img_tp
            img_iou = np.mean(img_tp[1:] / (img_tp[1:] + img_fp[1:] + img_fn[1:] + 1e-6))
            
            image_metrics.append((img_iou, global_idx, orig_img, labels, pred, att_map))
            
            central_img = orig_img[:, :, 1] if config.USE_25D else orig_img[:, :, 0]
            for c in [1, 2, 3]:
                gt_c = (labels == c)
                pred_c = (pred == c)
                if np.any(gt_c):
                    _, gt_num = ndimage.label(gt_c)
                    class_region_counts_gt[c].append(gt_num)
                    class_boundary_diffs[c].append(boundary_analyzer.get_boundary_contrast(central_img, gt_c))
                    class_pixel_areas_gt[c].append(np.sum(gt_c))
                if np.any(pred_c):
                    _, pred_num = ndimage.label(pred_c)
                    class_region_counts_pred[c].append(pred_num)
                if np.any(gt_c) and np.any(pred_c):
                    class_hd95_vals[c].append(hd95(pred_c, gt_c))
                    class_asd_vals[c].append(asd(pred_c, gt_c))
            global_idx += 1

    # 8. Final Visualization (Predictions vs GT vs Attention)
    plt.figure(figsize=(16, 12))
    for c_id, target_idx in vis_indices.items():
        # Find the cached metric for this index
        match = [m for m in image_metrics if m[1] == target_idx][0]
        _, _, img, gt, prd, att = match
        pos = c_id - 1
        
        vis_img = img[:, :, 1] if config.USE_25D else img[:, :, 0]
        
        # Plot OCT, GT, Pred, and Attention Map
        plt.subplot(3, 4, pos*4 + 1); plt.imshow(vis_img); plt.title(f"OCT (Best {config.CLASS_NAMES[c_id]})"); plt.axis("off")
        plt.subplot(3, 4, pos*4 + 2); plt.imshow(gt, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("GT Mask"); plt.axis("off")
        plt.subplot(3, 4, pos*4 + 3); plt.imshow(prd, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("Prediction"); plt.axis("off")
        
        # Attention Heatmap Overlay
        att_resized = cv2.resize(att, (512, 512))
        att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
        plt.subplot(3, 4, pos*4 + 4); plt.imshow(vis_img, cmap="gray"); plt.imshow(att_norm, cmap="jet", alpha=0.5); plt.title("Attention Map"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions.png"))
    plt.close()

    # 9. Save Failure Cases
    fail_dir = os.path.join(output_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)
    image_metrics.sort(key=lambda x: x[0])
    for i, (iou_val, g_idx, img, gt, prd, _) in enumerate(image_metrics[:5]):
        plt.figure(figsize=(12, 4))
        vis_img = img[:, :, 1] if config.USE_25D else img[:, :, 0]
        plt.subplot(1, 3, 1); plt.imshow(vis_img); plt.title("OCT Scan"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(gt, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("GT Mask"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(prd, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title(f"Pred (mIoU: {iou_val:.2f})"); plt.axis("off")
        plt.savefig(os.path.join(fail_dir, f"failure_{i+1}_idx_{g_idx}.png"))
        plt.close()

    # 10. Final Report Generation
    tp = np.diag(total_cm)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    ious = tp / (tp + fp + fn + 1e-6)
    dices = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    
    metrics = {
        'params': total_params,
        'mIoU': np.mean(ious),
        'mDice': np.mean(dices),
        'class_ious': {c: ious[c] for c in range(config.NUM_LABELS)},
        'class_dices': {c: dices[c] for c in range(config.NUM_LABELS)},
        'class_hd95': {c: np.mean(class_hd95_vals[c]) if class_hd95_vals[c] else 0.0 for c in [1, 2, 3]},
        'class_asd': {c: np.mean(class_asd_vals[c]) if class_asd_vals[c] else 0.0 for c in [1, 2, 3]},
        'class_avg_regions_gt': {c: np.mean(class_region_counts_gt[c]) if class_region_counts_gt[c] else 0.0 for c in [1, 2, 3]},
        'class_avg_regions_pred': {c: np.mean(class_region_counts_pred[c]) if class_region_counts_pred[c] else 0.0 for c in [1, 2, 3]},
        'class_boundary_precision': {c: np.mean(class_boundary_diffs[c]) if class_boundary_diffs[c] else 0.0 for c in [1, 2, 3]},
        'class_avg_pixel_area': {c: np.mean(class_pixel_areas_gt[c]) if class_pixel_areas_gt[c] else 0.0 for c in [1, 2, 3]},
        'mHD95': np.mean([np.mean(class_hd95_vals[c]) for c in [1, 2, 3] if class_hd95_vals[c]]),
        'mASD': np.mean([np.mean(class_asd_vals[c]) for c in [1, 2, 3] if class_asd_vals[c]]),
    }

    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(evaluate_model())
