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

class BoundaryPrecisionAnalyzer:
    """Analytical tool for measuring boundary clinical correctness (Anderson et al. 2023)"""
    def __init__(self, kernel_size=3):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get_boundary_contrast(self, image, mask):
        """Measures average intensity difference between inner and outer boundary"""
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
    """Anatomical prior: identifies the retina zone to filter out non-anatomical noise"""
    # Ensure img is float32 for processing
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= 255.0
    
    # Use a large blur to capture the general 'mass' of the eye tissue
    blurred = cv2.GaussianBlur(img_f, (31, 31), 0)
    row_means = np.mean(blurred, axis=1)
    
    # Threshold to find non-black regions
    # 0.03 is a safe floor for RETOUCH normalized scans
    mask_rows = row_means > 0.03
    
    retina_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if np.any(mask_rows):
        # Find first and last rows with signal
        rows = np.where(mask_rows)[0]
        start_row = max(0, rows[0] - 30) # Margin for Vitreous-Retina interface
        end_row = min(img.shape[0], rows[-1] + 50) # Margin for Choroid
        retina_mask[start_row:end_row, :] = 1
    else:
        # Fallback to full image if no signal detected (rare)
        retina_mask.fill(1)
    return retina_mask

def evaluate_model(model_path="best_model.pth", output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    device = config.DEVICE

    # Load test set
    all_files = sorted(os.listdir(config.IMG_DIR))
    if os.path.exists("test_patients.txt"):
        with open("test_patients.txt", "r") as f:
            test_patients = f.read().splitlines()
        logging.info("Loaded test set from test_patients.txt")
    else:
        _, _, test_patients = get_stratified_splits(all_files)
        logging.info("Recreated stratified test split")

    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]

    val_imgs, val_masks = test_imgs, test_masks # Evaluate on TEST SET now

    if "monai" not in config.MODEL_NAME:
        processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True)
    else:
        processor = None
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            in_channels=config.SWIN_CFG["in_channels"],
            out_channels=config.SWIN_CFG["out_channels"],
            feature_size=config.SWIN_CFG["feature_size"],
            drop_rate=config.SWIN_CFG["drop_rate"],
            attn_drop_rate=config.SWIN_CFG["attn_drop_rate"],
            spatial_dims=2
        )
    
    try:
        # Load weights
        state_dict = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=not "monai" in config.MODEL_NAME)
        logging.info(f"Loaded weights from {model_path}")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")

    model.to(device)
    model.eval()

    # Calculate parameter count (Requirement for Table in Thesis)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model Parameters: {total_params:.2f}M")

    # Final Eval Transform Setup (Match Validation logic in train.py)
    import albumentations as A
    eval_aug_list = []
    if getattr(config, "USE_CLAHE", False):
        eval_aug_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    eval_aug_list.append(A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1]))
    eval_transform = A.Compose(eval_aug_list)

    dataset = OCTDataset(val_imgs, val_masks, processor, transform=eval_transform, use_multimodal=config.USE_MULTIMODAL)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 1. Pre-scan to find best examples for each class
    logging.info("Scanning for best class-specific examples...")
    class_max_counts = {1: 0, 2: 0, 3: 0}
    vis_indices = {1: -1, 2: -1, 3: -1}
    for i in range(len(dataset)):
        m = dataset.get_raw_mask(i)
        for c in [1, 2, 3]:
            cnt = np.sum(m == c)
            if cnt > class_max_counts[c]:
                class_max_counts[c] = cnt
                vis_indices[c] = i
    
    target_indices = list(vis_indices.values())
    logging.info(f"Fixed vis indices: {vis_indices}")

    total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
    plt.figure(figsize=(15, 12))

    logging.info(f"Evaluating on {len(loader)} batches...")
    logging.info(f"Config: MODEL_NAME={config.MODEL_NAME}, USE_MULTIMODAL={config.USE_MULTIMODAL}, DEVICE={config.DEVICE}")
    
    class_hd95_vals = {1: [], 2: [], 3: []}
    class_asd_vals = {1: [], 2: [], 3: []}
    class_region_counts_gt = {1: [], 2: [], 3: []}
    class_region_counts_pred = {1: [], 2: [], 3: []}
    class_boundary_diffs = {1: [], 2: [], 3: []} # Teacher's metric
    class_pixel_areas_gt = {1: [], 2: [], 3: []} # Teacher's metric
    
    boundary_analyzer = BoundaryPrecisionAnalyzer()
    image_ious = [] # Track per-image IoU for failure analysis

    global_idx = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            if "monai" in config.MODEL_NAME:
                logits = model(pixel_values)
                if logits.shape[-2:] != config.AUG_SIZE:
                    logits = torch.nn.functional.interpolate(
                        logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                    )
            else:
                outputs = model(pixel_values=pixel_values)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                )
            
            if config.USE_TTA:
                scales = getattr(config, "TTA_SCALES", [1.0])
                all_logits = []
                
                for s in scales:
                    # Rescale input
                    if s != 1.0:
                        new_h = int(config.AUG_SIZE[0] * s)
                        new_w = int(config.AUG_SIZE[1] * s)
                        # SwinUNETR requirements: must be divisible by 32
                        if "monai" in config.MODEL_NAME:
                            new_h = (new_h // 32) * 32
                            new_w = (new_w // 32) * 32
                        
                        scaled_size = (new_h, new_w)
                        scaled_pixels = torch.nn.functional.interpolate(
                            pixel_values, size=scaled_size, mode="bilinear", align_corners=False
                        )
                    else:
                        scaled_pixels = pixel_values
                    
                    # Original pass at this scale
                    if "monai" in config.MODEL_NAME:
                        s_logits = model(scaled_pixels)
                        if s_logits.shape[-2:] != config.AUG_SIZE:
                            s_logits = torch.nn.functional.interpolate(
                                s_logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                            )
                    else:
                        s_outputs = model(pixel_values=scaled_pixels)
                        s_logits = torch.nn.functional.interpolate(
                            s_outputs.logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                        )
                    all_logits.append(s_logits)
                    
                    # Flipped pass at this scale
                    f_pixels = torch.flip(scaled_pixels, [3])
                    if "monai" in config.MODEL_NAME:
                        f_logits = model(f_pixels)
                        if f_logits.shape[-2:] != config.AUG_SIZE:
                            f_logits = torch.nn.functional.interpolate(
                                f_logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                            )
                    else:
                        f_outputs = model(pixel_values=f_pixels)
                        f_logits = torch.nn.functional.interpolate(
                            f_outputs.logits, size=config.AUG_SIZE, mode="bilinear", align_corners=False
                        )
                    # Unflip
                    uf_logits = torch.flip(f_logits, [3])
                    all_logits.append(uf_logits)
                
                # Average all TTA passes
                logits = torch.mean(torch.stack(all_logits), dim=0)

            # Threshold-based assignment instead of simple argmax
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            preds_batch = np.zeros(probs_batch.shape[0:1] + probs_batch.shape[2:], dtype=np.uint8)
            
            # Default thresholds or from config
            thresholds = getattr(config, "CLASS_THRESHOLDS", {1: 0.5, 2: 0.5, 3: 0.5})
            
            # Apply classes in reverse priority (IRF class 1 has final say)
            for c in [3, 2, 1]:
                thresh = thresholds.get(c, 0.5)
                preds_batch[probs_batch[:, c] > thresh] = c

            # Apply Morphological Cleaning & Smoothing
            if getattr(config, "MIN_REGION_SIZE", 0) > 0:
                cleaned_preds = []
                kernel = np.ones((3, 3), np.uint8)
                for b_idx in range(len(preds_batch)):
                    p = preds_batch[b_idx]
                    img_for_mask = orig_img_batch[b_idx]
                    
                    # Generate and apply anatomical retina mask
                    ret_mask = get_retina_mask(img_for_mask)
                    
                    new_p = np.zeros_like(p)
                    for c in range(1, config.NUM_LABELS):
                        c_mask = (p == c).astype(np.uint8)
                        
                        # Apply anatomical prior
                        c_mask = c_mask * ret_mask

                        # Selective Smoothing based on clinical morphology
                        if getattr(config, "USE_MORPH_SMOOTHING", True):
                            # Always close holes
                            c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
                            
                            # Only open (smooth spikes) for IRF (Class 1)
                            # PED (3) and SRF (2) should remain 'spiky' as per clinical reality
                            if c == 1:
                                c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)

                        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=8)
                        for label in range(1, num_labels):
                            if stats[label, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                                new_p[labels_im == label] = c
                    cleaned_preds.append(new_p)
                preds_batch = np.array(cleaned_preds)

        for b_idx in range(len(preds_batch)):
            labels = labels_batch[b_idx]
            pred = preds_batch[b_idx]
            orig_img = orig_img_batch[b_idx]
            i = global_idx
            global_idx += 1

            # Surface metrics for classes 1, 2, 3
            for c in [1, 2, 3]:
                gt_c = (labels == c)
                pred_c = (pred == c)
                
                if np.any(gt_c):
                    try:
                        _, gt_num = ndimage.label(gt_c)
                        class_region_counts_gt[c].append(gt_num)
                        
                        # Teacher's metric: Area and BP
                        diff = boundary_analyzer.get_boundary_contrast(orig_img, gt_c)
                        class_boundary_diffs[c].append(diff)
                        class_pixel_areas_gt[c].append(np.sum(gt_c))
                    except NameError:
                        pass
                        
                if np.any(pred_c):
                    try:
                        _, pred_num = ndimage.label(pred_c)
                        class_region_counts_pred[c].append(pred_num)
                    except NameError:
                        pass
                        
                if np.any(gt_c) and np.any(pred_c):
                    class_hd95_vals[c].append(hd95(pred_c, gt_c))
                    class_asd_vals[c].append(asd(pred_c, gt_c))

            # Incremental confusion matrix update
            mask = (labels >= 0) & (labels < config.NUM_LABELS)
            label_flat = labels[mask].astype(np.int64)
            pred_flat = pred[mask].astype(np.int64)
            
            total_cm += np.bincount(
                config.NUM_LABELS * label_flat + pred_flat,
                minlength=config.NUM_LABELS**2
            ).reshape(config.NUM_LABELS, config.NUM_LABELS)

            # Per-image mIoU for failure analysis
            img_cm = np.bincount(config.NUM_LABELS * label_flat + pred_flat, minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)
            img_tp = np.diag(img_cm)
            img_fp = img_cm.sum(axis=0) - img_tp
            img_fn = img_cm.sum(axis=1) - img_tp
            img_iou_vals = img_tp / (img_tp + img_fp + img_fn + 1e-6)
            
            relevant_classes = [c for c in range(1, config.NUM_LABELS) if (img_cm[c, :].sum() > 0 or img_cm[:, c].sum() > 0)]
            if relevant_classes:
                fluid_miou = np.mean(img_iou_vals[relevant_classes])
            else:
                fluid_miou = 1.0
            
            image_ious.append((fluid_miou, global_idx - 1, orig_img, labels, pred))

            # Visualization for fixed class-specific indices
            if i in target_indices:
                c_id = [k for k, v in vis_indices.items() if v == i][0]
                pos = list(vis_indices.keys()).index(c_id)
                plt.subplot(3, 3, pos*3 + 1); plt.imshow(orig_img); plt.title(f"OCT (Best {config.CLASS_NAMES[c_id]})"); plt.axis("off")
                plt.subplot(3, 3, pos*3 + 2); plt.imshow(labels, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title(f"GT (px: {class_max_counts[c_id]})"); plt.axis("off")
                plt.subplot(3, 3, pos*3 + 3); plt.imshow(pred, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title(f"Pred (IoU: {img_iou_vals[c_id]:.2f})"); plt.axis("off")

    # Calculate final metrics
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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions.png"))
    plt.close()

    # Save failure cases
    fail_dir = os.path.join(output_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)
    image_ious.sort(key=lambda x: x[0]) # Sort by mIoU ascending
    for idx, (iou_val, g_idx, img, gt, prd) in enumerate(image_ious[:5]):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(img); plt.title("OCT Scan"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(gt, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("GT Mask"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(prd, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title(f"Pred (mIoU: {iou_val:.2f})"); plt.axis("off")
        plt.savefig(os.path.join(fail_dir, f"failure_{idx+1}_idx_{g_idx}.png"))
        plt.close()

    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = evaluate_model()
    print(m)
