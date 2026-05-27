import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
from medpy.metric.binary import hd95, asd
from scipy import ndimage
import cv2
from dataset import OCTDataset
import logging
import config
from utils import get_stratified_splits
from monai.networks.nets import SwinUNETR

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class BoundaryPrecisionAnalyzer:
    """
    Analyzes boundary precision for fluid segmentation masks.
    Methodology: Anderson et al. 2023.
    """
    def extract_boundaries(self, mask):
        mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        outer_boundary = dilated - mask
        eroded = cv2.erode(mask, kernel, iterations=1)
        inner_boundary = mask - eroded
        return outer_boundary, inner_boundary

    def calculate_metrics(self, image, mask):
        # image expected as 0-1 float
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # If multi-channel, use only the first channel (original OCT)
        if len(image.shape) == 3:
            image = image[:, :, 0]

        # Filter out very small regions (noise) before precision check
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        clean_mask = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= 10:
                clean_mask[labels == label] = 1
        
        if not np.any(clean_mask):
            return 0.0

        outer_b, inner_b = self.extract_boundaries(clean_mask)
        outer_intensities = image[outer_b == 1]
        inner_intensities = image[inner_b == 1]
        
        if len(outer_intensities) > 0 and len(inner_intensities) > 0:
            return float(np.mean(outer_intensities) - np.mean(inner_intensities))
        return 0.0

def evaluate_model(model_path="best_model.pth", output_dir="."):
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
        model = SwinUNETR(
            img_size=config.SWIN_CFG["img_size"],
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
    except RuntimeError as e:
        logging.warning(f"Strict load failed: {e}. Trying non-strict load...")
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True), strict=False)
        logging.warning("Non-strict load complete. Results may be garbage if architectures mismatch!")

    model.to(config.DEVICE).eval()

    val_aug_list = []
    if getattr(config, "USE_CLAHE", False):
        val_aug_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    val_aug_list.append(A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1]))
    val_transform = A.Compose(val_aug_list)

    ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform, use_multimodal=config.USE_MULTIMODAL)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # find best example for EACH class (1: IRF, 2: SRF, 3: PED)
    vis_indices = {}
    class_max_counts = {1: 0, 2: 0, 3: 0}
    logging.info("Scanning for best class-specific examples...")
    for idx, m_path in enumerate(val_masks):
        m = np.array(Image.open(m_path))
        for c in [1, 2, 3]:
            count = np.sum(m == c)
            if count > class_max_counts[c]:
                class_max_counts[c] = count
                vis_indices[c] = idx
    
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
            else:
                outputs = model(pixel_values=pixel_values)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=(512, 512), mode="bilinear", align_corners=False
                )
            
            if config.USE_TTA:
                scales = getattr(config, "TTA_SCALES", [1.0])
                all_logits = []
                
                for s in scales:
                    # Rescale input
                    if s != 1.0:
                        scaled_size = (int(512 * s), int(512 * s))
                        scaled_pixels = torch.nn.functional.interpolate(
                            pixel_values, size=scaled_size, mode="bilinear", align_corners=False
                        )
                    else:
                        scaled_pixels = pixel_values
                    
                    # Original pass at this scale
                    s_outputs = model(pixel_values=scaled_pixels)
                    s_logits = torch.nn.functional.interpolate(
                        s_outputs.logits, size=(512, 512), mode="bilinear", align_corners=False
                    )
                    all_logits.append(s_logits)
                    
                    # Flipped pass at this scale
                    f_pixels = torch.flip(scaled_pixels, [3])
                    f_outputs = model(pixel_values=f_pixels)
                    f_logits = torch.nn.functional.interpolate(
                        f_outputs.logits, size=(512, 512), mode="bilinear", align_corners=False
                    )
                    # Unflip
                    uf_logits = torch.flip(f_logits, [3])
                    all_logits.append(uf_logits)
                
                # Average all TTA passes
                logits = torch.mean(torch.stack(all_logits), dim=0)

            preds_batch = logits.argmax(dim=1).cpu().numpy()

            # Apply Morphological Cleaning (Remove small regions)
            if getattr(config, "MIN_REGION_SIZE", 0) > 0:
                cleaned_preds = []
                for p in preds_batch:
                    new_p = np.zeros_like(p)
                    for c in range(1, config.NUM_LABELS):
                        c_mask = (p == c).astype(np.uint8)
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
                    # Area and Boundary Precision (Teacher's requirements)
                    class_pixel_areas_gt[c].append(int(np.sum(gt_c)))
                    diff = boundary_analyzer.calculate_metrics(orig_img, gt_c)
                    class_boundary_diffs[c].append(diff)

                    try:
                        _, gt_num = ndimage.label(gt_c)
                        class_region_counts_gt[c].append(gt_num)
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

            # Per-image mIoU for failure analysis (only consider classes present in GT or Pred)
            img_cm = np.bincount(config.NUM_LABELS * label_flat + pred_flat, minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)
            img_tp = np.diag(img_cm)
            img_fp = img_cm.sum(axis=0) - img_tp
            img_fn = img_cm.sum(axis=1) - img_tp
            
            # Intersection over Union for each class
            img_iou_vals = img_tp / (img_tp + img_fp + img_fn + 1e-6)
            
            # Find classes that should have been there (GT > 0) or were wrongly predicted (FP > 0)
            relevant_classes = [c for c in range(1, config.NUM_LABELS) if (img_cm[c, :].sum() > 0 or img_cm[:, c].sum() > 0)]
            
            if relevant_classes:
                fluid_miou = np.mean(img_iou_vals[relevant_classes])
            else:
                fluid_miou = 1.0 # Perfect score for background-only slice correctly predicted
            
            image_ious.append((fluid_miou, global_idx - 1, orig_img, labels, pred))

            # Visualization for fixed class-specific indices
            if i in target_indices:
                c_id = [k for k, v in vis_indices.items() if v == i][0]
                pos = list(vis_indices.keys()).index(c_id)
                plt.subplot(3, 3, pos*3 + 1); plt.imshow(orig_img); plt.title(f"OCT (Best {config.CLASS_NAMES[c_id]})"); plt.axis("off")
                plt.subplot(3, 3, pos*3 + 2); plt.imshow(labels, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title(f"GT (px: {class_max_counts[c_id]})"); plt.axis("off")
                plt.subplot(3, 3, pos*3 + 3); plt.imshow(pred, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("Pred"); plt.axis("off")

    vis_path = os.path.join(output_dir, "predictions.png")
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close() # Free memory
    logging.info(f"Saved visualization to {vis_path}")

    # Save Worst Predictions (Failure Analysis)
    failure_dir = os.path.join(output_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)
    image_ious.sort(key=lambda x: x[0]) # Sort by mIoU ascending
    for rank, (iou, idx, img, gt, pr) in enumerate(image_ious[:5]):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(img); plt.title(f"Worst {rank+1} (mIoU: {iou:.3f})"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(gt, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("Ground Truth"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(pr, cmap="jet", vmin=0, vmax=config.NUM_LABELS-1); plt.title("Prediction"); plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(failure_dir, f"failure_{rank+1}_idx_{idx}.png"))
        plt.close()
    logging.info(f"Saved 5 worst failure cases to {failure_dir}")

    # Calculate metrics from confusion matrix
    tp = np.diag(total_cm)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    
    metrics = {
        "class_ious": {}, "class_dices": {}, "class_hd95": {}, "class_asd": {}, 
        "class_avg_regions_gt": {}, "class_avg_regions_pred": {},
        "class_boundary_precision": {}, "class_avg_pixel_area": {}
    }
    for c in range(config.NUM_LABELS):
        denominator_iou = tp[c] + fp[c] + fn[c]
        iou = tp[c] / denominator_iou if denominator_iou > 0 else 1.0
        
        denominator_dice = 2 * tp[c] + fp[c] + fn[c]
        dice = (2 * tp[c]) / denominator_dice if denominator_dice > 0 else 1.0
        
        metrics["class_ious"][c] = float(iou)
        metrics["class_dices"][c] = float(dice)

        if c in [1, 2, 3]:
            metrics["class_hd95"][c] = float(np.mean(class_hd95_vals[c])) if class_hd95_vals[c] else 100.0
            metrics["class_asd"][c] = float(np.mean(class_asd_vals[c])) if class_asd_vals[c] else 50.0
            metrics["class_avg_regions_gt"][c] = float(np.mean(class_region_counts_gt[c])) if class_region_counts_gt[c] else 0.0
            metrics["class_avg_regions_pred"][c] = float(np.mean(class_region_counts_pred[c])) if class_region_counts_pred[c] else 0.0
            metrics["class_boundary_precision"][c] = float(np.mean(class_boundary_diffs[c])) if class_boundary_diffs[c] else 0.0
            metrics["class_avg_pixel_area"][c] = float(np.mean(class_pixel_areas_gt[c])) if class_pixel_areas_gt[c] else 0.0
    
    metrics["mIoU"] = float(np.mean(list(metrics["class_ious"].values())))
    metrics["mDice"] = float(np.mean(list(metrics["class_dices"].values())))
    metrics["mHD95"] = float(np.mean(list(metrics["class_hd95"].values())))
    metrics["mASD"] = float(np.mean(list(metrics["class_asd"].values())))
    
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = evaluate_model()
    print(m)
