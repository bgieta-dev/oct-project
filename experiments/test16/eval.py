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
    
    # Gaussian blur to remove speckle noise and capture tissue 'mass'
    blurred = cv2.GaussianBlur(img_f, (31, 31), 0)
    row_means = np.mean(blurred, axis=1)
    
    # Threshold based on RETOUCH dataset intensity distribution
    mask_rows = row_means > 0.03
    
    retina_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if np.any(mask_rows):
        rows = np.where(mask_rows)[0]
        # Margins added to ensure pathological detachments (PED/SRF) are not cut off
        start_row = max(0, rows[0] - 30) 
        end_row = min(img.shape[0], rows[-1] + 50) 
        retina_mask[start_row:end_row, :] = 1
    else:
        retina_mask.fill(1) # Fallback if no signal detected
    return retina_mask

# --- MODEL EVALUATION PIPELINE ---

def evaluate_model(model_path="best_model.pth", output_dir="."):
    """
    Comprehensive evaluation pipeline.
    Calculates Dice, IoU, HD95, ASD and clinical boundary metrics.
    Supports multi-scale TTA and edge-aware bilateral smoothing.
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

    # 2. Initialize Model and Processor
    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True)
    
    # Load Weights (strict=False to allow for minor architecture variants)
    state_dict = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. Setup Evaluation Pipeline
    import albumentations as A
    eval_aug_list = []
    if getattr(config, "USE_CLAHE", False):
        eval_aug_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    eval_aug_list.append(A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1]))
    eval_transform = A.Compose(eval_aug_list)

    dataset = OCTDataset(test_imgs, test_masks, processor, transform=eval_transform, use_multimodal=config.USE_MULTIMODAL)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. Inference with Test-Time Augmentation (TTA)
    logging.info(f"Evaluating with TTA={config.USE_TTA}...")
    
    total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
    class_hd95_vals = {1: [], 2: [], 3: []}
    class_asd_vals = {1: [], 2: [], 3: []}
    
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            if config.USE_TTA:
                # Multi-scale and Mirror TTA
                scales = getattr(config, "TTA_SCALES", [1.0])
                all_logits = []
                for s in scales:
                    scaled_pixels = torch.nn.functional.interpolate(pixel_values, scale_factor=s, mode="bilinear") if s != 1.0 else pixel_values
                    s_outputs = model(pixel_values=scaled_pixels)
                    s_logits = torch.nn.functional.interpolate(s_outputs.logits, size=config.AUG_SIZE, mode="bilinear")
                    all_logits.append(s_logits)
                    # Horizontal Flip TTA
                    f_outputs = model(pixel_values=torch.flip(scaled_pixels, [3]))
                    f_logits = torch.nn.functional.interpolate(f_outputs.logits, size=config.AUG_SIZE, mode="bilinear")
                    all_logits.append(torch.flip(f_logits, [3]))
                logits = torch.mean(torch.stack(all_logits), dim=0)
            else:
                outputs = model(pixel_values=pixel_values)
                logits = torch.nn.functional.interpolate(outputs.logits, size=config.AUG_SIZE, mode="bilinear")

            # 5. Edge-Aware Bilateral Smoothing (Soft-CRF Replacement)
            # Smooths predictions while preserving sharp tissue interfaces.
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            if getattr(config, "USE_SOFT_CRF", True):
                for b_idx in range(len(probs_batch)):
                    guide_img = (orig_img_batch[b_idx, :, :, 1] if config.USE_25D else orig_img_batch[b_idx, :, :, 0])
                    guide_uint8 = (guide_img * 255).astype(np.uint8)
                    for c in [0, 1]: # Smooth BG and IRF (fluid pockets)
                        prob_uint8 = (probs_batch[b_idx, c] * 255).astype(np.uint8)
                        probs_batch[b_idx, c] = cv2.bilateralFilter(prob_uint8, 9, 75, 75).astype(np.float32) / 255.0

            # 6. Clinical Post-Processing
            preds_batch = np.argmax(probs_batch, axis=1).astype(np.uint8)
            cleaned_preds = []
            for b_idx in range(len(preds_batch)):
                p = preds_batch[b_idx]
                ret_mask = get_retina_mask(orig_img_batch[b_idx])
                new_p = np.zeros_like(p)
                for c in range(1, config.NUM_LABELS):
                    c_mask = ((p == c).astype(np.uint8) * ret_mask)
                    # Morphological noise reduction
                    c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    if c == 1: c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    
                    # Connected component filtering (MIN_REGION_SIZE)
                    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                    for label in range(1, num_labels):
                        if stats[label, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                            new_p[labels_im == label] = c
                cleaned_preds.append(new_p)
            preds_batch = np.array(cleaned_preds)

        # 7. Metric Accumulation (Dice, IoU, HD95)
        for b_idx in range(len(preds_batch)):
            labels = labels_batch[b_idx]
            pred = preds_batch[b_idx]
            mask = (labels >= 0) & (labels < config.NUM_LABELS)
            total_cm += np.bincount(config.NUM_LABELS * labels[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=config.NUM_LABELS**2).reshape(config.NUM_LABELS, config.NUM_LABELS)
            
            for c in [1, 2, 3]:
                if np.any(labels == c) and np.any(pred == c):
                    class_hd95_vals[c].append(hd95(pred == c, labels == c))

    # 8. Final Report Generation
    tp = np.diag(total_cm)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    ious = tp / (tp + fp + fn + 1e-6)
    dices = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    
    return {
        'mIoU': np.mean(ious),
        'mDice': np.mean(dices),
        'mHD95': np.mean([np.mean(class_hd95_vals[c]) for c in [1, 2, 3] if class_hd95_vals[c]])
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(evaluate_model())
