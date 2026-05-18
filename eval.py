import os
import torch
import numpy as np
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
from dataset import OCTDataset
import logging
import config

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def get_stratified_splits(all_files):
    patient_to_device = {}
    for f in all_files:
        parts = f.split("_")
        device = parts[0]
        patient = "_".join(parts[:2])
        if patient not in patient_to_device:
            patient_to_device[patient] = device
    patients = np.array(list(patient_to_device.keys()))
    devices = np.array(list(patient_to_device.values()))
    _, temp_pts, _, temp_devs = train_test_split(
        patients, devices, test_size=0.20, random_state=42, stratify=devices
    )
    _, test_pts = train_test_split(temp_pts, test_size=0.50, random_state=42, stratify=temp_devs)
    return test_pts

def evaluate_model(model_path="best_model.pth", output_dir="."):
    all_files = sorted(os.listdir(config.IMG_DIR))

    if os.path.exists("test_patients.txt"):
        with open("test_patients.txt", "r") as f:
            test_patients = f.read().splitlines()
        logging.info("Loaded test set from test_patients.txt")
    else:
        test_patients = get_stratified_splits(all_files)
        logging.info("Recreated stratified test split")

    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]

    val_imgs, val_masks = test_imgs, test_masks # Evaluate on TEST SET now

    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True), strict=True)
        logging.info(f"Loaded weights from {model_path}")
    except RuntimeError as e:
        logging.warning(f"Strict load failed: {e}. Trying non-strict load...")
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True), strict=False)
        logging.warning("Non-strict load complete. Results may be garbage if architectures mismatch!")

    model.to(config.DEVICE).eval()

    val_transform = A.Compose([A.Resize(height=512, width=512)])
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

    global_idx = 0
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels_batch = batch["labels"].numpy()
        orig_img_batch = batch["orig_img"].numpy()
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=(512, 512), mode="bilinear", align_corners=False
            )
            preds_batch = logits.argmax(dim=1).cpu().numpy()

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

            # Visualization for fixed class-specific indices
            if i in target_indices:
                c_id = [k for k, v in vis_indices.items() if v == i][0]
                pos = list(vis_indices.keys()).index(c_id)
                plt.subplot(3, 3, pos*3 + 1); plt.imshow(orig_img); plt.title(f"OCT (Best {config.CLASS_NAMES[c_id]})"); plt.axis("off")
                plt.subplot(3, 3, pos*3 + 2); plt.imshow(labels, cmap="jet"); plt.title(f"GT (px: {class_max_counts[c_id]})"); plt.axis("off")
                plt.subplot(3, 3, pos*3 + 3); plt.imshow(pred, cmap="jet"); plt.title("Pred"); plt.axis("off")

    vis_path = os.path.join(output_dir, "predictions.png")
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close() # Free memory
    logging.info(f"Saved visualization to {vis_path}")

    # Calculate metrics from confusion matrix
    tp = np.diag(total_cm)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    
    metrics = {"class_ious": {}, "class_dices": {}, "class_hd95": {}, "class_asd": {}, "class_avg_regions_gt": {}, "class_avg_regions_pred": {}}
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
    
    metrics["mIoU"] = float(np.mean(list(metrics["class_ious"].values())))
    metrics["mDice"] = float(np.mean(list(metrics["class_dices"].values())))
    metrics["mHD95"] = float(np.mean(list(metrics["class_hd95"].values())))
    metrics["mASD"] = float(np.mean(list(metrics["class_asd"].values())))
    
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = evaluate_model()
    print(m)
