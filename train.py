import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
import time
from datetime import timedelta
from tqdm import tqdm
from dataset import OCTDataset
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from utils import get_stratified_splits
import torch.nn.functional as F
from eval import get_retina_mask
from scipy.ndimage import distance_transform_edt

# --- CUSTOM LOSS FUNCTIONS FOR MEDICAL SEGMENTATION ---

class BoundaryLoss(torch.nn.Module):
    """
    Boundary Loss based on Signed Distance Field (SDF).
    Penalizes deviations at the fluid-tissue interfaces.
    """
    def __init__(self, num_classes=config.NUM_LABELS):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def compute_sdf(self, img_gt, out_shape):
        """Compute Signed Distance Field for each class except background"""
        img_gt = img_gt.astype(np.uint8)
        sdf = np.zeros(out_shape)

        for b in range(out_shape[0]):
            for c in range(1, out_shape[1]):
                posmask = img_gt[b] == c
                if not posmask.any(): continue
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                sdf[b, c] = negdis - posdis
        return sdf

    def forward(self, probs, gt):
        with torch.no_grad():
            gt_numpy = gt.cpu().numpy()
            sdf_numpy = self.compute_sdf(gt_numpy, probs.shape)
            sdf = torch.from_numpy(sdf_numpy).float().to(probs.device)

        # Penalize only pixels outside the ground truth (positive SDF)
        loss = probs * torch.clamp(sdf, min=0)
        return loss.mean()

class FocalLoss(torch.nn.Module):
    """
    Focal Loss to handle extreme class imbalance by focusing on hard-to-classify pixels.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        if mask is not None:
            focal_loss = focal_loss * mask
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
                
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

class TverskyLoss(torch.nn.Module):
    """
    Tversky Loss: A generalization of Dice loss allowing control over FP vs FN.
    Essential for detecting small IRF cysts (maximizing Recall).
    """
    def __init__(self, alpha=getattr(config, "TVERSKY_ALPHA", 0.3), beta=getattr(config, "TVERSKY_BETA", 0.7), num_classes=config.NUM_LABELS):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, target, mask=None):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask

        dims = (0, 2, 3)
        tp = torch.sum(pred * target_one_hot, dims)
        fp = torch.sum(pred * (1 - target_one_hot), dims)
        fn = torch.sum((1 - pred) * target_one_hot, dims)
        
        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        # Focus averaging on fluid classes (1, 2, 3) to prevent BG dominance
        return 1 - tversky[1:].mean()

def calculate_dynamic_weights(mask_paths, num_classes=config.NUM_LABELS):
    """Statistical inverse-frequency weighting for class balance"""
    logging.info("Calculating dynamic class weights...")
    counts = np.zeros(num_classes)
    from PIL import Image
    for p in tqdm(mask_paths, desc="Scanning masks"):
        m = np.array(Image.open(p))
        counts += np.bincount(m.flatten(), minlength=num_classes)
    
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    weights[0] = min(weights[0], 0.2) # Cap background weight
    return weights

# --- MAIN TRAINING PIPELINE ---

def train_model(epochs=config.EPOCHS, save_path="best_model.pth", output_dir="."):
    """
    Core training loop for SegFormer MiT architectures.
    Implements 2.5D context loading, hybrid loss, and polynomial learning rate scheduling.
    """
    # 1. Dataset Splitting (Stratified by Patient and Device)
    all_files = sorted(os.listdir(config.IMG_DIR))
    train_patients, val_patients, test_patients = get_stratified_splits(all_files)
    
    def get_paths(patient_list):
        imgs, masks = [], []
        for f in all_files:
            p = "_".join(f.split("_")[:2])
            if p in patient_list:
                imgs.append(os.path.join(config.IMG_DIR, f))
                masks.append(os.path.join(config.MASK_DIR, f))
        return imgs, masks

    train_imgs, train_masks = get_paths(train_patients)
    val_imgs, val_masks = get_paths(val_patients)

    # 2. Weight Initialization
    if config.USE_DYNAMIC_WEIGHTS:
        dyn_weights = calculate_dynamic_weights(train_masks)
        class_weights = torch.tensor(dyn_weights, dtype=torch.float32).to(config.DEVICE)
    else:
        class_weights = torch.tensor(config.CLASS_WEIGHTS).to(config.DEVICE)

    # 3. Augmentation Pipeline (Clinical Precision focused)
    aug_list = []
    if getattr(config, "USE_CLAHE", False):
        aug_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))
    
    aug_list.extend([
        A.RandomResizedCrop(size=config.AUG_SIZE, scale=config.AUG_SCALE, p=1.0),
        A.HorizontalFlip(p=config.AUG_PROBS["flip"]),
        A.Rotate(limit=10, p=config.AUG_PROBS["rotate"]),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
        ], p=0.4),
        A.RandomBrightnessContrast(p=config.AUG_PROBS["brightness"]),
        A.GaussNoise(std_range=(0.02, 0.1), p=config.AUG_PROBS["noise"]),
    ])
    train_transform = A.Compose(aug_list)
    
    val_aug_list = [A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1])]
    val_transform = A.Compose(val_aug_list)

    # 4. Model Loading (SegFormer with ImageNet-21k Pre-training)
    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
    processor.do_reduce_labels = False

    train_ds = OCTDataset(train_imgs, train_masks, processor, transform=train_transform, use_multimodal=config.USE_MULTIMODAL)
    val_ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform, use_multimodal=config.USE_MULTIMODAL)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS, 
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=getattr(config, "DROPOUT_RATE", 0.0),
        attention_probs_dropout_prob=getattr(config, "DROPOUT_RATE", 0.0),
        classifier_dropout_prob=getattr(config, "DROPOUT_RATE", 0.0)
    ).to(config.DEVICE)

    # 5. Optimizer and Hybrid Scheduler (Warmup + Poly Decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=5e-2)
    def lr_lambda(current_step):
        if current_step < config.WARMUP_EPOCHS:
            return float(current_step) / float(max(1, config.WARMUP_EPOCHS))
        progress = float(current_step - config.WARMUP_EPOCHS) / float(max(1, epochs - config.WARMUP_EPOCHS))
        return max(0.0, (1.0 - progress) ** 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=getattr(config, "FOCAL_GAMMA", 2.0))
    tversky_criterion = TverskyLoss()
    boundary_criterion = BoundaryLoss()

    best_miou = 0.0
    history = {"loss": [], "miou": []}
    logging.info(f"Starting training for {epochs} epochs...")

    # 6. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                
                # Hybrid Loss: Focal + Tversky + Optional Boundary
                main_loss = focal_criterion(logits, labels)
                aux_loss = tversky_criterion(logits, labels)
                
                if getattr(config, "USE_BOUNDARY_LOSS", False):
                    # Adaptive annealing: increases boundary weight over time
                    annealed_weight = min(0.1, 0.01 + (epoch * 0.0025))
                    aux_loss += annealed_weight * boundary_criterion(F.softmax(logits, dim=1), labels)

                loss = (main_loss + aux_loss) / config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config.ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * config.ACCUMULATION_STEPS:.4f}"})
        
        scheduler.step()
        
        # 7. Validation and Metrics Calculation
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            model.eval()
            total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    pixel_values = batch["pixel_values"].to(config.DEVICE)
                    labels = batch["labels"].to(config.DEVICE)
                    with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                        outputs = model(pixel_values=pixel_values)
                    logits = torch.nn.functional.interpolate(
                        outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    
                    # Mathematical Argmax for class assignment
                    preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

                    # Anatomical Post-processing
                    if getattr(config, "MIN_REGION_SIZE", 0) > 0:
                        import cv2
                        cleaned_preds = []
                        kernel = np.ones((3, 3), np.uint8)
                        batch_start = i * config.BATCH_SIZE
                        batch_indices = list(range(batch_start, min(len(val_ds), batch_start + config.BATCH_SIZE)))

                        for b_idx in range(len(preds)):
                            p = preds[b_idx]
                            orig_img = val_ds.get_raw_image(batch_indices[b_idx])
                            if orig_img.shape != p.shape:
                                orig_img = cv2.resize(orig_img, (p.shape[1], p.shape[0]))
                            
                            ret_mask = get_retina_mask(orig_img)
                            new_p = np.zeros_like(p)
                            for c in range(1, config.NUM_LABELS):
                                c_mask = ((p == c).astype(np.uint8) * ret_mask)
                                c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
                                if c == 1: # Smooth only IRF
                                    c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
                                
                                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                                for lbl in range(1, num_labels):
                                    if stats[lbl, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                                        new_p[labels_im == lbl] = c
                            cleaned_preds.append(new_p)
                        preds = np.array(cleaned_preds)

                    mask_cm = (labels.cpu().numpy() >= 0) & (labels.cpu().numpy() < config.NUM_LABELS)
                    total_cm += np.bincount(
                        config.NUM_LABELS * labels.cpu().numpy()[mask_cm].astype(np.int64) + preds[mask_cm].astype(np.int64),
                        minlength=config.NUM_LABELS**2
                    ).reshape(config.NUM_LABELS, config.NUM_LABELS)
            
            tp = np.diag(total_cm)
            fp = total_cm.sum(axis=0) - tp
            fn = total_cm.sum(axis=1) - tp
            ious = tp / (tp + fp + fn + 1e-6)
            curr_miou = np.mean(ious)
            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            history["miou"].append(curr_miou)
            
            logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | mIoU: {curr_miou:.4f}")
            
            if curr_miou > best_miou:
                best_miou = curr_miou
                torch.save(model.state_dict(), save_path)
                logging.info(f"New best model saved! (mIoU: {best_miou:.4f})")
    
    logging.info(f"Training complete. Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
