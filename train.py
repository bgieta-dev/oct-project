import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from tqdm import tqdm
from dataset import OCTDataset
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from utils import get_stratified_splits
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from medpy.metric.binary import hd95
import cv2
from transformers import get_cosine_schedule_with_warmup

# --- CUSTOM LOSS FUNCTIONS FOR MEDICAL SEGMENTATION ---

class BoundaryLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def compute_sdf(self, img_gt, out_shape):
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
        loss = probs * torch.clamp(sdf, min=0)
        return loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=None, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        if self.alpha is not None:
            at = self.alpha[targets]
            focal_loss = focal_loss * at

        if mask is not None:
            focal_loss = focal_loss * mask
            if self.reduction == 'mean': 
                return focal_loss.sum() / (mask.sum() + 1e-8)
        if self.reduction == 'mean': 
            return focal_loss.mean()
        else: 
            return focal_loss.sum()

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, include_background=False):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.include_background = include_background

    def forward(self, pred, target, mask=None):
        pred = torch.softmax(pred, dim=1)
        target_long = target.long()
        target_one_hot = torch.nn.functional.one_hot(target_long, self.num_classes).permute(0, 3, 1, 2).float()
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
            
        dims = (0, 2, 3)
        tp = torch.sum(pred * target_one_hot, dims)
        fp = torch.sum(pred * (1 - target_one_hot), dims)
        fn = torch.sum((1 - pred) * target_one_hot, dims)
        
        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        
        if not self.include_background:
            return 1 - tversky[1:].mean()
        return 1 - tversky.mean()

def calculate_dynamic_weights(mask_paths, num_classes, target_class=None):
    logging.info("Calculating dynamic class weights...")
    counts = np.zeros(num_classes)
    from PIL import Image
    for p in tqdm(mask_paths, desc="Scanning masks"):
        m = np.array(Image.open(p))
        if target_class is not None:
            binary_m = np.zeros_like(m)
            binary_m[m == target_class] = 1
            m = binary_m
        counts += np.bincount(m.ravel(), minlength=num_classes)
    
    weights = 1.0 / (counts + 1.0) 
    weights = weights / weights.sum() * num_classes
    
    if weights[0] > 0.2:
        weights[0] = 0.2
        remaining_sum = num_classes - 0.2
        other_weights_sum = weights[1:].sum()
        if other_weights_sum > 0:
            weights[1:] = (weights[1:] / other_weights_sum) * remaining_sum
            
    return weights

# --- MAIN TRAINING PIPELINE ---

def distribute_files(all_files, train_p, val_p, cfg):
    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []
    for f in all_files:
        p = "_".join(f.split("_")[:2])
        img_path = os.path.join(cfg.IMG_DIR, f)
        mask_path = os.path.join(cfg.MASK_DIR, f)
        if p in train_p:
            train_imgs.append(img_path)
            train_masks.append(mask_path)
        elif p in val_p:
            val_imgs.append(img_path)
            val_masks.append(mask_path)
    return (train_imgs, train_masks), (val_imgs, val_masks)

def train_model(epochs=None, save_path=None, output_dir=".", cfg=config):
    if epochs is None: 
        epochs = cfg.EPOCHS
    if save_path is None: 
        save_path = "best_model.pth"

    all_files = sorted(os.listdir(cfg.IMG_DIR))
    train_patients, val_patients, test_patients = get_stratified_splits(all_files)
    
    (train_imgs, train_masks), (val_imgs, val_masks) = distribute_files(all_files, train_patients, val_patients, cfg)

    if getattr(cfg, "USE_DYNAMIC_WEIGHTS", False):
        dyn_weights = calculate_dynamic_weights(train_masks, num_classes=cfg.NUM_LABELS)
        class_weights = torch.tensor(dyn_weights, dtype=torch.float32).to(cfg.DEVICE)
    else:
        class_weights = torch.tensor(cfg.CLASS_WEIGHTS).to(cfg.DEVICE)

    train_transform = A.Compose([
        A.CLAHE(clip_limit=2.0, p=0.5) if getattr(cfg, "USE_CLAHE", False) else A.NoOp(),
        A.RandomResizedCrop(size=cfg.AUG_SIZE, scale=cfg.AUG_SCALE, p=1.0),
        A.HorizontalFlip(p=cfg.AUG_PROBS["flip"]),
        A.Rotate(limit=10, p=cfg.AUG_PROBS["rotate"]),
        A.OneOf([A.ElasticTransform(alpha=1, sigma=50, p=1.0), A.GridDistortion(p=1.0)], p=0.4),
        A.RandomBrightnessContrast(p=cfg.AUG_PROBS["brightness"]),
        A.GaussNoise(p=cfg.AUG_PROBS["noise"]),
    ])
    
    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])

    processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)
    processor.do_reduce_labels = False

    # Use target_class if defined in cfg (for Expert Binary models)
    target_cls = getattr(cfg, "TARGET_CLASS", None)
    train_ds = OCTDataset(train_imgs, train_masks, processor, transform=train_transform, 
                          use_multimodal=getattr(cfg, "USE_MULTIMODAL", False), target_class=target_cls)
    val_ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform, 
                        use_multimodal=getattr(cfg, "USE_MULTIMODAL", False), target_class=target_cls)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS, ignore_mismatched_sizes=True,
        hidden_dropout_prob=getattr(cfg, "DROPOUT_RATE", 0.0),
        attention_probs_dropout_prob=getattr(cfg, "DROPOUT_RATE", 0.0),
        classifier_dropout_prob=getattr(cfg, "DROPOUT_RATE", 0.0)
    ).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=5e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=cfg.WARMUP_EPOCHS, 
        num_training_steps=epochs
    )
    
    scaler = torch.amp.GradScaler(cfg.DEVICE.type, enabled=cfg.USE_AMP)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=getattr(cfg, "FOCAL_GAMMA", 2.0))
    
    tv_alpha = getattr(cfg, "TVERSKY_ALPHA", 0.3)
    tv_beta = getattr(cfg, "TVERSKY_BETA", 0.7)
    tversky_criterion = TverskyLoss(alpha=tv_alpha, beta=tv_beta, num_classes=cfg.NUM_LABELS)
    boundary_criterion = BoundaryLoss(num_classes=cfg.NUM_LABELS)

    best_miou = 0.0
    history = {"loss": [], "miou": [], "mhd95": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(cfg.DEVICE)
            labels = batch["labels"].to(cfg.DEVICE)
            with torch.amp.autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = torch.nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                
                focal_w = getattr(cfg, "FOCAL_WEIGHT", 0.5)
                main_loss = focal_w * focal_criterion(logits, labels)
                
                aux_loss = 0.0
                if getattr(cfg, "USE_TVERSKY", False) or getattr(cfg, "USE_FOCAL_TVERSKY", False):
                    tversky_w = getattr(cfg, "TVERSKY_WEIGHT", 0.5)
                    aux_loss += tversky_w * tversky_criterion(logits, labels)
                    
                if getattr(cfg, "USE_BOUNDARY_LOSS", False):
                    b_alpha = getattr(cfg, "BOUNDARY_ALPHA", 0.1)
                    b_w = min(b_alpha, (b_alpha / 10.0) + (epoch * (b_alpha / 40.0)))
                    aux_loss += b_w * boundary_criterion(F.softmax(logits, dim=1), labels)
                
                loss = (main_loss + aux_loss) / cfg.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}"})
        
        scheduler.step()

        if (epoch + 1) % cfg.VAL_INTERVAL == 0:
            model.eval()
            total_cm = np.zeros((cfg.NUM_LABELS, cfg.NUM_LABELS), dtype=np.int64)
            class_hd95_vals = {c: [] for c in range(1, cfg.NUM_LABELS)}
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    pixel_values = batch["pixel_values"].to(cfg.DEVICE)
                    labels_batch = batch["labels"].to(cfg.DEVICE)
                    outputs = model(pixel_values=pixel_values)
                    logits = torch.nn.functional.interpolate(outputs.logits, size=labels_batch.shape[-2:], mode="bilinear", align_corners=False)
                    preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                    labels_np = labels_batch.cpu().numpy()
                    
                    if getattr(cfg, "MIN_REGION_SIZE", 0) > 0:
                        cleaned_preds = []
                        for p in preds:
                            new_p = np.zeros_like(p)
                            for c in range(1, cfg.NUM_LABELS):
                                c_mask = (p == c).astype(np.uint8)
                                if np.any(c_mask):
                                    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=8)
                                    for lbl in range(1, num_labels):
                                        if stats[lbl, cv2.CC_STAT_AREA] >= cfg.MIN_REGION_SIZE:
                                            new_p[labels_im == lbl] = c
                            cleaned_preds.append(new_p)
                        preds = np.array(cleaned_preds)
                        
                    for b_idx in range(len(preds)):
                        p, l_np = preds[b_idx], labels_np[b_idx]
                        for c in range(1, cfg.NUM_LABELS):
                            p_c = p == c
                            l_c = l_np == c
                            if np.any(p_c) and np.any(l_c):
                                class_hd95_vals[c].append(hd95(p_c, l_c))
                            elif np.any(l_c):
                                class_hd95_vals[c].append(100.0) # Penalty for missing fluid entirely
                            elif np.any(p_c):
                                class_hd95_vals[c].append(100.0) # Penalty for false fluid prediction

                    mask_cm = (labels_np >= 0) & (labels_np < cfg.NUM_LABELS)
                    total_cm += np.bincount(cfg.NUM_LABELS * labels_np[mask_cm].astype(np.int64) + preds[mask_cm].astype(np.int64), minlength=cfg.NUM_LABELS**2).reshape(cfg.NUM_LABELS, cfg.NUM_LABELS)
            
            ious = np.diag(total_cm) / (total_cm.sum(axis=0) + total_cm.sum(axis=1) - np.diag(total_cm) + 1e-6)
            curr_miou = np.mean(ious)
            means = [np.mean(class_hd95_vals[c]) for c in range(1, cfg.NUM_LABELS) if class_hd95_vals[c]]
            curr_mhd95 = np.mean(means) if means else 0.0
            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            history["miou"].append(curr_miou)
            history["mhd95"].append(curr_mhd95)
            logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | mIoU: {curr_miou:.4f} | mHD95: {curr_mhd95:.2f}")
            if curr_miou > best_miou:
                best_miou = curr_miou
                torch.save(model.state_dict(), save_path)
                logging.info(f"New best model saved! (mIoU: {best_miou:.4f})")

        # Automated Plotting
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history["loss"])
        plt.title("Loss")
        plt.subplot(1, 3, 2) 
        plt.plot(history["miou"]) 
        plt.title("mIoU")
        plt.subplot(1, 3, 3)
        plt.plot(history["mhd95"]) 
        plt.title("mHD95")
        plt.savefig(os.path.join(output_dir, "metrics.png"))
        plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
