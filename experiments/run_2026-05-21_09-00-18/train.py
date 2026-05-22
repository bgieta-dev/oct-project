import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, get_cosine_schedule_with_warmup
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

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def dice_loss(pred, target, num_classes=config.NUM_LABELS):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(pred * target_one_hot, dims) 
    cardinality = torch.sum(pred + target_one_hot, dims)
    dice = (2. * intersection + 1e-6) / (cardinality + 1e-6)
    return 1 - dice.mean()

def calculate_dynamic_weights(mask_paths, num_classes=config.NUM_LABELS):
    logging.info("Calculating dynamic class weights...")
    counts = np.zeros(num_classes)
    from PIL import Image
    for p in tqdm(mask_paths, desc="Scanning masks"):
        m = np.array(Image.open(p))
        counts += np.bincount(m.flatten(), minlength=num_classes)
    
    # Inverse frequency weighting
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    # Ensure background doesn't dominate too much (cap it)
    weights[0] = min(weights[0], 0.2)
    return weights

def train_model(epochs=config.EPOCHS, save_path="best_model.pth", output_dir="."):
    all_files = sorted(os.listdir(config.IMG_DIR))
    train_patients, val_patients, test_patients = get_stratified_splits(all_files)
    
    with open("test_patients.txt", "w") as f:
        f.write("\n".join(test_patients))
    
    logging.info(f"Split: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")

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

    # Dynamic Weights
    if config.USE_DYNAMIC_WEIGHTS:
        dyn_weights = calculate_dynamic_weights(train_masks)
        logging.info(f"Dynamic weights: {dyn_weights}")
        class_weights = torch.tensor(dyn_weights, dtype=torch.float32).to(config.DEVICE)
    else:
        class_weights = torch.tensor(config.CLASS_WEIGHTS).to(config.DEVICE)

    train_transform = A.Compose([
        A.RandomResizedCrop(size=config.AUG_SIZE, scale=config.AUG_SCALE, p=1.0),
        A.HorizontalFlip(p=config.AUG_PROBS["flip"]),
        A.Rotate(limit=10, p=config.AUG_PROBS["rotate"]),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.RandomBrightnessContrast(p=config.AUG_PROBS["brightness"]),
        A.GaussNoise(std_range=(0.02, 0.1), p=config.AUG_PROBS["noise"]),
    ])
    val_transform = A.Compose([A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1])])

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

    if config.OPTIMIZER_TYPE == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=5e-2)
    elif config.OPTIMIZER_TYPE == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.9, weight_decay=5e-2)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=5e-2)
        
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=epochs)
    
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=getattr(config, "FOCAL_GAMMA", 2.0))

    best_miou = 0.0
    patience = 10
    epochs_no_improve = 0
    start_time = time.time()
    history = {"loss": [], "miou": []}
    logging.info(f"Starting training for {epochs} epochs...")
    logging.info(f"Config: MODEL_NAME={config.MODEL_NAME}, BATCH_SIZE={config.BATCH_SIZE}, ACCUM_STEPS={config.ACCUMULATION_STEPS}, LR={config.LR}, DEVICE={config.DEVICE}, AMP={config.USE_AMP}")

    for epoch in range(epochs):
        epoch_start = time.time()
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
                loss = (0.5 * focal_criterion(logits, labels) + 0.5 * dice_loss(logits, labels)) / config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config.ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * config.ACCUMULATION_STEPS:.4f}"})
        
        if (len(train_loader)) % config.ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        scheduler.step()
        
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            model.eval()
            total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(config.DEVICE)
                    labels = batch["labels"].to(config.DEVICE)
                    with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                        outputs = model(pixel_values=pixel_values)
                    logits = torch.nn.functional.interpolate(
                        outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    preds = logits.argmax(dim=1).cpu().numpy()
                    
                    # Morphological cleaning for validation metrics
                    if getattr(config, "MIN_REGION_SIZE", 0) > 0:
                        import cv2
                        cleaned_preds = []
                        for p in preds:
                            new_p = np.zeros_like(p)
                            for c in range(1, config.NUM_LABELS):
                                c_mask = (p == c).astype(np.uint8)
                                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=8)
                                for label in range(1, num_labels):
                                    if stats[label, cv2.CC_STAT_AREA] >= config.MIN_REGION_SIZE:
                                        new_p[labels_im == label] = c
                            cleaned_preds.append(new_p)
                        preds = np.array(cleaned_preds)

                    mask = (labels.cpu().numpy() >= 0) & (labels.cpu().numpy() < config.NUM_LABELS)
                    label_flat = labels.cpu().numpy()[mask].astype(np.int64)
                    pred_flat = preds[mask].astype(np.int64)
                    total_cm += np.bincount(
                        config.NUM_LABELS * label_flat + pred_flat,
                        minlength=config.NUM_LABELS**2
                    ).reshape(config.NUM_LABELS, config.NUM_LABELS)
            
            # Calculate mIoU from CM
            tp = np.diag(total_cm)
            fp = total_cm.sum(axis=0) - tp
            fn = total_cm.sum(axis=1) - tp
            ious = tp / (tp + fp + fn + 1e-6)
            curr_miou = np.mean(ious)
            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            history["miou"].append(curr_miou)
            
            epoch_dur = time.time() - epoch_start
            logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | mIoU: {curr_miou:.4f} | Time: {int(epoch_dur)}s")
            
            if curr_miou > best_miou:
                best_miou = curr_miou
                torch.save(model.state_dict(), save_path)
                logging.info(f"New best model saved! (mIoU: {best_miou:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs!")
                    break
        else:
            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            epoch_dur = time.time() - epoch_start
            logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | (Val Skipped) | Time: {int(epoch_dur)}s")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["loss"]); plt.title("Loss History"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(history["miou"]); plt.title("mIoU History"); plt.xlabel("Epoch"); plt.ylabel("mIoU")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics.png"))
        plt.close()

    logging.info(f"Training complete. Total time: {str(timedelta(seconds=int(time.time() - start_time)))}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
