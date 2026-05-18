import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, get_cosine_schedule_with_warmup
import albumentations as A
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
from tqdm import tqdm
from dataset import OCTDataset
import logging
import matplotlib.pyplot as plt
import config

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
    
    train_pts, temp_pts, _, temp_devs = train_test_split(
        patients, devices, test_size=0.20, random_state=42, stratify=devices
    )
    
    val_pts, test_pts = train_test_split(
        temp_pts, test_size=0.50, random_state=42, stratify=temp_devs
    )
    return train_pts, val_pts, test_pts

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

    train_transform = A.Compose([
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
    ])
    val_transform = A.Compose([A.Resize(height=512, width=512)])

    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
    processor.do_reduce_labels = False

    train_ds = OCTDataset(train_imgs, train_masks, processor, transform=train_transform, use_multimodal=config.USE_MULTIMODAL)
    val_ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform, use_multimodal=config.USE_MULTIMODAL)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=epochs)
    
    # Class 0: Background (0.2)
    # Class 1: IRF (5.0) - Hardest
    # Class 2: SRF (2.0)
    # Class 3: PED (2.0)
    class_weights = torch.tensor([0.2, 5.0, 2.0, 2.0]).to(config.DEVICE)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_miou = 0.0
    start_time = time.time()
    history = {"loss": [], "miou": []}
    logging.info(f"Starting training for {epochs} epochs...")
    logging.info(f"Config: MODEL_NAME={config.MODEL_NAME}, BATCH_SIZE={config.BATCH_SIZE}, ACCUM_STEPS={config.ACCUMULATION_STEPS}, LR={config.LR}, DEVICE={config.DEVICE}")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            
            loss = (0.5 * focal_criterion(logits, labels) + 0.5 * dice_loss(logits, labels)) / config.ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config.ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * config.ACCUMULATION_STEPS:.4f}"})
        
        if (len(train_loader)) % config.ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        
        model.eval()
        total_cm = np.zeros((config.NUM_LABELS, config.NUM_LABELS), dtype=np.int64)
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                outputs = model(pixel_values=pixel_values)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                preds = logits.argmax(dim=1)
                
                mask = (labels >= 0) & (labels < config.NUM_LABELS)
                label_flat = labels[mask].cpu().numpy().astype(np.int64)
                pred_flat = preds[mask].cpu().numpy().astype(np.int64)
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
