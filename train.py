import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
from tqdm import tqdm
from dataset import OCTDataset

# config
IMG_DIR = "data_folder/cropped_images"
MASK_DIR = "data_folder/cropped_masks"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
BATCH_SIZE = 8
LR = 1e-4 # learning rate
EPOCHS = 2
USE_MULTIMODAL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# split data by patient to avoid leakage
all_files = sorted(os.listdir(IMG_DIR))
patients = sorted(list(set(["_".join(f.split("_")[:2]) for f in all_files])))
train_patients, val_patients = train_test_split(patients, test_size=0.2, random_state=42)

def get_paths(patient_list):
    imgs, masks = [], []
    for f in all_files:
        p = "_".join(f.split("_")[:2])
        if p in patient_list:
            imgs.append(os.path.join(IMG_DIR, f))
            masks.append(os.path.join(MASK_DIR, f))
    return imgs, masks

train_imgs, train_masks = get_paths(train_patients)
val_imgs, val_masks = get_paths(val_patients)

# augmentation of data
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

val_transform = A.Compose([
    A.Resize(512, 512),
])

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
processor.do_reduce_labels = False

train_ds = OCTDataset(train_imgs, train_masks, processor, transform=train_transform, use_multimodal=USE_MULTIMODAL)
val_ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform, use_multimodal=USE_MULTIMODAL)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME, 
    num_labels=4, 
    ignore_mismatched_sizes=True
).to(DEVICE)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # weight that balances classes. if one class occurs less often, we can increase it.
        self.gamma = gamma # focus parameter. higher means model will more often lower loss value on easy eg.
        self.reduction = reduction # mean/sum

    def forward(self, inputs, targets):
        # calculating standard cross entropy for each element
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')

        # cross_entropy = -log(pt) -> pt = e^(cross_entropy). | close to 0 = hard eg, close to 1 = easy eg. 
        pt = torch.exp(-ce_loss)

        # FL(pt) = -alpha(1-pt)^(gamma)*log(pt)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def dice_loss(pred, target, num_classes=4):
    # normalize predictions
    pred = torch.softmax(pred, dim=1)
    # labels -> binary masks for each class ([B, C, H, W])
    target_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    dims = (0, 2, 3) # batch, height, width

    # common areas
    intersection = torch.sum(pred * target_one_hot, dims) 
    # sum of areas
    cardinality = torch.sum(pred + target_one_hot, dims)
    
    # Dice formula = (2*intersections)/(cardinality) | 1e-6 is added to prevent dividing by 0
    dice = (2. * intersection + 1e-6) / (cardinality + 1e-6)

    # 1 means exelent results, 0 means tragic
    return 1 - dice.mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# changes LR (learning rate) over time of training 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# loss function (focused on hard eg)
focal_criterion = FocalLoss(alpha=torch.tensor([1.0, 2.0, 1.0, 1.0]).to(DEVICE), gamma=2.0)

# ------------------------------------------
# TRAIN LOOP
# ------------------------------------------
best_miou = 0.0
start_time = time.time()
print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    # progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        # resize logits to match label size (512, 512)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        
        # hybrid loss: Focal + Dice
        f_loss = focal_criterion(logits, labels)
        d_loss = dice_loss(logits, labels)
        loss = 0.5 * f_loss + 0.5 * d_loss
        
        # backpropagation
        loss.backward()
        # change in weights 
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # changes LR on each epoch
    scheduler.step()
    curr_lr = optimizer.param_groups[0]['lr']
    
    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start
    total_elapsed = epoch_end - start_time
    
    print(f"\nEpoch {epoch+1}/{EPOCHS} Finished | Avg Loss: {total_loss/len(train_loader):.4f} | LR: {curr_lr:.6f} | Time: {str(timedelta(seconds=int(epoch_duration)))} | Total: {str(timedelta(seconds=int(total_elapsed)))}")

    # val (minimal)
    model.eval()
    val_ious = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1)
            
            # calculate IoU per class
            for c in range(4):
                intersection = ((preds == c) & (labels == c)).sum().item()
                union = ((preds == c) | (labels == c)).sum().item()
                if union > 0:
                    val_ious.append(intersection / union)
    
    curr_miou = np.mean(val_ious) if val_ious else 0
    print(f"Validation mIoU: {curr_miou:.4f}")
    
    if curr_miou > best_miou:
        best_miou = curr_miou
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best model saved! (mIoU: {best_miou:.4f})")

total_time = time.time() - start_time
print(f"\nTraining complete in {str(timedelta(seconds=int(total_time)))}")

torch.save(model.state_dict(), "segformer_oct.pth")
