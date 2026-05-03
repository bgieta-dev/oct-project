import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score
import warnings
from tqdm import tqdm
from dataset import OCTDataset

# config
IMG_DIR = "data_folder/cropped_images"
MASK_DIR = "data_folder/cropped_masks"
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
CHECKPOINT = "segformer_oct.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings from sklearn metrics for classes not present in the mask
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Split (same as train.py to get val set)
all_files = sorted(os.listdir(IMG_DIR))
patients = sorted(list(set(["_".join(f.split("_")[:2]) for f in all_files])))
_, val_patients = train_test_split(patients, test_size=0.2, random_state=42)
val_imgs = [os.path.join(IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in val_patients]
val_masks = [os.path.join(MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in val_patients]

# load model
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, num_labels=4, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

# transforms
val_transform = A.Compose([A.Resize(512, 512)])
ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform)
loader = DataLoader(ds, batch_size=1, shuffle=True)

# Metrics accumulation
all_preds = []
all_labels = []

plt.figure(figsize=(15, 10))
vis_count = 0

print(f"Evaluating on validation set ({len(loader)} images)...")
for i, batch in enumerate(tqdm(loader)):
    pixel_values = batch["pixel_values"].to(DEVICE)
    labels = batch["labels"].numpy()[0]
    orig_img = batch["orig_img"].numpy()[0]
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        # resize logits to match original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(512, 512), mode="bilinear", align_corners=False
        )
        pred = upsampled_logits.argmax(dim=1).cpu().numpy()[0]

    all_preds.append(pred.flatten())
    all_labels.append(labels.flatten())

    # plot only first 3
    if vis_count < 3:
        plt.subplot(3, 3, vis_count*3 + 1)
        plt.imshow(orig_img)
        plt.title("Original OCT")
        plt.axis("off")
        
        plt.subplot(3, 3, vis_count*3 + 2)
        plt.imshow(labels, cmap="jet")
        plt.title("Ground Truth")
        plt.axis("off")
        
        plt.subplot(3, 3, vis_count*3 + 3)
        plt.imshow(pred, cmap="jet")
        plt.title("Prediction")
        plt.axis("off")
        vis_count += 1

plt.tight_layout()
plt.savefig("eval_results.png")
print("Saved visualization to eval_results.png")

# Calculate metrics
print("\nCalculating metrics...")
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Assuming 4 classes based on train.py
num_classes = 4 
ious = []
dices = []

for c in range(num_classes):
    # Only calculate if the class is present in the ground truth
    if np.any(all_labels == c):
        iou = jaccard_score(all_labels == c, all_preds == c, zero_division=1.0)
        dice = f1_score(all_labels == c, all_preds == c, zero_division=1.0)
        ious.append(iou)
        dices.append(dice)
        print(f"Class {c} - IoU: {iou:.4f}, Dice: {dice:.4f}")
    else:
        print(f"Class {c} - Not present in validation set.")

if ious:
    print(f"\nMean IoU: {np.mean(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f}")
