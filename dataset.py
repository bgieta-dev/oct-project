import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class OCTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            
        # SegFormer wants pixel_values (C, H, W) and labels (H, W)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)
        labels = torch.from_numpy(mask).long()
        
        return {"pixel_values": pixel_values, "labels": labels, "orig_img": image}
