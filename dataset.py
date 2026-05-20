import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any
import config

class OCTDataset(Dataset):
    """
    Patient-aware OCT image loader.
    Supports multi-modal input (Original + Denoised + Edge map).
    """
    def __init__(
        self, 
        image_paths: List[str], 
        mask_paths: List[str], 
        processor: Any, 
        transform: Any = None, 
        use_multimodal: bool = False
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform
        self.use_multimodal = use_multimodal

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        
        if getattr(config, "USE_25D", False):
            # 2.5D logic: load t-1, t, t+1
            dir_path = os.path.dirname(img_path)
            base_name = os.path.basename(img_path)
            prefix = "_".join(base_name.split("_")[:-1])
            idx_str = base_name.split("_")[-1].split(".")[0]
            curr_idx = int(idx_str)
            ext = base_name.split(".")[-1]

            def load_slice(offset):
                neighbor_idx = curr_idx + offset
                neighbor_path = os.path.join(dir_path, f"{prefix}_{neighbor_idx:03d}.{ext}")
                if os.path.exists(neighbor_path):
                    return np.array(Image.open(neighbor_path).convert("L"))
                return None

            t_0 = np.array(Image.open(img_path).convert("L"))
            t_minus = load_slice(-1)
            t_plus = load_slice(1)
            
            # Use current slice as padding if neighbor is missing
            t_minus = t_minus if t_minus is not None else t_0
            t_plus = t_plus if t_plus is not None else t_0
            
            image = np.stack([t_minus, t_0, t_plus], axis=-1)
        elif self.use_multimodal:
            denoised_path = img_path.replace("cropped_images", "denoised_images")
            edge_path = img_path.replace("cropped_images", "edge_map_images")
            
            orig = np.array(Image.open(img_path).convert("L"))
            
            # Robust fallback: use original if denoised/edge map is missing
            if os.path.exists(denoised_path):
                denoised = np.array(Image.open(denoised_path).convert("L"))
            else:
                denoised = orig
                
            if os.path.exists(edge_path):
                edge = np.array(Image.open(edge_path).convert("L"))
            else:
                # Simple edge map on the fly if missing (optional enhancement)
                edge = orig 
            
            image = np.stack([orig, denoised, edge], axis=-1)
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        mask = np.array(Image.open(self.mask_paths[idx]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 1-99 Percentile Normalization
        image = image.astype(np.float32)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)
        
        # Scale back to 0-255 uint8 so processor works correctly
        image = (image * 255).astype(np.uint8)
            
        # SegFormer wants pixel_values (C, H, W) and labels (H, W)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)
        labels = torch.from_numpy(mask).long()
        
        return {"pixel_values": pixel_values, "labels": labels, "orig_img": image}
