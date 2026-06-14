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
    Supports 2.5D volumetric context and multimodal (Denoised/Edge) inputs.
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

    def get_raw_image(self, idx: int) -> np.ndarray:
        """
        Returns [0, 1] normalized single-slice image.
        Used for retina masking and post-processing visualization.
        """
        img_path = self.image_paths[idx]
        img_p = Image.open(img_path).convert("L")
        img_n = np.array(img_p).astype(np.float32)
        # Clinical normalization: 1-99 percentile clipping to remove noise spikes
        p1, p99 = np.percentile(img_n, (1, 99))
        img_n = np.clip(img_n, p1, p99)
        img_n = (img_n - p1) / (p99 - p1 + 1e-8)
        return img_n

    def get_raw_mask(self, idx: int) -> np.ndarray:
        """Returns raw categorical mask (0=BG, 1=IRF, 2=SRF, 3=PED)"""
        return np.array(Image.open(self.mask_paths[idx]))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        skip_norm = False
        
        # --- 2.5D LOGIC: VOLUMETRIC CONTEXT ---
        # Loads current slice (t) plus neighbors (t-1, t+1) to provide 3D spatial cues.
        if getattr(config, "USE_25D", False):
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
                    img_p = Image.open(neighbor_path).convert("L")
                    # Handle edge case where resize might be needed
                    if img_p.size != t_0_pil.size:
                        img_p = img_p.resize(t_0_pil.size, Image.BILINEAR)
                    
                    img_n = np.array(img_p).astype(np.float32)
                    p1, p99 = np.percentile(img_n, (1, 99))
                    img_n = np.clip(img_n, p1, p99)
                    img_n = (img_n - p1) / (p99 - p1 + 1e-8)
                    return img_n
                return None

            t_0_pil = Image.open(img_path).convert("L")
            t_0_np = np.array(t_0_pil).astype(np.float32)
            p1, p99 = np.percentile(t_0_np, (1, 99))
            t_0_norm = np.clip(t_0_np, p1, p99)
            t_0_norm = (t_0_norm - p1) / (p99 - p1 + 1e-8)
            
            t_minus = load_slice(-1)
            t_plus = load_slice(1)
            
            # Padding if slices are at the beginning/end of the volume
            t_minus = t_minus if t_minus is not None else t_0_norm
            t_plus = t_plus if t_plus is not None else t_0_norm
            
            image = np.stack([t_minus, t_0_norm, t_plus], axis=-1)
            image = (image * 255).astype(np.uint8)
            skip_norm = True
            
        # --- MULTIMODAL LOGIC: ENHANCED FEATURES ---
        elif self.use_multimodal:
            denoised_path = img_path.replace("cropped_images", "denoised_images")
            edge_path = img_path.replace("cropped_images", "edge_map_images")
            
            orig = np.array(Image.open(img_path).convert("L"))
            
            if os.path.exists(denoised_path):
                denoised = np.array(Image.open(denoised_path).convert("L"))
            else:
                denoised = orig
                
            if os.path.exists(edge_path):
                edge = np.array(Image.open(edge_path).convert("L"))
            else:
                edge = orig 
            
            image = np.stack([orig, denoised, edge], axis=-1)
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # --- BINARY EXPERT MODE ---
        if self.target_class is not None:
            binary_mask = np.zeros_like(mask)
            binary_mask[mask == self.target_class] = 1
            mask = binary_mask

        # --- AUGMENTATION ---
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # --- FINAL NORMALIZATION AND RESIZING ---
        if not skip_norm:
            image = image.astype(np.float32)
            p1, p99 = np.percentile(image, (1, 99))
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1 + 1e-8)
            image = (image * 255).astype(np.uint8)
            
        import cv2
        if hasattr(config, "AUG_SIZE") and (image.shape[0] != config.AUG_SIZE[0] or image.shape[1] != config.AUG_SIZE[1]):
            image = cv2.resize(image, (config.AUG_SIZE[1], config.AUG_SIZE[0]), interpolation=cv2.INTER_LINEAR)

        # Convert to transformer-friendly format (SegformerImageProcessor)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)
        
        if hasattr(config, "AUG_SIZE") and mask.shape != config.AUG_SIZE:
            mask = cv2.resize(mask, (config.AUG_SIZE[1], config.AUG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
            
        labels = torch.from_numpy(mask).long()
        
        return {"pixel_values": pixel_values, "labels": labels, "orig_img": image}
