import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import config as global_config

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
        use_multimodal: bool = False,
        target_class: Optional[int] = None,
        cfg: Any = global_config
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform
        self.use_multimodal = use_multimodal
        self.target_class = target_class
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.image_paths)

    def _normalize_slice(self, img_n: np.ndarray) -> np.ndarray:
        """Clinical normalization: 1-99 percentile clipping."""
        p1, p99 = np.percentile(img_n, (1, 99))
        img_n = np.clip(img_n, p1, p99)
        img_n = (img_n - p1) / (p99 - p1 + 1e-8)
        return img_n

    def get_raw_image(self, idx: int) -> np.ndarray:
        """Returns [0, 1] normalized single-slice image."""
        img_path = self.image_paths[idx]
        img_p = Image.open(img_path).convert("L")
        return self._normalize_slice(np.array(img_p).astype(np.float32))

    def get_raw_mask(self, idx: int) -> np.ndarray:
        """Returns raw categorical mask (0=BG, 1=IRF, 2=SRF, 3=PED)"""
        return np.array(Image.open(self.mask_paths[idx]))

    def _load_neighbor(self, dir_path, prefix, curr_idx, ext, offset, target_size):
        neighbor_idx = curr_idx + offset
        neighbor_path = os.path.join(dir_path, f"{prefix}_{neighbor_idx:03d}.{ext}")
        if os.path.exists(neighbor_path):
            img_p = Image.open(neighbor_path).convert("L")
            if img_p.size != target_size:
                img_p = img_p.resize(target_size, Image.BILINEAR)
            return self._normalize_slice(np.array(img_p).astype(np.float32))
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        skip_final_norm = False
        
        # --- 2.5D LOGIC: VOLUMETRIC CONTEXT ---
        if getattr(self.cfg, "USE_25D", False):
            dir_path = os.path.dirname(img_path)
            base_name = os.path.basename(img_path)
            
            # More robust splitting: Spectralis_TRAIN001_001.png
            parts = base_name.split(".")
            name_part = parts[0]
            ext = parts[1]
            
            name_segments = name_part.split("_")
            prefix = "_".join(name_segments[:-1])
            curr_idx = int(name_segments[-1])

            t_0_pil = Image.open(img_path).convert("L")
            t_0_norm = self._normalize_slice(np.array(t_0_pil).astype(np.float32))
            
            t_minus = self._load_neighbor(dir_path, prefix, curr_idx, ext, -1, t_0_pil.size)
            t_plus = self._load_neighbor(dir_path, prefix, curr_idx, ext, 1, t_0_pil.size)
            
            t_minus = t_minus if t_minus is not None else t_0_norm
            t_plus = t_plus if t_plus is not None else t_0_norm
            
            image = np.stack([t_minus, t_0_norm, t_plus], axis=-1)
            image = (image * 255).astype(np.uint8)
            skip_final_norm = True
            
        elif self.use_multimodal:
            # Fallback pathing logic
            denoised_path = img_path.replace("cropped_images", "denoised_images")
            edge_path = img_path.replace("cropped_images", "edge_map_images")
            
            orig = np.array(Image.open(img_path).convert("L"))
            denoised = np.array(Image.open(denoised_path).convert("L")) if os.path.exists(denoised_path) else orig
            edge = np.array(Image.open(edge_path).convert("L")) if os.path.exists(edge_path) else orig
            
            image = np.stack([orig, denoised, edge], axis=-1)
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        mask = np.array(Image.open(self.mask_paths[idx]))
        
        if self.target_class is not None:
            binary_mask = np.zeros_like(mask)
            binary_mask[mask == self.target_class] = 1
            mask = binary_mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if not skip_final_norm:
            image = self._normalize_slice(image.astype(np.float32))
            image = (image * 255).astype(np.uint8)
            
        aug_size = getattr(self.cfg, "AUG_SIZE", None)
        if aug_size:
            if image.shape[0] != aug_size[0] or image.shape[1] != aug_size[1]:
                image = cv2.resize(image, (aug_size[1], aug_size[0]), interpolation=cv2.INTER_LINEAR)
            if mask.shape != aug_size:
                mask = cv2.resize(mask, (aug_size[1], aug_size[0]), interpolation=cv2.INTER_NEAREST)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)
        labels = torch.from_numpy(mask).long()
        
        return {"pixel_values": pixel_values, "labels": labels, "orig_img": image}
