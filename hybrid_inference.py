import torch
import numpy as np
import cv2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import config
import config_irf_expert as expert_config
import torch.nn.functional as F

class HybridInference:
    def __init__(self, base_model_path, expert_model_path, ensemble_mode="soft", expert_weight=0.4, blend_strategy="linear", irf_threshold=None):
        self.device = config.DEVICE
        self.ensemble_mode = ensemble_mode
        self.expert_weight = expert_weight
        self.blend_strategy = blend_strategy
        self.irf_threshold = irf_threshold
        
        # 1. Load Base Model (mit-b2, multi-class)
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True
        ).to(self.device)
        self.base_model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        self.base_model.eval()
        
        # 2. Load IRF Expert Model (mit-b0, binary)
        self.expert_model = SegformerForSemanticSegmentation.from_pretrained(
            expert_config.MODEL_NAME, num_labels=expert_config.NUM_LABELS, ignore_mismatched_sizes=True
        ).to(self.device)
        self.expert_model.load_state_dict(torch.load(expert_model_path, map_location=self.device))
        self.expert_model.eval()
        
        self.processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)

    def _predict_logits(self, model, processor, image, cfg):
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        target_size = image.shape[:2]
        
        if getattr(cfg, "USE_TTA", False):
            scales = getattr(cfg, "TTA_SCALES", [1.0])
            all_logits = []
            for s in scales:
                if s != 1.0:
                    scaled_size = (int(cfg.AUG_SIZE[0] * s), int(cfg.AUG_SIZE[1] * s))
                    scaled_pixels = torch.nn.functional.interpolate(
                        pixel_values, size=scaled_size, mode="bilinear", align_corners=False
                    )
                else:
                    scaled_pixels = pixel_values
                
                s_outputs = model(pixel_values=scaled_pixels)
                s_logits = torch.nn.functional.interpolate(
                    s_outputs.logits, size=target_size, mode="bilinear", align_corners=False
                )
                all_logits.append(s_logits)
                
                f_pixels = torch.flip(scaled_pixels, [3])
                f_outputs = model(pixel_values=f_pixels)
                f_logits = torch.nn.functional.interpolate(
                    f_outputs.logits, size=target_size, mode="bilinear", align_corners=False
                )
                uf_logits = torch.flip(f_logits, [3])
                all_logits.append(uf_logits)
            
            return torch.mean(torch.stack(all_logits), dim=0)
        else:
            out = model(pixel_values=pixel_values).logits
            return torch.nn.functional.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def segment(self, image_np):
        """
        Input: Raw 2.5D/Multimodal image [H, W, 3] from OCTDataset
        Output: Merged Mask [H, W] (0=BG, 1=IRF, 2=SRF, 3=PED)
        """
        # --- PASS 1: BASE MODEL ---
        base_logits = self._predict_logits(self.base_model, self.processor, image_np, config)
        base_probs = F.softmax(base_logits, dim=1).squeeze(0).cpu().numpy()
        
        # --- PASS 2: EXPERT MODEL (Pure 2D) ---
        # Extract middle slice (S_n) from 2.5D context and replicate to 3 channels for processor
        image_2d = image_np[:, :, 1] if len(image_np.shape) == 3 else image_np
        image_2d_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
        
        expert_logits = self._predict_logits(self.expert_model, self.processor, image_2d_rgb, expert_config)
        expert_probs = F.softmax(expert_logits, dim=1).squeeze(0).cpu().numpy()
        expert_irf_prob = expert_probs[1] # Probability of IRF
        
        # --- ENSEMBLE MERGING LOGIC ---
        target_classes = list(range(1, config.NUM_LABELS))
        
        if self.ensemble_mode == "soft":
            # Probability-level blending for IRF (Class 1)
            merged_probs = base_probs.copy()
            
            P_base = base_probs[1]
            P_expert = expert_irf_prob
            w = self.expert_weight
            
            if self.blend_strategy == "linear":
                merged_probs[1] = (1 - w) * P_base + w * P_expert
            elif self.blend_strategy == "geometric":
                merged_probs[1] = np.clip((P_base ** (1 - w)) * (P_expert ** w), 0, 1)
            elif self.blend_strategy == "harmonic":
                merged_probs[1] = np.clip(1.0 / ((1 - w) / (P_base + 1e-8) + w / (P_expert + 1e-8) + 1e-8), 0, 1)
            elif self.blend_strategy == "max":
                merged_probs[1] = np.maximum(P_base, P_expert)
            elif self.blend_strategy == "min":
                merged_probs[1] = np.minimum(P_base, P_expert)
            elif self.blend_strategy == "confidence":
                # Only blend in the uncertain region of the base model
                uncertain_mask = (P_base > 0.15) & (P_base < 0.45)
                blend_val = (1 - w) * P_base + w * P_expert
                merged_probs[1] = np.where(uncertain_mask, blend_val, P_base)
            
            # Clinical Sharpening for PED (Class 3)
            if config.NUM_LABELS > 3:
                sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                sharp_ped = cv2.filter2D(merged_probs[3], -1, sharpen_kernel)
                merged_probs[3] = np.clip(sharp_ped, 0, 1)
                
            # Reverse-Priority Thresholding
            final_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            thresholds = getattr(config, "CLASS_THRESHOLDS", {c: 0.5 for c in target_classes}).copy()
            if self.irf_threshold is not None:
                thresholds[1] = self.irf_threshold
                
            for c in reversed(target_classes):
                thresh = thresholds.get(c, 0.5)
                final_mask[merged_probs[c] > thresh] = c
                
        else: # "hard" mask-level merging (original logic, but aligned with eval heuristics)
            # Base Model prediction with sharpening and thresholding
            base_probs_copy = base_probs.copy()
            if config.NUM_LABELS > 3:
                sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                sharp_ped = cv2.filter2D(base_probs_copy[3], -1, sharpen_kernel)
                base_probs_copy[3] = np.clip(sharp_ped, 0, 1)
                
            base_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            thresholds = getattr(config, "CLASS_THRESHOLDS", {c: 0.5 for c in target_classes}).copy()
            if self.irf_threshold is not None:
                thresholds[1] = self.irf_threshold
                
            for c in reversed(target_classes):
                thresh = thresholds.get(c, 0.5)
                base_mask[base_probs_copy[c] > thresh] = c
                
            # Threshold for Expert IRF (aggressive)
            irf_threshold = getattr(expert_config, "CLASS_THRESHOLDS", {1: 0.25})[1]
            expert_irf_mask = (expert_irf_prob > irf_threshold)
            
            final_mask = base_mask.copy()
            # If expert says IRF, it overrides BG (0) and existing Base IRF (1).
            # It does NOT override SRF (2) / PED (3) to avoid anatomical corruption.
            final_mask[(final_mask <= 1) & expert_irf_mask] = 1
            
        # --- CLINICAL POST-PROCESSING HEURISTICS (aligned with eval.py) ---
        cleaned_mask = np.zeros_like(final_mask)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        min_region_size = getattr(config, "MIN_REGION_SIZE", 0)
        
        for c in target_classes:
            c_mask = (final_mask == c).astype(np.uint8)
            if np.any(c_mask):
                # Morphological separation for IRF (Class 1) to break thin false-positive bridges
                if c == 1 and config.NUM_LABELS > 1:
                    c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel_3x3, iterations=1)
                
                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, 8)
                for label in range(1, num_labels):
                    if stats[label, cv2.CC_STAT_AREA] >= min_region_size:
                        cleaned_mask[labels_im == label] = c
                        
        return cleaned_mask

if __name__ == "__main__":
    # Example usage / Test stub
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    
    BASE_PATH = "best_model.pth"
    EXPERT_PATH = "irf_expert_best.pth"
    
    if os.path.exists(BASE_PATH) and os.path.exists(EXPERT_PATH):
        engine = HybridInference(BASE_PATH, EXPERT_PATH)
        print("Hybrid Engine Initialized.")
    else:
        print("Missing weights. Train expert first!")
