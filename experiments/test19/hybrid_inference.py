import torch
import numpy as np
import cv2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import config
import config_irf_expert as expert_config
import torch.nn.functional as F

class HybridInference:
    def __init__(self, base_model_path, expert_model_path):
        self.device = config.DEVICE
        
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

    @torch.no_grad()
    def segment(self, image_np):
        """
        Input: Raw 2.5D/Multimodal image [H, W, 3] from OCTDataset
        Output: Merged Mask [H, W] (0=BG, 1=IRF, 2=SRF, 3=PED)
        """
        # --- PASS 1: BASE MODEL ---
        inputs_base = self.processor(images=image_np, return_tensors="pt").to(self.device)
        base_out = self.base_model(**inputs_base).logits
        base_logits = F.interpolate(base_out, size=image_np.shape[:2], mode="bilinear", align_corners=False)
        base_probs = F.softmax(base_logits, dim=1).squeeze(0).cpu().numpy()
        base_mask = np.argmax(base_probs, axis=0).astype(np.uint8)
        
        # --- PASS 2: EXPERT MODEL (Pure 2D) ---
        # Extract middle slice (S_n) from 2.5D context and replicate to 3 channels for processor
        image_2d = image_np[:, :, 1] if len(image_np.shape) == 3 else image_np
        image_2d_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
        
        inputs_expert = self.processor(images=image_2d_rgb, return_tensors="pt").to(self.device)
        expert_out = self.expert_model(**inputs_expert).logits
        expert_logits = F.interpolate(expert_out, size=image_np.shape[:2], mode="bilinear", align_corners=False)
        expert_probs = F.softmax(expert_logits, dim=1).squeeze(0).cpu().numpy()
        expert_irf_prob = expert_probs[1] # Probability of IRF
        
        # --- ENSEMBLE MERGING LOGIC ---
        final_mask = base_mask.copy()
        
        # Threshold for Expert IRF (aggressive)
        irf_threshold = getattr(expert_config, "CLASS_THRESHOLDS", {1: 0.25})[1]
        expert_irf_mask = (expert_irf_prob > irf_threshold)
        
        # Conflict Resolution: Vectorized
        # If expert says IRF, it overrides BG (0) and existing Base IRF (1).
        # It does NOT override SRF (2) / PED (3) to avoid anatomical corruption.
        final_mask[(final_mask <= 1) & expert_irf_mask] = 1
            
        # Optional: Clean up tiny noise artifacts
        if getattr(config, "MIN_REGION_SIZE", 0) > 0:
            for c in [1, 2, 3]:
                c_mask = (final_mask == c).astype(np.uint8)
                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=8)
                for lbl in range(1, num_labels):
                    if stats[lbl, cv2.CC_STAT_AREA] < config.MIN_REGION_SIZE:
                        final_mask[labels_im == lbl] = 0
                        
        return final_mask

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
