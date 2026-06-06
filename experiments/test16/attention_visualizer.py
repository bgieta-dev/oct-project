import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from dataset import OCTDataset
import config
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A

def generate_attention_maps(model_path="best_model.pth", output_dir="attention_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model with attention output enabled
    print(f"Loading model for attention visualization: {model_path}...")
    
    # Check architecture
    if "monai" in config.MODEL_NAME:
        print("SwinUNETR attention visualization is not yet supported in this script. Skipping.")
        return

    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        ignore_mismatched_sizes=True,
        output_attentions=True # CRITICAL
    )
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load exact weights for attention ({e}). Using base model.")
        
    model.to(device)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)

    # 2. Get test images and find BEST slices for visualization
    all_files = sorted(os.listdir(config.IMG_DIR))
    if not os.path.exists("test_patients.txt"):
        from utils import get_stratified_splits
        _, _, test_patients = get_stratified_splits(all_files)
    else:
        with open("test_patients.txt", "r") as f:
            test_patients = f.read().splitlines()
    
    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    
    # Pre-scan to find best examples for each class (1: IRF, 2: SRF, 3: PED)
    print("Scanning for slices with significant pathology for visualization...")
    vis_indices = {}
    class_max_counts = {1: 0, 2: 0, 3: 0}
    for idx, m_path in enumerate(test_masks):
        m = np.array(Image.open(m_path))
        for c in [1, 2, 3]:
            count = np.sum(m == c)
            if count > class_max_counts[c]:
                class_max_counts[c] = count
                vis_indices[c] = idx
    
    target_indices = sorted(list(set(vis_indices.values())))
    print(f"Targeting indices: {target_indices}")

    # Use proper validation transform
    val_aug_list = [A.Resize(height=config.AUG_SIZE[0], width=config.AUG_SIZE[1])]
    val_transform = A.Compose(val_aug_list)

    dataset = OCTDataset(test_imgs, test_masks, processor, transform=val_transform, use_multimodal=config.USE_MULTIMODAL)
    
    print("Generating attention maps...")
    
    with torch.no_grad():
        for target_idx in target_indices:
            batch = dataset[target_idx]
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device)
            labels = batch["labels"].numpy()
            orig_img = batch["orig_img"] 
            
            # Use middle slice (central anatomy) for grayscale visualization
            if orig_img.shape[-1] == 3:
                # If 2.5D: 0=t-1, 1=t, 2=t+1. Use 1 for central slice.
                # If Multimodal: 0=Orig, 1=Denoised, 2=Edge. Use 0.
                vis_idx = 1 if getattr(config, "USE_25D", False) else 0
                vis_img = orig_img[:, :, vis_idx]
            else:
                vis_img = orig_img

            # Diagnostic: Count pathology pixels
            px_count = np.sum(labels > 0)
            class_label = "unknown"
            for c, idx in vis_indices.items():
                if idx == target_idx: class_label = config.CLASS_NAMES[c]
            
            print(f"-> Generating maps for {class_label} (Index {target_idx}, Pathology pixels: {px_count})")

            outputs = model(pixel_values=pixel_values)
            attentions = outputs.attentions # List of layers
            
            # Save representative layers (Stages 0, 1, 2, 3)
            # SegFormer has 4 stages of transformer blocks
            for stage_idx, stage_att in enumerate(attentions):
                # stage_att shape: (batch, heads, seq_len, seq_len)
                avg_att = torch.mean(stage_att, dim=1) 
                spatial_att = torch.mean(avg_att, dim=1) 
                
                grid_size = int(np.sqrt(spatial_att.shape[1]))
                heatmap = spatial_att.view(grid_size, grid_size).cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                heatmap_resized = cv2.resize(heatmap, (512, 512))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                vis_uint8 = np.uint8(vis_img * 255) if vis_img.max() <= 1.0 else np.uint8(vis_img)
                orig_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_GRAY2BGR)
                
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
                
                fname = f"att_idx{target_idx}_{class_label}_stage{stage_idx}.png"
                cv2.imwrite(os.path.join(output_dir, fname), overlay)
            
            # Save GT overlays for clinical validation
            gt_norm = np.uint8(labels * 60)
            gt_color = cv2.applyColorMap(gt_norm, cv2.COLORMAP_JET)
            gt_color[labels == 0] = 0 # Background black
            
            vis_uint8 = np.uint8(vis_img * 255) if vis_img.max() <= 1.0 else np.uint8(vis_img)
            orig_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_GRAY2BGR)
            
            gt_overlay = cv2.addWeighted(orig_bgr, 0.5, gt_color, 0.5, 0)
            cv2.imwrite(os.path.join(output_dir, f"att_idx{target_idx}_{class_label}_GT_OVERLAY.png"), gt_overlay)
            cv2.imwrite(os.path.join(output_dir, f"att_idx{target_idx}_{class_label}_GT_MASK.png"), gt_color)

    print(f"Done! Results saved in {output_dir}/")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    generate_attention_maps()
