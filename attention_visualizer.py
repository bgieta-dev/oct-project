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

def generate_attention_maps(model_path="best_model.pth", output_dir="attention_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model with attention output enabled
    print(f"Loading model from {model_path}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        ignore_mismatched_sizes=True,
        output_attentions=True # CRITICAL
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)

    # 2. Get some test images
    all_files = sorted(os.listdir(config.IMG_DIR))
    with open("test_patients.txt", "r") as f:
        test_patients = f.read().splitlines()
    
    test_imgs = [os.path.join(config.IMG_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    test_masks = [os.path.join(config.MASK_DIR, f) for f in all_files if "_".join(f.split("_")[:2]) in test_patients]
    
    dataset = OCTDataset(test_imgs[:10], test_masks[:10], processor, use_multimodal=config.USE_MULTIMODAL)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Generating attention maps for sample images...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].numpy()[0]
            orig_img = batch["orig_img"].numpy()[0] # (512, 512)

            outputs = model(pixel_values=pixel_values)
            attentions = outputs.attentions # List of 4 stages
            
            # Process each stage
            for stage_idx, stage_att in enumerate(attentions):
                # stage_att shape: (batch, heads, seq_len, seq_len)
                # For SegFormer, seq_len is H*W of the feature map at that stage
                # We aggregate attention by averaging over heads and then averaging over the query dimension
                # to see which spatial locations are attended to MOST across all pixels.
                
                # Average across heads
                avg_att = torch.mean(stage_att, dim=1) # (1, seq_len, seq_len)
                
                # Aggregate across queries (rows) to get spatial importance map
                spatial_att = torch.mean(avg_att, dim=1) # (1, seq_len)
                
                # Reshape to 2D grid
                grid_size = int(np.sqrt(spatial_att.shape[1]))
                heatmap = spatial_att.view(grid_size, grid_size).cpu().numpy()
                
                # Normalize heatmap
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # Resize to 512x512
                heatmap_resized = cv2.resize(heatmap, (512, 512))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                # Overlay on original image
                # Convert orig_img to BGR if it's grayscale
                if len(orig_img.shape) == 2:
                    orig_bgr = cv2.cvtColor(np.uint8(orig_img * 255), cv2.COLOR_GRAY2BGR)
                else:
                    orig_bgr = np.uint8(orig_img * 255)
                
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
                
                # Save result
                fname = f"att_img{i}_stage{stage_idx}.png"
                cv2.imwrite(os.path.join(output_dir, fname), overlay)
            
            # Also save GT for reference
            gt_color = cv2.applyColorMap(np.uint8(labels * 60), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_dir, f"att_img{i}_GT.png"), gt_color)
            
            if i >= 2: break # Only 3 images

    print(f"Done! Results saved in {output_dir}/")

if __name__ == "__main__":
    generate_attention_maps()
