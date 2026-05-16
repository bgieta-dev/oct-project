import os
import torch

# Paths
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# Model
MODEL_NAME = "nvidia/mit-b2" # nvidia/mit-b0, nvidia/mit-b1, nvidia/mit-b2, nvidia/mit-b3
NUM_LABELS = 4
USE_MULTIMODAL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Hyperparameters
LR = 1e-4
EPOCHS = 80

# Class Definitions
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}

def get_vram_config(model_name):
    """Auto-adjust for VRAM safety (target effective batch 32)"""
    if "b3" in model_name:
        return {"batch_size": 2, "accum_steps": 16}
    elif "b2" in model_name:
        return {"batch_size": 8, "accum_steps": 4}
    else: # B0 or others
        return {"batch_size": 16, "accum_steps": 2}


VRAM = get_vram_config(MODEL_NAME)
BATCH_SIZE = VRAM["batch_size"]
ACCUMULATION_STEPS = VRAM["accum_steps"]
