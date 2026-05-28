import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# Model Configuration
MODEL_NAME = "monai/swinunetr" 
NUM_LABELS = 4
USE_MULTIMODAL = True
USE_25D = True # Stack adjacent slices (t-1, t, t+1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SwinUNETR Specifics (2D version for 2.5D context)
SWIN_CFG = {
    "img_size": (512, 512),
    "in_channels": 3,
    "out_channels": 4,
    "feature_size": 48,
    "drop_rate": 0.1,
    "attn_drop_rate": 0.1,
}

# Training Hyperparameters
LR = 5e-5
EPOCHS = 80
OPTIMIZER_TYPE = "AdamW" # AdamW, SGD
USE_AMP = True # Mixed Precision Training
VAL_INTERVAL = 1 # Validate every N epochs
USE_DYNAMIC_WEIGHTS = True # Calculate weights from training set
USE_TVERSKY = True # Use Tversky Loss instead of Dice Loss
USE_FOCAL_TVERSKY = True # Combine Focal with Tversky for extreme focus
USE_CLAHE = True # Use CLAHE for contrast enhancement
FOCAL_GAMMA = 3.0 # Increased Focal Gamma to force focus on difficult structures
DROPOUT_RATE = 0.2 # Increased for heavier regularization
WARMUP_EPOCHS = 15 # Extended to stabilize extreme augmentations

# Post-processing
MIN_REGION_SIZE = 50 # Morphological cleaning

# Class Definitions
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}
CLASS_WEIGHTS = [0.2, 5.0, 2.0, 2.0] # Fallback weights

# Evaluation
USE_TTA = True # Test-Time Augmentation
TTA_SCALES = [0.8, 1.0, 1.2] # Multi-scale inference

# Augmentation Settings
AUG_SIZE = (512, 512)
AUG_SCALE = (0.8, 1.0)
AUG_PROBS = {
    "flip": 0.5,
    "rotate": 0.5,
    "brightness": 0.2,
    "noise": 0.2
}

# Notifications
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def get_vram_config(model_name: str):
    """Auto-adjust for VRAM safety (target effective batch 32)"""
    if "swinunetr" in model_name:
        return {"batch_size": 4, "accum_steps": 8}
    elif "b4" in model_name:
        return {"batch_size": 4, "accum_steps": 8}
    elif "b3" in model_name:
        return {"batch_size": 8, "accum_steps": 4}
    elif "b2" in model_name:
        return {"batch_size": 8, "accum_steps": 4}
    elif "b1" in model_name:
        return {"batch_size": 16, "accum_steps": 2}
    else: # B0 or others
        return {"batch_size": 16, "accum_steps": 2}

VRAM = get_vram_config(MODEL_NAME)
BATCH_SIZE = VRAM["batch_size"]
ACCUMULATION_STEPS = VRAM["accum_steps"]
