import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# Model Configuration
MODEL_NAME = "nvidia/mit-b2" # Best stability + Advanced logic
NUM_LABELS = 4
USE_MULTIMODAL = True
USE_25D = False # Multimodal (Denoised/Edge) is more stable for boundaries than 2.5D
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Hyperparameters
LR = 6e-5 
EPOCHS = 80
OPTIMIZER_TYPE = "AdamW" # AdamW, SGD
USE_AMP = True # Mixed Precision Training
VAL_INTERVAL = 1 # Validate every N epochs
USE_DYNAMIC_WEIGHTS = True # Calculate weights from training set
USE_CLAHE = True # sharpening the fluid-tissue interface
USE_TVERSKY = True # Use Tversky Loss instead of Dice Loss
USE_FOCAL_TVERSKY = True # Combine Focal with Tversky for extreme focus
USE_BOUNDARY_LOSS = True # Proven success in Test 12 for HD95 optimization
BOUNDARY_ALPHA = 0.1     
# Standard gamma for balanced focus
FOCAL_GAMMA = 2.0 
DROPOUT_RATE = 0.1 # Reduced from 0.2 to prevent underfitting
WARMUP_EPOCHS = 5 # Reduced from 15 for faster convergence

# Class Definitions
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}
CLASS_WEIGHTS = [0.2, 5.0, 2.0, 2.0] # Fallback weights
TVERSKY_ALPHA = 0.2 # Complement to Beta
TVERSKY_BETA = 0.8 # Boosted to reduce False Negatives (Recall priority)

# Evaluation

USE_TTA = True # Test-Time Augmentation
TTA_SCALES = [0.8, 1.0, 1.2, 1.5] # Multi-scale inference


CLASS_THRESHOLDS = {1: 0.35, 2: 0.5, 3: 0.5} # Lower threshold for IRF to boost recall


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
    if "b4" in model_name:
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
