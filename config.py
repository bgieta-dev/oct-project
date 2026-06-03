import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# Model Configuration
MODEL_NAME = "nvidia/mit-b2" # Golden Model: Best stability + Advanced logic
NUM_LABELS = 4
USE_MULTIMODAL = True
USE_25D = True # Stack adjacent slices (t-1, t, t+1) for volumetric context
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
USE_BOUNDARY_LOSS = False # Disabled: Causes loss plateau. Soft-CRF handles boundaries now.
BOUNDARY_ALPHA = 0.1     
# Increased Focal Gamma to force focus on difficult structures (from Test 11)
FOCAL_GAMMA = 3.0 
DROPOUT_RATE = 0.2 # Heavier regularization for 56 patients
WARMUP_EPOCHS = 15 # Extended to stabilize extreme augmentations (from Test 11)

# Class Definitions
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}
CLASS_WEIGHTS = [0.2, 5.0, 2.0, 2.0] # Fallback weights
TVERSKY_ALPHA = 0.2 # Restored to 0.2/0.8 balance for higher recall (Test 11 style)
TVERSKY_BETA = 0.8 

# Evaluation
MIN_REGION_SIZE = 50 # Restored from 50 (Test 11) to be more generous with small cysts
USE_TTA = True # Test-Time Augmentation
TTA_SCALES = [0.8, 1.0, 1.2] # Multi-scale inference
USE_SOFT_CRF = True # Edge-aware bilateral smoothing for boundary alignment (from Test 14)


CLASS_THRESHOLDS = {1: 0.50, 2: 0.50, 3: 0.50} # Restored IRF threshold to 0.5 to stop over-prediction


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
