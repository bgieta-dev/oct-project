import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# --- PATH CONFIGURATION ---
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# --- SEGFORMER ARCHITECTURE CONFIGURATION ---
# Model: MiT-B2 (Golden Model for Stability)
MODEL_NAME = "nvidia/mit-b2" 
NUM_LABELS = 4
USE_MULTIMODAL = True
USE_25D = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRAINING HYPERPARAMETERS ---
LR = 5e-5
EPOCHS = 80
OPTIMIZER_TYPE = "AdamW" 
USE_AMP = True 
VAL_INTERVAL = 1 

# --- LOSS FUNCTION STRATEGY ---
USE_DYNAMIC_WEIGHTS = True 
USE_CLAHE = True 
USE_TVERSKY = True 
USE_FOCAL_TVERSKY = True 
USE_BOUNDARY_LOSS = False
BOUNDARY_ALPHA = 0.1     

FOCAL_GAMMA = 3.0 
DROPOUT_RATE = 0.2 
WARMUP_EPOCHS = 15 

# --- CLASS DEFINITIONS AND CLINICAL WEIGHTS ---
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}
CLASS_WEIGHTS = [0.2, 5.0, 2.0, 2.0] 
# CLINICAL HIGH RECALL SETUP: High penalty for False Negatives (Beta), low for False Positives (Alpha)
TVERSKY_ALPHA = 0.1 
TVERSKY_BETA = 0.9 

# --- EVALUATION AND POST-PROCESSING ---
MIN_REGION_SIZE = 50 
USE_TTA = True 
TTA_SCALES = [0.8, 1.0, 1.2] 
USE_SOFT_CRF = False # Disabled due to pydensecrf compilation issues on py3.13
CRF_ITERATIONS = 5 # How hard to snap to edges

# CLINICAL HIGH RECALL: Lower thresholds force the model to predict fluid even with lower confidence
# IRF is particularly dangerous to miss, so it triggers at just 30% confidence.
CLASS_THRESHOLDS = {1: 0.30, 2: 0.40, 3: 0.40} 

# --- DATA AUGMENTATION SETTINGS ---
AUG_SIZE = (512, 512)
AUG_SCALE = (0.8, 1.0)
AUG_PROBS = {
    "flip": 0.5,
    "rotate": 0.5,
    "brightness": 0.2,
    "noise": 0.2
}

# --- NOTIFICATIONS ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def get_vram_config(model_name: str):
    """Auto-adjust batch size for 12GB VRAM (Target effective batch = 32)"""
    if "b4" in model_name:
        return {"batch_size": 4, "accum_steps": 8}
    elif "b3" in model_name:
        return {"batch_size": 8, "accum_steps": 4} # Verified for 12GB VRAM
    elif "b2" in model_name:
        return {"batch_size": 8, "accum_steps": 4}
    elif "b1" in model_name:
        return {"batch_size": 16, "accum_steps": 2}
    else: 
        return {"batch_size": 16, "accum_steps": 2}

VRAM = get_vram_config(MODEL_NAME)
BATCH_SIZE = VRAM["batch_size"]
ACCUMULATION_STEPS = VRAM["accum_steps"]
