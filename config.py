import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# --- PATH CONFIGURATION ---
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# --- SEGFORMER ARCHITECTURE CONFIGURATION ---
# Model: MiT-B3 (Mix Vision Transformer). 
# Test 17: Scaling up to B3 (~44M parameters) using the stabilized Test 16 pipeline.
MODEL_NAME = "nvidia/mit-b2" 
NUM_LABELS = 4
USE_MULTIMODAL = True
# 2.5D Logic: Utilizing 3 adjacent B-scans as input channels (t-1, t, t+1).
USE_25D = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRAINING HYPERPARAMETERS ---
LR = 5e-5 # Slightly lower LR for B3 stability
EPOCHS = 80
OPTIMIZER_TYPE = "AdamW" 
USE_AMP = True 
VAL_INTERVAL = 1 

# --- LOSS FUNCTION STRATEGY ---
USE_DYNAMIC_WEIGHTS = False 
USE_CLAHE = True 
USE_TVERSKY = True 
USE_FOCAL_TVERSKY = True 
USE_BOUNDARY_LOSS = False # Disabled per Test 13/15 lessons.
BOUNDARY_ALPHA = 0.1     

# Focal Gamma: 2.0 (Balanced focus)
FOCAL_GAMMA = 2.0 
# [MODIFICATION] Increased to 0.3 to prevent B3 from memorizing noise on the small RETOUCH set.
DROPOUT_RATE = 0.3 
WARMUP_EPOCHS = 15 

# --- CLASS DEFINITIONS AND CLINICAL WEIGHTS ---
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}
# Static Weights (Test 16 success): Higher background penalty (0.5) to keep HD95 low.
CLASS_WEIGHTS = [0.5, 5.0, 2.0, 2.0] 
TVERSKY_ALPHA = 0.3 
TVERSKY_BETA = 0.7 

# --- EVALUATION AND POST-PROCESSING ---
# [MODIFICATION] Reduced to 5px. B3 has higher capacity for detail; we want to preserve its precise detections.
MIN_REGION_SIZE = 5 
USE_TTA = True 
TTA_SCALES = [0.75, 1.0, 1.25, 1.5] 
USE_SOFT_CRF = True # Edge-aware smoothing

# Class thresholds for manual calibration (optional fallback)
CLASS_THRESHOLDS = {1: 0.35, 2: 0.50, 3: 0.50} 

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
