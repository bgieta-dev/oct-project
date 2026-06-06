import os
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

# --- PATH CONFIGURATION ---
DATA_DIR = "data_folder"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
MASK_DIR = os.path.join(DATA_DIR, "cropped_masks")

# --- SEGFORMER ARCHITECTURE CONFIGURATION ---
# Model: MiT-B2 (Mix Vision Transformer). 
# B2 provides the best balance between parameter count and generalization on small medical datasets (56 patients).
MODEL_NAME = "nvidia/mit-b2" 
NUM_LABELS = 4
USE_MULTIMODAL = True
# 2.5D Logic: Utilizing 3 adjacent B-scans as input channels (t-1, t, t+1).
# This provides the transformer with volumetric/anatomical context.
USE_25D = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRAINING HYPERPARAMETERS ---
LR = 6e-5 # Learning Rate tuned for ImageNet weight fine-tuning
EPOCHS = 80
OPTIMIZER_TYPE = "AdamW" 
USE_AMP = True # Mixed Precision for VRAM efficiency and speed
VAL_INTERVAL = 1 

# --- LOSS FUNCTION STRATEGY ---
# Disabled dynamic weights (inverse freq) to stabilize HD95 by maintaining background penalty.
USE_DYNAMIC_WEIGHTS = False 
USE_CLAHE = True # Local contrast enhancement at fluid-tissue interfaces
USE_TVERSKY = True # Tversky Index optimization (DSC with FP/FN control)
USE_FOCAL_TVERSKY = True # Focal + Tversky combination - critical for small IRF cysts
USE_BOUNDARY_LOSS = False # Disabled. Past experiments (Test 13) showed that SDF-based boundary loss destabilizes training on small datasets, increasing HD95.
BOUNDARY_ALPHA = 0.1     

# Focal Gamma: 2.0 (Balanced focus on hard pixels). 
# Values above 3.0 caused convergence regression on this specific dataset.
FOCAL_GAMMA = 2.0 
DROPOUT_RATE = 0.2 # Regularization to prevent overfitting on the RETOUCH dataset
WARMUP_EPOCHS = 15 # Extended warmup (15 epochs) to stabilize transformer weights initially

# --- CLASS DEFINITIONS AND CLINICAL WEIGHTS ---
CLASS_NAMES = {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"}
# Static Weights: 
# Background (0.5) - prevents fluid "bleeding" into healthy tissue (reduces HD95).
# IRF (5.0) - highest priority for the smallest and hardest-to-segment cysts.
CLASS_WEIGHTS = [0.5, 5.0, 2.0, 2.0] 
TVERSKY_ALPHA = 0.3 
TVERSKY_BETA = 0.7 

# --- EVALUATION AND POST-PROCESSING ---
# Minimum Region Size: 10px. 
# Reduced from 50px to preserve small, clinically significant IRF microcysts.
MIN_REGION_SIZE = 10 
USE_TTA = True # Test-Time Augmentation (multi-scale prediction averaging)
TTA_SCALES = [0.75, 1.0, 1.25, 1.5] 
USE_SOFT_CRF = True # Edge-aware smoothing to align boundaries with scan intensities

# Class thresholds for manual calibration (optional fallback for non-argmax logic)
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
        return {"batch_size": 8, "accum_steps": 4}
    elif "b2" in model_name:
        return {"batch_size": 8, "accum_steps": 4}
    elif "b1" in model_name:
        return {"batch_size": 16, "accum_steps": 2}
    else: 
        return {"batch_size": 16, "accum_steps": 2}

VRAM = get_vram_config(MODEL_NAME)
BATCH_SIZE = VRAM["batch_size"]
ACCUMULATION_STEPS = VRAM["accum_steps"]
