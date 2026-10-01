import os
import torch
from config import * # Inherit base settings

# --- EXPERT OVERRIDES ---
MODEL_NAME = "nvidia/mit-b0" # Lightweight expert
NUM_LABELS = 2 # Binary: Background vs IRF
TARGET_CLASS = 1 # Focus exclusively on IRF
USE_25D = False # Force 2D for expert model

# --- BALANCED EXPERT HYPERPARAMETERS ---
USE_DYNAMIC_WEIGHTS = False # Use fixed weights for stability
LR = 5e-5 # Lower LR for stability
EPOCHS = 60
CLASS_WEIGHTS = [1.0, 8.0] # More balanced ratio

# Loss weighting (tuning the balance between Focal and Tversky)
FOCAL_WEIGHT = 0.5
TVERSKY_WEIGHT = 0.5

# Tversky setup: softer recall bias to prevent "blob" predictions
TVERSKY_ALPHA = 0.3 
TVERSKY_BETA = 0.7

FOCAL_GAMMA = 2.0

# Inference thresholds (for hybrid_inference.py)
CLASS_THRESHOLDS = {1: 0.25} # Aggressive threshold for IRF recall

# Higher resolution if possible, or same as base for consistency
AUG_SIZE = (512, 512)

# Specific directory for expert weights
EXPERT_SAVE_PATH = "irf_expert_best.pth"
