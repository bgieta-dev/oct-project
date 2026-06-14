import os
import torch
from config import * # Inherit base settings

# --- EXPERT OVERRIDES ---
MODEL_NAME = "nvidia/mit-b0" # Lightweight expert
NUM_LABELS = 2 # Binary: Background vs IRF
TARGET_CLASS = 1 # Focus exclusively on IRF

# --- AGGRESSIVE RECALL HYPERPARAMETERS ---
LR = 7e-5 # Slightly higher for smaller model
EPOCHS = 60
CLASS_WEIGHTS = [0.1, 10.0] # Massive weight on IRF

# Tversky setup for high recall (Beta = 0.95)
TVERSKY_ALPHA = 0.05 
TVERSKY_BETA = 0.95 

FOCAL_GAMMA = 3.0 

# Higher resolution if possible, or same as base for consistency
AUG_SIZE = (512, 512)

# Specific directory for expert weights
EXPERT_SAVE_PATH = "irf_expert_best.pth"
