import os
import logging
import torch
from train import train_model
import config_irf_expert as config

def train_expert():
    """
    Trains the Binary IRF Expert using the modular train_model logic.
    Target: mit-b0 architecture with high-recall Tversky loss.
    """
    logging.info(f"--- STARTING IRF EXPERT TRAINING ---")
    logging.info(f"Target Class: {config.CLASS_NAMES[config.TARGET_CLASS]}")
    logging.info(f"Model: {config.MODEL_NAME} | Tversky Beta: {config.TVERSKY_BETA}")
    
    # Ensure specific weights for multi-class don't interfere
    if os.path.exists(config.EXPERT_SAVE_PATH):
        os.remove(config.EXPERT_SAVE_PATH)
        logging.info(f"Removed stale {config.EXPERT_SAVE_PATH}")

    # Call the modular training loop with the expert config
    train_model(
        epochs=config.EPOCHS,
        save_path=config.EXPERT_SAVE_PATH,
        cfg=config
    )
    
    logging.info(f"Expert training finished. Best weights saved to {config.EXPERT_SAVE_PATH}")

if __name__ == "__main__":
    # Setup logging to console for immediate feedback
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    train_expert()
