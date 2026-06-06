import os
import logging
import shutil
import gc
import torch
import requests
import json
import sys
from datetime import datetime
from train import train_model
from eval import evaluate_model
from attention_visualizer import generate_attention_maps
import config

# --- UTILITY FUNCTIONS ---

def send_discord_notification(message):
    """Sends experiment status updates to a Discord channel via Webhook"""
    if not config.DISCORD_WEBHOOK_URL:
        return
    try:
        payload = {"content": message}
        requests.post(config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Failed to send Discord notification: {e}")


def setup_logging(exp_dir):
    """Configures dual logging: Console (short) and File (verbose)"""
    log_file = os.path.join(exp_dir, "experiment.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    logging.captureWarnings(True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return log_file

# --- MAIN EXECUTION PIPELINE ---

def main():
    """
    Orchestrates the full OCT segmentation research pipeline:
    1. Environment setup and logging
    2. SegFormer Model Training
    3. Comprehensive Metric Evaluation
    4. Transformer Attention Visualization
    5. Artifact Archiving
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.abspath(f"experiments/run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    setup_logging(exp_dir)
    logging.info(f"Pipeline started. Results directory: {exp_dir}")
    
    # Hardware Diagnostics
    logging.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # PHASE 1: MODEL TRAINING
    best_model_name = "best_model.pth"
    logging.info("--- PHASE 1: TRAINING ---")
    try:
        train_model(epochs=config.EPOCHS, save_path=best_model_name, output_dir=exp_dir)
        
        if os.path.exists(best_model_name):
            shutil.copy(best_model_name, os.path.join(exp_dir, "best_model.pth"))

        # Memory Management: Clear VRAM for the evaluation pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        print("Error in train")

    # PHASE 2: EVALUATION
    logging.info("--- PHASE 2: EVALUATION ---")
    try:
        metrics = evaluate_model(model_path=best_model_name, output_dir=exp_dir)
    
        # Log Aggregated Metrics
        logging.info(f"Final mIoU: {metrics['mIoU']:.4f} | Final mDice: {metrics['mDice']:.4f}")
        logging.info(f"Final mHD95: {metrics['mHD95']:.4f} | Final mASD: {metrics['mASD']:.4f}")
        logging.info(f"Model Parameters: {metrics.get('params', 0):.2f}M")
        
        # Log Class-Specific Findings (Essential for Thesis Tables)
        for c in [1, 2, 3]:
            name = config.CLASS_NAMES[c]
            iou = metrics['class_ious'][c]
            dice = metrics['class_dices'][c]
            hd = metrics['class_hd95'][c]
            asd_val = metrics['class_asd'][c]
            avg_reg_gt = metrics['class_avg_regions_gt'][c]
            avg_reg_pred = metrics['class_avg_regions_pred'][c]
            bp = metrics['class_boundary_precision'][c]
            area = metrics['class_avg_pixel_area'][c]
            
            logging.info(f"Class {c} ({name}) | IoU: {iou:.4f} | Dice: {dice:.4f} | HD95: {hd:.2f} | ASD: {asd_val:.2f}")
            logging.info(f"  Regions GT/Pred: {avg_reg_gt:.1f}/{avg_reg_pred:.1f} | BP: {bp:.4f} | Avg Area: {area:.1f} px")
    except:
        print("Error in eval")

    # PHASE 3: INTERPRETABILITY (ATTENTION MAPS)
    try:
        logging.info("--- PHASE 3: ATTENTION VISUALIZATION ---")
        att_dir = os.path.join(exp_dir, "attention_maps")
        generate_attention_maps(model_path=best_model_name, output_dir=att_dir)
    except:
        print("Error in ATTENTION MAPS")

    # PHASE 4: ARCHIVING
    # Save the exact version of the code and plan used for this specific run
    try: 
        scripts_to_archive = ["README.md", "plan.md", "train.py", "eval.py", "main.py", "dataset.py", "config.py", "test_patients.txt", "attention_visualizer.py", "utils.py"]
        for f in scripts_to_archive:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(exp_dir, f))
                
        logging.info(f"Pipeline finished successfully. Artifacts saved in {exp_dir}")

        # Send Notification
        msg = f"**OCT Research Update**\nRun: `{os.path.basename(exp_dir)}` completed.\nmDice: `{metrics['mDice']:.4f}` | mHD95: `{metrics['mHD95']:.2f}`"
        send_discord_notification(msg)
    except:
        print("Error in ARCHIVING")
        

if __name__ == "__main__":
    main()
