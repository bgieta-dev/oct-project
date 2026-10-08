import os
import logging
import shutil
import gc
import torch
import requests
import json
import sys
import traceback
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

def check_vram_diagnostics():
    """Reports explicit VRAM usage metrics prior to heavy training loops to catch potential out-of-memory states."""
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            logging.info(f"VRAM Status Diagnostic: {free_gb:.2f} GB Free / {total_gb:.2f} GB Total")
            if free_gb < 2.0:
                logging.warning("Extremely low VRAM detected (< 2.0 GB). Risk of training Out-Of-Memory failure is elevated.")
        except Exception as e:
            logging.error(f"Could not retrieve precise VRAM information: {e}")

def archive_experiment_sources(exp_dir: str):
    """Dynamically archives project code and configurations into the run directory, skipping experiments/ and system files."""
    logging.info("--- PHASE 4: DYNAMIC CODE ARCHIVING ---")
    try:
        # Scan current project root dynamically for relevant source/configuration artifacts
        for entry in os.listdir("."):
            if os.path.isfile(entry):
                if entry.endswith((".py", ".md", ".txt", ".sh", ".bat")) or entry == "requirements.txt":
                    shutil.copy(entry, os.path.join(exp_dir, entry))
        logging.info("Dynamic source code and configurations archiving complete.")
    except Exception as e:
        logging.error(f"Error encountered during dynamic code archiving: {e}")
        logging.error(traceback.format_exc())

# --- MAIN EXECUTION PIPELINE ---

def main():
    """
    Orchestrates the full OCT segmentation research pipeline:
    1. Environment setup and logging
    2. SegFormer Model Training
    3. Comprehensive Metric Evaluation
    4. Transformer Attention Visualization
    5. Dynamic Source Artifact Archiving
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.abspath(f"experiments/run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    setup_logging(exp_dir)
    logging.info(f"Pipeline started. Results directory: {exp_dir}")
    
    # [CLEANUP] Remove stale weights to prevent architecture mismatch (e.g., B2 weights loading into B3)
    best_model_name = "best_model.pth"
    if os.path.exists(best_model_name):
        os.remove(best_model_name)
        logging.info(f"Removed stale weights at {best_model_name}")
    
    # Hardware Diagnostics
    logging.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logging.info(f"GPU Max Memory Capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        check_vram_diagnostics()

    # PHASE 1: MODEL TRAINING
    logging.info("--- PHASE 1: TRAINING ---")
    training_success = False
    try:
        train_model(epochs=config.EPOCHS, save_path=best_model_name, output_dir=exp_dir)
        
        if os.path.exists(best_model_name):
            shutil.copy(best_model_name, os.path.join(exp_dir, "best_model.pth"))
            training_success = True
            logging.info("Training complete. Model weights stored safely.")
        else:
            logging.error("Model training did not raise an exception, but best_model.pth was not created.")

        # Memory Management: Clear VRAM for the evaluation pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logging.error("CRITICAL ERROR during training:")
        logging.error(traceback.format_exc())

    # PHASE 2: EVALUATION
    logging.info("--- PHASE 2: EVALUATION ---")
    metrics = None
    if os.path.exists(best_model_name):
        try:
            metrics = evaluate_model(model_path=best_model_name, output_dir=exp_dir)
        
            # Log Aggregated Metrics
            logging.info(f"Final mIoU: {metrics['mIoU']:.4f} | Final mDice: {metrics['mDice']:.4f}")
            logging.info(f"Final mHD95: {metrics['mHD95']:.4f} | Final mASD: {metrics['mASD']:.4f}")
            logging.info(f"Model Parameters: {metrics.get('params', 0):.2f}M")
            
            # Log Class-Specific Findings (Essential for Thesis Tables)
            for c in range(1, config.NUM_LABELS):
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
        except Exception:
            logging.error("ERROR during evaluation phase:")
            logging.error(traceback.format_exc())
    else:
        logging.error(f"Skipping Evaluation: Expected weights file '{best_model_name}' was not found.")

    # PHASE 3: INTERPRETABILITY (ATTENTION MAPS)
    logging.info("--- PHASE 3: ATTENTION VISUALIZATION ---")
    if os.path.exists(best_model_name):
        try:
            att_dir = os.path.join(exp_dir, "attention_maps")
            generate_attention_maps(model_path=best_model_name, output_dir=att_dir)
        except Exception:
            logging.error("ERROR during attention visualization phase:")
            logging.error(traceback.format_exc())
    else:
        logging.error(f"Skipping Attention Mapping: Expected weights file '{best_model_name}' was not found.")

    # PHASE 4: ARCHIVING
    archive_experiment_sources(exp_dir)
    logging.info(f"Pipeline run fully finalized. All output saved to {exp_dir}")

    # Send Notification
    try:
        if metrics:
            msg = f"**OCT Research Update**\nRun: `{os.path.basename(exp_dir)}` completed.\nmDice: `{metrics['mDice']:.4f}` | mHD95: `{metrics['mHD95']:.2f}`"
            send_discord_notification(msg)
        elif training_success:
            send_discord_notification(f"**OCT Research Update**\nRun: `{os.path.basename(exp_dir)}` weights saved, but failed/skipped during evaluation metrics pass.")
        else:
            send_discord_notification(f"**OCT Research Update**\nRun: `{os.path.basename(exp_dir)}` CRITICAL FAILURE during training pass.")
    except Exception as e:
        logging.error(f"Failed to post final notification: {e}")

if __name__ == "__main__":
    main()
