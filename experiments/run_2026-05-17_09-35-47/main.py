import os
import logging
import shutil
import gc
import torch
from datetime import datetime
from train import train_model
from eval import evaluate_model
import config

def setup_logging(exp_dir):
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

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.abspath(f"experiments/run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    setup_logging(exp_dir)
    logging.info(f"Pipeline started. Results directory: {exp_dir}")
    
    # Log hardware info
    logging.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # training
    best_model_name = "best_model.pth"
    logging.info("--- TRAINING ---")
    train_model(epochs=config.EPOCHS, save_path=best_model_name, output_dir=exp_dir)
    
    if os.path.exists(best_model_name):
        shutil.copy(best_model_name, os.path.join(exp_dir, "best_model.pth"))

    # Force cleanup before evaluation - OOM prevention
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # evaluation
    logging.info("--- EVALUATION ---")
    metrics = evaluate_model(model_path=best_model_name, output_dir=exp_dir)
    
    logging.info(f"Final mIoU: {metrics['mIoU']:.4f} | Final mDice: {metrics['mDice']:.4f}")
    logging.info(f"Final mHD95: {metrics['mHD95']:.4f} | Final mASD: {metrics['mASD']:.4f}")
    
    for c in [1, 2, 3]:
        name = config.CLASS_NAMES[c]
        iou = metrics['class_ious'][c]
        dice = metrics['class_dices'][c]
        hd = metrics['class_hd95'][c]
        logging.info(f"Class {c} ({name}) | IoU: {iou:.4f} | Dice: {dice:.4f} | HD95: {hd:.2f}")

    # archiving project docs and scripts
    for f in ["README.md", "plan.md", "train.py", "eval.py", "main.py", "dataset.py", "config.py", "test_patients.txt"]:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(exp_dir, f))
            
    logging.info(f"Pipeline finished.")

if __name__ == "__main__":
    main()
