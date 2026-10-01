import argparse
import gc
import logging
import os
import traceback

import torch

from octseg.config import load_config, save_config
from octseg.attention_visualizer import generate_attention_maps
from octseg.evaluate import evaluate_model, log_metrics
from octseg.runs import (archive_experiment_sources, check_vram_diagnostics, make_run_dir,
                         send_discord_notification, setup_logging)
from octseg.train import train_model

log = logging.getLogger(__name__)


def main():
    """
    Orchestrates the full OCT segmentation research pipeline in ``outputs/runs/<timestamp>_<name>/``:
    1. Environment setup and logging
    2. SegFormer model training
    3. Metric evaluation
    4. Transformer attention visualization
    5. Source/config/git snapshot
    """
    parser = argparse.ArgumentParser(description="Train, evaluate and visualise a SegFormer run")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--output", default=None, help="Run directory (default: outputs/runs/<timestamp>_<name>)")
    parser.add_argument("--name", default="run", help="Run name suffix (default: run)")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_dir = str(make_run_dir("runs", args.name, args.output))
    model_path = os.path.join(exp_dir, "best_model.pth")
    setup_logging(exp_dir)
    save_config(config, os.path.join(exp_dir, "config.yaml"))
    log.info(f"Pipeline started. Results directory: {exp_dir}")

    log.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        log.info(f"GPU Max Memory Capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        check_vram_diagnostics()

    # PHASE 1: MODEL TRAINING
    log.info("--- PHASE 1: TRAINING ---")
    training_success = False
    try:
        train_model(exp_dir, cfg=config, epochs=args.epochs)
        if os.path.exists(model_path):
            training_success = True
            log.info("Training complete. Model weights stored safely.")
        else:
            log.error("Model training did not raise an exception, but best_model.pth was not created.")

        # Memory Management: Clear VRAM for the evaluation pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        log.error("CRITICAL ERROR during training:")
        log.error(traceback.format_exc())

    # PHASE 2: EVALUATION
    log.info("--- PHASE 2: EVALUATION ---")
    metrics = None
    if training_success:
        try:
            metrics = evaluate_model(model_path=model_path, output_dir=exp_dir, cfg=config)
            log_metrics(metrics, config)
        except Exception:
            log.error("ERROR during evaluation phase:")
            log.error(traceback.format_exc())
    else:
        log.error(f"Skipping Evaluation: Expected weights file '{model_path}' was not found.")

    # PHASE 3: INTERPRETABILITY (ATTENTION MAPS)
    log.info("--- PHASE 3: ATTENTION VISUALIZATION ---")
    if training_success:
        try:
            generate_attention_maps(model_path=model_path, output_dir=os.path.join(exp_dir, "attention_maps"), cfg=config)
        except Exception:
            log.error("ERROR during attention visualization phase:")
            log.error(traceback.format_exc())
    else:
        log.error(f"Skipping Attention Mapping: Expected weights file '{model_path}' was not found.")

    # PHASE 4: ARCHIVING
    archive_experiment_sources(exp_dir)
    log.info(f"Pipeline run fully finalized. All output saved to {exp_dir}")

    # Send Notification
    run = os.path.basename(exp_dir)
    if metrics:
        msg = f"**OCT Research Update**\nRun: `{run}` completed.\nmDice: `{metrics['mDice']:.4f}` | mHD95: `{metrics['mHD95']:.2f}`"
    elif training_success:
        msg = f"**OCT Research Update**\nRun: `{run}` weights saved, but failed/skipped during evaluation metrics pass."
    else:
        msg = f"**OCT Research Update**\nRun: `{run}` CRITICAL FAILURE during training pass."
    send_discord_notification(msg)


if __name__ == "__main__":
    main()
