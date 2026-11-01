import argparse
import gc
import logging
import os
import shutil
import sys
import traceback
from typing import Any, Dict, Optional

import torch

from octseg.config import ROOT, Config, load_config, save_config
from octseg.evaluate_hybrid import evaluate_hybrid
from octseg.runs import (
    archive_experiment_sources,
    check_vram_diagnostics,
    get_next_experiment_name,
    make_run_dir,
    promote_run_to_experiment,
    send_discord_notification,
    setup_logging,
)
from octseg.sweep_hybrid import sweep_hybrid
from octseg.train_expert import train_expert

log = logging.getLogger(__name__)


def pipeline_hybrid(
    base_weights: str,
    output_dir: str,
    cfg: Config,
    expert_cfg: Config,
    expert_weights: Optional[str] = None,
    epochs: Optional[int] = None,
    promote: bool = False,
    exp_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the complete hybrid pipeline: IRF expert training, validation sweep,

    test evaluation, source snapshotting, and optional experiment promotion.
    """
    if not os.path.exists(base_weights):
        raise FileNotFoundError(f"Base model checkpoint not found: {base_weights}")

    eval_dir = str(output_dir)
    os.makedirs(eval_dir, exist_ok=True)
    save_config(cfg, os.path.join(eval_dir, "config.yaml"))
    save_config(expert_cfg, os.path.join(eval_dir, "expert_config.yaml"))

    # Step 1: Expert Training (or reuse)
    if not expert_weights:
        expert_run_dir = os.path.join(eval_dir, "expert")
        os.makedirs(expert_run_dir, exist_ok=True)
        log.info(f"--- STEP 1: TRAINING IRF EXPERT ({expert_run_dir}) ---")
        train_expert(expert_run_dir, expert_cfg, epochs=epochs)
        expert_weights = os.path.join(expert_run_dir, "best_model.pth")
        if not os.path.exists(expert_weights):
            raise FileNotFoundError(
                f"Expert training completed but '{expert_weights}' was not found"
            )
        log.info(f"Expert training complete. Checkpoint: {expert_weights}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if not os.path.exists(expert_weights):
            raise FileNotFoundError(
                f"Expert model checkpoint not found: {expert_weights}"
            )
        log.info(f"--- STEP 1: REUSING IRF EXPERT WEIGHTS ({expert_weights}) ---")

    # Step 2: Validation Sweep
    sweep_dir = os.path.join(eval_dir, "sweep")
    log.info(f"--- STEP 2: HYBRID SWEEP (VAL) in {sweep_dir} ---")
    sweep_res = sweep_hybrid(
        base_weights=base_weights,
        expert_weights=expert_weights,
        output_dir=sweep_dir,
        cfg=cfg,
        expert_cfg=expert_cfg,
        split="val",
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Test Evaluation
    log.info("--- STEP 3: HYBRID TEST EVALUATION ---")
    best = sweep_res.get("best")
    if best is not None and "params" in best:
        eval_params = best["params"]
        log.info(f"Evaluating winning sweep configuration on test set: {eval_params}")
    else:
        eval_params = {}
        log.info(
            "No sweep configuration beat base model on val; using default hybrid config."
        )

    test_metrics = evaluate_hybrid(
        base_weights=base_weights,
        expert_weights=expert_weights,
        output_dir=eval_dir,
        cfg=cfg,
        expert_cfg=expert_cfg,
        ensemble_mode=eval_params.get("ensemble_mode"),
        expert_weight=eval_params.get("expert_weight"),
        blend_strategy=eval_params.get("blend_strategy"),
        irf_threshold=eval_params.get("irf_threshold"),
        irf_min_region_size=eval_params.get("irf_min_region_size"),
        irf_override=eval_params.get("irf_override"),
    )

    # Step 4: Snapshot sweep.yaml and configs into eval_dir
    log.info("--- STEP 4: SNAPSHOTTING CONFIGS AND SWEEP ---")
    sweep_yaml_src = os.path.join(sweep_dir, "sweep.yaml")
    sweep_yaml_dst = os.path.join(eval_dir, "sweep.yaml")
    if os.path.exists(sweep_yaml_src) and sweep_yaml_src != sweep_yaml_dst:
        shutil.copy2(sweep_yaml_src, sweep_yaml_dst)
    save_config(cfg, os.path.join(eval_dir, "config.yaml"))
    save_config(expert_cfg, os.path.join(eval_dir, "expert_config.yaml"))

    # Step 5: Archiving sources and optional promotion
    log.info("--- STEP 5: ARCHIVING SOURCES ---")
    archive_experiment_sources(eval_dir)
    if promote and exp_name:
        promote_run_to_experiment(eval_dir, exp_name)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Turnkey pipeline for hybrid base + IRF expert training, sweep, and evaluation"
    )
    parser.add_argument(
        "--base-weights",
        default="outputs/runs/2026-10-27_14-57-10_test20/best_model.pth",
        help="Path to multi-class base checkpoint (default: outputs/runs/2026-10-27_14-57-10_test20/best_model.pth)",
    )
    parser.add_argument(
        "--expert-weights",
        default=None,
        help="Optional path to existing expert checkpoint. If omitted, triggers train_expert",
    )
    parser.add_argument(
        "--config",
        default="configs/hybrid.yaml",
        help="Path to hybrid config YAML (default: configs/hybrid.yaml)",
    )
    parser.add_argument(
        "--expert-config",
        default="configs/irf_expert.yaml",
        help="Path to expert config YAML (default: configs/irf_expert.yaml)",
    )
    parser.add_argument(
        "--name",
        default="hybrid_eval",
        help="Suffix for run dir (default: hybrid_eval)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output directory (default: outputs/runs/<timestamp>_<name>)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for expert training",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Automatically copy run outputs into experiments/<exp_name> and archive",
    )
    args = parser.parse_args()

    if args.promote and args.name == "hybrid_eval":
        args.name = get_next_experiment_name()

    cfg = load_config(args.config)
    expert_cfg = load_config(args.expert_config)
    eval_dir = str(make_run_dir("runs", args.name, args.output))
    setup_logging(eval_dir)

    log.info(f"Hybrid pipeline started. Run directory: {eval_dir}")
    log.info(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    if torch.cuda.is_available():
        log.info(
            f"GPU Max Memory Capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        check_vram_diagnostics()

    base_weights = args.base_weights
    if not os.path.exists(base_weights):
        root_cand = str(ROOT / base_weights)
        if os.path.exists(root_cand):
            base_weights = root_cand

    expert_weights = args.expert_weights
    if expert_weights and not os.path.exists(expert_weights):
        root_cand = str(ROOT / expert_weights)
        if os.path.exists(root_cand):
            expert_weights = root_cand

    test_metrics = None
    try:
        test_metrics = pipeline_hybrid(
            base_weights=base_weights,
            output_dir=eval_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            expert_weights=expert_weights,
            epochs=args.epochs,
            promote=args.promote,
            exp_name=args.name,
        )
    except Exception:
        log.error("CRITICAL ERROR during hybrid pipeline:")
        log.error(traceback.format_exc())
        run = os.path.basename(eval_dir)
        send_discord_notification(
            f"**OCT Hybrid Pipeline Update**\nRun: `{run}` CRITICAL FAILURE during execution."
        )
        sys.exit(1)

    run = os.path.basename(eval_dir)
    irf_dice = test_metrics.get("class_dices", {}).get(
        1, test_metrics.get("class_dices", {}).get("1", 0.0)
    )
    msg = (
        f"**OCT Hybrid Pipeline Update**\nRun: `{run}` completed.\n"
        f"mDice: `{test_metrics['mDice']:.4f}` | IRF Dice: `{irf_dice:.4f}` | mHD95: `{test_metrics['mHD95']:.2f}`"
    )
    send_discord_notification(msg)


if __name__ == "__main__":
    main()
