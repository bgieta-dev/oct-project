import argparse
import logging
import os
import sys
import traceback
from datetime import datetime

import albumentations as A
from tqdm import tqdm

from octseg.config import CONFIG_DIR, ROOT, Config, load_config, save_config
from octseg.dataset import OCTDataset
from octseg.hybrid_inference import BLEND_STRATEGIES, HybridInference
from octseg.metrics import SegmentationMetrics, save_metrics_yaml
from octseg.runs import setup_logging
from octseg.splits import load_splits, patient_of
from octseg.viz import save_predictions_grid, select_vis_indices

log = logging.getLogger(__name__)


def evaluate_hybrid(base_weights, expert_weights, output_dir, cfg: Config, expert_cfg: Config, ensemble_mode=None, expert_weight=None,
                    blend_strategy=None, irf_threshold=None, irf_min_region_size=None, irf_override=None):
    """Evaluates the Base (multi-class) + Expert (binary IRF) ensemble on the frozen test patients.

    Unset hybrid parameters default to the ``HYBRID_*`` config entries. Raises FileNotFoundError
    if a checkpoint is missing. Returns the metrics dict (also written to metrics.yaml).
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info("--- STARTING HYBRID EVALUATION ---")

    engine = HybridInference(
        base_weights, expert_weights, cfg, expert_cfg,
        ensemble_mode=ensemble_mode, expert_weight=expert_weight, blend_strategy=blend_strategy,
        irf_threshold=irf_threshold, irf_min_region_size=irf_min_region_size, irf_override=irf_override,
    )
    log.info(f"Ensemble Mode: {engine.ensemble_mode} | Expert Weight: {engine.expert_weight} | Blend Strategy: {engine.blend_strategy} | IRF Threshold: {engine.irf_threshold} | IRF Min Region Size: {engine.irf_min_region_size} | IRF Override: {engine.irf_override}")

    all_files = sorted(os.listdir(cfg.IMG_DIR))
    test_patients = set(load_splits(cfg.SPLIT_DIR)[2])
    test_imgs = [os.path.join(cfg.IMG_DIR, f) for f in all_files if patient_of(f) in test_patients]
    test_masks = [os.path.join(cfg.MASK_DIR, f) for f in all_files if patient_of(f) in test_patients]

    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])
    ds = OCTDataset(image_paths=test_imgs, mask_paths=test_masks, processor=engine.processor,
                    transform=val_transform, cfg=cfg)
    log.info(f"Testing on {len(ds)} images from {len(test_patients)} patients.")

    vis_indices, class_max_counts = select_vis_indices(test_masks, list(range(1, cfg.NUM_LABELS)))
    log.info(f"Dynamic vis indices: {vis_indices}")

    metrics = SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg)
    vis_data = {}

    for idx in tqdm(range(len(ds)), desc="Evaluating Hybrid"):
        item = ds[idx]
        image_np = item["orig_img"]  # 2.5D context [H, W, 3]
        gt_mask = item["labels"].numpy().astype("uint8")
        pred_mask, att_map = engine.segment(image_np, return_attention=True)
        metrics.update(gt_mask, pred_mask, image_np)
        for c_key, t_idx in vis_indices.items():
            if idx == t_idx:
                vis_data[c_key] = (image_np, gt_mask, pred_mask, att_map)

    log.info("Generating predictions.png...")
    save_predictions_grid(vis_data, class_max_counts, cfg, os.path.join(output_dir, "predictions.png"))

    result = metrics.compute()
    save_metrics_yaml(result, os.path.join(output_dir, "metrics.yaml"))

    log.info("=" * 30)
    log.info("FINAL HYBRID METRICS")
    log.info("=" * 30)
    log.info(f"mIoU: {result['mIoU']:.4f} | mDice: {result['mDice']:.4f}")
    log.info(f"mHD95: {result['mHD95']:.4f} | mASD: {result['mASD']:.4f}")
    for c in range(1, cfg.NUM_LABELS):
        log.info(f"Class {c} ({cfg.CLASS_NAMES[c]}) | IoU: {result['class_ious'][c]:.4f} | Dice: {result['class_dices'][c]:.4f} | HD95: {result['class_hd95'][c]:.2f} | ASD: {result['class_asd'][c]:.2f}")
        log.info(f"  Regions GT/Pred: {result['class_avg_regions_gt'][c]:.1f}/{result['class_avg_regions_pred'][c]:.1f} | BP: {result['class_boundary_precision'][c]:.4f} | Avg Area: {result['class_avg_pixel_area'][c]:.1f} px")
    return result


def main():
    parser = argparse.ArgumentParser(description="Clinical Hybrid Ensemble Evaluation Pipeline")
    parser.add_argument("--config", default=str(CONFIG_DIR / "hybrid.yaml"), help="Base-model YAML config incl. HYBRID_* defaults (default: configs/hybrid.yaml)")
    parser.add_argument("--expert-config", default=str(CONFIG_DIR / "irf_expert.yaml"), help="Expert YAML config (default: configs/irf_expert.yaml)")
    parser.add_argument("--base-weights", type=str, required=True, help="Path to base model weights")
    parser.add_argument("--expert-weights", type=str, required=True, help="Path to expert model weights")
    parser.add_argument("--output", type=str, help="Output directory (default: outputs/hybrid_eval/<timestamp>)")
    # Hybrid options default to the HYBRID_* entries of --config.
    parser.add_argument("--ensemble-mode", type=str, default=None, choices=["soft", "hard"], help="soft (probability blend) or hard (mask override)")
    parser.add_argument("--expert-weight", type=float, default=None, help="Weight of expert predictions in soft ensemble (0.0 to 1.0)")
    parser.add_argument("--blend-strategy", type=str, default=None, choices=BLEND_STRATEGIES, help="Blending strategy for soft ensembling")
    parser.add_argument("--irf-threshold", type=float, default=None, help="Decision threshold for IRF; lower = more sensitive")
    parser.add_argument("--irf-min-region-size", type=int, default=None, help="Minimum IRF region size in pixels")
    parser.add_argument("--irf-override", action=argparse.BooleanOptionalAction, default=None, help="Allow IRF to override/fragment SRF and PED (not recommended)")
    args = parser.parse_args()

    cfg, expert_cfg = load_config(args.config), load_config(args.expert_config)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.abspath(args.output or ROOT / "outputs" / "hybrid_eval" / timestamp)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, "hybrid_eval.log")
    save_config(cfg, os.path.join(output_dir, "config.yaml"))
    save_config(expert_cfg, os.path.join(output_dir, "expert_config.yaml"))
    try:
        evaluate_hybrid(
            base_weights=args.base_weights, expert_weights=args.expert_weights, output_dir=output_dir,
            cfg=cfg, expert_cfg=expert_cfg,
            ensemble_mode=args.ensemble_mode, expert_weight=args.expert_weight,
            blend_strategy=args.blend_strategy, irf_threshold=args.irf_threshold,
            irf_min_region_size=args.irf_min_region_size, irf_override=args.irf_override,
        )
    except Exception:
        log.error("CRITICAL ERROR encountered during hybrid evaluation:")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
