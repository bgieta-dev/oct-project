import argparse
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
from tqdm import tqdm

from octseg.config import CONFIG_DIR, Config, load_config, save_config
from octseg.dataset import OCTDataset
from octseg.hybrid_inference import HybridInference
from octseg.metrics import SegmentationMetrics, save_metrics_yaml
from octseg.postprocess import apply_thresholds, clean_regions, sharpen_ped
from octseg.runs import make_run_dir, setup_logging
from octseg.splits import split_files

log = logging.getLogger(__name__)

SWEEP_GRID = (
    [{"ensemble_mode": "soft", "blend_strategy": "linear", "expert_weight": w, "irf_threshold": t}
     for w in (0.3, 0.4, 0.5, 0.6) for t in (0.25, 0.35)]
    + [{"ensemble_mode": "soft", "blend_strategy": s, "expert_weight": 0.5, "irf_threshold": t}
       for s in ("confidence", "max") for t in (0.25, 0.35)]
    + [{"ensemble_mode": "hard"}, {"ensemble_mode": "replace"}]
)
SRF_PED_TOLERANCE = 0.01

SUMMARY_KEYS = (
    "mIoU",
    "mDice",
    "mHD95",
    "class_dices",
    "class_ious",
    "class_avg_regions_gt",
    "class_avg_regions_pred",
)


def _get_dice(d: Dict[str, Any], class_id: int) -> float:
    dices = d["class_dices"] if "class_dices" in d else d
    if class_id in dices:
        return float(dices[class_id])
    if str(class_id) in dices:
        return float(dices[str(class_id)])
    raise KeyError(f"Class {class_id} not found in class_dices")


def select_best(rows: List[Dict[str, Any]], base_row: Dict[str, Any], tolerance: float = SRF_PED_TOLERANCE) -> Optional[Dict[str, Any]]:
    """Row with the highest IRF Dice among rows that beat base IRF Dice and keep SRF/PED Dice within
    `tolerance` of the base; None if no row qualifies. Ties: first in grid order.
    """
    base_irf = _get_dice(base_row, 1)
    base_srf = _get_dice(base_row, 2)
    base_ped = _get_dice(base_row, 3)

    best = None
    best_irf = base_irf
    for row in rows:
        irf = _get_dice(row, 1)
        srf = _get_dice(row, 2)
        ped = _get_dice(row, 3)
        if irf > base_irf and srf >= base_srf - tolerance and ped >= base_ped - tolerance:
            if irf > best_irf:
                best = row
                best_irf = irf
    return best


def _format_row_label(params: Dict[str, Any]) -> str:
    mode = params.get("ensemble_mode")
    if mode == "soft":
        strat = params.get("blend_strategy")
        w = params.get("expert_weight")
        t = params.get("irf_threshold")
        return f"{mode}/{strat}/{w}/{t}"
    return str(mode)


def _format_metrics_line(label: str, row: Dict[str, Any]) -> str:
    irf_dice = _get_dice(row, 1)
    srf_dice = _get_dice(row, 2)
    ped_dice = _get_dice(row, 3)
    reg_pred = row.get("class_avg_regions_pred", {})
    reg_gt = row.get("class_avg_regions_gt", {})
    irf_pred = reg_pred.get(1, reg_pred.get("1", 0.0))
    irf_gt = reg_gt.get(1, reg_gt.get("1", 0.0))
    return (
        f"{label:32s} | IRF Dice: {irf_dice:.4f} | SRF Dice: {srf_dice:.4f} "
        f"| PED Dice: {ped_dice:.4f} | IRF regions pred/gt: {irf_pred:.1f}/{irf_gt:.1f}"
    )


def sweep_hybrid(base_weights: str, expert_weights: str, output_dir: str, cfg: Config, expert_cfg: Config, split: str = "val") -> Dict[str, Any]:
    """Sweeps hybrid ensemble hyperparameter combinations on the given split."""
    if cfg.ARCH != "segformer" or expert_cfg.ARCH != "segformer":
        raise ValueError("Hybrid sweep requires ARCH=segformer")

    os.makedirs(output_dir, exist_ok=True)
    log.info(f"--- STARTING HYBRID SWEEP (split={split}) ---")

    engine = HybridInference(base_weights, expert_weights, cfg, expert_cfg)

    imgs, masks = split_files(cfg, split)
    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])
    ds = OCTDataset(image_paths=imgs, mask_paths=masks, processor=engine.processor,
                    transform=val_transform, cfg=cfg)
    log.info(f"Sweeping across {len(SWEEP_GRID)} configurations on {len(ds)} images ({split} split).")

    base_metrics = SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg)
    grid_metrics = [
        SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg)
        for _ in SWEEP_GRID
    ]

    for idx in tqdm(range(len(ds)), desc=f"Sweeping Hybrid ({split})"):
        item = ds[idx]
        image_np = item["orig_img"]
        gt_mask = item["labels"].numpy()

        base_probs, expert_irf_prob = engine.probabilities(image_np)

        # Base row postprocessing
        base_probs_copy = sharpen_ped(base_probs.copy(), factor=cfg.PED_SHARPEN_FACTOR)
        base_pred = apply_thresholds(base_probs_copy, cfg.CLASS_THRESHOLDS, irf_override=True)
        base_pred = clean_regions(base_pred, cfg.MIN_REGION_SIZE, irf_open=True)
        base_metrics.update(gt_mask, base_pred, image_np)

        # Grid rows postprocessing
        for params, m in zip(SWEEP_GRID, grid_metrics):
            engine.ensemble_mode = params.get("ensemble_mode", cfg.HYBRID_ENSEMBLE_MODE)
            engine.blend_strategy = params.get("blend_strategy", cfg.HYBRID_BLEND_STRATEGY)
            engine.expert_weight = params.get("expert_weight", cfg.HYBRID_EXPERT_WEIGHT)
            engine.irf_threshold = params.get("irf_threshold", cfg.HYBRID_IRF_THRESHOLD)
            engine.irf_min_region_size = cfg.HYBRID_IRF_MIN_REGION_SIZE
            engine.irf_override = cfg.HYBRID_IRF_OVERRIDE

            pred_mask = engine.postprocess(base_probs, expert_irf_prob)
            m.update(gt_mask, pred_mask, image_np)

    base_summary = {k: base_metrics.compute()[k] for k in SUMMARY_KEYS}
    rows = []
    for params, m in zip(SWEEP_GRID, grid_metrics):
        summary = {k: m.compute()[k] for k in SUMMARY_KEYS}
        rows.append({"params": dict(params), **summary})

    best = select_best(rows, base_summary, tolerance=SRF_PED_TOLERANCE)

    sweep_result = {
        "split": split,
        "base": base_summary,
        "rows": rows,
        "best": best,
    }
    save_metrics_yaml(sweep_result, os.path.join(output_dir, "sweep.yaml"))

    log.info("=" * 70)
    log.info("SWEEP SUMMARY")
    log.info("=" * 70)
    log.info(_format_metrics_line("base", base_summary))
    for row in rows:
        log.info(_format_metrics_line(_format_row_label(row["params"]), row))
    log.info("=" * 70)

    if best is not None:
        p = best["params"]
        mode = p["ensemble_mode"]
        cmd_parts = [
            "python -m octseg.evaluate_hybrid",
            f"--base-weights {base_weights}",
            f"--expert-weights {expert_weights}",
            f"--ensemble-mode {mode}",
        ]
        if mode == "soft":
            cmd_parts.extend([
                f"--blend-strategy {p['blend_strategy']}",
                f"--expert-weight {p['expert_weight']}",
                f"--irf-threshold {p['irf_threshold']}",
            ])
        cmd = " ".join(cmd_parts)
        log.info(f"Recommended test evaluation command:\n{cmd}")
    else:
        log.info(f"No hybrid configuration beats the base model on {split}; hybrid not adopted.")

    return sweep_result


def main():
    parser = argparse.ArgumentParser(description="Clinical Hybrid Ensemble Hyperparameter Sweep")
    parser.add_argument("--config", default=str(CONFIG_DIR / "hybrid.yaml"), help="Base-model YAML config (default: configs/hybrid.yaml)")
    parser.add_argument("--expert-config", default=str(CONFIG_DIR / "irf_expert.yaml"), help="Expert YAML config (default: configs/irf_expert.yaml)")
    parser.add_argument("--base-weights", type=str, required=True, help="Path to base model weights")
    parser.add_argument("--expert-weights", type=str, required=True, help="Path to expert model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate (default: val)")
    parser.add_argument("--name", type=str, default="hybrid_sweep", help="Run name suffix (default: hybrid_sweep)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg, expert_cfg = load_config(args.config), load_config(args.expert_config)
    run_dir = str(make_run_dir("hybrid_sweep", name=args.name, output=args.output))
    setup_logging(run_dir, "sweep.log")
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    save_config(expert_cfg, os.path.join(run_dir, "expert_config.yaml"))

    try:
        sweep_hybrid(
            base_weights=args.base_weights,
            expert_weights=args.expert_weights,
            output_dir=run_dir,
            cfg=cfg,
            expert_cfg=expert_cfg,
            split=args.split,
        )
    except Exception:
        log.error("CRITICAL ERROR encountered during hybrid sweep:")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
