"""Zero-compute calibration of SRF/PED thresholds, PED sharpening, and 3D volumetric consistency filtering."""

import argparse
from datetime import datetime
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from tqdm import tqdm

from octseg.config import CONFIG_DIR, ROOT, Config, load_config
from octseg.dataset import OCTDataset
from octseg.hybrid_inference import HybridInference
from octseg.metrics import SegmentationMetrics, save_metrics_yaml
from octseg.postprocess import volumetric_consistency_filter
from octseg.runs import setup_logging
from octseg.splits import split_files
from octseg.volume_data import group_volumes

log = logging.getLogger(__name__)

DEFAULT_BASE_WEIGHTS = "outputs/runs/2026-10-27_14-57-10_test20/best_model.pth"
DEFAULT_EXPERT_WEIGHTS = "outputs/runs/2026-11-01_15-44-27_test22/expert/best_model.pth"

SRF_THRESHOLDS = [0.45, 0.55, 0.65, 0.75, 0.80]
PED_THRESHOLDS = [0.45, 0.55, 0.65, 0.75, 0.80]
PED_SHARPEN_FACTORS = [0.0, 0.3, 0.7, 1.0]
VOLUMETRIC_FILTER_OPTIONS = [False, True]
MIN_SLICES = 2

TEST22_BASELINE: Dict[str, Any] = {
    "mIoU": 0.7233,
    "mDice": 0.8306,
    "mHD95": 73.82,
    "class_ious": {1: 0.5786, 2: 0.7216, 3: 0.6174},
    "class_dices": {1: 0.7331, 2: 0.8383, 3: 0.7634},
}


def predict_volume_masks(
    engine: HybridInference,
    base_probs_vol: np.ndarray,
    expert_irf_probs_vol: np.ndarray,
    srf_threshold: float,
    ped_threshold: float,
    ped_sharpen_factor: float,
    use_volumetric_filter: bool,
    min_slices: int = MIN_SLICES,
) -> np.ndarray:
    """Applies hybrid merging, thresholding, region cleaning, and optional 3D filtering.

    Args:
        engine: HybridInference instance.
        base_probs_vol: Base model probabilities [Z, C, H, W].
        expert_irf_probs_vol: Expert IRF probabilities [Z, H, W].
        srf_threshold: Decision threshold for SRF (class 2).
        ped_threshold: Decision threshold for PED (class 3).
        ped_sharpen_factor: Blend factor for PED sharpening.
        use_volumetric_filter: Whether to apply 3D connected-component consistency filtering.
        min_slices: Minimum slice span along Z-axis for 3D components.

    Returns:
        Predicted label masks volume [Z, H, W] (uint8).
    """
    prev_thresholds = dict(engine.cfg.CLASS_THRESHOLDS)
    prev_factor = getattr(engine.cfg, "PED_SHARPEN_FACTOR", 1.0)

    try:
        engine.cfg.CLASS_THRESHOLDS[2] = srf_threshold
        engine.cfg.CLASS_THRESHOLDS[3] = ped_threshold
        engine.cfg.PED_SHARPEN_FACTOR = ped_sharpen_factor

        z_dim = base_probs_vol.shape[0]
        slice_masks = [
            engine.postprocess(base_probs_vol[z], expert_irf_probs_vol[z])
            for z in range(z_dim)
        ]
        vol_masks = np.stack(slice_masks, axis=0)

        if use_volumetric_filter:
            vol_masks = volumetric_consistency_filter(vol_masks, min_slices=min_slices)

        return vol_masks
    finally:
        engine.cfg.CLASS_THRESHOLDS = prev_thresholds
        engine.cfg.PED_SHARPEN_FACTOR = prev_factor


def extract_probability_volumes(
    engine: HybridInference,
    split: str,
    cfg: Config,
) -> List[Dict[str, Any]]:
    """Runs inference across all volumes in `split` and caches probabilities and ground truth.

    Args:
        engine: HybridInference model wrapper.
        split: Dataset split ('train', 'val', or 'test').
        cfg: Configuration object.

    Returns:
        List of dicts containing patient ID, probabilities, masks, and original images.
    """
    image_paths, mask_paths = split_files(cfg, split)
    records = group_volumes(image_paths, mask_paths)
    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])

    volume_data: List[Dict[str, Any]] = []
    log.info(
        f"Extracting probabilities for {len(records)} volumes on '{split}' split..."
    )

    for record in tqdm(records, desc=f"Inference ({split})"):
        ds = OCTDataset(
            image_paths=record.image_paths,
            mask_paths=record.mask_paths,
            processor=engine.processor,
            transform=val_transform,
            cfg=cfg,
        )
        base_probs_list: List[np.ndarray] = []
        expert_irf_probs_list: List[np.ndarray] = []
        gt_masks_list: List[np.ndarray] = []
        orig_imgs_list: List[np.ndarray] = []

        for idx in range(len(ds)):
            item = ds[idx]
            image_np = item["orig_img"]
            gt_mask = item["labels"].numpy().astype(np.uint8)

            base_probs, expert_irf_prob = engine.probabilities(image_np)

            base_probs_list.append(base_probs)
            expert_irf_probs_list.append(expert_irf_prob)
            gt_masks_list.append(gt_mask)
            orig_imgs_list.append(image_np)

        volume_data.append(
            {
                "patient": record.patient,
                "base_probs": np.stack(base_probs_list, axis=0),
                "expert_irf_probs": np.stack(expert_irf_probs_list, axis=0),
                "gt_masks": np.stack(gt_masks_list, axis=0),
                "orig_imgs": orig_imgs_list,
            }
        )

    return volume_data


def compute_cm_metrics(
    cm: np.ndarray, num_labels: int
) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Computes mIoU, mDice, class_ious, and class_dices from a confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    ious = tp / (tp + fp + fn + 1e-6)
    dices = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    miou = float(np.mean(ious))
    mdice = float(np.mean(dices))
    class_ious = {c: float(ious[c]) for c in range(num_labels)}
    class_dices = {c: float(dices[c]) for c in range(num_labels)}
    return miou, mdice, class_ious, class_dices


def evaluate_grid(
    volume_data: List[Dict[str, Any]],
    engine: HybridInference,
    cfg: Config,
    srf_thresholds: List[float] = SRF_THRESHOLDS,
    ped_thresholds: List[float] = PED_THRESHOLDS,
    ped_sharpen_factors: List[float] = PED_SHARPEN_FACTORS,
    volumetric_filter_options: List[bool] = VOLUMETRIC_FILTER_OPTIONS,
    min_slices: int = MIN_SLICES,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Runs fast grid search on precomputed probability volumes.

    Returns:
        (base_summary, rows, best_row)
    """
    num_labels = cfg.NUM_LABELS
    total_configs = (
        len(srf_thresholds)
        * len(ped_thresholds)
        * len(ped_sharpen_factors)
        * len(volumetric_filter_options)
    )
    log.info(
        f"Running zero-compute grid search across {total_configs} configurations..."
    )

    # 1. Base / default summary
    base_srf = cfg.CLASS_THRESHOLDS.get(2, 0.8)
    base_ped = cfg.CLASS_THRESHOLDS.get(3, 0.8)
    base_sharpen = getattr(cfg, "PED_SHARPEN_FACTOR", 1.0)
    base_vol = False

    base_cm = np.zeros((num_labels, num_labels), dtype=np.int64)
    for vol in volume_data:
        pred_vol = predict_volume_masks(
            engine,
            vol["base_probs"],
            vol["expert_irf_probs"],
            srf_threshold=base_srf,
            ped_threshold=base_ped,
            ped_sharpen_factor=base_sharpen,
            use_volumetric_filter=base_vol,
            min_slices=min_slices,
        )
        gt_vol = vol["gt_masks"]
        valid = (gt_vol >= 0) & (gt_vol < num_labels)
        base_cm += np.bincount(
            num_labels * gt_vol[valid].astype(np.int64)
            + pred_vol[valid].astype(np.int64),
            minlength=num_labels**2,
        ).reshape(num_labels, num_labels)

    b_miou, b_mdice, b_ious, b_dices = compute_cm_metrics(base_cm, num_labels)
    base_summary = {
        "params": {
            "srf_threshold": float(base_srf),
            "ped_threshold": float(base_ped),
            "ped_sharpen_factor": float(base_sharpen),
            "use_volumetric_filter": base_vol,
        },
        "mIoU": b_miou,
        "mDice": b_mdice,
        "class_ious": b_ious,
        "class_dices": b_dices,
    }

    # 2. Grid evaluation
    rows: List[Dict[str, Any]] = []
    threshold_triplets = [
        (srf_t, ped_t, ped_f)
        for srf_t in srf_thresholds
        for ped_t in ped_thresholds
        for ped_f in ped_sharpen_factors
    ]

    for srf_t, ped_t, ped_f in tqdm(threshold_triplets, desc="Sweeping grid"):
        cm_unfiltered = np.zeros((num_labels, num_labels), dtype=np.int64)
        cm_filtered = np.zeros((num_labels, num_labels), dtype=np.int64)

        for vol in volume_data:
            vol_masks = predict_volume_masks(
                engine,
                vol["base_probs"],
                vol["expert_irf_probs"],
                srf_threshold=srf_t,
                ped_threshold=ped_t,
                ped_sharpen_factor=ped_f,
                use_volumetric_filter=False,
                min_slices=min_slices,
            )
            gt_vol = vol["gt_masks"]
            valid = (gt_vol >= 0) & (gt_vol < num_labels)

            cm_unfiltered += np.bincount(
                num_labels * gt_vol[valid].astype(np.int64)
                + vol_masks[valid].astype(np.int64),
                minlength=num_labels**2,
            ).reshape(num_labels, num_labels)

            if True in volumetric_filter_options:
                filtered_masks = volumetric_consistency_filter(
                    vol_masks, min_slices=min_slices
                )
                cm_filtered += np.bincount(
                    num_labels * gt_vol[valid].astype(np.int64)
                    + filtered_masks[valid].astype(np.int64),
                    minlength=num_labels**2,
                ).reshape(num_labels, num_labels)

        if False in volumetric_filter_options:
            u_miou, u_mdice, u_ious, u_dices = compute_cm_metrics(
                cm_unfiltered, num_labels
            )
            rows.append(
                {
                    "params": {
                        "srf_threshold": float(srf_t),
                        "ped_threshold": float(ped_t),
                        "ped_sharpen_factor": float(ped_f),
                        "use_volumetric_filter": False,
                    },
                    "mIoU": u_miou,
                    "mDice": u_mdice,
                    "class_ious": u_ious,
                    "class_dices": u_dices,
                }
            )

        if True in volumetric_filter_options:
            f_miou, f_mdice, f_ious, f_dices = compute_cm_metrics(
                cm_filtered, num_labels
            )
            rows.append(
                {
                    "params": {
                        "srf_threshold": float(srf_t),
                        "ped_threshold": float(ped_t),
                        "ped_sharpen_factor": float(ped_f),
                        "use_volumetric_filter": True,
                    },
                    "mIoU": f_miou,
                    "mDice": f_mdice,
                    "class_ious": f_ious,
                    "class_dices": f_dices,
                }
            )

    best_row = max(rows, key=lambda r: (r["mIoU"], r["mDice"]))
    return base_summary, rows, best_row


def evaluate_test_split(
    engine: HybridInference,
    best_params: Dict[str, Any],
    cfg: Config,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates uncalibrated baseline and calibrated winning configuration on test split."""
    test_volumes = extract_probability_volumes(engine, "test", cfg)

    # 1. Uncalibrated test metrics
    log.info("Evaluating uncalibrated base model on test split...")
    uncalibrated_metrics = SegmentationMetrics(
        cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg
    )
    base_srf = cfg.CLASS_THRESHOLDS.get(2, 0.8)
    base_ped = cfg.CLASS_THRESHOLDS.get(3, 0.8)
    base_sharpen = getattr(cfg, "PED_SHARPEN_FACTOR", 1.0)
    base_vol = False

    for vol in test_volumes:
        pred_vol = predict_volume_masks(
            engine,
            vol["base_probs"],
            vol["expert_irf_probs"],
            srf_threshold=base_srf,
            ped_threshold=base_ped,
            ped_sharpen_factor=base_sharpen,
            use_volumetric_filter=base_vol,
            min_slices=MIN_SLICES,
        )
        for z in range(pred_vol.shape[0]):
            uncalibrated_metrics.update(
                vol["gt_masks"][z], pred_vol[z], vol["orig_imgs"][z]
            )
    test_uncalibrated = uncalibrated_metrics.compute()

    # 2. Calibrated test metrics
    log.info(f"Evaluating calibrated model on test split with params: {best_params}...")
    calibrated_metrics = SegmentationMetrics(
        cfg.NUM_LABELS, hd95_missing="skip", extended=True, cfg=cfg
    )

    for vol in test_volumes:
        pred_vol = predict_volume_masks(
            engine,
            vol["base_probs"],
            vol["expert_irf_probs"],
            srf_threshold=best_params["srf_threshold"],
            ped_threshold=best_params["ped_threshold"],
            ped_sharpen_factor=best_params["ped_sharpen_factor"],
            use_volumetric_filter=best_params["use_volumetric_filter"],
            min_slices=MIN_SLICES,
        )
        for z in range(pred_vol.shape[0]):
            calibrated_metrics.update(
                vol["gt_masks"][z], pred_vol[z], vol["orig_imgs"][z]
            )
    test_calibrated = calibrated_metrics.compute()

    return test_uncalibrated, test_calibrated


def calibrate_postprocess(
    base_weights: str = DEFAULT_BASE_WEIGHTS,
    expert_weights: str = DEFAULT_EXPERT_WEIGHTS,
    config_path: str = str(CONFIG_DIR / "hybrid.yaml"),
    expert_config_path: str = str(CONFIG_DIR / "irf_expert.yaml"),
    split: str = "val",
    eval_test: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs probability extraction, grid search calibration, and optional test evaluation."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = str(ROOT / "outputs" / "calibration" / timestamp)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, "calibration.log")

    log.info("=" * 70)
    log.info("STARTING ZERO-COMPUTE POST-PROCESSING CALIBRATION")
    log.info("=" * 70)
    log.info(f"Base Weights: {base_weights}")
    log.info(f"Expert Weights: {expert_weights}")
    log.info(f"Config: {config_path} | Expert Config: {expert_config_path}")
    log.info(f"Sweep Split: {split} | Eval Test: {eval_test}")
    log.info(f"Output Directory: {output_dir}")

    cfg, expert_cfg = load_config(config_path), load_config(expert_config_path)
    engine = HybridInference(base_weights, expert_weights, cfg, expert_cfg)

    # Step 1: Precompute / extract raw probability volumes
    val_volumes = extract_probability_volumes(engine, split, cfg)

    # Step 2: Grid search on validation split
    base_summary, rows, best_row = evaluate_grid(val_volumes, engine, cfg)

    log.info("=" * 70)
    log.info(f"CALIBRATION GRID SEARCH COMPLETED ({len(rows)} configurations)")
    log.info("=" * 70)
    log.info(
        f"Base config: srf={base_summary['params']['srf_threshold']:.2f}, "
        f"ped={base_summary['params']['ped_threshold']:.2f}, "
        f"ped_sharpen={base_summary['params']['ped_sharpen_factor']:.1f}, "
        f"vol_filter={base_summary['params']['use_volumetric_filter']} -> "
        f"mIoU: {base_summary['mIoU']:.4f} | mDice: {base_summary['mDice']:.4f} | "
        f"SRF Dice: {base_summary['class_dices'][2]:.4f} | PED Dice: {base_summary['class_dices'][3]:.4f}"
    )
    log.info(
        f"Best config: srf={best_row['params']['srf_threshold']:.2f}, "
        f"ped={best_row['params']['ped_threshold']:.2f}, "
        f"ped_sharpen={best_row['params']['ped_sharpen_factor']:.1f}, "
        f"vol_filter={best_row['params']['use_volumetric_filter']} -> "
        f"mIoU: {best_row['mIoU']:.4f} | mDice: {best_row['mDice']:.4f} | "
        f"SRF Dice: {best_row['class_dices'][2]:.4f} | PED Dice: {best_row['class_dices'][3]:.4f}"
    )
    delta_val_miou = best_row["mIoU"] - base_summary["mIoU"]
    delta_val_mdice = best_row["mDice"] - base_summary["mDice"]
    log.info(
        f"Validation Gain: Delta mIoU = {delta_val_miou:+.4f} | Delta mDice = {delta_val_mdice:+.4f}"
    )

    results: Dict[str, Any] = {
        "split": split,
        "best_params": best_row["params"],
        "val_baseline": base_summary,
        "val_calibrated": best_row,
        "grid_rows": rows,
    }

    # Step 3: Optional test set evaluation & benchmark comparison
    if eval_test:
        test_uncalibrated, test_calibrated = evaluate_test_split(
            engine, best_row["params"], cfg
        )

        delta_vs_uncal = {
            "mIoU": float(test_calibrated["mIoU"] - test_uncalibrated["mIoU"]),
            "mDice": float(test_calibrated["mDice"] - test_uncalibrated["mDice"]),
            "mHD95": float(test_calibrated["mHD95"] - test_uncalibrated["mHD95"]),
        }
        delta_vs_t22 = {
            "mIoU": float(test_calibrated["mIoU"] - TEST22_BASELINE["mIoU"]),
            "mDice": float(test_calibrated["mDice"] - TEST22_BASELINE["mDice"]),
            "mHD95": float(test_calibrated["mHD95"] - TEST22_BASELINE["mHD95"]),
        }

        log.info("=" * 70)
        log.info("TEST SPLIT COMPARISON: CALIBRATED vs UNCALIBRATED BASE vs TEST22")
        log.info("=" * 70)
        log.info(
            f"test22 Milestone : mIoU: {TEST22_BASELINE['mIoU']:.4f} | "
            f"mDice: {TEST22_BASELINE['mDice']:.4f} | "
            f"mHD95: {TEST22_BASELINE['mHD95']:.2f} mm"
        )
        log.info(
            f"Uncalibrated Base: mIoU: {test_uncalibrated['mIoU']:.4f} | "
            f"mDice: {test_uncalibrated['mDice']:.4f} | "
            f"mHD95: {test_uncalibrated['mHD95']:.2f} mm"
        )
        log.info(
            f"Calibrated Best  : mIoU: {test_calibrated['mIoU']:.4f} | "
            f"mDice: {test_calibrated['mDice']:.4f} | "
            f"mHD95: {test_calibrated['mHD95']:.2f} mm"
        )
        log.info(
            f"Delta vs Uncalibrated : Delta mIoU: {delta_vs_uncal['mIoU']:+.4f} | "
            f"Delta mDice: {delta_vs_uncal['mDice']:+.4f} | "
            f"Delta mHD95: {delta_vs_uncal['mHD95']:+.2f} mm"
        )
        log.info(
            f"Delta vs test22       : Delta mIoU: {delta_vs_t22['mIoU']:+.4f} | "
            f"Delta mDice: {delta_vs_t22['mDice']:+.4f} | "
            f"Delta mHD95: {delta_vs_t22['mHD95']:+.2f} mm"
        )

        for c in range(1, cfg.NUM_LABELS):
            c_name = cfg.CLASS_NAMES.get(c, str(c))
            log.info(
                f"Class {c} ({c_name}): "
                f"IoU {test_uncalibrated['class_ious'][c]:.4f} -> {test_calibrated['class_ious'][c]:.4f} "
                f"(test22: {TEST22_BASELINE['class_ious'].get(c, 0.0):.4f}) | "
                f"Dice {test_uncalibrated['class_dices'][c]:.4f} -> {test_calibrated['class_dices'][c]:.4f} "
                f"(test22: {TEST22_BASELINE['class_dices'].get(c, 0.0):.4f}) | "
                f"HD95 {test_uncalibrated['class_hd95'][c]:.2f} -> {test_calibrated['class_hd95'][c]:.2f} mm"
            )

        results["test_baseline_test22"] = TEST22_BASELINE
        results["test_uncalibrated"] = test_uncalibrated
        results["test_calibrated"] = test_calibrated
        results["comparison"] = {
            "delta_vs_uncalibrated": delta_vs_uncal,
            "delta_vs_test22": delta_vs_t22,
        }

    results_yaml_path = os.path.join(output_dir, "calibration_results.yaml")
    save_metrics_yaml(results, results_yaml_path)
    log.info(f"Saved calibration results to {results_yaml_path}")
    return results


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-Compute Calibration of Post-Processing Hyperparameters on OCT Volumes"
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default=DEFAULT_BASE_WEIGHTS,
        help=f"Path to base model weights (default: {DEFAULT_BASE_WEIGHTS})",
    )
    parser.add_argument(
        "--expert-weights",
        type=str,
        default=DEFAULT_EXPERT_WEIGHTS,
        help=f"Path to expert model weights (default: {DEFAULT_EXPERT_WEIGHTS})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "hybrid.yaml"),
        help="Path to base/hybrid YAML config (default: configs/hybrid.yaml)",
    )
    parser.add_argument(
        "--expert-config",
        type=str,
        default=str(CONFIG_DIR / "irf_expert.yaml"),
        help="Path to expert YAML config (default: configs/irf_expert.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split for sweep (default: val)",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate winning parameters on test split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for logs and metrics",
    )
    return parser.parse_args(args)


def main():
    """CLI entry point for octseg.calibrate_postprocess."""
    args = parse_args()
    try:
        calibrate_postprocess(
            base_weights=args.base_weights,
            expert_weights=args.expert_weights,
            config_path=args.config,
            expert_config_path=args.expert_config,
            split=args.split,
            eval_test=args.eval_test,
            output_dir=args.output,
        )
    except FileNotFoundError as e:
        log.error(f"Checkpoint or data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unhandled error during calibration: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
