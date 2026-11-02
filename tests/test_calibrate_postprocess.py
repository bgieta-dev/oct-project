"""Unit tests for octseg.calibrate_postprocess."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from octseg.calibrate_postprocess import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_EXPERT_WEIGHTS,
    calibrate_postprocess,
    compute_cm_metrics,
    evaluate_grid,
    evaluate_test_split,
    extract_probability_volumes,
    parse_args,
    predict_volume_masks,
)
from octseg.config import Config
from octseg.volume_data import VolumeRecord


def test_parse_args_defaults():
    args = parse_args([])
    assert args.base_weights == DEFAULT_BASE_WEIGHTS
    assert args.expert_weights == DEFAULT_EXPERT_WEIGHTS
    assert "hybrid.yaml" in args.config
    assert "irf_expert.yaml" in args.expert_config
    assert args.split == "val"
    assert args.eval_test is False
    assert args.output is None


def test_parse_args_custom():
    custom_args = [
        "--base-weights",
        "weights/base.pth",
        "--expert-weights",
        "weights/expert.pth",
        "--config",
        "configs/custom_hybrid.yaml",
        "--expert-config",
        "configs/custom_expert.yaml",
        "--split",
        "test",
        "--eval-test",
        "--output",
        "outputs/calibration_test",
    ]
    args = parse_args(custom_args)
    assert args.base_weights == "weights/base.pth"
    assert args.expert_weights == "weights/expert.pth"
    assert args.config == "configs/custom_hybrid.yaml"
    assert args.expert_config == "configs/custom_expert.yaml"
    assert args.split == "test"
    assert args.eval_test is True
    assert args.output == "outputs/calibration_test"


def test_compute_cm_metrics():
    # 1. Identity confusion matrix (perfect prediction)
    cm_perfect = 20 * np.eye(4, dtype=np.int64)
    miou, mdice, class_ious, class_dices = compute_cm_metrics(cm_perfect, 4)
    assert miou == pytest.approx(1.0)
    assert mdice == pytest.approx(1.0)
    for c in range(4):
        assert class_ious[c] == pytest.approx(1.0)
        assert class_dices[c] == pytest.approx(1.0)

    # 2. Asymmetric 2x2 confusion matrix
    cm_imbalanced = np.array([[10, 2], [3, 5]], dtype=np.int64)
    miou_imb, mdice_imb, class_ious_imb, class_dices_imb = compute_cm_metrics(
        cm_imbalanced, 2
    )
    # class 0: tp=10, fp=3, fn=2 -> iou=10/15, dice=20/25
    assert class_ious_imb[0] == pytest.approx(10 / 15)
    assert class_dices_imb[0] == pytest.approx(20 / 25)
    # class 1: tp=5, fp=2, fn=3 -> iou=5/10, dice=10/15
    assert class_ious_imb[1] == pytest.approx(5 / 10)
    assert class_dices_imb[1] == pytest.approx(10 / 15)
    assert miou_imb == pytest.approx((10 / 15 + 5 / 10) / 2)
    assert mdice_imb == pytest.approx((20 / 25 + 10 / 15) / 2)


def test_predict_volume_masks_restores_engine_cfg():
    engine = MagicMock()
    engine.cfg.CLASS_THRESHOLDS = {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.8}
    engine.cfg.PED_SHARPEN_FACTOR = 1.0

    def mock_postprocess(base_p, expert_p):
        return np.zeros((16, 16), dtype=np.uint8)

    engine.postprocess.side_effect = mock_postprocess

    base_probs_vol = np.zeros((4, 4, 16, 16), dtype=np.float32)
    expert_probs_vol = np.zeros((4, 16, 16), dtype=np.float32)

    vol_masks = predict_volume_masks(
        engine=engine,
        base_probs_vol=base_probs_vol,
        expert_irf_probs_vol=expert_probs_vol,
        srf_threshold=0.55,
        ped_threshold=0.65,
        ped_sharpen_factor=0.3,
        use_volumetric_filter=False,
    )

    assert vol_masks.shape == (4, 16, 16)
    assert vol_masks.dtype == np.uint8
    assert engine.postprocess.call_count == 4
    # Ensure thresholds & sharpen factor restored in finally block
    assert engine.cfg.CLASS_THRESHOLDS[2] == 0.8
    assert engine.cfg.CLASS_THRESHOLDS[3] == 0.8
    assert engine.cfg.PED_SHARPEN_FACTOR == 1.0


def test_predict_volume_masks_volumetric_filter():
    engine = MagicMock()
    engine.cfg.CLASS_THRESHOLDS = {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.8}
    engine.cfg.PED_SHARPEN_FACTOR = 1.0

    # Return a mask with class 2 present only on slice 0 (z_span = 1)
    call_idx = [0]

    def mock_postprocess(base_p, expert_p):
        mask = np.zeros((16, 16), dtype=np.uint8)
        if call_idx[0] == 0:
            mask[4:12, 4:12] = 2
        call_idx[0] += 1
        return mask

    engine.postprocess.side_effect = mock_postprocess

    base_probs_vol = np.zeros((3, 4, 16, 16), dtype=np.float32)
    expert_probs_vol = np.zeros((3, 16, 16), dtype=np.float32)

    # Without 3D filter: slice 0 retains class 2
    vol_unfiltered = predict_volume_masks(
        engine=engine,
        base_probs_vol=base_probs_vol,
        expert_irf_probs_vol=expert_probs_vol,
        srf_threshold=0.6,
        ped_threshold=0.6,
        ped_sharpen_factor=1.0,
        use_volumetric_filter=False,
    )
    assert (vol_unfiltered[0] == 2).any()

    # Reset call counter
    call_idx[0] = 0

    # With 3D filter (min_slices=2): 1-slice component removed
    vol_filtered = predict_volume_masks(
        engine=engine,
        base_probs_vol=base_probs_vol,
        expert_irf_probs_vol=expert_probs_vol,
        srf_threshold=0.6,
        ped_threshold=0.6,
        ped_sharpen_factor=1.0,
        use_volumetric_filter=True,
        min_slices=2,
    )
    assert not (vol_filtered[0] == 2).any()
    assert (vol_filtered == 0).all()


def test_evaluate_grid():
    engine = MagicMock()
    engine.cfg.CLASS_THRESHOLDS = {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.8}
    engine.cfg.PED_SHARPEN_FACTOR = 1.0

    # Dynamic postprocess: return class 2 if srf_threshold <= 0.6, else class 0
    def mock_postprocess(base_p, expert_p):
        mask = np.zeros((16, 16), dtype=np.uint8)
        if engine.cfg.CLASS_THRESHOLDS[2] <= 0.6:
            mask[2:8, 2:8] = 2
        return mask

    engine.postprocess.side_effect = mock_postprocess

    # Synthetic ground truth has class 2 at [2:8, 2:8] for slices 0 and 1
    gt = np.zeros((3, 16, 16), dtype=np.uint8)
    gt[0, 2:8, 2:8] = 2
    gt[1, 2:8, 2:8] = 2

    volume_data = [
        {
            "patient": "test_patient",
            "base_probs": np.zeros((3, 4, 16, 16), dtype=np.float32),
            "expert_irf_probs": np.zeros((3, 16, 16), dtype=np.float32),
            "gt_masks": gt,
            "orig_imgs": [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)],
        }
    ]

    cfg = Config()
    srf_thresholds = [0.55, 0.75]
    ped_thresholds = [0.55, 0.75]
    ped_sharpen_factors = [0.0, 1.0]
    volumetric_filter_options = [False, True]

    base_summary, rows, best_row = evaluate_grid(
        volume_data=volume_data,
        engine=engine,
        cfg=cfg,
        srf_thresholds=srf_thresholds,
        ped_thresholds=ped_thresholds,
        ped_sharpen_factors=ped_sharpen_factors,
        volumetric_filter_options=volumetric_filter_options,
        min_slices=2,
    )

    expected_total_rows = (
        len(srf_thresholds)
        * len(ped_thresholds)
        * len(ped_sharpen_factors)
        * len(volumetric_filter_options)
    )
    assert len(rows) == expected_total_rows
    assert "params" in base_summary
    assert "mIoU" in base_summary
    assert "mDice" in base_summary
    assert "class_ious" in base_summary
    assert "class_dices" in base_summary

    # Ensure best_row is selected by maximum (mIoU, mDice)
    max_row = max(rows, key=lambda r: (r["mIoU"], r["mDice"]))
    assert best_row == max_row
    # srf_threshold 0.55 accurately predicts class 2, so best_row must have srf_threshold=0.55
    assert best_row["params"]["srf_threshold"] == 0.55
    assert best_row["mIoU"] > base_summary["mIoU"]


def test_extract_probability_volumes():
    cfg = Config()
    record = VolumeRecord(
        patient="patient_A",
        image_paths=["img_0.png", "img_1.png"],
        mask_paths=["mask_0.png", "mask_1.png"],
    )

    mock_engine = MagicMock()
    mock_engine.probabilities.return_value = (
        np.zeros((4, 16, 16), dtype=np.float32),
        np.zeros((16, 16), dtype=np.float32),
    )

    dummy_item = {
        "orig_img": np.zeros((16, 16, 3), dtype=np.uint8),
        "labels": torch.zeros((16, 16), dtype=torch.uint8),
    }

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 2
    mock_ds.__getitem__.return_value = dummy_item

    with (
        patch(
            "octseg.calibrate_postprocess.split_files",
            return_value=(["img_0.png", "img_1.png"], ["mask_0.png", "mask_1.png"]),
        ),
        patch("octseg.calibrate_postprocess.group_volumes", return_value=[record]),
        patch("octseg.calibrate_postprocess.OCTDataset", return_value=mock_ds),
    ):
        vols = extract_probability_volumes(mock_engine, "val", cfg)
        assert len(vols) == 1
        vol = vols[0]
        assert vol["patient"] == "patient_A"
        assert vol["base_probs"].shape == (2, 4, 16, 16)
        assert vol["expert_irf_probs"].shape == (2, 16, 16)
        assert vol["gt_masks"].shape == (2, 16, 16)
        assert len(vol["orig_imgs"]) == 2


def test_evaluate_test_split():
    cfg = Config()
    engine = MagicMock()
    engine.cfg.CLASS_THRESHOLDS = {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.8}
    engine.cfg.PED_SHARPEN_FACTOR = 1.0
    engine.postprocess.return_value = np.zeros((16, 16), dtype=np.uint8)

    test_volume = {
        "patient": "patient_test",
        "base_probs": np.zeros((2, 4, 16, 16), dtype=np.float32),
        "expert_irf_probs": np.zeros((2, 16, 16), dtype=np.float32),
        "gt_masks": np.zeros((2, 16, 16), dtype=np.uint8),
        "orig_imgs": [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)],
    }

    best_params = {
        "srf_threshold": 0.65,
        "ped_threshold": 0.65,
        "ped_sharpen_factor": 0.7,
        "use_volumetric_filter": True,
    }

    with patch(
        "octseg.calibrate_postprocess.extract_probability_volumes",
        return_value=[test_volume],
    ):
        uncal, cal = evaluate_test_split(engine, best_params, cfg)
        assert "mIoU" in uncal and "mDice" in uncal
        assert "mIoU" in cal and "mDice" in cal
        assert uncal["class_ious"][0] == pytest.approx(1.0)
        assert cal["class_ious"][0] == pytest.approx(1.0)
        assert uncal["mIoU"] == pytest.approx(0.25)
        assert cal["mIoU"] == pytest.approx(0.25)


def test_calibrate_postprocess_full_flow(tmp_path):
    output_dir = tmp_path / "calibration_run"

    cfg = Config()
    expert_cfg = Config(TARGET_CLASS=1)

    mock_engine = MagicMock()
    mock_engine.cfg.CLASS_THRESHOLDS = {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.8}
    mock_engine.cfg.PED_SHARPEN_FACTOR = 1.0
    mock_engine.postprocess.return_value = np.zeros((16, 16), dtype=np.uint8)

    dummy_volume = {
        "patient": "patient_val",
        "base_probs": np.zeros((2, 4, 16, 16), dtype=np.float32),
        "expert_irf_probs": np.zeros((2, 16, 16), dtype=np.float32),
        "gt_masks": np.zeros((2, 16, 16), dtype=np.uint8),
        "orig_imgs": [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)],
    }

    with (
        patch(
            "octseg.calibrate_postprocess.load_config", side_effect=[cfg, expert_cfg]
        ),
        patch("octseg.calibrate_postprocess.HybridInference", return_value=mock_engine),
        patch(
            "octseg.calibrate_postprocess.extract_probability_volumes",
            return_value=[dummy_volume],
        ),
    ):
        results = calibrate_postprocess(
            base_weights="dummy_base.pth",
            expert_weights="dummy_expert.pth",
            config_path="configs/hybrid.yaml",
            expert_config_path="configs/irf_expert.yaml",
            split="val",
            eval_test=True,
            output_dir=str(output_dir),
        )

        assert results["split"] == "val"
        assert "best_params" in results
        assert "val_baseline" in results
        assert "val_calibrated" in results
        assert "grid_rows" in results
        assert "test_baseline_test22" in results
        assert "test_uncalibrated" in results
        assert "test_calibrated" in results
        assert "comparison" in results

        yaml_path = output_dir / "calibration_results.yaml"
        assert yaml_path.exists()
        loaded = yaml.safe_load(yaml_path.read_text())
        assert loaded["split"] == "val"
        assert loaded["best_params"] == results["best_params"]
