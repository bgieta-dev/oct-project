import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from octseg.benchmark import benchmark, parse_args, peak_vram_mb, time_calls
from octseg.config import Config


def test_time_calls():
    calls = []

    def dummy_fn(x):
        calls.append(x)
        return x * 2

    inputs = [1, 2, 3]
    device = torch.device("cpu")
    stats = time_calls(dummy_fn, inputs, warmup=2, device=device)

    # 2 warmup calls on inputs[0] + 3 timed calls = 5 total
    assert len(calls) == 5
    assert calls[:2] == [1, 1]
    assert calls[2:] == [1, 2, 3]

    assert "mean_ms" in stats
    assert "median_ms" in stats
    assert "p95_ms" in stats
    assert stats["n"] == 3
    assert stats["mean_ms"] >= 0.0


def test_time_calls_empty_inputs_raises():
    with pytest.raises(ValueError, match="inputs must not be empty"):
        time_calls(lambda x: x, [], warmup=1, device=torch.device("cpu"))


def test_peak_vram_mb_cpu():
    device = torch.device("cpu")
    res = peak_vram_mb(lambda x: x, 42, device)
    assert res is None


def test_peak_vram_mb_cuda():
    device = torch.device("cuda")
    with patch("torch.cuda.reset_peak_memory_stats") as mock_reset, \
         patch("torch.cuda.max_memory_allocated", return_value=1048576 * 128) as mock_max, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.synchronize"):
        res = peak_vram_mb(lambda x: x, 42, device)
        mock_reset.assert_called_once()
        mock_max.assert_called_once()
        assert res == 128.0


def test_parse_args_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_expert_pair_validation():
    # Only expert-config -> error
    with pytest.raises(SystemExit):
        parse_args(["--config", "configs/base.yaml", "--model", "model.pth", "--expert-config", "exp.yaml"])

    # Only expert-weights -> error
    with pytest.raises(SystemExit):
        parse_args(["--config", "configs/base.yaml", "--model", "model.pth", "--expert-weights", "exp.pth"])

    # Both provided -> valid
    args = parse_args([
        "--config", "configs/base.yaml",
        "--model", "model.pth",
        "--expert-config", "exp.yaml",
        "--expert-weights", "exp.pth",
        "--device", "cpu",
        "--threads", "4",
        "--slices", "10",
        "--volumes", "1",
        "--warmup", "2",
        "--name", "test_bench",
    ])
    assert args.config == "configs/base.yaml"
    assert args.model == "model.pth"
    assert args.expert_config == "exp.yaml"
    assert args.expert_weights == "exp.pth"
    assert args.device == "cpu"
    assert args.threads == 4
    assert args.slices == 10
    assert args.volumes == 1
    assert args.warmup == 2
    assert args.name == "test_bench"


def test_missing_model_raises_file_not_found(tmp_path):
    cfg_segformer = Config(ARCH="segformer")
    missing_path = str(tmp_path / "non_existent.pth")

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        benchmark(cfg_segformer, missing_path, tmp_path)

    cfg_swin = Config(ARCH="swin_unetr_3d", SWIN_FEATURE_SIZE=12, SWIN_ROI=(32, 32, 32))
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        benchmark(cfg_swin, missing_path, tmp_path)


def test_benchmark_segformer_modes(tmp_path):
    cfg = Config(ARCH="segformer", DEVICE_OVERRIDE="cpu")
    weights_path = tmp_path / "model.pth"
    # Create dummy checkpoint
    dummy_model = torch.nn.Linear(10, 10)
    torch.save(dummy_model.state_dict(), weights_path)

    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.zeros(100, 100)]
    mock_model.return_value = MagicMock()

    dummy_px = torch.zeros((1, 3, 64, 64))
    dummy_orig = np.zeros((64, 64, 3), dtype=np.uint8)

    with patch("octseg.benchmark.load_segformer", return_value=mock_model), \
         patch("octseg.benchmark.split_files", return_value=(["img1.png", "img2.png"], ["m1.png", "m2.png"])), \
         patch("octseg.benchmark.OCTDataset") as mock_ds_cls, \
         patch("octseg.benchmark.predict_logits", return_value=torch.zeros((1, 4, 64, 64))):

        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 2
        mock_ds.__getitem__.side_effect = [
            {"pixel_values": dummy_px[0], "orig_img": dummy_orig},
            {"pixel_values": dummy_px[0], "orig_img": dummy_orig},
        ]
        mock_ds_cls.return_value = mock_ds

        result = benchmark(
            cfg=cfg,
            model_path=str(weights_path),
            output_dir=tmp_path,
            slices=2,
            warmup=1,
        )

        assert result["arch"] == "segformer"
        assert result["device"] == "cpu"
        assert "forward" in result["modes"]
        assert "pipeline" in result["modes"]
        assert "hybrid" not in result["modes"]
        assert result["modes"]["forward"]["n"] == 2
        assert result["modes"]["pipeline"]["n"] == 2

        yaml_file = tmp_path / "benchmark.yaml"
        assert yaml_file.exists()
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["arch"] == "segformer"
        assert loaded["modes"]["forward"]["peak_vram_mb"] is None


def test_benchmark_swin_unetr_3d_modes(tmp_path):
    cfg = Config(ARCH="swin_unetr_3d", SWIN_FEATURE_SIZE=12, SWIN_ROI=(32, 32, 32), DEVICE_OVERRIDE="cpu")
    weights_path = tmp_path / "model.pth"
    dummy_model = torch.nn.Linear(10, 10)
    torch.save(dummy_model.state_dict(), weights_path)

    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.zeros(100, 100)]

    dummy_vol = np.zeros((16, 32, 32), dtype=np.uint8)

    with patch("octseg.benchmark.load_swin_unetr", return_value=mock_model), \
         patch("octseg.benchmark.split_files", return_value=(["img1.png"], ["m1.png"])), \
         patch("octseg.benchmark.group_volumes", return_value=[MagicMock()]), \
         patch("octseg.benchmark.load_volume", return_value=(dummy_vol, dummy_vol)), \
         patch("octseg.benchmark.predict_volume", return_value=np.zeros((16, 4, 32, 32), dtype=np.float32)):

        result = benchmark(
            cfg=cfg,
            model_path=str(weights_path),
            output_dir=tmp_path,
            volumes=1,
            warmup=1,
        )

        assert result["arch"] == "swin_unetr_3d"
        assert "volume_forward" in result["modes"]
        assert "volume_pipeline" in result["modes"]
        assert "ms_per_bscan" in result["modes"]["volume_forward"]
        assert "ms_per_bscan" in result["modes"]["volume_pipeline"]

        yaml_file = tmp_path / "benchmark.yaml"
        assert yaml_file.exists()
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["modes"]["volume_forward"]["ms_per_bscan"] >= 0.0

def test_benchmark_segformer_with_hybrid(tmp_path):
    cfg = Config(ARCH="segformer", DEVICE_OVERRIDE="cpu")
    expert_cfg = Config(ARCH="segformer", DEVICE_OVERRIDE="cpu", TARGET_CLASS=1)
    weights_path = tmp_path / "model.pth"
    expert_weights = tmp_path / "expert.pth"
    dummy_model = torch.nn.Linear(10, 10)
    torch.save(dummy_model.state_dict(), weights_path)
    torch.save(dummy_model.state_dict(), expert_weights)

    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.zeros(100, 100)]
    dummy_px = torch.zeros((1, 3, 64, 64))
    dummy_orig = np.zeros((64, 64, 3), dtype=np.uint8)

    mock_hybrid = MagicMock()
    mock_hybrid.segment.return_value = np.zeros((64, 64), dtype=np.uint8)

    with patch("octseg.benchmark.load_segformer", return_value=mock_model), \
         patch("octseg.benchmark.split_files", return_value=(["img1.png"], ["m1.png"])), \
         patch("octseg.benchmark.OCTDataset") as mock_ds_cls, \
         patch("octseg.benchmark.predict_logits", return_value=torch.zeros((1, 4, 64, 64))), \
         patch("octseg.benchmark.HybridInference", return_value=mock_hybrid):

        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 1
        mock_ds.__getitem__.return_value = {"pixel_values": dummy_px[0], "orig_img": dummy_orig}
        mock_ds_cls.return_value = mock_ds

        result = benchmark(
            cfg=cfg,
            model_path=str(weights_path),
            output_dir=tmp_path,
            expert_cfg=expert_cfg,
            expert_weights_path=str(expert_weights),
            slices=1,
            warmup=1,
        )

        assert "forward" in result["modes"]
        assert "pipeline" in result["modes"]
        assert "hybrid" in result["modes"]
        assert result["modes"]["hybrid"]["n"] == 1
        mock_hybrid.segment.assert_called()


def test_benchmark_swin_unetr_3d_with_expert_raises(tmp_path):
    cfg = Config(ARCH="swin_unetr_3d", SWIN_FEATURE_SIZE=12, SWIN_ROI=(32, 32, 32), DEVICE_OVERRIDE="cpu")
    expert_cfg = Config(ARCH="segformer", DEVICE_OVERRIDE="cpu", TARGET_CLASS=1)
    with pytest.raises(ValueError, match="Expert model is not supported for ARCH=swin_unetr_3d"):
        benchmark(
            cfg=cfg,
            model_path="dummy.pth",
            output_dir=tmp_path,
            expert_cfg=expert_cfg,
            expert_weights_path="expert.pth",
        )
