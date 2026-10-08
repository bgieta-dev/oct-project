import pytest
import torch

from octseg.config import Config
from octseg.model import build_swin_unetr, load_segformer, load_swin_unetr


def test_load_segformer_missing_file_raises_not_found(tmp_path):
    cfg = Config()
    missing_path = tmp_path / "non_existent_model.pth"
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_segformer(cfg, missing_path)


def test_load_segformer_strict_loading_mismatch(tmp_path):
    cfg = Config()
    bad_ckpt = tmp_path / "corrupt_keys.pth"
    torch.save({"bogus_layer.weight": torch.randn(10, 10)}, bad_ckpt)

    # When strict=True, mismatched state dict must raise RuntimeError
    with pytest.raises(RuntimeError):
        load_segformer(cfg, bad_ckpt)


def test_load_swin_unetr_missing_file_raises_not_found(tmp_path):
    cfg = Config(ARCH="swin_unetr_3d", SWIN_FEATURE_SIZE=12, SWIN_ROI=(32, 32, 32))
    missing_path = tmp_path / "non_existent_model.pth"
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_swin_unetr(cfg, missing_path)


def test_load_swin_unetr_strict_loading_mismatch(tmp_path):
    cfg = Config(ARCH="swin_unetr_3d", SWIN_FEATURE_SIZE=12, SWIN_ROI=(32, 32, 32))
    bad_ckpt = tmp_path / "corrupt_keys.pth"
    torch.save({"bogus_layer.weight": torch.randn(10, 10)}, bad_ckpt)
    with pytest.raises(RuntimeError):
        load_swin_unetr(cfg, bad_ckpt)


def test_build_swin_unetr_missing_pretrained_raises_not_found(tmp_path):
    cfg = Config(
        ARCH="swin_unetr_3d",
        SWIN_FEATURE_SIZE=48,
        SWIN_ROI=(32, 32, 32),
        SWIN_PRETRAINED=str(tmp_path / "missing_pretrained.pt"),
    )
    with pytest.raises(FileNotFoundError, match="SwinUNETR pretrained weights not found"):
        build_swin_unetr(cfg, pretrained=True)
