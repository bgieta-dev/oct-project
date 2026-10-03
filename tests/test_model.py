import pytest
import torch

from octseg.config import Config
from octseg.model import load_segformer


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
