"""SegFormer and SwinUNETR construction and strict checkpoint loading."""
import logging
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

from octseg.config import ROOT

log = logging.getLogger(__name__)


def load_segformer(cfg, weights_path, output_attentions=False):
    """Builds the SegFormer described by `cfg` and loads `weights_path` into it.

    Fails loudly: raises FileNotFoundError if the checkpoint is missing and RuntimeError
    if its keys do not match the architecture (strict loading).

    Args:
        cfg: config providing MODEL_NAME, NUM_LABELS, DEVICE.
        weights_path: path to a ``state_dict`` saved by `octseg.train`.
        output_attentions: return attention tensors from forward passes.

    Returns:
        Model on `cfg.DEVICE` in eval mode.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS, ignore_mismatched_sizes=True, output_attentions=output_attentions
    )
    state_dict = torch.load(weights_path, map_location=cfg.DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(cfg.DEVICE).eval()


def build_swin_unetr(cfg, pretrained=True):
    """Builds a MONAI SwinUNETR 3D model according to `cfg`.

    Args:
        cfg: configuration providing NUM_LABELS, SWIN_FEATURE_SIZE, SWIN_DROP_PATH, SWIN_PRETRAINED.
        pretrained: whether to load self-supervised pretrained weights if configured.

    Returns:
        SwinUNETR model instance.
    """
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        in_channels=1,
        out_channels=cfg.NUM_LABELS,
        feature_size=cfg.SWIN_FEATURE_SIZE,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=cfg.SWIN_DROP_PATH,
        use_checkpoint=True,
        spatial_dims=3,
    )
    if pretrained and cfg.SWIN_PRETRAINED:
        weights_path = Path(cfg.SWIN_PRETRAINED)
        if not weights_path.is_absolute():
            weights_path = ROOT / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"SwinUNETR pretrained weights not found: {weights_path}")
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            # Official MONAI release asset may require weights_only=False
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        weights = ckpt if "state_dict" in ckpt else {"state_dict": ckpt}
        model.load_from(weights=weights)
        log.info(f"Loaded SwinUNETR self-supervised weights from {weights_path}")
    return model


def load_swin_unetr(cfg, weights_path):
    """Builds the SwinUNETR described by `cfg` and loads `weights_path` into it.

    Fails loudly: raises FileNotFoundError if checkpoint missing and RuntimeError
    if keys do not match the architecture (strict loading).

    Returns:
        Model on `cfg.DEVICE` in eval mode.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
    model = build_swin_unetr(cfg, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(cfg.DEVICE).eval()
