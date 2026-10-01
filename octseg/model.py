"""SegFormer construction and strict checkpoint loading."""
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


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
