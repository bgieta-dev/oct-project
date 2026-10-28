"""Inference-time helpers shared by training validation, evaluation and hybrid inference.

Pipeline: `predict_logits` (TTA) -> softmax -> `sharpen_ped` -> `apply_thresholds` -> `clean_regions`.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scipy import ndimage
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
OPEN_KERNEL = np.ones((3, 3), np.uint8)
PED_CLASS = 3
IRF_CLASS = 1


def predict_logits(model, pixel_values, target_size, cfg):
    """Runs `model` on `pixel_values` [B,3,H,W] and returns logits [B,C,*target_size].

    With `cfg.USE_TTA`, averages logits over `cfg.TTA_SCALES` and horizontal flips.

    Args:
        model: SegFormer model (callable with `pixel_values=`).
        pixel_values: Preprocessed input tensor.
        target_size: (H, W) of the returned logits.
        cfg: Config providing USE_TTA, TTA_SCALES and AUG_SIZE.

    Returns:
        Tensor of logits at `target_size`.
    """
    def run(px):
        out = model(pixel_values=px).logits
        return F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

    if not cfg.USE_TTA:
        return run(pixel_values)

    all_logits = []
    for s in cfg.TTA_SCALES:
        if s != 1.0:
            scaled_size = (int(cfg.AUG_SIZE[0] * s), int(cfg.AUG_SIZE[1] * s))
            px = F.interpolate(pixel_values, size=scaled_size, mode="bilinear", align_corners=False)
        else:
            px = pixel_values
        all_logits.append(run(px))
        all_logits.append(torch.flip(run(torch.flip(px, [3])), [3]))
    return torch.mean(torch.stack(all_logits), dim=0)


def sharpen_ped(probs, factor=1.0):
    """Sharpens the PED probability channel in place and returns `probs`.

    Compensates for bilinear blurring of PED peaks. No-op if there are <= 3 classes or factor <= 0.

    Args:
        probs: float32 array [C,H,W] or [B,C,H,W].
        factor: blend factor between 0.0 (original) and 1.0 (fully sharpened).
    """
    if probs.shape[-3] <= PED_CLASS or factor <= 0:
        return probs
    if probs.ndim == 3:
        orig = probs[PED_CLASS]
        filtered = np.clip(cv2.filter2D(orig, -1, SHARPEN_KERNEL), 0, 1)
        probs[PED_CLASS] = np.clip((1.0 - factor) * orig + factor * filtered, 0, 1)
    else:
        for b in range(probs.shape[0]):
            orig = probs[b, PED_CLASS]
            filtered = np.clip(cv2.filter2D(orig, -1, SHARPEN_KERNEL), 0, 1)
            probs[b, PED_CLASS] = np.clip((1.0 - factor) * orig + factor * filtered, 0, 1)
    return probs


def apply_thresholds(probs, thresholds, irf_override=True):
    """Converts class probabilities to a label mask with per-class thresholds.

    Classes are laid down in reverse order so IRF (1) is written last. With
    `irf_override=False`, IRF is only written where no SRF/PED was laid down.

    Args:
        probs: array [C,H,W] or [B,C,H,W].
        thresholds: dict class -> threshold (missing classes default to 0.5).
        irf_override: whether IRF may overwrite higher-numbered classes.

    Returns:
        uint8 mask [H,W] or [B,H,W].
    """
    num_classes = probs.shape[-3]
    mask = np.zeros(probs.shape[:-3] + probs.shape[-2:], dtype=np.uint8)
    classes = list(range(num_classes - 1, 0, -1))
    if irf_override:
        for c in classes:
            mask[probs[..., c, :, :] > thresholds.get(c, 0.5)] = c
    else:
        for c in classes[:-1]:
            mask[probs[..., c, :, :] > thresholds.get(c, 0.5)] = c
        mask[(mask == 0) & (probs[..., IRF_CLASS, :, :] > thresholds.get(IRF_CLASS, 0.5))] = IRF_CLASS
    return mask


def clean_regions(mask, min_sizes, irf_open=True):
    """Removes connected components smaller than the per-class minimum size.

    Args:
        mask: uint8 label mask [H,W].
        min_sizes: int (same for all classes) or dict class -> minimum area in pixels.
        irf_open: apply a 3x3 morphological opening to IRF before filtering.

    Returns:
        New uint8 mask [H,W].
    """
    cleaned = np.zeros_like(mask)
    for c in np.unique(mask):
        if c == 0:
            continue
        c_mask = (mask == c).astype(np.uint8)
        if irf_open and c == IRF_CLASS:
            c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=1)
        min_size = min_sizes[int(c)] if isinstance(min_sizes, dict) else min_sizes
        num, labels_im, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=8)
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels_im == lbl] = c
    return cleaned


def _filter_3d(vol: np.ndarray, min_slices: int) -> np.ndarray:
    if min_slices <= 1:
        return vol.copy()
    cleaned = vol.copy()
    struct = np.ones((3, 3, 3), bool)
    for c in (1, 2, 3):
        c_mask = (cleaned == c)
        if not np.any(c_mask):
            continue
        labeled, num_features = ndimage.label(c_mask, structure=struct)
        if num_features == 0:
            continue
        slices = ndimage.find_objects(labeled)
        for comp_id, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            z_span = sl[0].stop - sl[0].start
            if z_span < min_slices:
                sub_lbl = labeled[sl]
                sub_cleaned = cleaned[sl]
                sub_cleaned[sub_lbl == comp_id] = 0
                cleaned[sl] = sub_cleaned
    return cleaned


def volumetric_consistency_filter(volume_masks: np.ndarray, min_slices: int = 2) -> np.ndarray:
    """Removes 3D connected components that span fewer than `min_slices` along the Z axis.

    Args:
        volume_masks: numpy array of shape [Z, H, W] or [B, Z, H, W].
        min_slices: minimum number of slices a 3D component must span to be kept.

    Returns:
        Cleaned volume masks array.
    """
    if volume_masks.ndim == 3:
        return _filter_3d(volume_masks, min_slices)
    elif volume_masks.ndim == 4:
        return np.stack([_filter_3d(v, min_slices) for v in volume_masks], axis=0)
    else:
        raise ValueError(f"Expected 3D [Z, H, W] or 4D [B, Z, H, W] array, got shape {volume_masks.shape}")
