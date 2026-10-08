"""Volumetric data utilities and transforms for 3D SwinUNETR."""
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from octseg.dataset import normalize_slice
from octseg.splits import patient_of

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeRecord:
    patient: str
    image_paths: List[str]
    mask_paths: List[str]


def group_volumes(image_paths: List[str], mask_paths: List[str]) -> List[VolumeRecord]:
    """Groups slice paths into VolumeRecords by patient, ordered numerically by slice index.

    Args:
        image_paths: list of slice image paths.
        mask_paths: list of corresponding slice mask paths.

    Returns:
        List of VolumeRecord objects sorted by patient.
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Mismatched image_paths ({len(image_paths)}) and mask_paths ({len(mask_paths)})")

    mask_by_name = {os.path.basename(m): m for m in mask_paths}
    if all(os.path.basename(img) in mask_by_name for img in image_paths):
        pairs = [(img, mask_by_name[os.path.basename(img)]) for img in image_paths]
    else:
        pairs = list(zip(image_paths, mask_paths))

    patient_slices = {}
    for img_p, mask_p in pairs:
        stem = Path(img_p).stem
        slice_str = stem.split("_")[-1]
        slice_idx = int(slice_str)
        patient = patient_of(os.path.basename(img_p))
        if patient not in patient_slices:
            patient_slices[patient] = []
        patient_slices[patient].append((slice_idx, img_p, mask_p))

    records = []
    for patient in sorted(patient_slices.keys()):
        slices = sorted(patient_slices[patient], key=lambda x: x[0])
        indices = [s[0] for s in slices]
        if any(b - a != 1 for a, b in zip(indices[:-1], indices[1:])):
            log.warning(f"Non-contiguous slice indices for {patient}")
        records.append(
            VolumeRecord(
                patient=patient,
                image_paths=[s[1] for s in slices],
                mask_paths=[s[2] for s in slices],
            )
        )
    return records


def load_volume(record: VolumeRecord, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all slices for a volume, normalizes and resizes them.

    Args:
        record: VolumeRecord describing the patient's slices.
        size: target (H, W) spatial dimensions for each slice.

    Returns:
        Tuple of (image_volume [Z, H, W] uint8, mask_volume [Z, H, W] uint8).
    """
    imgs = []
    masks = []
    for img_p, mask_p in zip(record.image_paths, record.mask_paths):
        img_raw = np.array(Image.open(img_p).convert("L")).astype(np.float32)
        norm = normalize_slice(img_raw)
        if norm.shape[:2] != size:
            norm = cv2.resize(norm, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        img_u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        imgs.append(img_u8)

        mask_raw = np.array(Image.open(mask_p))
        if mask_raw.shape[:2] != size:
            mask_resized = cv2.resize(mask_raw, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_raw
        masks.append(mask_resized.astype(np.uint8))

    return np.stack(imgs, axis=0), np.stack(masks, axis=0)


def train_transforms_3d(cfg):
    """Returns MONAI 3D augmentation pipeline for patch-based SwinUNETR training."""
    import monai.transforms as mt

    keys = ["image", "label"]
    return mt.Compose([
        mt.ScaleIntensityRanged("image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        mt.SpatialPadd(keys, spatial_size=cfg.SWIN_ROI),
        mt.RandCropByLabelClassesd(
            keys,
            label_key="label",
            spatial_size=cfg.SWIN_ROI,
            ratios=[1] * cfg.NUM_LABELS,
            num_classes=cfg.NUM_LABELS,
            num_samples=cfg.SWIN_SAMPLES_PER_VOLUME,
        ),
        mt.RandFlipd(keys, prob=cfg.AUG_PROBS["flip"], spatial_axis=2),
        mt.RandScaleIntensityd("image", factors=0.1, prob=cfg.AUG_PROBS["brightness"]),
        mt.RandShiftIntensityd("image", offsets=0.1, prob=cfg.AUG_PROBS["brightness"]),
        mt.RandGaussianNoised("image", prob=cfg.AUG_PROBS["noise"], std=0.02),
        mt.EnsureTyped(keys, track_meta=False),
    ])
