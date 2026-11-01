"""Segmentation metrics: confusion-matrix IoU/Dice, HD95/ASD, region counts, boundary contrast."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from medpy.metric.binary import asd, hd95
from scipy import ndimage

HD95_MISSING_POLICIES = ("skip", "penalty100")
HD95_PENALTY = 100.0


def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0

def slice_mean_iou(labels, pred, num_classes) -> Optional[float]:
    """Mean IoU over fluid classes present in labels or pred; None if the slice has no fluid at all."""
    if not (np.any(labels > 0) or np.any(pred > 0)):
        return None
    ious = []
    for c in range(1, num_classes):
        intersection = np.sum((labels == c) & (pred == c))
        union = np.sum((labels == c) | (pred == c))
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


class BoundaryPrecisionAnalyzer:
    """Mean intensity contrast between the inner and outer edge of a mask."""

    def __init__(self, kernel_size=3, cfg=None):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.central_slice_idx = cfg.CENTRAL_SLICE_IDX if cfg is not None else 1

    def get_boundary_contrast(self, image, mask):
        if not np.any(mask):
            return 0.0
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, self.central_slice_idx]
        dilated = cv2.dilate(mask.astype(np.uint8), self.kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), self.kernel, iterations=1)
        outer_edge = dilated - mask.astype(np.uint8)
        inner_edge = mask.astype(np.uint8) - eroded
        outer_vals = image[outer_edge > 0]
        inner_vals = image[inner_edge > 0]
        if len(outer_vals) == 0 or len(inner_vals) == 0:
            return 0.0
        return np.abs(np.mean(inner_vals) - np.mean(outer_vals))


class SegmentationMetrics:
    """Accumulates per-slice statistics and aggregates them into the reported metrics.

    HD95 policy for slices where only one of prediction/ground truth contains a class:
      * ``"skip"`` (default): the slice is ignored. This reproduces the thesis numbers.
      * ``"penalty100"``: a penalty of 100 is recorded (used by train-time validation).
    ASD is always skipped in those cases.

    Args:
        num_classes: number of labels including background.
        hd95_missing: ``"skip"`` or ``"penalty100"``.
        extended: also collect ASD, region counts, boundary contrast and pixel areas.
        cfg: config (only CENTRAL_SLICE_IDX is read, for boundary contrast).
    """

    def __init__(self, num_classes, hd95_missing="skip", extended=True, cfg=None):
        if hd95_missing not in HD95_MISSING_POLICIES:
            raise ValueError(f"hd95_missing must be one of {HD95_MISSING_POLICIES}, got {hd95_missing!r}")
        self.num_classes = num_classes
        self.hd95_missing = hd95_missing
        self.extended = extended
        self.classes = list(range(1, num_classes))
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.hd95 = {c: [] for c in self.classes}
        self.asd = {c: [] for c in self.classes}
        self.regions_gt = {c: [] for c in self.classes}
        self.regions_pred = {c: [] for c in self.classes}
        self.boundary = {c: [] for c in self.classes}
        self.areas_gt = {c: [] for c in self.classes}
        self._analyzer = BoundaryPrecisionAnalyzer(cfg=cfg)

    def update(self, labels, pred, image=None):
        """Adds one slice. `labels`/`pred`: integer masks [H,W]; `image`: raw image for boundary contrast (skipped if None)."""
        valid = (labels >= 0) & (labels < self.num_classes)
        self.cm += np.bincount(
            self.num_classes * labels[valid].astype(np.int64) + pred[valid].astype(np.int64),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        for c in self.classes:
            gt_c, pred_c = labels == c, pred == c
            has_gt, has_pred = bool(np.any(gt_c)), bool(np.any(pred_c))
            if has_gt and has_pred:
                self.hd95[c].append(hd95(pred_c, gt_c))
                if self.extended:
                    self.asd[c].append(asd(pred_c, gt_c))
            elif (has_gt or has_pred) and self.hd95_missing == "penalty100":
                self.hd95[c].append(HD95_PENALTY)
            if not self.extended:
                continue
            if has_gt:
                self.regions_gt[c].append(ndimage.label(gt_c)[1])
                if image is not None:
                    self.boundary[c].append(self._analyzer.get_boundary_contrast(image, gt_c))
                self.areas_gt[c].append(np.sum(gt_c))
            if has_pred:
                self.regions_pred[c].append(ndimage.label(pred_c)[1])

    def ious_dices(self):
        tp = np.diag(self.cm)
        fp, fn = self.cm.sum(axis=0) - tp, self.cm.sum(axis=1) - tp
        return tp / (tp + fp + fn + 1e-6), (2 * tp) / (2 * tp + fp + fn + 1e-6)

    def compute(self):
        """Returns the aggregate metrics dict (same keys the pipeline logs)."""
        ious, dices = self.ious_dices()
        cls = self.classes
        return {
            "mIoU": np.mean(ious), "mDice": np.mean(dices),
            "class_ious": {c: ious[c] for c in range(self.num_classes)},
            "class_dices": {c: dices[c] for c in range(self.num_classes)},
            "class_hd95": {c: safe_mean(self.hd95[c]) for c in cls},
            "class_asd": {c: safe_mean(self.asd[c]) for c in cls},
            "class_avg_regions_gt": {c: safe_mean(self.regions_gt[c]) for c in cls},
            "class_avg_regions_pred": {c: safe_mean(self.regions_pred[c]) for c in cls},
            "class_boundary_precision": {c: safe_mean(self.boundary[c]) for c in cls},
            "class_avg_pixel_area": {c: safe_mean(self.areas_gt[c]) for c in cls},
            "mHD95": safe_mean([safe_mean(self.hd95[c]) for c in cls if self.hd95[c]]),
            "mASD": safe_mean([safe_mean(self.asd[c]) for c in cls if self.asd[c]]),
            "hd95_missing": self.hd95_missing,
        }


def _plain(obj):
    """Converts numpy scalars/containers to plain Python for YAML."""
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_metrics_yaml(metrics, path):
    """Writes the metrics dict as YAML (``*.json`` is gitignored in this repo)."""
    Path(path).write_text(yaml.safe_dump(_plain(metrics), sort_keys=False))
