"""Figures: prediction grid, training curves, and test-slice selection for visualisation."""
import os
import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def select_vis_indices(mask_paths, classes, target_class=None):
    """Finds, for every class, the slice whose mask has the most pixels of that class.

    Args:
        mask_paths: ordered list of mask file paths.
        classes: class ids to look for (background excluded).
        target_class: if set (binary expert), masks are remapped to {0, 1} first.

    Returns:
        (vis_indices, class_max_counts): class -> slice index (-1 if never present), class -> pixel count.
    """
    class_max_counts = {c: 0 for c in classes}
    vis_indices = {c: -1 for c in classes}
    for i, m_path in enumerate(mask_paths):
        m = np.array(Image.open(m_path))
        if target_class is not None:
            m = (m == target_class).astype(m.dtype)
        for c in classes:
            cnt = np.sum(m == c)
            if cnt > class_max_counts[c]:
                class_max_counts[c] = cnt
                vis_indices[c] = i
    return vis_indices, class_max_counts


def save_predictions_grid(vis_data, class_max_counts, cfg, path, dpi=120):
    """Saves the 4-column (or 3-column if attention is None) grid (OCT | GT | prediction [| self-attention]).

    Args:
        vis_data: class -> (image, gt_mask, pred_mask, attention_map). attention_map may be None (3-column grid).
        class_max_counts: class -> GT pixel count shown in the title.
        cfg: config providing NUM_LABELS, CLASS_NAMES, ATTENTION_CONTRAST.
        path: output PNG path.
        dpi: figure resolution.
    """
    num_plots = len(vis_data)
    if num_plots == 0:
        return
    ncols = 4 if any(v[3] is not None for v in vis_data.values()) else 3
    plt.figure(figsize=(5.5 * ncols, 5 * num_plots))
    # Rows ordered by class id (1=IRF, 2=SRF, 3=PED), not by insertion order.
    for i, c_id in enumerate(c for c in range(1, cfg.NUM_LABELS) if c in vis_data):
        img, gt, prd, att = vis_data[c_id]

        ax1 = plt.subplot(num_plots, ncols, i * ncols + 1)
        ax1.imshow(img)
        ax1.set_title(f"OCT: {cfg.CLASS_NAMES.get(c_id, f'Class {c_id}')}", fontsize=14, fontweight="bold")
        ax1.axis("off")

        ax2 = plt.subplot(num_plots, ncols, i * ncols + 2)
        ax2.imshow(gt, cmap="jet", vmin=0, vmax=cfg.NUM_LABELS - 1)
        ax2.set_title(f"GT (px: {class_max_counts[c_id]})", fontsize=12)
        ax2.axis("off")

        tp = np.sum((gt == c_id) & (prd == c_id))
        fp = np.sum((gt != c_id) & (prd == c_id))
        fn = np.sum((gt == c_id) & (prd != c_id))
        c_iou = tp / (tp + fp + fn + 1e-6)

        ax3 = plt.subplot(num_plots, ncols, i * ncols + 3)
        ax3.imshow(prd, cmap="jet", vmin=0, vmax=cfg.NUM_LABELS - 1)
        ax3.set_title(f"Prediction (IoU: {c_iou:.2f})", fontsize=12)
        ax3.axis("off")

        if att is not None:
            ax4 = plt.subplot(num_plots, ncols, i * ncols + 4)
            att_resized = cv2.resize(att, (img.shape[1], img.shape[0]))
            att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
            att_norm = np.power(att_norm, cfg.ATTENTION_CONTRAST)
            gray_bg = img[:, :, 1] if img.ndim == 3 else img
            ax4.imshow(gray_bg, cmap="gray")
            ax4.imshow(att_norm, cmap="jet", alpha=0.5)
            ax4.set_title("Self-Attention (encoder block 5)", fontsize=12)
            ax4.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_history(history, path):
    """Saves loss / mIoU / mHD95 curves."""
    plt.figure(figsize=(12, 4))
    for i, (key, title) in enumerate((("loss", "Loss"), ("miou", "mIoU"), ("mhd95", "mHD95")), start=1):
        plt.subplot(1, 3, i)
        plt.plot(history[key])
        plt.title(title)
    plt.savefig(path)
    plt.close()


def save_failure_cases(failure_records, output_dir, cfg, top_k=5, dpi=120):
    """Saves worst-performing failure slices (OCT | GT | Prediction) sorted by IoU ascending.

    Args:
        failure_records: list of (iou, slice_idx, img_np, gt_mask, pred_mask).
        output_dir: base evaluation output directory.
        cfg: config providing NUM_LABELS.
        top_k: maximum number of failure cases to save.
        dpi: figure resolution.
    """
    if not failure_records:
        return
    failures_dir = os.path.join(output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)
    sorted_records = sorted(failure_records, key=lambda x: x[0])
    for rank, (iou, slice_idx, img, gt, prd) in enumerate(sorted_records[:top_k]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax1, ax2, ax3 = axes

        ax1.imshow(img)
        ax1.set_title(f"OCT (Slice {slice_idx})", fontsize=12, fontweight="bold")
        ax1.axis("off")

        vmax = cfg.NUM_LABELS - 1 if cfg is not None and hasattr(cfg, "NUM_LABELS") else 3
        ax2.imshow(gt, cmap="jet", vmin=0, vmax=vmax)
        ax2.set_title("Ground Truth", fontsize=12)
        ax2.axis("off")

        ax3.imshow(prd, cmap="jet", vmin=0, vmax=vmax)
        ax3.set_title(f"Prediction (mIoU: {iou:.2f})", fontsize=12)
        ax3.axis("off")

        plt.tight_layout()
        save_path = os.path.join(failures_dir, f"failure_{rank + 1}_idx_{slice_idx}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
