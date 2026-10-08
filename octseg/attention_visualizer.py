import argparse
import logging
import os

import albumentations as A
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor

from octseg.config import Config, ROOT, load_config
from octseg.dataset import OCTDataset
from octseg.model import load_segformer
from octseg.runs import setup_logging
from octseg.splits import split_files
from octseg.viz import select_vis_indices

log = logging.getLogger(__name__)


def generate_attention_maps(model_path, output_dir, cfg: Config):
    """Saves per-encoder-block attention overlays and GT overlays for the most pathological test slices.

    Files are named ``att_idx<slice>_<class>_block<i>.png``; ``i`` indexes all encoder blocks
    (not SegFormer stages). Raises FileNotFoundError if the checkpoint is missing.
    """
    if cfg.ARCH != "segformer":
        raise ValueError("generate_attention_maps requires ARCH=segformer")
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading model for attention visualization: {model_path}...")
    model = load_segformer(cfg, model_path, output_attentions=True)
    processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)

    # Find the slices with the most pixels per class (1: IRF, 2: SRF, 3: PED)
    test_imgs, test_masks = split_files(cfg, "test")

    log.info("Scanning for slices with significant pathology for visualization...")
    vis_indices, _ = select_vis_indices(test_masks, list(range(1, cfg.NUM_LABELS)))
    target_indices = sorted({i for i in vis_indices.values() if i != -1})
    log.info(f"Targeting indices: {target_indices}")

    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])
    dataset = OCTDataset(test_imgs, test_masks, processor, transform=val_transform,
                         cfg=cfg)

    log.info("Generating attention maps...")
    with torch.no_grad():
        for target_idx in target_indices:
            batch = dataset[target_idx]
            pixel_values = batch["pixel_values"].unsqueeze(0).to(cfg.DEVICE)
            labels = batch["labels"].numpy()
            orig_img = batch["orig_img"]

            # 2.5D: channels are t-1, t, t+1 (use t). Multimodal: orig, denoised, edge (use orig).
            if orig_img.shape[-1] == 3:
                vis_img = orig_img[:, :, 1 if cfg.USE_25D else 0]
            else:
                vis_img = orig_img

            class_label = "unknown"
            for c, idx in vis_indices.items():
                if idx == target_idx:
                    class_label = cfg.CLASS_NAMES[c]
            log.info(f"-> Generating maps for {class_label} (Index {target_idx}, Pathology pixels: {np.sum(labels > 0)})")

            attentions = model(pixel_values=pixel_values).attentions  # one tensor per encoder block
            vis_uint8 = np.uint8(vis_img * 255) if vis_img.max() <= 1.0 else np.uint8(vis_img)
            orig_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_GRAY2BGR)

            for block_idx, block_att in enumerate(attentions):
                # block_att: (batch, heads, seq_len, seq_len)
                spatial_att = torch.mean(torch.mean(block_att, dim=1), dim=1)
                grid_size = int(np.sqrt(spatial_att.shape[1]))
                heatmap = spatial_att.view(grid_size, grid_size).cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                heatmap_resized = cv2.resize(heatmap, (512, 512))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
                cv2.imwrite(os.path.join(output_dir, f"att_idx{target_idx}_{class_label}_block{block_idx}.png"), overlay)

            # GT overlays for clinical validation
            gt_color = cv2.applyColorMap(np.uint8(labels * 60), cv2.COLORMAP_JET)
            gt_color[labels == 0] = 0  # Background black
            gt_overlay = cv2.addWeighted(orig_bgr, 0.5, gt_color, 0.5, 0)
            cv2.imwrite(os.path.join(output_dir, f"att_idx{target_idx}_{class_label}_GT_OVERLAY.png"), gt_overlay)
            cv2.imwrite(os.path.join(output_dir, f"att_idx{target_idx}_{class_label}_GT_MASK.png"), gt_color)

    log.info(f"Done! Results saved in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Attention-map visualisation for a trained SegFormer")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs" / "attention"), help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output, "attention.log")
    generate_attention_maps(args.model, args.output, load_config(args.config))


if __name__ == "__main__":
    main()
