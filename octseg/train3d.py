"""Volumetric training pipeline for 3D SwinUNETR."""
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from octseg.config import Config
from octseg.evaluate3d import predict_volume
from octseg.losses import FocalLoss, TverskyLoss
from octseg.metrics import SegmentationMetrics
from octseg.model import build_swin_unetr
from octseg.postprocess import clean_regions
from octseg.splits import split_files
from octseg.train import _seed_worker, calculate_dynamic_weights, seed_everything
from octseg.viz import plot_history
from octseg.volume_data import group_volumes, load_volume, train_transforms_3d

log = logging.getLogger(__name__)


def validate_3d(model, val_volumes: List, cfg: Config) -> Tuple[float, float]:
    """Runs one validation pass over 3D volumes using sliding-window inference.

    Args:
        model: SwinUNETR model.
        val_volumes: list of (img_u8, gt_mask) tuples.
        cfg: configuration object.

    Returns:
        (mIoU, mHD95) with penalty100 for missing/false classes.
    """
    model.eval()
    metrics = SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="penalty100", extended=False)
    with torch.no_grad():
        for item in val_volumes:
            img_u8, gt_mask = item if isinstance(item, tuple) else (item["image"], item["mask"])
            probs = predict_volume(model, img_u8, cfg, tta=False)
            preds = np.argmax(probs, axis=1).astype(np.uint8)

            for z in range(len(preds)):
                p = preds[z]
                if cfg.MIN_REGION_SIZE > 0:
                    p = clean_regions(p, cfg.MIN_REGION_SIZE, irf_open=False)
                metrics.update(gt_mask[z], p)

    result = metrics.compute()
    return result["mIoU"], result["mHD95"]


def train_model_3d(output_dir: str, cfg: Config, epochs: int = None, save_path: str = None):
    """Trains a 3D SwinUNETR model and saves the best checkpoint by validation mIoU.

    Args:
        output_dir: directory to store run artifacts (metrics.png, config.yaml).
        cfg: configuration object.
        epochs: overrides cfg.EPOCHS if provided.
        save_path: path to save best model checkpoint.
    """
    epochs = cfg.EPOCHS if epochs is None else epochs
    save_path = os.path.join(output_dir, "best_model.pth") if save_path is None else save_path
    seed_everything(cfg.SEED)

    train_imgs, train_masks = split_files(cfg, "train")
    val_imgs, val_masks = split_files(cfg, "val")

    train_records = group_volumes(train_imgs, train_masks)
    val_records = group_volumes(val_imgs, val_masks)

    if cfg.USE_DYNAMIC_WEIGHTS:
        dyn_weights = calculate_dynamic_weights(train_masks, num_classes=cfg.NUM_LABELS, target_class=cfg.TARGET_CLASS)
        class_weights = torch.tensor(dyn_weights, dtype=torch.float32).to(cfg.DEVICE)
    else:
        class_weights = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32).to(cfg.DEVICE)

    log.info(f"Loading {len(train_records)} training volumes at {cfg.SWIN_VOLUME_SIZE} into memory...")
    train_data = []
    for r in train_records:
        img, mask = load_volume(r, cfg.SWIN_VOLUME_SIZE)
        train_data.append({"image": img[None], "label": mask[None]})

    log.info(f"Loading {len(val_records)} validation volumes into memory...")
    val_volumes = []
    for r in val_records:
        img = load_volume(r, cfg.SWIN_VOLUME_SIZE)[0]
        mask = load_volume(r, cfg.AUG_SIZE)[1]
        val_volumes.append((img, mask))

    transforms = train_transforms_3d(cfg)
    transforms.set_random_state(seed=cfg.SEED)

    from monai.data import DataLoader, Dataset

    train_ds = Dataset(train_data, transform=transforms)
    generator = torch.Generator().manual_seed(cfg.SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        generator=generator,
        worker_init_fn=_seed_worker,
    )

    model = build_swin_unetr(cfg, pretrained=True).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.WARMUP_EPOCHS, num_training_steps=epochs
    )

    scaler = torch.amp.GradScaler(cfg.DEVICE.type, enabled=cfg.USE_AMP)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=cfg.FOCAL_GAMMA)
    tversky_criterion = TverskyLoss(alpha=cfg.TVERSKY_ALPHA, beta=cfg.TVERSKY_BETA, num_classes=cfg.NUM_LABELS)

    best_miou = 0.0
    history = {"loss": [], "miou": [], "mhd95": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            images = batch["image"].to(cfg.DEVICE)
            labels = batch["label"][:, 0].long().to(cfg.DEVICE)

            with torch.amp.autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
                logits = model(images)
                main_loss = cfg.FOCAL_WEIGHT * focal_criterion(logits, labels)

                aux_loss = 0.0
                if cfg.USE_TVERSKY:
                    aux_loss += cfg.TVERSKY_WEIGHT * tversky_criterion(logits, labels)

                loss = (main_loss + aux_loss) / cfg.ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}"})

        scheduler.step()

        if (epoch + 1) % cfg.VAL_INTERVAL == 0:
            curr_miou, curr_mhd95 = validate_3d(model, val_volumes, cfg)
            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            history["miou"].append(curr_miou)
            history["mhd95"].append(curr_mhd95)
            log.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | mIoU: {curr_miou:.4f} | mHD95: {curr_mhd95:.2f}")
            if curr_miou > best_miou:
                best_miou = curr_miou
                torch.save(model.state_dict(), save_path)
                log.info(f"New best model saved! (mIoU: {best_miou:.4f})")

        plot_history(history, os.path.join(output_dir, "metrics.png"))

    if torch.cuda.is_available():
        log.info(f"Peak training VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
