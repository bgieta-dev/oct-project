import logging
import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, get_cosine_schedule_with_warmup

from octseg.config import Config, load_config, save_config
from octseg.dataset import OCTDataset
from octseg.losses import BoundaryLoss, FocalLoss, TverskyLoss
from octseg.metrics import SegmentationMetrics
from octseg.postprocess import clean_regions
from octseg.splits import split_files
from octseg.viz import plot_history

log = logging.getLogger(__name__)


def seed_everything(seed):
    """Seeds python, numpy and torch (CPU + CUDA). cudnn determinism is deliberately not forced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_dynamic_weights(mask_paths, num_classes, target_class=None):
    log.info("Calculating dynamic class weights...")
    counts = np.zeros(num_classes)
    for p in tqdm(mask_paths, desc="Scanning masks"):
        m = np.array(Image.open(p))
        if target_class is not None:
            binary_m = np.zeros_like(m)
            binary_m[m == target_class] = 1
            m = binary_m
        counts += np.bincount(m.ravel(), minlength=num_classes)

    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * num_classes

    if weights[0] > 0.2:
        weights[0] = 0.2
        remaining_sum = num_classes - 0.2
        other_weights_sum = weights[1:].sum()
        if other_weights_sum > 0:
            weights[1:] = (weights[1:] / other_weights_sum) * remaining_sum

    return weights


def validate(model, val_loader, cfg):
    """One validation pass. Returns (mIoU, mHD95); HD95 uses the 100-px penalty for missed/false classes."""
    model.eval()
    metrics = SegmentationMetrics(cfg.NUM_LABELS, hd95_missing="penalty100", extended=False)
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(cfg.DEVICE)
            labels_batch = batch["labels"].to(cfg.DEVICE)
            outputs = model(pixel_values=pixel_values)
            logits = F.interpolate(outputs.logits, size=labels_batch.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            labels_np = labels_batch.cpu().numpy()

            if cfg.MIN_REGION_SIZE > 0:
                preds = np.array([clean_regions(p, cfg.MIN_REGION_SIZE, irf_open=False) for p in preds])

            for p, l_np in zip(preds, labels_np):
                metrics.update(l_np, p)
    result = metrics.compute()
    return result["mIoU"], result["mHD95"]


def train_model(output_dir, cfg: Config, epochs=None, save_path=None):
    """Trains a SegFormer and keeps the checkpoint with the best validation mIoU.

    Args:
        output_dir: run directory (receives metrics.png and, by default, best_model.pth).
        cfg: configuration object.
        epochs: overrides cfg.EPOCHS.
        save_path: checkpoint path (default: <output_dir>/best_model.pth).
    """
    epochs = cfg.EPOCHS if epochs is None else epochs
    save_path = os.path.join(output_dir, "best_model.pth") if save_path is None else save_path
    seed_everything(cfg.SEED)

    train_imgs, train_masks = split_files(cfg, "train")
    val_imgs, val_masks = split_files(cfg, "val")

    if cfg.USE_DYNAMIC_WEIGHTS:
        dyn_weights = calculate_dynamic_weights(train_masks, num_classes=cfg.NUM_LABELS, target_class=cfg.TARGET_CLASS)
        class_weights = torch.tensor(dyn_weights, dtype=torch.float32).to(cfg.DEVICE)
    else:
        class_weights = torch.tensor(cfg.CLASS_WEIGHTS).to(cfg.DEVICE)

    train_transform = A.Compose([
        A.CLAHE(clip_limit=2.0, p=0.5) if cfg.USE_CLAHE else A.NoOp(),
        A.RandomResizedCrop(size=cfg.AUG_SIZE, scale=cfg.AUG_SCALE, p=1.0),
        A.HorizontalFlip(p=cfg.AUG_PROBS["flip"]),
        A.Rotate(limit=10, p=cfg.AUG_PROBS["rotate"]),
        A.OneOf([A.ElasticTransform(alpha=1, sigma=50, p=1.0), A.GridDistortion(p=1.0)], p=0.4),
        A.RandomBrightnessContrast(p=cfg.AUG_PROBS["brightness"]),
        A.GaussNoise(p=cfg.AUG_PROBS["noise"]),
    ])

    val_transform = A.Compose([A.Resize(height=cfg.AUG_SIZE[0], width=cfg.AUG_SIZE[1])])

    processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)
    processor.do_reduce_labels = False

    train_ds = OCTDataset(train_imgs, train_masks, processor, transform=train_transform, cfg=cfg)
    val_ds = OCTDataset(val_imgs, val_masks, processor, transform=val_transform, cfg=cfg)

    generator = torch.Generator().manual_seed(cfg.SEED)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              generator=generator, worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS, ignore_mismatched_sizes=True,
        hidden_dropout_prob=cfg.DROPOUT_RATE,
        attention_probs_dropout_prob=cfg.DROPOUT_RATE,
        classifier_dropout_prob=cfg.DROPOUT_RATE,
    ).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.WARMUP_EPOCHS, num_training_steps=epochs
    )

    scaler = torch.amp.GradScaler(cfg.DEVICE.type, enabled=cfg.USE_AMP)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=cfg.FOCAL_GAMMA)
    tversky_criterion = TverskyLoss(alpha=cfg.TVERSKY_ALPHA, beta=cfg.TVERSKY_BETA, num_classes=cfg.NUM_LABELS)
    boundary_criterion = BoundaryLoss(num_classes=cfg.NUM_LABELS)

    best_miou = 0.0
    history = {"loss": [], "miou": [], "mhd95": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(cfg.DEVICE)
            labels = batch["labels"].to(cfg.DEVICE)
            with torch.amp.autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
                outputs = model(pixel_values=pixel_values)
                logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

                main_loss = cfg.FOCAL_WEIGHT * focal_criterion(logits, labels)

                aux_loss = 0.0
                if cfg.USE_TVERSKY:
                    aux_loss += cfg.TVERSKY_WEIGHT * tversky_criterion(logits, labels)

                if cfg.USE_BOUNDARY_LOSS:
                    b_alpha = cfg.BOUNDARY_ALPHA
                    b_w = min(b_alpha, (b_alpha / 10.0) + (epoch * (b_alpha / 40.0)))
                    aux_loss += b_w * boundary_criterion(F.softmax(logits, dim=1), labels)

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
            curr_miou, curr_mhd95 = validate(model, val_loader, cfg)
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


def main():
    """Training only (no evaluation); `python -m octseg.pipeline` runs the full train/eval/attention pipeline."""
    import argparse
    from octseg.runs import make_run_dir, setup_logging
    parser = argparse.ArgumentParser(description="Train a SegFormer (training only)")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--output", default=None, help="Run directory (default: outputs/runs/<timestamp>_<name>)")
    parser.add_argument("--name", default="train", help="Run name suffix")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = str(make_run_dir("runs", args.name, args.output))
    setup_logging(run_dir)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    if cfg.ARCH == "swin_unetr_3d":
        from octseg.train3d import train_model_3d
        train_model_3d(run_dir, cfg, epochs=args.epochs)
    else:
        train_model(run_dir, cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
