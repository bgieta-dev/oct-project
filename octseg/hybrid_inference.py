from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor

from octseg.config import Config
from octseg.model import load_segformer
from octseg.postprocess import apply_thresholds, clean_regions, predict_logits, sharpen_ped

ENSEMBLE_MODES = ("soft", "hard", "replace")
BLEND_STRATEGIES = ("linear", "geometric", "harmonic", "max", "min", "confidence")

def blend_irf(p_base, p_expert, strategy, w, conf_low=0.15, conf_high=0.45):
    """Blends base and expert IRF probability maps; `w` is the expert weight."""
    if strategy == "linear":
        return (1 - w) * p_base + w * p_expert
    if strategy == "geometric":
        return np.clip((p_base ** (1 - w)) * (p_expert ** w), 0, 1)
    if strategy == "harmonic":
        return np.clip(1.0 / ((1 - w) / (p_base + 1e-8) + w / (p_expert + 1e-8) + 1e-8), 0, 1)
    if strategy == "max":
        return np.maximum(p_base, p_expert)
    if strategy == "min":
        return np.minimum(p_base, p_expert)
    if strategy == "confidence":
        # Only blend in the uncertain region of the base model
        uncertain = (p_base > conf_low) & (p_base < conf_high)
        return np.where(uncertain, (1 - w) * p_base + w * p_expert, p_base)
    raise ValueError(f"Unknown blend strategy {strategy!r}; expected one of {BLEND_STRATEGIES}")


class HybridInference:
    """Multi-class base SegFormer + binary IRF expert, merged via "soft", "hard", or "replace" mode.

    Unset keyword arguments default to the ``HYBRID_*`` entries of `cfg`.
    Both models receive the same input image, so `cfg` and `expert_cfg` must agree on USE_25D.
    """

    def __init__(self, base_model_path, expert_model_path, cfg: Config, expert_cfg: Config,
                 ensemble_mode=None, expert_weight=None, blend_strategy=None,
                 irf_threshold=None, irf_min_region_size=None, irf_override=None):
        if cfg.USE_25D != expert_cfg.USE_25D:
            raise ValueError("Base and expert configs must share USE_25D: the hybrid feeds both the same image.")
        self.cfg, self.expert_cfg = cfg, expert_cfg
        self.device = cfg.DEVICE
        self.ensemble_mode = cfg.HYBRID_ENSEMBLE_MODE if ensemble_mode is None else ensemble_mode
        if self.ensemble_mode not in ENSEMBLE_MODES:
            raise ValueError(f"Unknown ensemble mode {self.ensemble_mode!r}; expected one of {ENSEMBLE_MODES}")
        self.expert_weight = cfg.HYBRID_EXPERT_WEIGHT if expert_weight is None else expert_weight
        self.blend_strategy = cfg.HYBRID_BLEND_STRATEGY if blend_strategy is None else blend_strategy
        self.irf_threshold = cfg.HYBRID_IRF_THRESHOLD if irf_threshold is None else irf_threshold
        self.irf_min_region_size = cfg.HYBRID_IRF_MIN_REGION_SIZE if irf_min_region_size is None else irf_min_region_size
        self.irf_override = cfg.HYBRID_IRF_OVERRIDE if irf_override is None else irf_override

        self.base_model = load_segformer(cfg, base_model_path)
        self.expert_model = load_segformer(expert_cfg, expert_model_path)
        self.processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)

    def _merge(self, base_probs, expert_irf_prob):
        """Merges class probabilities [C,H,W] with the expert IRF map [H,W] into a label mask."""
        thresholds = dict(self.cfg.CLASS_THRESHOLDS)
        thresholds[1] = self.irf_threshold

        conf_low = getattr(self.cfg, "HYBRID_CONFIDENCE_LOW", 0.15)
        conf_high = getattr(self.cfg, "HYBRID_CONFIDENCE_HIGH", 0.45)
        ped_factor = getattr(self.cfg, "PED_SHARPEN_FACTOR", 1.0)

        if self.ensemble_mode == "soft":
            merged = base_probs.copy()
            p1_old = base_probs[1]
            p1_new = blend_irf(
                p1_old,
                expert_irf_prob,
                self.blend_strategy,
                self.expert_weight,
                conf_low=conf_low,
                conf_high=conf_high,
            )
            p1_new = np.clip(p1_new, 0.0, 1.0)
            scale = (1.0 - p1_new) / (1.0 - p1_old + 1e-8)
            for c in range(merged.shape[0]):
                if c != 1:
                    merged[c] = np.clip(base_probs[c] * scale, 0.0, 1.0)
            merged[1] = p1_new
            sharpen_ped(merged, factor=ped_factor)
            return apply_thresholds(merged, thresholds, irf_override=self.irf_override)

        if self.ensemble_mode == "replace":
            probs = sharpen_ped(base_probs.copy(), factor=ped_factor)
            mask = apply_thresholds(probs, thresholds, irf_override=False)   # base SRF/PED kept (priority 1)
            mask[mask == 1] = 0                                               # base IRF discarded
            mask[(mask == 0) & (expert_irf_prob > self.expert_cfg.CLASS_THRESHOLDS[1])] = 1  # expert IRF (priority 2)
            return mask

        # "hard": threshold the base model, then let the expert add IRF over background/IRF only
        # (never over SRF/PED, to avoid anatomical corruption).
        probs = sharpen_ped(base_probs.copy(), factor=ped_factor)
        mask = apply_thresholds(probs, thresholds, irf_override=True)
        expert_mask = expert_irf_prob > self.expert_cfg.CLASS_THRESHOLDS[1]
        mask[(mask <= 1) & expert_mask] = 1
        return mask

    @torch.no_grad()
    def probabilities(self, image_np) -> Tuple[np.ndarray, np.ndarray]:
        """Runs base and expert models on `image_np` and returns their probabilities.

        Args:
            image_np: [H, W, 3] image from OCTDataset (``orig_img``).

        Returns:
            (base_probs [C, H, W], expert_irf_prob [H, W]).
        """
        pixel_values = self.processor(images=image_np, return_tensors="pt").pixel_values.to(self.device)
        target_size = image_np.shape[:2]

        base_logits = predict_logits(self.base_model, pixel_values, target_size, self.cfg)
        base_probs = F.softmax(base_logits, dim=1).squeeze(0).cpu().numpy()
        expert_logits = predict_logits(self.expert_model, pixel_values, target_size, self.expert_cfg)
        expert_irf_prob = F.softmax(expert_logits, dim=1).squeeze(0).cpu().numpy()[1]
        return base_probs, expert_irf_prob

    def postprocess(self, base_probs, expert_irf_prob) -> np.ndarray:
        """Merges class probabilities and cleans regions. Does not mutate inputs.

        Args:
            base_probs: Base model class probabilities [C, H, W].
            expert_irf_prob: Expert model IRF probabilities [H, W].

        Returns:
            Cleaned label mask [H, W].
        """
        mask = self._merge(base_probs, expert_irf_prob)

        # Class-specific minimum sizes keep small expert IRF detections.
        min_sizes = {c: self.cfg.MIN_REGION_SIZE for c in range(1, self.cfg.NUM_LABELS)}
        min_sizes[1] = self.irf_min_region_size
        return clean_regions(mask, min_sizes, irf_open=True)

    @torch.no_grad()
    def segment(self, image_np, return_attention=False):
        """Segments one image.

        Args:
            image_np: [H, W, 3] image from OCTDataset (``orig_img``).
            return_attention: also return the base model's encoder-block-5 attention map.

        Returns:
            Mask [H, W] (0=BG, 1=IRF, 2=SRF, 3=PED); with `return_attention`, ``(mask, attention_map)``.
        """
        base_probs, expert_irf_prob = self.probabilities(image_np)
        mask = self.postprocess(base_probs, expert_irf_prob)

        if not return_attention:
            return mask
        pixel_values = self.processor(images=image_np, return_tensors="pt").pixel_values.to(self.device)
        att = self.base_model(pixel_values=pixel_values, output_attentions=True).attentions[5]
        spatial_att = torch.mean(torch.mean(att, dim=1), dim=1)
        grid_size = int(np.sqrt(spatial_att.shape[1]))
        return mask, spatial_att.view(-1, grid_size, grid_size).squeeze(0).cpu().numpy()
