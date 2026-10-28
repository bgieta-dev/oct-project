"""SegFormer, RETFound and SwinUNETR construction and strict checkpoint loading."""
import logging
import math
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput

from octseg.config import ROOT

# RETFound-MAE-OCT encoder (official models_vit.py: ViT-L/16). Tests monkeypatch a tiny ViT here.
RETFOUND_VIT_KWARGS = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True)
RETFOUND_IMG_SIZE = 512  # SegformerImageProcessor output size; other sizes (TTA scales) via dynamic_img_size
RETFOUND_DETAIL_STAGES = 2  # MiT stages kept: stride 4 and stride 8

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


class MLPProj(nn.Module):
    """Per-token linear projection of a feature map (same as HF ``SegformerMLP``)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Linear(cin, cout)

    def forward(self, x):
        b, _, h, w = x.shape
        x = self.proj(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(b, -1, h, w)


class RetFoundSegModel(nn.Module):
    """RETFound ViT encoder + MiT detail branch (stages 1-2) + SegFormer-style all-MLP decoder.

    Pure transformer: the only convolutions are the ViT/MiT patch embeddings. Logits come out at
    stride 4 of the (16-px padded) input, like SegFormer; callers interpolate to the target size.
    """

    def __init__(self, cfg, pretrained_detail=True):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer

        self.encoder = VisionTransformer(
            img_size=RETFOUND_IMG_SIZE, num_classes=0, global_pool="", dynamic_img_size=True,
            drop_path_rate=cfg.RETFOUND_DROP_PATH, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **RETFOUND_VIT_KWARGS,
        )
        depth = len(self.encoder.blocks)
        if max(cfg.RETFOUND_TAP_LAYERS) >= depth:
            raise ValueError(f"RETFOUND_TAP_LAYERS {tuple(cfg.RETFOUND_TAP_LAYERS)} exceed encoder depth {depth}")
        self.taps = list(cfg.RETFOUND_TAP_LAYERS)
        embed_dim, dim = self.encoder.embed_dim, cfg.RETFOUND_DECODER_DIM

        if cfg.RETFOUND_DETAIL_BRANCH:
            from transformers import SegformerConfig, SegformerModel

            if pretrained_detail:
                mit = SegformerModel.from_pretrained(cfg.RETFOUND_DETAIL_MODEL)
            else:
                mit = SegformerModel(SegformerConfig.from_pretrained(cfg.RETFOUND_DETAIL_MODEL))
            self.detail_embed = nn.ModuleList(mit.encoder.patch_embeddings[:RETFOUND_DETAIL_STAGES])
            self.detail_blocks = nn.ModuleList(mit.encoder.block[:RETFOUND_DETAIL_STAGES])
            self.detail_norm = nn.ModuleList(mit.encoder.layer_norm[:RETFOUND_DETAIL_STAGES])
            detail_dims = list(mit.config.hidden_sizes[:RETFOUND_DETAIL_STAGES])
        else:
            self.detail_embed = None
            detail_dims = []

        self.tap_proj = nn.ModuleList([MLPProj(embed_dim, dim) for _ in self.taps])
        self.detail_proj = nn.ModuleList([MLPProj(c, dim) for c in detail_dims])
        n_in = len(self.taps) + len(detail_dims)
        self.fuse = nn.Linear(n_in * dim, dim)
        self.fuse_norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.DROPOUT_RATE)
        self.classifier = nn.Linear(dim, cfg.NUM_LABELS)

    def forward(self, pixel_values, output_attentions=False):
        """`output_attentions` is accepted for signature parity with SegFormer and ignored."""
        b = pixel_values.shape[0]
        h, w = pixel_values.shape[-2:]
        hp, wp = math.ceil(h / 16) * 16, math.ceil(w / 16) * 16
        if (hp, wp) != (h, w):
            x = F.interpolate(pixel_values, (hp, wp), mode="bilinear", align_corners=False)
        else:
            x = pixel_values
        h4, w4 = hp // 4, wp // 4

        vit = self.encoder.forward_intermediates(
            x, indices=self.taps, stop_early=True, intermediates_only=True, output_fmt="NCHW"
        )
        feats = [proj(f) for proj, f in zip(self.tap_proj, vit)]

        if self.detail_embed is not None:
            # Mirrors SegformerEncoder.forward for the first RETFOUND_DETAIL_STAGES stages.
            t = x
            for embed, blocks, norm, proj in zip(self.detail_embed, self.detail_blocks, self.detail_norm,
                                                 self.detail_proj):
                tok, th, tw = embed(t)
                for layer in blocks:
                    tok = layer(tok, th, tw)[0]
                tok = norm(tok)
                t = tok.transpose(1, 2).reshape(b, -1, th, tw)
                feats.append(proj(t))

        feats = [
            f if f.shape[-2:] == (h4, w4) else F.interpolate(f, (h4, w4), mode="bilinear", align_corners=False)
            for f in feats
        ]
        tokens = torch.cat(feats, dim=1).flatten(2).transpose(1, 2)
        tokens = self.dropout(self.act(self.fuse_norm(self.fuse(tokens))))
        logits = self.classifier(tokens).transpose(1, 2).reshape(b, -1, h4, w4)
        return SemanticSegmenterOutput(logits=logits)


def _load_retfound_encoder(encoder, weights):
    """Loads RETFound MAE weights (HF repo id or local .pth) into the timm ViT `encoder`.

    Raises RuntimeError if the repo is gated without access or encoder keys are missing.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    from timm.layers import resample_abs_pos_embed

    local = Path(weights)
    if not local.is_absolute():
        local = ROOT / local
    if local.is_file():
        path = local
    else:
        try:
            path = hf_hub_download(repo_id=weights, filename=f"{weights.split('/')[-1]}.pth")
        except GatedRepoError as e:
            raise RuntimeError(
                f"RETFound weights are gated: accept the licence at https://huggingface.co/{weights} "
                "and run `hf auth login` (or set HF_TOKEN in .env)"
            ) from e

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        log.warning("RETFound checkpoint needs weights_only=False (official MAE checkpoint stores argparse args)")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    cleaned = {}
    for k, v in sd.items():
        for prefix in ("backbone.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if k.startswith(("decoder", "mask_token", "head", "fc_norm")):
            continue
        cleaned[k] = v
    sd = cleaned

    if "pos_embed" in sd and sd["pos_embed"].shape != encoder.pos_embed.shape:
        sd["pos_embed"] = resample_abs_pos_embed(
            sd["pos_embed"], new_size=list(encoder.patch_embed.grid_size), num_prefix_tokens=1
        )

    res = encoder.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("norm.")]
    if missing:
        raise RuntimeError(f"RETFound checkpoint missing encoder keys: {missing[:10]}")
    log.info(f"Loaded RETFound encoder from {path} ({len(sd)} tensors; ignored {len(res.unexpected_keys)} unexpected)")


def build_retfound(cfg, pretrained=True):
    """Builds the RETFound segmentation model: load encoder weights, freeze, inject LoRA.

    The construction order is identical for training and loading, so state-dict keys match.

    Args:
        cfg: config providing the RETFOUND_* fields, NUM_LABELS, DROPOUT_RATE.
        pretrained: load RETFound encoder weights and the pretrained MiT detail branch.

    Returns:
        RetFoundSegModel (not moved to a device).
    """
    model = RetFoundSegModel(cfg, pretrained_detail=pretrained)
    if pretrained:
        if cfg.RETFOUND_WEIGHTS:
            _load_retfound_encoder(model.encoder, cfg.RETFOUND_WEIGHTS)
        else:
            log.warning("RETFOUND_WEIGHTS is null: encoder randomly initialised (tests/smoke only)")

    for p in model.encoder.parameters():
        p.requires_grad = False
    if cfg.RETFOUND_LORA_RANK > 0:
        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=cfg.RETFOUND_LORA_RANK, lora_alpha=cfg.RETFOUND_LORA_ALPHA, lora_dropout=cfg.RETFOUND_LORA_DROPOUT,
            target_modules=r"blocks\.\d+\.attn\.(qkv|proj)", bias="none",
        )
        inject_adapter_in_model(lora_cfg, model.encoder)
        for name, p in model.encoder.named_parameters():
            p.requires_grad = "lora_" in name
    model.encoder.set_grad_checkpointing(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"RETFound trainable params: {trainable:.2f}M / {total:.2f}M ({100 * trainable / total:.1f}%)")
    return model


def build_processor(cfg):
    """Image processor for the 2D model described by `cfg` (labels never reduced)."""
    from transformers import SegformerImageProcessor

    if cfg.ARCH == "segformer":
        processor = SegformerImageProcessor.from_pretrained(cfg.MODEL_NAME)
    elif cfg.ARCH == "retfound":
        # RETFound uses ImageNet mean/std at 512x512, identical to the SegFormer defaults.
        processor = SegformerImageProcessor()
    else:
        raise ValueError(f"build_processor does not support ARCH={cfg.ARCH}")
    processor.do_reduce_labels = False
    return processor


def build_2d_model(cfg):
    """Builds the trainable 2D model for `cfg.ARCH` with pretrained weights (not moved to a device)."""
    if cfg.ARCH == "segformer":
        return SegformerForSemanticSegmentation.from_pretrained(
            cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS, ignore_mismatched_sizes=True,
            hidden_dropout_prob=cfg.DROPOUT_RATE,
            attention_probs_dropout_prob=cfg.DROPOUT_RATE,
            classifier_dropout_prob=cfg.DROPOUT_RATE,
        )
    if cfg.ARCH == "retfound":
        return build_retfound(cfg)
    raise ValueError(f"build_2d_model does not support ARCH={cfg.ARCH}")


def load_2d_model(cfg, weights_path, output_attentions=False):
    """Builds the 2D model for `cfg.ARCH` and strictly loads `weights_path`.

    Raises FileNotFoundError if the checkpoint is missing and RuntimeError on key mismatch.
    RETFound needs no HF token here: only the MiT config JSON is fetched.

    Returns:
        Model on `cfg.DEVICE` in eval mode.
    """
    if cfg.ARCH == "segformer":
        return load_segformer(cfg, weights_path, output_attentions=output_attentions)
    if cfg.ARCH == "retfound":
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
        model = build_retfound(cfg, pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model.to(cfg.DEVICE).eval()
    raise ValueError(f"load_2d_model does not support ARCH={cfg.ARCH}")


def optimizer_param_groups(model, cfg):
    """AdamW parameter groups; RETFound LoRA params get their own lr and no weight decay."""
    if cfg.ARCH == "segformer":
        return [{"params": list(model.parameters())}]
    if cfg.ARCH == "retfound":
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in params if "lora_" not in n]}]
        if cfg.RETFOUND_LORA_RANK > 0:
            groups.append({
                "params": [p for n, p in params if "lora_" in n],
                "lr": cfg.RETFOUND_LORA_LR,
                "weight_decay": 0.0,
            })
        return groups
    raise ValueError(f"optimizer_param_groups does not support ARCH={cfg.ARCH}")


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
