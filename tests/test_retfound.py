from functools import partial

import pytest
import torch
from torch import nn

import octseg.model
from octseg.config import Config
from octseg.model import build_retfound, load_2d_model

TINY_VIT = dict(patch_size=16, embed_dim=64, depth=4, num_heads=2, mlp_ratio=2, qkv_bias=True)
DECODER_PREFIXES = ("tap_proj.", "detail_proj.", "fuse.", "fuse_norm.", "classifier.")
DETAIL_PREFIXES = ("detail_embed.", "detail_blocks.", "detail_norm.")


@pytest.fixture
def make_cfg(monkeypatch):
    """Tiny-ViT RETFound config factory (random encoder, no gated download, CPU)."""
    monkeypatch.setattr(octseg.model, "RETFOUND_VIT_KWARGS", TINY_VIT)

    def _make(**overrides):
        kwargs = dict(
            ARCH="retfound", RETFOUND_WEIGHTS=None, RETFOUND_TAP_LAYERS=(0, 1, 2, 3), RETFOUND_DECODER_DIM=32,
            RETFOUND_DETAIL_BRANCH=False, BATCH_SIZE=2, ACCUMULATION_STEPS=1, DEVICE_OVERRIDE="cpu",
        )
        kwargs.update(overrides)
        return Config(**kwargs)

    return _make


def _tiny_source_vit(img_size):
    from timm.models.vision_transformer import VisionTransformer

    return VisionTransformer(
        img_size=img_size, num_classes=0, global_pool="", norm_layer=partial(nn.LayerNorm, eps=1e-6), **TINY_VIT
    )


@pytest.mark.parametrize("detail", [False, True])
def test_forward_shape_pads_to_16_and_outputs_stride_4(make_cfg, detail):
    model = build_retfound(make_cfg(RETFOUND_DETAIL_BRANCH=detail)).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 72, 100)).logits
    # 72x100 is padded to 80x112; logits at stride 4.
    assert logits.shape == (2, 4, 20, 28)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("detail", [False, True])
def test_lora_rank_16_trains_only_lora_in_encoder(make_cfg, detail):
    model = build_retfound(make_cfg(RETFOUND_LORA_RANK=16, RETFOUND_DETAIL_BRANCH=detail))
    trainable_enc = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert trainable_enc
    assert all("lora_" in n for n in trainable_enc)

    rest = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert any(n.startswith("classifier.") for n, _ in rest)
    assert all(p.requires_grad for n, p in rest if n.startswith(DECODER_PREFIXES))
    detail_params = [p for n, p in rest if n.startswith(DETAIL_PREFIXES)]
    assert bool(detail_params) == detail
    assert all(p.requires_grad for p in detail_params)


@pytest.mark.parametrize("detail", [False, True])
def test_lora_rank_0_freezes_encoder(make_cfg, detail):
    model = build_retfound(make_cfg(RETFOUND_LORA_RANK=0, RETFOUND_DETAIL_BRANCH=detail))
    assert not any("lora_" in n for n, _ in model.encoder.named_parameters())
    assert not any(p.requires_grad for p in model.encoder.parameters())

    rest = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert rest and all(p.requires_grad for _, p in rest)
    assert any(n.startswith(DETAIL_PREFIXES) for n, _ in rest) == detail


def test_lora_grads_flow_through_grad_checkpointing(make_cfg):
    model = build_retfound(make_cfg(RETFOUND_LORA_RANK=16)).train()
    assert model.encoder.grad_checkpointing

    model(torch.randn(2, 3, 64, 64)).logits.square().mean().backward()

    qkv = model.encoder.blocks[0].attn.qkv
    # peft initialises lora_B to zero, so lora_A's first-step grad is zero but must exist;
    # a non-zero lora_B grad proves gradients reach the adapters through the checkpointed blocks.
    assert qkv.lora_A.default.weight.grad is not None
    grad_b = qkv.lora_B.default.weight.grad
    assert grad_b is not None and torch.isfinite(grad_b).all() and grad_b.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)


def test_pretrained_load_resamples_pos_embed_and_ignores_mae_extras(make_cfg, tmp_path):
    src = _tiny_source_vit(img_size=64)  # 4x4 grid; model encoder is 512 px -> 32x32 grid
    sd = src.state_dict()
    path = tmp_path / "fake.pth"
    torch.save({"model": {**sd, "decoder_embed.weight": torch.zeros(1), "mask_token": torch.zeros(1, 1, 64)}}, path)

    model = build_retfound(make_cfg(RETFOUND_WEIGHTS=str(path)))
    assert model.encoder.pos_embed.shape[1] == 32 * 32 + 1
    assert torch.equal(model.encoder.blocks[0].attn.qkv.base_layer.weight, sd["blocks.0.attn.qkv.weight"])
    assert torch.equal(model.encoder.cls_token, sd["cls_token"])

    del sd["blocks.0.norm1.weight"]
    torch.save({"model": sd}, path)
    with pytest.raises(RuntimeError, match="missing encoder keys"):
        build_retfound(make_cfg(RETFOUND_WEIGHTS=str(path)))


def test_load_2d_model_strict_roundtrip(make_cfg, tmp_path):
    cfg = make_cfg()
    model = build_retfound(cfg).eval()
    with torch.no_grad():
        for name, p in model.encoder.named_parameters():
            if "lora_B" in name:  # make adapters non-trivial so the round-trip covers them
                p.normal_()
    path = tmp_path / "best_model.pth"
    torch.save(model.state_dict(), path)

    loaded = load_2d_model(cfg, path)
    assert not loaded.training
    x = torch.randn(1, 3, 64, 80)
    with torch.no_grad():
        assert torch.equal(model(x).logits, loaded(x).logits)


def test_load_2d_model_missing_file_raises_not_found(make_cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_2d_model(make_cfg(), tmp_path / "missing.pth")


def test_load_2d_model_bogus_state_dict_raises(make_cfg, tmp_path):
    bad = tmp_path / "bogus.pth"
    torch.save({"bogus_layer.weight": torch.randn(10, 10)}, bad)
    with pytest.raises(RuntimeError):
        load_2d_model(make_cfg(), bad)
