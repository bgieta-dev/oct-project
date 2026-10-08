import pytest

from octseg.config import CONFIG_DIR, Config, load_config, save_config


def test_base_yaml_matches_dataclass_defaults():
    assert load_config(CONFIG_DIR / "base.yaml") == Config()


def test_expert_batch_size_follows_its_backbone():
    """Regression for B2: expert (mit-b0) must not inherit the mit-b2 batch size."""
    base, expert = load_config(CONFIG_DIR / "base.yaml"), load_config(CONFIG_DIR / "irf_expert.yaml")
    assert (base.BATCH_SIZE, base.ACCUMULATION_STEPS) == (8, 4)
    assert (expert.BATCH_SIZE, expert.ACCUMULATION_STEPS) == (16, 2)
    assert expert.NUM_LABELS == 2 and expert.TARGET_CLASS == 1 and expert.MODEL_NAME.endswith("mit-b0")


def test_unknown_key_fails_loudly(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("base: " + str(CONFIG_DIR / "base.yaml") + "\nLEARNING_RATE: 1\n")
    with pytest.raises(ValueError, match="LEARNING_RATE"):
        load_config(p)


def test_hybrid_config_inherits_base():
    hybrid = load_config(CONFIG_DIR / "hybrid.yaml")
    assert hybrid.MODEL_NAME == "nvidia/mit-b2" and hybrid.HYBRID_IRF_MIN_REGION_SIZE == 12


def test_swin_unetr_config_validation():
    with pytest.raises(ValueError, match="Unknown ARCH"):
        Config(ARCH="invalid_arch")

    with pytest.raises(ValueError, match="SWIN_ROI dims must be divisible by 32"):
        Config(ARCH="swin_unetr_3d", SWIN_ROI=(32, 100, 160))

    with pytest.raises(ValueError, match="SWIN_PRETRAINED requires SWIN_FEATURE_SIZE 48"):
        Config(ARCH="swin_unetr_3d", SWIN_PRETRAINED="x", SWIN_FEATURE_SIZE=24)


def test_swin_unetr_config_yaml_roundtrip(tmp_path):
    cfg = Config(ARCH="swin_unetr_3d", SWIN_ROI=(32, 160, 160))
    out_yaml = tmp_path / "swin.yaml"
    save_config(cfg, out_yaml)
    loaded = load_config(out_yaml)
    assert loaded.SWIN_ROI == (32, 160, 160)
    assert isinstance(loaded.SWIN_ROI, tuple)
