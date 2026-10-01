import pytest

from octseg.config import CONFIG_DIR, Config, load_config


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
