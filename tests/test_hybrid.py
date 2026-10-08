import numpy as np
import pytest

from octseg.config import Config
from octseg.hybrid_inference import ENSEMBLE_MODES, HybridInference, blend_irf


def test_blend_irf_strategies():
    p_base = np.array([0.2, 0.4, 0.8], dtype=np.float32)
    p_expert = np.array([0.6, 0.1, 0.9], dtype=np.float32)
    w = 0.5

    # Linear
    lin = blend_irf(p_base, p_expert, "linear", w)
    assert np.allclose(lin, 0.5 * p_base + 0.5 * p_expert)

    # Max / Min
    assert np.allclose(blend_irf(p_base, p_expert, "max", w), np.maximum(p_base, p_expert))
    assert np.allclose(blend_irf(p_base, p_expert, "min", w), np.minimum(p_base, p_expert))

    # Geometric
    geo = blend_irf(p_base, p_expert, "geometric", w)
    assert np.allclose(geo, np.sqrt(p_base * p_expert))

    # Harmonic
    harm = blend_irf(p_base, p_expert, "harmonic", w)
    expected_harm = 2.0 / (1.0 / (p_base + 1e-8) + 1.0 / (p_expert + 1e-8) + 1e-8)
    assert np.allclose(harm, np.clip(expected_harm, 0, 1), atol=1e-5)

    # Confidence (only between 0.15 and 0.45)
    conf = blend_irf(p_base, p_expert, "confidence", 0.5)
    # p_base[0]=0.2 is uncertain -> blended with 0.6 -> 0.4
    # p_base[1]=0.4 is uncertain -> blended with 0.1 -> 0.25
    # p_base[2]=0.8 is certain (>0.45) -> remains 0.8
    assert np.isclose(conf[0], 0.4)
    assert np.isclose(conf[1], 0.25)
    assert np.isclose(conf[2], 0.8)


def test_blend_irf_custom_confidence():
    p_base = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    p_expert = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    res = blend_irf(p_base, p_expert, "confidence", 0.5, conf_low=0.2, conf_high=0.7)
    assert np.isclose(res[0], 0.1)
    assert np.isclose(res[1], 0.4)
    assert np.isclose(res[2], 0.55)
    assert np.isclose(res[3], 0.9)

def test_blend_irf_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown blend strategy"):
        blend_irf(np.array([0.5]), np.array([0.5]), "non_existent_strategy", 0.5)


def test_hybrid_inference_mismatched_25d_rejected():
    cfg_base = Config(USE_25D=True)
    cfg_expert = Config(USE_25D=False)
    with pytest.raises(ValueError, match="must share USE_25D"):
        HybridInference("dummy_base.pth", "dummy_expert.pth", cfg_base, cfg_expert)


def test_hybrid_inference_invalid_ensemble_mode():
    cfg = Config(USE_25D=False)
    expert_cfg = Config(USE_25D=False)
    with pytest.raises(ValueError, match="Unknown ensemble mode"):
        HybridInference("dummy_base.pth", "dummy_expert.pth", cfg, expert_cfg, ensemble_mode="bogus")


class _DummyHybrid(HybridInference):
    """Subclass bypassing neural net loading to test _merge logic in isolation."""
    def __init__(self, cfg, expert_cfg, ensemble_mode="soft", expert_weight=0.4, blend_strategy="linear"):
        if ensemble_mode not in ENSEMBLE_MODES:
            raise ValueError(f"Unknown ensemble mode {ensemble_mode!r}; expected one of {ENSEMBLE_MODES}")
        self.cfg = cfg
        self.expert_cfg = expert_cfg
        self.ensemble_mode = ensemble_mode
        self.expert_weight = expert_weight
        self.blend_strategy = blend_strategy
        self.irf_threshold = cfg.HYBRID_IRF_THRESHOLD
        self.irf_min_region_size = cfg.HYBRID_IRF_MIN_REGION_SIZE
        self.irf_override = cfg.HYBRID_IRF_OVERRIDE


def test_hybrid_merge_soft_mode():
    cfg = Config(CLASS_THRESHOLDS={1: 0.3, 2: 0.8, 3: 0.8}, HYBRID_IRF_THRESHOLD=0.25)
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode="soft", expert_weight=0.5, blend_strategy="linear")

    # Base: BG=0.8, IRF=0.1, SRF=0.1, PED=0.0
    # Expert IRF=0.5 -> linear blend: 0.5*0.1 + 0.5*0.5 = 0.30 >= IRF_THRESHOLD (0.25)
    base_probs = np.zeros((4, 4, 4), dtype=np.float32)
    base_probs[0] = 0.8
    base_probs[1] = 0.1
    expert_irf = np.full((4, 4), 0.5, dtype=np.float32)

    mask = dummy._merge(base_probs, expert_irf)
    assert (mask == 1).all()


def test_hybrid_merge_soft_mode_renormalization():
    cfg = Config(
        CLASS_THRESHOLDS={1: 0.35, 2: 0.18, 3: 0.8},
        HYBRID_IRF_THRESHOLD=0.35,
        PED_SHARPEN_FACTOR=0.0,
    )
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode="soft", expert_weight=0.5, blend_strategy="linear")

    # Base: BG=0.6, IRF=0.2, SRF=0.2, PED=0.0
    # Without normalization, SRF=0.2 > 0.18, so SRF would trigger.
    # With normalization:
    # p1_old = 0.2, p1_new = 0.5*0.2 + 0.5*0.6 = 0.4
    # scale = (1.0 - 0.4) / (1.0 - 0.2) = 0.75
    # SRF becomes 0.2 * 0.75 = 0.15 < 0.18, so SRF is suppressed!
    # IRF becomes 0.4 >= 0.35, so IRF triggers.
    base_probs = np.zeros((4, 4, 4), dtype=np.float32)
    base_probs[0] = 0.6
    base_probs[1] = 0.2
    base_probs[2] = 0.2
    base_probs[3] = 0.0
    expert_irf = np.full((4, 4), 0.6, dtype=np.float32)

    mask = dummy._merge(base_probs, expert_irf)
    assert (mask == 1).all()

def test_hybrid_merge_hard_mode_preserves_srf_ped():
    cfg = Config(CLASS_THRESHOLDS={1: 0.3, 2: 0.8, 3: 0.8}, HYBRID_IRF_THRESHOLD=0.25)
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode="hard")

    # Base has SRF (class 2) in top-left, background elsewhere
    base_probs = np.zeros((4, 8, 8), dtype=np.float32)
    base_probs[0] = 0.9
    base_probs[2, :4, :4] = 0.95
    base_probs[0, :4, :4] = 0.05

    # Expert says IRF everywhere with high confidence
    expert_irf = np.full((8, 8), 0.99, dtype=np.float32)

    mask = dummy._merge(base_probs, expert_irf)
    # SRF must NOT be overridden by IRF expert in hard mode!
    assert (mask[:4, :4] == 2).all()
    # Background should be claimed by expert as IRF
    assert (mask[4:, :] == 1).all()
    assert (mask[:4, 4:] == 1).all()


def test_hybrid_merge_replace_drops_base_irf_when_expert_below_threshold():
    cfg = Config(CLASS_THRESHOLDS={1: 0.3, 2: 0.8, 3: 0.8})
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode="replace")

    # Base has IRF (class 1) everywhere with high confidence
    base_probs = np.zeros((4, 8, 8), dtype=np.float32)
    base_probs[1] = 0.95
    base_probs[0] = 0.05

    # Expert IRF is low confidence (below threshold 0.25)
    expert_irf = np.full((8, 8), 0.10, dtype=np.float32)

    mask = dummy._merge(base_probs, expert_irf)
    # Base IRF must be discarded and replaced with 0 since expert <= 0.25
    assert (mask == 0).all()


def test_hybrid_merge_replace_adds_expert_irf_on_background():
    cfg = Config(CLASS_THRESHOLDS={1: 0.3, 2: 0.8, 3: 0.8})
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode="replace")

    # Base has background everywhere
    base_probs = np.zeros((4, 8, 8), dtype=np.float32)
    base_probs[0] = 1.0

    # Expert has high IRF probability
    expert_irf = np.full((8, 8), 0.80, dtype=np.float32)

    mask = dummy._merge(base_probs, expert_irf)
    # Background claimed by expert IRF
    assert (mask == 1).all()


def test_hybrid_merge_replace_never_overwrites_srf_ped():
    cfg = Config(CLASS_THRESHOLDS={1: 0.3, 2: 0.8, 3: 0.8})
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode="replace")

    base_probs = np.zeros((4, 8, 8), dtype=np.float32)
    # Both IRF and SRF/PED have high probability, but SRF/PED priority 1 should be kept
    base_probs[1] = 0.90
    base_probs[2, :4, :] = 0.95   # SRF top half
    base_probs[3, 4:, :] = 0.95   # PED bottom half

    # Expert IRF is very confident
    expert_irf = np.full((8, 8), 0.99, dtype=np.float32)

    mask = dummy._merge(base_probs, expert_irf)
    assert (mask[:4, :] == 2).all()
    assert (mask[4:, :] == 3).all()


@pytest.mark.parametrize("mode", ["soft", "hard", "replace"])
def test_hybrid_postprocess_does_not_mutate_inputs(mode):
    cfg = Config(CLASS_THRESHOLDS={1: 0.3, 2: 0.8, 3: 0.8}, HYBRID_IRF_MIN_REGION_SIZE=1)
    expert_cfg = Config(CLASS_THRESHOLDS={1: 0.25})
    dummy = _DummyHybrid(cfg, expert_cfg, ensemble_mode=mode)

    base_probs = np.zeros((4, 16, 16), dtype=np.float32)
    base_probs[0] = 0.4
    base_probs[1] = 0.35
    base_probs[2] = 0.15
    base_probs[3] = 0.1
    expert_irf = np.full((16, 16), 0.6, dtype=np.float32)

    base_copy = base_probs.copy()
    expert_copy = expert_irf.copy()

    _ = dummy.postprocess(base_probs, expert_irf)

    assert np.array_equal(base_probs, base_copy)
    assert np.array_equal(expert_irf, expert_copy)


def test_select_best_filtering():
    from octseg.sweep_hybrid import select_best

    base_row = {
        "class_dices": {1: 0.50, 2: 0.70, 3: 0.60},
    }

    # Case 1: Row beats IRF, but drops SRF by more than tolerance (0.01) -> skipped
    row_violating_srf = {
        "params": {"ensemble_mode": "soft", "blend_strategy": "linear", "expert_weight": 0.4, "irf_threshold": 0.25},
        "class_dices": {1: 0.55, 2: 0.68, 3: 0.60},  # SRF dropped by 0.02 > 0.01
    }
    # Case 2: Row beats IRF, but drops PED by more than tolerance -> skipped
    row_violating_ped = {
        "params": {"ensemble_mode": "soft", "blend_strategy": "linear", "expert_weight": 0.5, "irf_threshold": 0.25},
        "class_dices": {1: 0.56, 2: 0.70, 3: 0.58},  # PED dropped by 0.02 > 0.01
    }
    assert select_best([row_violating_srf, row_violating_ped], base_row, tolerance=0.01) is None

    # Case 3: No row beats base IRF Dice -> None
    row_lower_irf = {
        "params": {"ensemble_mode": "hard"},
        "class_dices": {1: 0.49, 2: 0.70, 3: 0.60},
    }
    row_equal_irf = {
        "params": {"ensemble_mode": "replace"},
        "class_dices": {1: 0.50, 2: 0.70, 3: 0.60},
    }
    assert select_best([row_lower_irf, row_equal_irf], base_row, tolerance=0.01) is None

    # Case 4: Qualifying rows -> picks highest IRF; ties -> first in grid order
    row_qualifying_1 = {
        "params": {"ensemble_mode": "soft", "expert_weight": 0.3},
        "class_dices": {1: 0.52, 2: 0.695, 3: 0.60},
    }
    row_qualifying_2 = {
        "params": {"ensemble_mode": "soft", "expert_weight": 0.4},
        "class_dices": {1: 0.54, 2: 0.692, 3: 0.595},  # SRF/PED within 0.01
    }
    row_qualifying_tie = {
        "params": {"ensemble_mode": "soft", "expert_weight": 0.5},
        "class_dices": {1: 0.54, 2: 0.70, 3: 0.60},
    }
    best = select_best([row_qualifying_1, row_qualifying_2, row_qualifying_tie], base_row, tolerance=0.01)
    assert best == row_qualifying_2  # first of the 0.54 ties
