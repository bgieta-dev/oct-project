"""Configuration: a `Config` dataclass (defaults = test18 lineage) plus YAML loading.

Usage:
    cfg = load_config("configs/irf_expert.yaml")   # merges `base:` chain, then the file itself

YAML keys are the `Config` field names (UPPER_CASE). `DEVICE`, data paths and the Discord
URL are resolved at runtime and never stored in YAML. No import-time side effects.
"""
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

# (batch_size, accumulation_steps) for a 12 GB GPU, keyed by MiT variant.
_VRAM_TABLE = {"b5": (2, 16), "b4": (4, 8), "b3": (8, 4), "b2": (8, 4), "b1": (16, 2), "b0": (16, 2)}


def get_vram_config(model_name: str) -> Tuple[int, int]:
    """Batch size and gradient-accumulation steps that fit 12 GB VRAM, with a defensive fallback."""
    name = model_name.lower()
    for variant, cfg in _VRAM_TABLE.items():
        if variant in name:
            return cfg
    logging.warning(f"Unknown MODEL_NAME '{model_name}'. Using safe defensive VRAM limits to avoid OOM.")
    return 4, 8


@dataclass
class Config:
    # --- architecture ---
    MODEL_NAME: str = "nvidia/mit-b2"
    NUM_LABELS: int = 4
    TARGET_CLASS: Optional[int] = None  # set (e.g. 1) for binary expert models
    USE_MULTIMODAL: bool = True  # only takes effect when USE_25D is False
    USE_25D: bool = True
    SEED: int = 42

    # --- training ---
    LR: float = 5e-5
    EPOCHS: int = 80
    USE_AMP: bool = True
    VAL_INTERVAL: int = 1
    WARMUP_EPOCHS: int = 15
    DROPOUT_RATE: float = 0.2
    BATCH_SIZE: Optional[int] = None  # derived from MODEL_NAME when None
    ACCUMULATION_STEPS: Optional[int] = None  # derived from MODEL_NAME when None

    # --- loss ---
    USE_DYNAMIC_WEIGHTS: bool = True
    CLASS_WEIGHTS: List[float] = field(default_factory=lambda: [0.2, 5.0, 2.0, 2.0])  # used when USE_DYNAMIC_WEIGHTS is False
    USE_CLAHE: bool = True
    USE_TVERSKY: bool = True
    FOCAL_WEIGHT: float = 0.5
    TVERSKY_WEIGHT: float = 0.5
    FOCAL_GAMMA: float = 3.0
    # High recall: large penalty for false negatives (beta), small for false positives (alpha)
    TVERSKY_ALPHA: float = 0.1
    TVERSKY_BETA: float = 0.9
    USE_BOUNDARY_LOSS: bool = False
    BOUNDARY_ALPHA: float = 0.1

    # --- classes ---
    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {0: "Background", 1: "IRF", 2: "SRF", 3: "PED"})

    # --- evaluation & post-processing ---
    MIN_REGION_SIZE: int = 80
    USE_TTA: bool = True
    TTA_SCALES: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    # Lower thresholds force fluid predictions at lower confidence (clinical high recall)
    CLASS_THRESHOLDS: Dict[int, float] = field(default_factory=lambda: {1: 0.30, 2: 0.80, 3: 0.80})
    ATTENTION_CONTRAST: float = 0.6
    PED_SHARPEN_FACTOR: float = 1.0
    SAVE_FAILURES: bool = True
    TOP_K_FAILURES: int = 5
    CENTRAL_SLICE_IDX: int = 1  # centre channel of the 2.5D stack

    # --- augmentation ---
    AUG_SIZE: Tuple[int, int] = (512, 512)
    AUG_SCALE: Tuple[float, float] = (0.8, 1.0)
    AUG_PROBS: Dict[str, float] = field(default_factory=lambda: {"flip": 0.5, "rotate": 0.5, "brightness": 0.2, "noise": 0.2})

    # --- hybrid ensemble (octseg.evaluate_hybrid defaults) ---
    HYBRID_ENSEMBLE_MODE: str = "soft"
    HYBRID_EXPERT_WEIGHT: float = 0.4
    HYBRID_BLEND_STRATEGY: str = "linear"
    HYBRID_IRF_THRESHOLD: float = 0.25
    HYBRID_IRF_MIN_REGION_SIZE: int = 12
    HYBRID_IRF_OVERRIDE: bool = False

    HYBRID_CONFIDENCE_LOW: float = 0.15
    HYBRID_CONFIDENCE_HIGH: float = 0.45
    def __post_init__(self):
        self.AUG_SIZE = tuple(self.AUG_SIZE)
        self.AUG_SCALE = tuple(self.AUG_SCALE)
        if self.BATCH_SIZE is None or self.ACCUMULATION_STEPS is None:
            batch_size, accum_steps = get_vram_config(self.MODEL_NAME)
            if self.BATCH_SIZE is None:
                self.BATCH_SIZE = batch_size
            if self.ACCUMULATION_STEPS is None:
                self.ACCUMULATION_STEPS = accum_steps

    # --- runtime-resolved values (never serialised) ---
    @property
    def DEVICE(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def DATA_DIR(self) -> Path:
        return Path(os.getenv("OCT_DATA_DIR", ROOT / "data_folder"))

    # Strings, because dataset.py derives multimodal paths via str.replace.
    @property
    def IMG_DIR(self) -> str:
        return str(self.DATA_DIR / "cropped_images")

    @property
    def MASK_DIR(self) -> str:
        return str(self.DATA_DIR / "cropped_masks")

    @property
    def SPLIT_DIR(self) -> Path:
        return ROOT / "splits"

    def to_dict(self) -> dict:
        """Plain-Python dict of all fields (YAML-serialisable)."""
        d = asdict(self)
        d["AUG_SIZE"], d["AUG_SCALE"] = list(self.AUG_SIZE), list(self.AUG_SCALE)
        return d


def _read_yaml(path: Path, seen: Tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic `base:` chain through {path}")
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("base", None)
    merged = _read_yaml(path.parent / base, seen + (path,)) if base else {}
    merged.update(data)
    return merged


def load_config(path=None) -> Config:
    """Loads `path` (default: configs/base.yaml), resolving its `base:` chain.

    Raises ValueError for keys that are not `Config` fields, so typos fail loudly.
    """
    load_dotenv(ROOT / ".env", override=True)
    path = CONFIG_DIR / "base.yaml" if path is None else Path(path)
    values = _read_yaml(path)
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return Config(**values)


def save_config(cfg: Config, path) -> None:
    """Writes the fully resolved config (no `base:`) for run reproducibility."""
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
