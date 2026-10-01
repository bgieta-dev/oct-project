# Changelog

## [2.0.0] - 2026-10-01 — project reorganisation (see ORGANIZATION_PLAN.md)

### Changed (layout; tag `pre-reorg` marks the old layout)
- All code moved into the `octseg/` package; run modules with `python -m octseg.<module>`.
  Renames: `eval.py` -> `evaluate.py`, `eval_hybrid.py` -> `evaluate_hybrid.py`, `main.py` -> `pipeline.py`, `utils.py` -> `splits.py`.
- `run.sh`/`run.bat` -> `scripts/` (now `python -m octseg.pipeline "$@"`); `test_patients.txt` -> `splits/`, together with the new frozen `train_patients.txt` / `val_patients.txt` (seed-42 split; all code reads them instead of recomputing).
- Root leftovers `predictions.png`, `failures/` -> `experiments/legacy_root/`. Plan/summary docs -> `docs/archive/`, `changelog.md` -> `CHANGELOG.md`, thesis `dokumentacja/` -> `docs/thesis/` (sources now tracked; build products, `OCT.pdf`, `CNN/` ignored), `_help_data/` -> `analysis/external_roche/` (ignored).
- Outputs go to `outputs/{runs,eval,hybrid_eval,attention}/` (gitignored) instead of the repo root / `experiments/run_*`; the root `best_model.pth` / `irf_expert_best.pth` files are gone, checkpoints live in the run directory.
- Run snapshot is now `src/` (package copy) + `requirements.txt` + `splits/` + `config.yaml` + `git.txt` instead of copying all root files.
- `requirements.txt` pinned to the versions used on the dev host (torch per platform); `requirements-dev.txt`, `pyproject.toml` (ruff/pytest config), `.env.example` added; `update_git.sh` removed (`git add .` could commit outputs).

### Added
- `configs/{base,irf_expert,hybrid}.yaml` + `octseg.config.Config` dataclass / `load_config`; unknown keys fail loudly; no import-time side effects; `DEVICE`, data paths resolved at runtime (`OCT_DATA_DIR` overrides the data folder).
- Shared modules replacing duplicated code: `losses.py`, `metrics.py` (`SegmentationMetrics` with explicit `hd95_missing` policy), `postprocess.py` (TTA, PED sharpening, thresholds, region cleanup), `viz.py`, `runs.py`, `model.py` (strict checkpoint loading).
- CLIs: every module takes `--config`; `pipeline`/`train_expert`/`train` take `--name`, `--epochs`, `--output`. `metrics.yaml` is written in every run/eval directory.
- `tests/` (config, splits, dataset, losses, metrics, postprocess).

### Fixed
- **IRF expert config (B1/B2):** `train_model` now passes `cfg` to `OCTDataset`, so expert settings (`USE_25D`, `TARGET_CLASS`) are honoured; expert batch size derives from its backbone. Expert input decided as 2.5D and the hybrid feeds both models the same image (previously the expert got the middle slice replicated x3).
- Missing or mismatching checkpoints raise (`strict=True`, `weights_only=True`) instead of evaluating an untrained model (B4); `evaluate` logs INFO again in standalone mode (B5); eval/hybrid output dirs are gitignored (B6).
- `HybridInference`: `segment` and `segment_with_attention` merged into `segment(image, return_attention=False)`; defaults come from `HYBRID_*` only (the divergent `min_region_size` fallbacks are gone).
- Reproducibility: `seed_everything(SEED)`, seeded `DataLoader` generator/workers.

### Changed (behavior-neutral)
- Training step no longer passes `labels=` to the HF model (it computed an unused internal CE loss).
- `predictions.png` saved at dpi 120 (was 200); the attention column is titled "encoder block 5" and new attention files are named `..._block{i}.png` (were `..._stage{i}.png`; same index).
- Removed dead keys (`OPTIMIZER_TYPE`, `USE_SOFT_CRF`, `CRF_ITERATIONS`, `USE_FOCAL_TVERSKY`) and the SwinUNETR/monai branch.
- Verified numerically against tag `pre-reorg`: split hash, dataset item, loss values, TTA logits, threshold/cleanup masks and all hybrid merge modes are identical.

### Docs
- README rewritten; `docs/METHODS.md` (merged and corrected `Dokumentacja.md` + `summary.md`), `docs/EXPERIMENTS.md`, `docs/ROADMAP.md` added.

## [1.1.1] - 2026-06-20

### Added
- **Unified Post-Processing & Heuristics Parity**: Added full Test-Time Augmentation (TTA), clinical sharpening for PED (Class 3), reverse-priority thresholding, and morphological post-processing (opening and size filtering) to `hybrid_inference.py` to match the baseline `eval.py` pipeline.
- **Soft Probability-Level Ensembling**: Added support for soft-blending base and expert softmax probabilities before thresholding, with multiple blending strategies (`linear`, `geometric`, `harmonic`, `max`, `min`, `confidence`).
- **Class-Specific Minimum Region Sizes**: Added support for class-specific minimum region sizes, allowing tiny intraretinal fluid (IRF) cysts (defaulting to 12 pixels) to be preserved while keeping a higher noise filter (50 pixels) for SRF and PED.
- **Anatomical Layer Protection (`irf_override`)**: Added an override flag (defaulting to `False`) to prevent IRF predictions from overriding and segmenting/fragmenting subretinal fluid (SRF) and pigment epithelial detachment (PED) boundaries.
- **Dynamic Slice Selection and Predictions Grid**: Automated dynamic slice scanning and generation of a 4-column visual prediction comparison grid (`predictions.png`) inside `eval_hybrid.py` to mirror the layout of `eval.py`.
- **Runtime Argument Parsing**: Equipped `eval_hybrid.py` with runtime configuration flags (`--irf-threshold`, `--irf-min-region-size`, `--ensemble-mode`, `--expert-weight`, `--blend-strategy`, and `--irf-override`) for direct server-side tuning.
- **Centralized Hybrid Configurations**: Centralized optimal parameters (`HYBRID_ENSEMBLE_MODE = "soft"`, `HYBRID_IRF_THRESHOLD = 0.25`, and `HYBRID_IRF_MIN_REGION_SIZE = 12`) in `config.py`.

## [1.1.0] - 2026-06-18

### Added
- **Configurable Loss Weights**: Added `FOCAL_WEIGHT`, `TVERSKY_WEIGHT`, and `BOUNDARY_ALPHA` to configuration overrides in `train_model`.
- **Include Background Option**: Added `include_background` parameter to `TverskyLoss` to allow toggling background contribution to the loss.
- **Missing Class Penalty**: mHD95 metric now penalizes total detection failures (False Negatives/False Positives) with a constant factor (100.0) to prevent metric bias.

### Changed
- **FocalLoss Mathematical Correction**: Refactored `FocalLoss` to calculate unweighted Cross Entropy for the probability term ($p_t$) before applying alpha weights. This prevents corruption of the focal term.
- **Dynamic Weight Optimization**: 
    - Switched from `flatten()` to `ravel()` for faster mask scanning.
    - Implemented Laplace smoothing (`counts + 1.0`) to prevent weight explosion on missing classes.
    - Added full re-normalization after background weight capping to maintain constant gradient scale.
- **Train Loop Refactor**:
    - Optimized data loading: replaced redundant $O(N^2)$ loops with a single-pass file distribution logic.
    - Improved device portability: replaced hardcoded `'cuda'` strings with `cfg.DEVICE.type` for AMP components.
    - Optimized validation: added early exits for connected component analysis on empty predictions.
- **Dataset Robustness & Clean-up**:
    - Decoupled `OCTDataset` from global configuration, supporting injection of modular `cfg` parameters.
    - Standardized medical normalization into a single internal helper `_normalize_slice`, reducing computing redundancy.
    - Refactored 2.5D volumetric loading into an independent function `_load_neighbor` to avoid closure scope risks.
    - Hardened filename segment parsing to gracefully handle variations in underscore placement.
- **Evaluation Engine Modernization & Standalone CLI (`eval.py`)**:
    - Decoupled script from global imports: `evaluate_model` now accepts an explicit configuration object `cfg`.
    - Automated dynamic label handling: eliminated all hardcoded `[1, 2, 3]` arrays, dynamically constructing loops via `range(1, cfg.NUM_LABELS)`.
    - Added Binary Expert support: implemented mask re-mapping logic during visual selection when evaluating binary sub-models.
    - Optimized throughput: increased `num_workers=2` and enabled `pin_memory=True` for accelerated asynchronous disk I/O throughput.
    - Robust plotting layout: dynamically adjust grid generation sizing depending on the count of parsed target classes.
    - Removed Dead Code: Completely stripped out the vestigial `get_retina_mask` function and its dead imports from `train.py`.
    - Implemented CLI argument parsing (`argparse`) supporting `--model` and `--output` flags to allow flexible standalone evaluation runs without hardcoded weights path assumptions.
    - Wrapped execution logic inside an encapsulated `main()` entry block with clean, isolated logger instances to prevent downstream logging module conflicts or unexpected console side-effects.
- **Resilient Pipeline Orchestration (`main.py`)**:
    - Implemented comprehensive fault tolerance by execution isolation across distinct trial phases, preventing down-stream visual or interpretability script glitches from aborting clinical metric captures.
    - Replaced hardcoded script archive tracking arrays with a dynamic filesystem crawl across the workspace root, preserving architectural state snapshots while adhering strictly to historical dataset folder constraints.
    - Embedded proactive available-VRAM safety diagnostic queries (`torch.cuda.mem_get_info`) to issue runtime structural bottleneck warnings early.
- **Stratified Patient Split Hardening (`utils.py`)**:
    - Generalized the regular expression `FILENAME_PATTERN` to support format-neutral filename suffixes (e.g., matching `.tiff` alongside `.png`), fixing data-discovery blockages during patient-stratified split generation.
- **Configuration Centralization (`config.py`)**:
    - Eliminated implementation magic number leakage by explicitly centralizing `FOCAL_WEIGHT`, `TVERSKY_WEIGHT`, and `ATTENTION_CONTRAST`.
    - Centralized clinical anatomical heuristics parameters: defined `CENTRAL_SLICE_IDX` to enable rapid heuristic tuning without code refactoring.
    - Hardened VRAM manager logic: added defensive boundaries for `nvidia/mit-b5` and implemented safe conservative fallbacks for unknown, custom, or larger models to actively prevent Out-Of-Memory (OOM) failures.

### Fixed
- **TverskyLoss Type Safety**: Added explicit `.long()` cast for targets to prevent runtime crashes during one-hot encoding.
- **mHD95 Inflation**: Fixed a logic error where classes entirely missed by the model were excluded from the mean Hausdorff distance calculation.
- **Dataset Missing Property Bug**: Fixed a major bug where `target_class` passed by `train.py` was ignored by `OCTDataset.__init__`, restoring the capability to train Binary Expert models.
