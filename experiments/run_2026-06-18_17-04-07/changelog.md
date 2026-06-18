# Changelog

All notable changes to the OCT Segmentation project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
- **Evaluation Engine Modernization (`eval.py`)**:
    - Decoupled script from global imports: `evaluate_model` now accepts an explicit configuration object `cfg`.
    - Automated dynamic label handling: eliminated all hardcoded `[1, 2, 3]` arrays, dynamically constructing loops via `range(1, cfg.NUM_LABELS)`.
    - Added Binary Expert support: implemented mask re-mapping logic during visual selection when evaluating binary sub-models.
    - Optimized throughput: increased `num_workers=2` and enabled `pin_memory=True` for accelerated asynchronous disk I/O throughput.
    - Robust plotting layout: dynamically adjust grid generation sizing depending on the count of parsed target classes.
    - Removed Dead Code: Completely stripped out the vestigial `get_retina_mask` function and its dead imports from `train.py`.
- **Configuration Centralization (`config.py`)**:
    - Eliminated implementation magic number leakage by explicitly centralizing `FOCAL_WEIGHT`, `TVERSKY_WEIGHT`, and `ATTENTION_CONTRAST`.
    - Centralized clinical anatomical heuristics parameters: defined `CENTRAL_SLICE_IDX` to enable rapid heuristic tuning without code refactoring.
    - Hardened VRAM manager logic: added defensive boundaries for `nvidia/mit-b5` and implemented safe conservative fallbacks for unknown, custom, or larger models to actively prevent Out-Of-Memory (OOM) failures.

### Fixed
- **TverskyLoss Type Safety**: Added explicit `.long()` cast for targets to prevent runtime crashes during one-hot encoding.
- **mHD95 Inflation**: Fixed a logic error where classes entirely missed by the model were excluded from the mean Hausdorff distance calculation.
- **Dataset Missing Property Bug**: Fixed a major bug where `target_class` passed by `train.py` was ignored by `OCTDataset.__init__`, restoring the capability to train Binary Expert models.
