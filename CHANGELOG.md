# Changelog

## [Unreleased]

### Added
- **Autonomous Subagent Templates** (`.omp/agents/ml-research-scout.md`, `.omp/agents/ml-strategy-architect.md`): Domain-specific specialist subagents for deep medical ML research discovery, literature benchmarking, and critical architectural synthesis/stress-testing against homelab hardware boundaries (Tesla P100 @ 125W).
- **Stabilized Adaptive SAM (ASAM)** (`octseg/optim.py`, `octseg/config.py`): Implemented Kwon et al. (ICML 2021) scale-invariant perturbation: $T_w = |w| + \eta$, $e_w = \rho \frac{T_w^2 \odot \nabla_w \mathcal{L}}{\| T_w \odot \nabla_w \mathcal{L} \|_2}$ with numerical floor `SAM_ETA: float = 0.01` and scale $\rho = 0.5$. Prevents over-smoothing of delicate micro-cyst feature extractors in early MiT-B2 blocks.
- **Decoupled Post-Processing** (`octseg/config.py`, `octseg/postprocess.py`, `octseg/evaluate.py`, `octseg/train.py`): Introduced `CLASS_MIN_REGION_SIZES: Optional[Dict[int, int]]` and `IRF_OPEN: bool = True` fields, enabling class-specific area filtering (e.g. 20 px for tiny IRF cysts vs 80 px for large SRF/PED macro-lesions) and disabling destructive $3\times3$ morphological opening that was erasing micro-cysts narrower than 3 px.
- **Experiment Configuration `test21`** (`configs/test21.yaml`): Unified recipe combining ASAM-AdamW ($\rho=0.5, \eta=0.01$), softened Focal loss ($\gamma=2.0$ restoring $2\times$ gradient flow to ambiguous borders), rebalanced Tversky ($\alpha=0.2, \beta=0.8$), decoupled region cleaning (`{1: 20, 2: 80, 3: 80}`, `IRF_OPEN: false`), and saturated 15.1 GB P100 memory profile (`BATCH_SIZE: 10`, `ACCUMULATION_STEPS: 3`).
- **SAM-AdamW Optimizer & Loss Geometry Regularization** (`octseg/optim.py`, `configs/sam_adamw.yaml`): Sharpness-Aware Minimization wrapper around AdamW to enforce flat minima $\min_w \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}(w + \epsilon)$ ($\rho = 0.05$). Fully integrated with AMP GradScaler and multi-step gradient accumulation across 2D SegFormer (`octseg/train.py`) and 3D SwinUNETR (`octseg/train3d.py`). New config parameters `USE_SAM: bool = False`, `SAM_RHO: float = 0.05`, and `SAM_ADAPTIVE: bool = False` in `octseg/config.py` and `configs/base.yaml`. Comprehensive unit test suite added in `tests/test_optim.py`.
- **Remote P100 One-Click Launchers** (`run_remote_p100.sh`, `run_remote_p100.bat`, `scripts/run_remote_p100.*`): Automated SSH dispatchers for Linux and Windows. Connects to `192.168.1.131`, verifies SSH keys, queries running sessions, pulls latest code from GitHub, starts detached `oct_training` tmux sessions with `run_p100.sh`, displays live GPU telemetry, and outputs interactive attach commands.
- **Windows & Cross-Platform Launchers** (`run.bat`, `run.sh`, `scripts/run.bat`, `scripts/run.sh`): Windows batch launcher supporting automatic venv activation, default execution of `configs/test21.yaml`, CLI argument forwarding (`%*`), auto-promotion to `experiments/testN`, and automatic GitHub sync via `git_update.bat` upon completion.
- **P100 Dedicated Launcher & GitHub Auto-Sync** (`scripts/run_p100.sh`): Execution script for Tesla P100 bare-metal node (`192.168.1.131`). Automates temporary VRAM reclamation from inference containers (`qwen-inference-p100`), runs pipeline with auto-promotion, invokes `git_update.sh` on completion to commit and push experiment metrics/predictions/logs to GitHub, and safely restores container service on exit or interrupt.
- **CUDA Device Compatibility Fix** (`tests/test_volume_data.py`): Explicitly moves `build_swin_unetr` test instance to `cfg.DEVICE`, resolving tensor device mismatch on CUDA-enabled hosts.
- **Sequential experiment runner** (`octseg/run_series.py`, `scripts/run_retfound_ablation.{sh,bat}`): runs configs one by one as consecutive `testN` experiments (`pipeline --promote`), preflights configs and RETFound weights before training, continues after a failed run, stops when `nvidia-smi` reports the GPU lost, writes `experiments/series_<first>-<last>.md` summary, optional `--push`.
- **SwinUNETR 3D** (`ARCH: swin_unetr_3d`, `monai==1.6.1`, `configs/swin_unetr_3d.yaml`): native volumetric model on whole OCT volumes. `octseg/volume_data.py` (`VolumeRecord`, `group_volumes` with numeric slice order, `load_volume` -> uint8 `[Z,H,W]` after per-slice `normalize_slice` + resize, MONAI `train_transforms_3d` with `RandCropByLabelClassesd` patches), `octseg/train3d.py` (`train_model_3d`: Focal + Tversky on patches, AdamW + cosine warmup, validation every `VAL_INTERVAL` with `penalty100`), `octseg/evaluate3d.py` (`predict_volume`: Gaussian sliding window, optional lateral-flip TTA, softmax upsampled to `AUG_SIZE`; `evaluate_model_3d` applies the 2D post-processing chain per slice). `build_swin_unetr` / `load_swin_unetr` in `model.py` (gradient checkpointing, optional self-supervised `model_swinvit.pt` via `SWIN_PRETRAINED`, strict checkpoint loading). New `SWIN_*` config fields. `train`, `evaluate` and `pipeline` dispatch on `cfg.ARCH`; the attention phase is skipped for non-SegFormer models.
- **Hybrid `replace` mode** (`octseg/hybrid_inference.py`, `ENSEMBLE_MODES = ("soft", "hard", "replace")`): base SRF/PED kept, base IRF discarded, expert IRF (> expert IRF threshold) written over background only. `HybridInference.segment` split into `probabilities` (both forward passes) and `postprocess` (merge + cleanup, never mutates its inputs); unknown modes raise. `evaluate_hybrid --ensemble-mode` restricted to `ENSEMBLE_MODES`.
- **Hybrid sweep** (`octseg/sweep_hybrid.py`): runs base + expert once per slice and evaluates 14 merge variants (soft linear w ∈ {0.3, 0.4, 0.5, 0.6} × IRF threshold ∈ {0.25, 0.35}; soft `confidence`/`max` at w = 0.5 × both thresholds; `hard`; `replace`) plus the base row on `--split val` (default). `select_best` picks the highest IRF Dice among variants that beat the base IRF Dice and keep SRF/PED Dice within `SRF_PED_TOLERANCE = 0.01`; writes `sweep.yaml` and logs the exact `evaluate_hybrid` test command or "hybrid not adopted".
- **Inference benchmark** (`octseg/benchmark.py`): per-call latency (mean/median/p95 ms, CUDA-synchronised) and peak VRAM; modes `forward`, `pipeline` (TTA + post-processing), `hybrid` (with `--expert-config/--expert-weights`) for 2D models, `volume_forward`/`volume_pipeline` with `ms_per_bscan` for SwinUNETR 3D. `--device cpu|cuda`, `--threads`; writes `outputs/benchmark/<run>/benchmark.yaml`.
- **Experiment auto-numbering** (`octseg/runs.py`): `get_next_experiment_name` returns `test<N+1>`, N = highest `testN` in the `docs/EXPERIMENTS.md` index or among `experiments/testN*` directories; `promote_run_to_experiment` copies a run into `experiments/<name>` without `*.pth`/`*.pt`. `python -m octseg.pipeline --promote` uses the next free name when `--name` is not given and promotes at the end; `scripts/run.sh` / `run.bat` pass `--promote`.
- **Git sync scripts** `git_update.sh` / `git_update.bat` (repo root, wrappers in `scripts/`): `git add .`, commit (message = first argument, default `chore: sync main repo`), `pull --rebase`, push; the rebase is aborted on conflict.
- **Windows GPU recovery** `recover_gpu.ps1` / `recover_gpu.bat` (repo root, wrappers in `scripts/`) for `nvidia-smi` "GPU is lost": self-elevates, kills python processes, restarts the NVIDIA container services, restarts/power-cycles the PnP display device, checks `nvidia-smi`, offers a reboot (`-ForceRebootOnFailure` reboots without asking).

### Changed
- `split_files(cfg, split)` / `SPLIT_NAMES` (`octseg/splits.py`) replace the duplicated split listing in `train`, `evaluate`, `evaluate_hybrid`, `attention_visualizer`; `distribute_files` removed.
- `Config`: `ARCH` (`segformer` | `retfound` | `swin_unetr_3d`, validated), `DEVICE_OVERRIDE`, `WEIGHT_DECAY` (0.05, previously hard-coded; behaviour unchanged).
- `TverskyLoss` works on N-D inputs (`[B,C,H,W]` and `[B,C,D,H,W]`).
- `slice_mean_iou` (`octseg/metrics.py`) shared by 2D/3D failure-case selection; `normalize_slice` is a module-level function in `octseg/dataset.py` (same math).
- `save_predictions_grid` draws 3 columns when no attention map is given.
- Training logs `Peak training VRAM: <MB>` at the end (2D and 3D).
- `train_expert`, `evaluate_hybrid`, `sweep_hybrid`, `attention_visualizer` raise `ValueError` unless `ARCH=segformer`.
- **Workspace declutter & git hygiene**: Removed stray unreferenced root `metrics.png`; updated `.gitignore` with `/metrics.png`, `/predictions.png`, `data_folder/`, and `.pytest_cache` patterns to prevent future clutter from local runs and tooling.

### Docs
- **test20 Promoted (SegFormer-B2 + SAM-AdamW)**: Evaluated on test split (674 slices). Achieved mIoU **0.7035** (beating test19 0.7029), SRF Dice **0.8383** (+1.71%), PED Dice **0.7634** (+0.48%), and superior boundary metrics (**mHD95 73.58 mm** vs 75.31 mm, **mASD 25.42 mm** vs 27.24 mm). Artifacts archived in `experiments/test20/`.
- test19 (SegFormer-B2 base) added to the `docs/EXPERIMENTS.md` index; planned RETFound ablation test21–23; `docs/METHODS.md` sections on RETFound + LoRA, SwinUNETR 3D and the hybrid selection on val; `docs/ROADMAP.md` status of the vacation items.

## [2.1.0] - 2026-10-03

### Added
- **Volumetric Z-axis Consistency Filter** (`octseg/postprocess.py`): `volumetric_consistency_filter(volume_masks, min_slices=2)` using 3D connected component analysis across consecutive B-scans to filter isolated single-slice false positives.
- **Configurable PED Sharpening Factor** (`octseg/config.py`, `octseg/postprocess.py`): `PED_SHARPEN_FACTOR` (default 1.0) allowing fine-tuned blending between original and sharpened probabilities.
- **Calibrated Probability Soft Hybrid Blending** (`octseg/hybrid_inference.py`, `octseg/config.py`): `HYBRID_CONFIDENCE_LOW` (0.15) and `HYBRID_CONFIDENCE_HIGH` (0.45) parameters for confidence-based IRF blending; added probability renormalization of non-IRF classes ($c \neq 1$) to ensure sum of class probabilities equals 1.
- **Automated Failure Cases Extraction** (`octseg/viz.py`, `octseg/evaluate.py`): `save_failure_cases` automatically exports the top-k worst performing slices by IoU to `output_dir/failures/`.
- **Unit Test Suite Expansion**: Added unit tests in `tests/test_hybrid.py`, `tests/test_model.py`, `tests/test_viz.py`, and `tests/test_runs.py`, expanding test coverage from 26 to 46 unit tests.

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
