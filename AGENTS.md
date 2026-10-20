# AGENTS.md

## Purpose
OCT retinal fluid segmentation (IRF / SRF / PED) on the RETOUCH dataset: SegFormer (MiT) 2.5D baseline, plus a pure-transformer RETFound-OCT + LoRA model (`ARCH: retfound`) and a native 3D SwinUNETR (`ARCH: swin_unetr_3d`). Part of an engineering thesis at Poznan University of Technology (this repo = Maksymilian's transformer part). Details: `README.md`, `docs/METHODS.md` (Polish), `docs/EXPERIMENTS.md`, `docs/ROADMAP.md`.

## Layout
- `octseg/` importable package; run modules via `python -m octseg.<module>` (config, dataset, splits, losses, metrics, postprocess, model, train, train_expert, evaluate, evaluate_hybrid, hybrid_inference, sweep_hybrid, benchmark, volume_data, train3d, evaluate3d, attention_visualizer, pipeline, run_series, viz, runs).
- `configs/` `base.yaml`, `irf_expert.yaml`, `hybrid.yaml`, `swin_unetr_3d.yaml`, `retfound_lora.yaml`, `retfound_lora_nodetail.yaml`, `retfound_frozen.yaml`. Every CLI accepts `--config <yaml>`.
- `splits/` frozen patient train/val/test lists (seed 42); list a split's files with `octseg.splits.split_files(cfg, split)`.
- `tests/` pytest, synthetic data, CPU.
- `scripts/` `run.sh` / `run.bat` (pipeline with `--promote`), `run_retfound_ablation.sh` / `.bat` (three RETFound configs via `run_series`), `git_update.sh` / `.bat` and `recover_gpu.ps1` / `.bat` (wrappers of the root scripts).
- Root: `git_update.sh` / `.bat` (add, commit, `pull --rebase`, push), `recover_gpu.ps1` / `.bat` (Windows, admin: recover an NVIDIA GPU after `nvidia-smi` "GPU is lost").
- `docs/` METHODS, EXPERIMENTS, ROADMAP, `archive/`, `thesis/` (LaTeX).
- `experiments/` frozen history — do not edit. Tracked: `test19`, `legacy_root/`; `test2`..`test18` were removed from the working tree in commit `0851b69` and live in commit `4657305`. Index of all runs: `docs/EXPERIMENTS.md`.
- Gitignored: `outputs/` (runs, checkpoints), `data_folder/` (dataset), `analysis/` (unrelated code from another project).
- `.omp/agents/` specialist subagent templates for omp (tech-lead-architect, ml-training-engineer, segmentation-eval-specialist, interpretability-and-viz-specialist, thesis-and-docs-specialist, qa-test-engineer, codebase-cleaner).

## Setup / run
All commands from repo root.
```bash
python3.13 -m venv venv && source venv/bin/activate
pip install torch==2.11.0   # platform-specific build
pip install -r requirements.txt -r requirements-dev.txt
python -m octseg.pipeline --name my-run [--epochs N]   # train + eval + attention + snapshot
python -m octseg.pipeline --promote                   # same, named testN (next free) and copied to experiments/
python -m octseg.train --name train-only
python -m octseg.evaluate --model outputs/runs/<run>/best_model.pth
python -m octseg.sweep_hybrid --base-weights <base> --expert-weights <expert>   # hybrid variant selection on val
python -m octseg.benchmark --config configs/base.yaml --model <ckpt> --device cpu
python -m octseg.run_series --configs a.yaml b.yaml   # consecutive testN experiments
pytest
```
Data: `data_folder/cropped_images/` + `data_folder/cropped_masks/`, names `{device}_{patient_id}_{slice_id}.png`; masks 0=bg, 1=IRF, 2=SRF, 3=PED. Override with `OCT_DATA_DIR`. `.env.example`: optional `DISCORD_WEBHOOK_URL`; `HF_TOKEN` needed to download the gated RETFound weights (`YukunZhou/RETFound_mae_natureOCT`) for training `configs/retfound*.yaml` (evaluation does not need it). SwinUNETR pretrained weights: `outputs/pretrained/model_swinvit.pt` (`SWIN_PRETRAINED`).

## Conventions / invariants
- Do not regenerate splits (`python -m octseg.splits`) except deliberately; they are frozen on disk.
- Never commit `*.pth` checkpoints.
- Missing/mismatching checkpoint must raise an error, not evaluate an untrained model.
- Run dirs: `outputs/runs/<timestamp>_<name>/` (config.yaml, git.txt, src/, metrics.yaml, predictions.png, logs, checkpoint); sweeps `outputs/hybrid_sweep/`, benchmarks `outputs/benchmark/`, series logs `outputs/series/`.
- Experiment numbering: `runs.get_next_experiment_name()` = `test<N+1>`, N = max `testN` over `| testN` rows in the `docs/EXPERIMENTS.md` index and `experiments/testN*` dirs. Never add planned runs as `| testN` index rows. `promote_run_to_experiment` never copies `*.pth` / `*.pt`.
- Model code paths dispatch on `cfg.ARCH` (`segformer` | `retfound` | `swin_unetr_3d`), never on `MODEL_NAME`. Expert, hybrid, sweep and attention maps are SegFormer-only and raise otherwise.
- Ruff excludes `experiments`, `docs/thesis`, `analysis`, `venv` (see `pyproject.toml`); pytest `testpaths = ["tests"]`.
- Reported results come from the old `experiments/test*/` snapshots on the lab machine (test2..test19), not this refactored package.
- Docs: `CHANGELOG.md` and `ORGANIZATION_PLAN.md` exist at repo root.

## Guardrails
- Tooling: Python 3.13 (`.python-version`). `uv` preferred: `uv venv --python 3.13 venv && uv pip install -r requirements.txt -r requirements-dev.txt`.
- Quality gate: `ruff check .` (blocking set in `pyproject.toml` is clean 0 errors). Touched files must also be clean under strict set.
- Local pre-commit: `uv tool run pre-commit run --all-files`.
- Verification: Deterministic playbook at `.omp/skills/verify/SKILL.md` (`skill://verify`).
- Invariants: Never commit `*.pth`/`*.pt` checkpoints; never commit `.env` with real credentials; unit tests run CPU-only on synthetic data; external calls require explicit timeouts.
- Rules reference: `.omp/rules/engineering-guardrails.md`.
