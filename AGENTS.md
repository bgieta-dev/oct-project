# AGENTS.md

## Purpose
OCT retinal fluid segmentation (IRF / SRF / PED) with SegFormer (MiT) on the RETOUCH dataset. Part of an engineering thesis at Poznan University of Technology (this repo = Maksymilian's SegFormer part). Details: `README.md`, `docs/METHODS.md` (Polish), `docs/EXPERIMENTS.md`, `docs/ROADMAP.md`.

## Layout
- `octseg/` importable package; run modules via `python -m octseg.<module>` (config, dataset, splits, losses, metrics, postprocess, train, train_expert, evaluate, evaluate_hybrid, hybrid_inference, attention_visualizer, pipeline, viz, runs, model).
- `configs/` `base.yaml`, `irf_expert.yaml`, `hybrid.yaml`. Every CLI accepts `--config <yaml>`.
- `splits/` frozen patient train/val/test lists (seed 42).
- `tests/` pytest, synthetic data, CPU.
- `scripts/` `run.sh` / `run.bat` (full pipeline).
- `docs/` METHODS, EXPERIMENTS, ROADMAP, `archive/`, `thesis/` (LaTeX).
- `experiments/` frozen history test2..test18 (old flat-import snapshots) — do not edit.
- Gitignored: `outputs/` (runs, checkpoints), `data_folder/` (dataset), `analysis/` (unrelated code from another project).
- `.omp/agents/` specialist subagent templates for omp (tech-lead-architect, ml-training-engineer, segmentation-eval-specialist, interpretability-and-viz-specialist, thesis-and-docs-specialist, qa-test-engineer, codebase-cleaner).

## Setup / run
All commands from repo root.
```bash
python3.13 -m venv venv && source venv/bin/activate
pip install torch==2.11.0   # platform-specific build
pip install -r requirements.txt -r requirements-dev.txt
python -m octseg.pipeline --name my-run [--epochs N]   # train + eval + attention + snapshot
python -m octseg.train --name train-only
python -m octseg.evaluate --model outputs/runs/<run>/best_model.pth
pytest
```
Data: `data_folder/cropped_images/` + `data_folder/cropped_masks/`, names `{device}_{patient_id}_{slice_id}.png`; masks 0=bg, 1=IRF, 2=SRF, 3=PED. Override with `OCT_DATA_DIR`. `.env.example` has optional `DISCORD_WEBHOOK_URL`.

## Conventions / invariants
- Do not regenerate splits (`python -m octseg.splits`) except deliberately; they are frozen on disk.
- Never commit `*.pth` checkpoints.
- Missing/mismatching checkpoint must raise an error, not evaluate an untrained model.
- Run dirs: `outputs/runs/<timestamp>_<name>/` (config.yaml, git.txt, src/, metrics.yaml, predictions.png, logs, checkpoint).
- Ruff excludes `experiments`, `docs/thesis`, `analysis`, `venv` (see `pyproject.toml`); pytest `testpaths = ["tests"]`.
- Reported results come from the old `experiments/test*/` snapshots on the lab machine, not this refactored package.
- Docs: `CHANGELOG.md` and `ORGANIZATION_PLAN.md` exist at repo root.
