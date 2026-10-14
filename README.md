# OCT Fluid Segmentation (SegFormer, RETOUCH)

Automated segmentation of retinal fluid in OCT B-scans with SegFormer (MiT) transformers — part of an engineering
thesis at Poznan University of Technology (2026; authors: Maksymilian Naskret, Eryk Naumienko; this repository is
Maksymilian's SegFormer part, the nnUNet baseline is Eryk's).

Classes: **IRF** (intraretinal fluid), **SRF** (subretinal fluid), **PED** (pigment epithelium detachment).

Highlights: patient-stratified split frozen on disk, 2.5D input (slices *t-1, t, t+1*), Focal + Tversky loss,
clinical high-recall thresholds, test-time augmentation, a binary IRF expert with hybrid ensembling, and attention
maps for interpretability. Methods are described in [docs/METHODS.md](docs/METHODS.md) (Polish).

## Results (test split: 7 patients)

| Metric | Test 11-Gold (balanced) | Test 18 (high-recall) |
| :--- | :---: | :---: |
| mIoU | **0.7529** | 0.6990 |
| mDice | **0.8521** | 0.8118 |
| mHD95 (lower is better) | **67.31** | 74.84 |
| mASD (lower is better) | **22.32** | 26.16 |
| IRF / SRF / PED Dice | **0.7483 / 0.8489 / 0.8220** | 0.6800 / 0.8187 / 0.7634 |

Test 18 trades overlap for sensitivity (Tversky 0.1/0.9, IRF threshold 0.30). The full run history is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Test 19 retrained the SegFormer-B2 base (mIoU 0.7029, mDice 0.8150,
IRF / SRF / PED Dice 0.6958 / 0.8222 / 0.7564); its IRF expert and hybrid have not been run yet, see
[docs/ROADMAP.md](docs/ROADMAP.md). Reported numbers were produced on the Windows lab machine with the code of the
corresponding `experiments/test*/` snapshot, not with this refactored package.

## Layout

```
octseg/            importable package (run modules with `python -m octseg.<module>`)
  config.py          Config dataclass + YAML loading     splits.py        patient splits (frozen files)
  dataset.py         2.5D patient-aware loader           losses.py        Focal / Tversky / Boundary
  postprocess.py     TTA, thresholds, region cleanup     metrics.py       IoU/Dice/HD95/ASD, HD95 policy
  train.py           2D training loop                    evaluate.py      test-set evaluation
  model.py           SegFormer / RETFound+LoRA / SwinUNETR builders, strict checkpoint loading
  hybrid_inference.py  base + IRF expert ensemble        evaluate_hybrid.py
  sweep_hybrid.py    hybrid merge-variant selection on val
  volume_data.py     3D volumes + MONAI transforms       train3d.py, evaluate3d.py   SwinUNETR 3D
  benchmark.py       inference latency / peak VRAM
  attention_visualizer.py                                viz.py, runs.py   figures, run dirs, experiment promotion
  pipeline.py        train -> eval -> attention -> snapshot [-> experiments/testN]
  run_series.py      several configs as consecutive testN experiments
  train_expert.py    binary IRF expert
configs/           base.yaml, irf_expert.yaml, hybrid.yaml, swin_unetr_3d.yaml,
                   retfound_lora.yaml, retfound_lora_nodetail.yaml, retfound_frozen.yaml
splits/            train/val/test patient lists (frozen, seed 42)
tests/             pytest (synthetic data, CPU, seconds)
scripts/           run.sh / run.bat (pipeline + auto-promotion), run_retfound_ablation.sh / .bat,
                   git_update.sh / .bat, recover_gpu.ps1 / .bat (wrappers of the root scripts)
git_update.sh/.bat git add + commit + pull --rebase + push
recover_gpu.ps1/.bat  Windows: recover an NVIDIA GPU reported as "GPU is lost" without a reboot (admin)
docs/              METHODS.md, EXPERIMENTS.md, ROADMAP.md, archive/, thesis/ (LaTeX sources)
experiments/       frozen history: test19 + legacy_root/ (test2..test18 in git history, commit 4657305)
outputs/           gitignored: new runs, evaluations, sweeps, benchmarks, checkpoints
data_folder/       gitignored: dataset
analysis/          gitignored: unrelated code from another project (kept for reference)
```

## Setup

```bash
python3.13 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch==2.11.0                                # pick the build for your platform (CUDA / ROCm / CPU)
pip install -r requirements.txt
pip install -r requirements-dev.txt                      # pytest, ruff (optional)
cp .env.example .env                                     # optional: DISCORD_WEBHOOK_URL for run notifications
```

`nvidia/mit-b*` backbones are downloaded from the Hugging Face hub on first use. `ARCH: retfound` configs
(`configs/retfound_*.yaml`) need the gated `YukunZhou/RETFound_mae_natureOCT` weights: accept the access form on
Hugging Face and set `HF_TOKEN` in `.env` (training only; evaluation loads the trained checkpoint without it).

## Data

Place the RETOUCH slices as `data_folder/cropped_images/` and `data_folder/cropped_masks/` (same file names,
`{device}_{patient_id}_{slice_id}.png`, masks: 0 = background, 1 = IRF, 2 = SRF, 3 = PED). Use another location with
`OCT_DATA_DIR=/path/to/data`. The split is read from `splits/*.txt`; regenerate it only deliberately with
`python -m octseg.splits`.

## Usage

All commands run from the repository root. Every CLI accepts `--config <yaml>`; run directories are
`outputs/runs/<timestamp>_<name>/` unless `--output` is given.

```bash
python -m octseg.pipeline --name my-run [--epochs N] [--config configs/base.yaml]   # train + eval + attention + snapshot
python -m octseg.train --name train-only [--epochs N]                                # training only
python -m octseg.train_expert --name irf-expert                                      # binary IRF expert (configs/irf_expert.yaml)
python -m octseg.evaluate --model outputs/runs/<run>/best_model.pth [--output DIR]
python -m octseg.evaluate --config configs/irf_expert.yaml --model <expert>.pth      # expert alone
python -m octseg.evaluate_hybrid --base-weights <base>.pth --expert-weights <expert>.pth [--ensemble-mode soft|hard|replace ...]
python -m octseg.sweep_hybrid --base-weights <base>.pth --expert-weights <expert>.pth [--split val]   # pick the merge variant on val
python -m octseg.benchmark --config configs/base.yaml --model <base>.pth [--device cpu|cuda]          # latency + peak VRAM
python -m octseg.pipeline --config configs/swin_unetr_3d.yaml --name swin3d                            # native 3D SwinUNETR
python -m octseg.attention_visualizer --model <base>.pth
python -m octseg.pipeline --promote [--config ...]                                   # + copy to experiments/testN (next free N)
scripts/run.sh [--config ...]           # pipeline --promote, activates venv (scripts\run.bat on Windows)
scripts/run_retfound_ablation.sh [--push]   # RETFound LoRA / no-detail / frozen, one after another (testN..N+2; .bat on Windows)
python -m octseg.run_series --configs a.yaml b.yaml [--epochs N] [--push]   # any configs sequentially, summary table at the end
pytest                                  # unit tests
```

Each run directory holds `config.yaml`, `git.txt`, `src/`, `metrics.yaml`, `predictions.png`, logs and the
checkpoint. Missing or mismatching checkpoints raise an error instead of silently evaluating an untrained model.

## Checkpoints

`*.pth` files are never committed. No SegFormer checkpoint is stored in this repository; the checkpoints of the
reported runs live on the lab machine (see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).

## Thesis

LaTeX sources are in [docs/thesis/](docs/thesis/) (`latexmk -pdf Maksymilian.tex`; build products are ignored).

## License

No license file is present; all rights reserved by the authors unless they add one.
