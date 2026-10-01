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
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The IRF-expert ensemble (test 19) has not been run yet, see
[docs/ROADMAP.md](docs/ROADMAP.md). Reported numbers were produced on the Windows lab machine with the code of the
corresponding `experiments/test*/` snapshot, not with this refactored package.

## Layout

```
octseg/            importable package (run modules with `python -m octseg.<module>`)
  config.py          Config dataclass + YAML loading     splits.py        patient splits (frozen files)
  dataset.py         2.5D patient-aware loader           losses.py        Focal / Tversky / Boundary
  postprocess.py     TTA, thresholds, region cleanup     metrics.py       IoU/Dice/HD95/ASD, HD95 policy
  train.py           training loop                       evaluate.py      test-set evaluation
  hybrid_inference.py  base + IRF expert ensemble        evaluate_hybrid.py
  attention_visualizer.py                                viz.py, runs.py, model.py   figures, run dirs, checkpoint loading
  pipeline.py        train -> eval -> attention -> snapshot
  train_expert.py    binary IRF expert
configs/           base.yaml, irf_expert.yaml, hybrid.yaml
splits/            train/val/test patient lists (frozen, seed 42)
tests/             pytest (synthetic data, CPU, seconds)
scripts/           run.sh / run.bat (full pipeline)
docs/              METHODS.md, EXPERIMENTS.md, ROADMAP.md, archive/, thesis/ (LaTeX sources)
experiments/       frozen history test2..test18 (old flat-import snapshots, do not edit)
outputs/           gitignored: new runs, evaluations, checkpoints
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

`nvidia/mit-b*` backbones are downloaded from the Hugging Face hub on first use.

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
python -m octseg.evaluate_hybrid --base-weights <base>.pth --expert-weights <expert>.pth [--blend-strategy linear ...]
python -m octseg.attention_visualizer --model <base>.pth
scripts/run.sh --name my-run            # same as the pipeline, activates venv (scripts\run.bat on Windows)
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
