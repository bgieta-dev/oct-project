# ORGANIZATION_PLAN — oct-project

Scope: repository layout, hygiene, code quality, and docs for the SegFormer OCT fluid-segmentation thesis project (IRF/SRF/PED, RETOUCH).
Everything below comes from reading the files at commit `ea79f2a` (working tree clean, 117 commits). Claims marked **[INFERENCE]** were not directly observed.

Guiding rule: keep it boring and proportionate to a thesis repo. That means one importable package, YAML configs, a gitignored `outputs/` folder, a few focused tests, and **no** new frameworks (no Hydra, no MLflow/W&B, no DVC).

---

## 1. Current state findings

### 1.1 Entry points and module graph

| Entry point | How it is run | Imports |
|---|---|---|
| `main.py` (full pipeline: train → eval → attention → archive) | `run.sh` / `run.bat` → `python main.py $1` (the `$1`/`%1` arg is ignored: `main()` takes no CLI args, `main.py:90-200`) | `train`, `eval`, `attention_visualizer`, `config` (`main.py:11-14`) |
| `train.py` | `python train.py` (`train.py:323-325`) | `dataset`, `config`, `utils` (`train.py:8,13,14`) |
| `train_expert.py` | `python train_expert.py` (`test19_prep.md:38`) | `train`, `config_irf_expert as config` (`train_expert.py:4-5`) |
| `eval.py` | `python eval.py --model --output` (`eval.py:318-321`) | `dataset`, `config`, `utils` (`eval.py:10,12,14`) |
| `eval_hybrid.py` | `python eval_hybrid.py --base-weights … --blend-strategy …` (`eval_hybrid.py:207-216`) | `hybrid_inference`, `config`, `utils`, `dataset`, `eval.BoundaryPrecisionAnalyzer` (`eval_hybrid.py:10-14`) |
| `hybrid_inference.py` | library; `__main__` is a stub (`hybrid_inference.py:365-378`) | `config`, `config_irf_expert` (`hybrid_inference.py:5-6`) |
| `attention_visualizer.py` | called from `main.py:176`; standalone `__main__` (`:142-145`) | `dataset`, `config`, `utils` (lazy, `:47`) |

All imports are flat and only work when the CWD is the repo root.

### 1.2 Duplication (most to least costly)

1. **`HybridInference.segment` and `segment_with_attention` are copy-pasted.** The ensemble merge, PED sharpening, thresholding and region cleanup appear twice: `hybrid_inference.py:90-192` and `:261-363` (~100 identical lines). Only `segment_with_attention` is called (`eval_hybrid.py:101`).
2. **The TTA loop is written three times:** `eval.py:131-158`, `hybrid_inference.py:40-66` (`_predict_logits`), and `hybrid_inference.py:215-247`.
3. **Post-processing (PED sharpening + reverse-priority thresholds + IRF opening + min-region filter) is written four times:** `eval.py:162-200`, `hybrid_inference.py:117-192`, `:288-363`, plus a reduced variant in train-time validation (`train.py:266-278`).
4. **Metric accumulation and aggregation are duplicated** between `eval.py:101-226,273-291` and `eval_hybrid.py:83-125,176-183`. So is the per-class log block in `main.py:152-164` and `eval_hybrid.py:191-201`.
5. **The `predictions.png` 4-column grid is duplicated** in `eval.py:228-271` and `eval_hybrid.py:132-174`. Both titles say "Stage 2".
6. **"Find the max-pixel slice per class" is duplicated** in `eval.py:60-79`, `eval_hybrid.py:69-81` and `attention_visualizer.py:56-66`.
7. **Test-patient selection is implemented four ways:** `train.py:150-153`, `eval.py:48-55` (prefers `test_patients.txt`), `eval_hybrid.py:45-49` (ignores `test_patients.txt`), `attention_visualizer.py:45-54`. I checked on the current data: `test_patients.txt` equals the seed-42 split (7 patients; 56/7/7 over 6922 slices). Every tracked `experiments/test*/test_patients.txt` has the same hash (`a846b0d6`). The four paths agree today only because sklearn's output is unchanged.
8. **`config.py` vs `config_irf_expert.py`:** the expert file is `from config import *` plus overrides (`config_irf_expert.py:3`). This is inheritance through star import (see bugs below).
9. **`utils.py` vs `_help_data/utils_segmentation.py` are not duplicates.** `_help_data/*` is code from a different project. It imports a package that does not exist here (`from preprocess.segmentation.utils_segmentation import …`, `_help_data/calc_fluid_features.py:24`, `calc_boundary_precision.py:25`, `boundary_precision_analysis.py:7`). It hardcodes Windows/foreign paths (`E:/Newcastle2024/DATA/MP4/CVAT_val_round22/`, `calc_fluid_features.py:34`, `calc_boundary_precision.py:37`; `../../../../DATA/ROCHE/CVAT/`, `boundary_precision_analysis.py:400`), reads Roche clinical Excel files (`calc_fluid_features.py:94`), and needs `seaborn`/`openpyxl`, which are not installed. It is gitignored (`.gitignore:276`) and cannot run in this repo. Separately, `eval.py:22-40 BoundaryPrecisionAnalyzer` is a much simpler intensity-contrast metric that shares only a name with the Anderson-2023 analyzer in `_help_data/boundary_precision_analysis.py:9`.
10. **Experiment snapshots duplicate all code:** `main.py:74-86` copies every root `*.py/*.md/*.txt/*.sh/*.bat` into each run. `experiments/` has 92 tracked `.py` copies, and grep/IDE searches over `*.py` return them alongside live code.

### 1.3 Correctness bugs found while reading (these affect results)

- **B1 — the IRF expert is trained on 2.5D input but run on 2D input.** `train.py:182-185` builds `OCTDataset` without `cfg=cfg`, so the dataset falls back to the global `config` (`dataset.py:28`, `dataset.py:36`). As a result `USE_25D=False` from `config_irf_expert.py:9` is ignored and the expert trains on the base `USE_25D=True` (`config.py:18`). At inference the expert gets the middle slice replicated ×3 (`hybrid_inference.py:83-86`, `:254-257`). `IRF_EXPERT_PLAN.md:8` asks for 2.5D input; `config_irf_expert.py:9` says "Force 2D". The intent conflicts with itself, and the code matches neither consistently. There are no test19 results yet (`experiments/` stops at `test18`), so fix this before running test19.
- **B2 — the expert inherits the B2 batch size.** `BATCH_SIZE`/`ACCUMULATION_STEPS` are computed once from `nvidia/mit-b2` (`config.py:109-111`). `config_irf_expert.py:6` changes `MODEL_NAME` to `mit-b0` without recomputing them.
- **B3 — HD95 policy differs between validation and test.** Train-time validation adds a 100.0 penalty for missed or false classes (`train.py:285-290`, described in `changelog.md:19`). Final evaluation and hybrid evaluation skip those slices (`eval.py:223-225`, `eval_hybrid.py:123-125`). Reported thesis mHD95 is the skip variant.
- **B4 — checkpoint loading can fail silently.** `eval.py:85-88` evaluates the untrained ImageNet model if the file is missing and loads with `strict=False`. `attention_visualizer.py:33-37` catches any load error and silently keeps the base model.
- **B5 — the standalone eval CLI loses its INFO logs.** `eval.py:293-310` configures a named logger `eval_standalone`, but `evaluate_model` logs through the root logger (`eval.py:61,86,113,229`), which has no handler in that mode.
- **B6 — the `hybrid_eval_results/` output path is not gitignored,** and neither is `eval_results_<ts>/` (`eval.py:324`). `update_git.sh` runs `git add .` and would commit them.

### 1.4 Hardcoded / CWD-relative paths

- Data: `config.py:9-11` (`"data_folder"`). Multimodal paths are built by string replace (`dataset.py:101-102`).
- Split file: `eval.py:49`, `attention_visualizer.py:46-51` (`test_patients.txt` in the CWD).
- Checkpoints in the CWD: `main.py:107-110` (deletes and recreates `best_model.pth` at root, then copies it into the run dir at `:125`), `train.py:148`, `eval.py:44,319`, `eval_hybrid.py:19,208-209`, `hybrid_inference.py:371-372`, `config_irf_expert.py:34` + `train_expert.py:17-18` (deletes `irf_expert_best.pth` at root).
- Outputs: `main.py:100` (`experiments/run_<ts>`), `eval.py:324` (`eval_results_<ts>` at root), `eval_hybrid.py:19,210` (`hybrid_eval_results`), `attention_visualizer.py:14` (`attention_results`).
- Device is duplicated: `attention_visualizer.py:16` ignores `config.DEVICE`.

### 1.5 Config handling

- Module-level globals with side effects at import time: `load_dotenv(override=True)` (`config.py:6`), CUDA probe (`:19`), VRAM table (`:89-111`).
- **Dead keys** (defined, never read by live code): `OPTIMIZER_TYPE` (`config.py:24`; AdamW is hardcoded at `train.py:197`), `USE_SOFT_CRF`, `CRF_ITERATIONS` (`config.py:58-59`), `HYBRID_ENSEMBLE_MODE/IRF_THRESHOLD/IRF_MIN_REGION_SIZE` (`config.py:114-116`; `eval_hybrid.py:19,211-215` repeats the values as CLI defaults, and `HybridInference` has a third default `irf_min_region_size=20` at `hybrid_inference.py:10`). `USE_FOCAL_TVERSKY` (`config.py:32`) is only OR-ed with `USE_TVERSKY` (`train.py:231`); no focal exponent is applied to Tversky.
- **Effectively dead:** `USE_MULTIMODAL=True` (`config.py:17`) never takes effect while `USE_25D=True`, because the 2.5D branch wins (`dataset.py:73` vs `:99`). That makes `data_folder/denoised_images` and `edge_map_images` (1.1 GB each) unused in the current setup. `CLASS_WEIGHTS` (`config.py:48`) is ignored while `USE_DYNAMIC_WEIGHTS=True` (`config.py:29`, `train.py:155-163`). The `monai` branch in `attention_visualizer.py:22-24` is a leftover of SwinUNETR (test13/14).
- Usage is mixed: `getattr(cfg, "X", default)` with defaults that differ from `config.py` (e.g. `TVERSKY_ALPHA` 0.3 at `train.py:207` vs 0.1 at `config.py:51`; `MIN_REGION_SIZE` fallback 50 at `hybrid_inference.py:175` vs 80 at `config.py:55`).
- No config is serialized per run; reproducibility relies on the copied `config.py`.

### 1.6 Experiments: how outputs are produced and named

- `main.py:99-101` writes `experiments/run_YYYY-MM-DD_HH-MM-SS/`. The tracked directories are named `test2…test18`, `test11-gold`, so runs were renamed by hand after the fact. Each log's first line keeps the original name and shows runs happened on Windows (`C:\Users\Student\Documents\python\oct-project\experiments\run_2026-06-08_10-31-03`, `experiments/test18/experiment.log:1`).
- Metrics exist only as log lines (`main.py:147-164`); there is no machine-readable metrics file. Observed index (from `experiment.log`):

  | dir | original run | backbone | final mIoU / mDice |
  |---|---|---|---|
  | test2 | run_2026-05-17_09-35-47 | mit-b2 | 0.7125 / 0.8220 |
  | test3 | run_2026-05-18_11-08-54 | mit-b2 | 0.7149 / 0.8247 |
  | test4 | run_2026-05-19_09-35-43 | mit-b2 | 0.6879 / 0.8047 |
  | test5 | run_2026-05-19_22-53-23 | mit-b2 | 0.7322 / 0.8378 |
  | test6 | run_2026-05-20_09-10-48 | mit-b3 | 0.6897 / 0.8047 |
  | test7 | run_2026-05-20_19-17-35 | mit-b3 | 0.6850 / 0.7987 |
  | test8 | run_2026-05-21_09-00-18 | mit-b3 | 0.7080 / 0.8189 |
  | test9 | run_2026-05-22_09-53-35 | mit-b2 | 0.7409 / 0.8434 |
  | test10 | run_2026-05-26_08-11-36 | mit-b3 | 0.6844 / 0.8014 |
  | test11-gold | run_2026-05-29_13-59-49 | mit-b2 | 0.7529 / 0.8521 |
  | test12 | run_2026-06-01_19-28-20 | — (only log + metrics.png) | — |
  | test13 | run_2026-06-02_02-38-14 | mit-b2 | 0.7186 / 0.8273 |
  | test14 | run_2026-06-02_22-51-11 | mit-b2 | 0.7405 / 0.8431 |
  | test15 | run_2026-06-03_09-48-00 | mit-b2 | 0.7013 / 0.8147 |
  | test16 | run_2026-06-06_07-58-47 | mit-b2 | 0.7037 / 0.8169 (first "Final" line) |
  | test17 | run_2026-06-06_21-26-47 | mit-b3 | — (no "Final mIoU" line) |
  | test18 | run_2026-06-08_10-31-03 | mit-b2 | 0.6990 / 0.8118 |

- **No SegFormer checkpoints exist anywhere on disk.** `find -name '*.pth'` (excluding venv) finds only 9 CNN checkpoints under `dokumentacja/CNN/`. test11-gold and test18 cannot be re-evaluated locally, and test19's hybrid eval needs a freshly trained base model.
- Root `predictions.png` and `failures/*.png` (10 files) are tracked leftovers from older eval versions. The current `eval.py` has no failure-case code, but `plan.md:20` still lists the feature as done.

### 1.7 Repo hygiene

- **Tracked:** 29 root files + 484 under `experiments/` (326 png, 92 py, 31 md, 20 log, 15 txt) + 10 under `failures/` + `.codegraph/.gitignore`. `.git` is 203 MB (pack 101 MiB). The largest blobs are ~10 MB `predictions.png` files (test11-gold/16/17/18), produced by `dpi=200` (`eval.py:270`).
- **Correctly ignored and never committed:** `venv/` (19 GB, `.gitignore:125`), `.env` (`:122`; `git log --all -- .env` is empty), `data_folder` (7.1 GB, `:274`), `*.pth` (`:169`).
- **`.env` exists** with one key, `DISCORD_WEBHOOK_URL` (value not read). There is no `.env.example`.
- **Ignored but should be versioned:** `/dokumentacja` (`.gitignore:278`). The whole LaTeX thesis (`Maksymilian.tex`, `bibliografia.bib`, figures) has **no version control**. The folder also holds LaTeX build byproducts (`.aux .bbl .blg .fdb_latexmk .fls .lof .log .lot .out .synctex.gz .toc`), a 21 MB `OCT.pdf`, and 862 MB of CNN runs (798 MB of which are `best_model.pth`).
- **`.gitignore` traps for any redesign:** `*.json` and `*.csv` are ignored globally (`.gitignore:185-186`), and `outputs/`/`conf/` too (Hydra block, `:212,215`). A `configs/*.json` or `metrics.json` would be silently untracked, so use YAML or `.txt`. The `outputs/` rule is reused intentionally below.
- **`requirements.txt` (14 lines) is unpinned and incomplete.** `torch` is missing, and so is `requests` (used at `main.py:6`). `monai`, `einops` and `SimpleITK` are listed but not imported by any live root module (monai appears only in snapshots `experiments/test13|14/eval.py:88`). Installed versions in `venv/lib/python3.13/site-packages`: torch 2.11.0+rocm7.2, transformers 5.7.0, albumentations 2.0.8, numpy 2.4.3, scikit-learn 1.6.1, scipy 1.17.1, medpy 0.5.2, opencv-python-headless 4.13.0.92, pillow 12.1.1, matplotlib 3.10.9, tqdm 4.67.3, python-dotenv 1.2.2, requests 2.33.1, pyyaml 6.0.3, monai 1.5.2, einops 0.8.2, simpleitk 2.5.5.
- **`venv/` is broken on this host.** `venv/pyvenv.cfg` says 3.13.12, but `venv/bin/python` resolves to `/usr/bin/python3.14`, so `venv/bin/python -c "import torch"` fails. Packages are reachable only via `PYTHONPATH=venv/lib/python3.13/site-packages /usr/bin/python3.13`.
- **No seeding anywhere** in live code (no `manual_seed`, `np.random.seed`, `random.seed`). `utils.get_stratified_splits(seed=42)` is the only seeded step (`utils.py:9`).
- Unused imports: `json` and `sys` in `main.py:7-8`.
- **`update_git.sh` runs `git add . && git commit -m "$1" && git push`.** Combined with B6 it commits arbitrary outputs.

### 1.8 Docs drift (stated vs code)

| Claim | Where | Code reality |
|---|---|---|
| Soft-CRF bilateral refinement | `README.md:18`, `improvement_plan.md:15` | `USE_SOFT_CRF=False` and no CRF code (`config.py:58`) |
| Retina zone masking | `summary.md:29-31`, `README.md:17` | removed (`changelog.md:42`) |
| LR 6e-5, polynomial decay | `summary.md:13-14` | cosine with warmup (`train.py:198-202`), `LR=5e-5` (`config.py:22`) |
| TTA scales incl. 1.5 | `summary.md:35` | `[0.8, 1.0, 1.2]` (`config.py:57`) |
| Boundary Loss in use | `Dokumentacja.md:34` | `USE_BOUNDARY_LOSS=False` (`config.py:39`) |
| Expert Tversky α=0.05/β=0.95, 2.5D input | `IRF_EXPERT_PLAN.md:8,22`, `test19_prep.md:15-16` | α=0.3/β=0.7, `USE_25D=False` (`config_irf_expert.py:9,22-23`) |
| Failure-case extraction | `plan.md:20` | not in current `eval.py` (existed in `experiments/test10/eval.py:269-281`) |
| Project structure list | `README.md:94-99` | missing the hybrid/expert files |
| "Stage 0/6/11" attention figures | `dokumentacja/Maksymilian.tex:470-530` | `stage{idx}` is the **block index** across all encoder blocks (`attention_visualizer.py:108,125`), not a SegFormer stage |

---

## 2. Target directory layout

```
oct-project/
├── octseg/                      # the only importable code (flat package, no src/)
│   ├── __init__.py
│   ├── config.py                # Config dataclass + load_config(yaml) + paths anchored to repo root
│   ├── splits.py                # ← utils.py (get_stratified_splits) + read/write split files
│   ├── dataset.py               # ← dataset.py
│   ├── losses.py                # ← train.py:23-101 (BoundaryLoss, FocalLoss, TverskyLoss)
│   ├── metrics.py               # confusion-matrix IoU/Dice, HD95/ASD policy, region counts, boundary contrast (← eval.py:22-40)
│   ├── postprocess.py           # predict_logits (TTA), PED sharpening, thresholds, min-region cleanup
│   ├── train.py                 # ← train.py (loop only)
│   ├── train_expert.py          # ← train_expert.py
│   ├── evaluate.py              # ← eval.py
│   ├── hybrid_inference.py      # ← hybrid_inference.py (single segment(); attention optional)
│   ├── evaluate_hybrid.py       # ← eval_hybrid.py
│   ├── attention_visualizer.py  # ← attention_visualizer.py (name kept: cited in Maksymilian.tex:305)
│   ├── viz.py                   # predictions grid + metrics curves (← eval.py:228-271, train.py:309-321)
│   ├── runs.py                  # run dir creation, logging setup, source/config snapshot, Discord notify (← main.py:18-86)
│   └── pipeline.py              # ← main.py
├── configs/
│   ├── base.yaml                # = current config.py values (test18 lineage)
│   ├── irf_expert.yaml          # = config_irf_expert.py overrides
│   └── hybrid.yaml              # = HYBRID_* + eval_hybrid CLI defaults
├── splits/
│   ├── train_patients.txt       # frozen from seed-42 split
│   ├── val_patients.txt
│   └── test_patients.txt        # ← test_patients.txt
├── tests/                       # pytest, synthetic arrays only, no GPU/data
├── scripts/
│   ├── run.sh                   # ← run.sh
│   └── run.bat                  # ← run.bat
├── analysis/
│   └── external_roche/          # ← _help_data/ (stays gitignored; foreign project code)
├── docs/
│   ├── METHODS.md               # ← Dokumentacja.md + summary.md (corrected)
│   ├── EXPERIMENTS.md           # ← README "Experimental History" + index table §1.6
│   ├── ROADMAP.md               # ← open items of plan.md, improvement_plan.md, IRF_EXPERT_PLAN.md, test19_prep.md
│   ├── archive/                 # original plan files + PODSUMOWANIE_*.odt/.pdf, unchanged
│   └── thesis/                  # ← dokumentacja/ (sources tracked; build products, OCT.pdf, CNN/ ignored)
├── experiments/                 # frozen tracked history test2…test18 (+ legacy_root/); new runs land here only by explicit promotion
├── outputs/                     # gitignored: runs/, eval/, hybrid_eval/, checkpoints
├── data_folder/                 # gitignored, unchanged name (Windows lab machine + run history depend on it)
├── README.md  CHANGELOG.md  ORGANIZATION_PLAN.md
├── requirements.txt             # pinned
├── requirements-dev.txt         # pytest, ruff
├── pyproject.toml               # tool config only (ruff/pytest excludes); no packaging needed
├── .env.example
└── .gitignore
```

Why each choice:

- **Flat `octseg/` package instead of `src/`.** `python -m octseg.pipeline` works from the repo root with no `pip install -e .`. That matters on the Windows lab machine, which today just activates a venv (`run.bat:2-3`). A `src/` layout would add an install step for no benefit in a non-distributed repo.
- **Module filenames are mostly kept** (`dataset.py`, `train.py`, `hybrid_inference.py`, `attention_visualizer.py`), so docs and thesis prose (`Maksymilian.tex:305`) stay accurate. Two renames: `eval.py → evaluate.py` (avoid the `octseg.eval` vs builtin `eval` confusion) and `main.py → pipeline.py` (describes what it does). `utils.py → splits.py` because splitting is all it contains (`utils.py:9-63`).
- **New `losses.py` / `metrics.py` / `postprocess.py`.** These are the only new modules, and each exists to remove a duplication listed in §1.2. They are also the units worth testing.
- **`configs/*.yaml`** because `*.json` is gitignored (`.gitignore:186`) and `conf/` is ignored (`:215`). PyYAML 6.0.3 is already installed. A dataclass gives one source of defaults, replacing the scattered `getattr` defaults (§1.5).
- **`splits/` as text files** keeps the current `test_patients.txt` format (`eval.py:50`) and makes splits independent of the sklearn version.
- **`outputs/` already matches the ignore rule** (`.gitignore:212`). Runs and checkpoints stop polluting the root and `experiments/`.
- **`experiments/` stays tracked and frozen.** `README.md` and the thesis discussion refer to "Test 11-Gold"/"Test 18" by name. Its code snapshots are historical evidence; don't refactor them.
- **`docs/thesis/` versioned without build products.** Today the thesis has zero history. The sources are tiny (`Maksymilian.tex` 44 KB, `bibliografia.bib` 4 KB, figures ~37 MB incl. `test18/`, `test11/`, `test2/`).
- **`analysis/external_roche/` stays ignored.** The code belongs to another project (§1.2 item 9). It is kept for reference, not integrated.
- **Not proposed:** notebooks (none exist), Hydra, MLflow/W&B (W&B is mentioned only in `PODSUMOWANIE_…pdf` p.1 and was never wired in), DVC.

---

## 3. Migration steps (structure only, behavior-preserving)

Each step is one commit. Steps 0–7 change **no numerics**; behavioral fixes come in §4. Commands assume the repo root as CWD and Linux. On the Windows lab machine, use the same `git mv` lines in Git Bash.

### Step 0 — Safety net and a working interpreter

- [ ] `git tag pre-reorg`
- [ ] Repoint the broken venv to 3.13 **[INFERENCE: `venv --upgrade` relinks `bin/python` to the given interpreter without reinstalling packages]**:
  `/usr/bin/python3.13 -m venv --upgrade venv`
- [ ] Capture a parity snapshot with the **old** layout. Save as `/tmp/parity.py` (throwaway, not committed). It needs network once for `nvidia/mit-b2`, because no HF cache exists on this host:
  ```python
  import sys, hashlib, numpy as np, torch
  sys.path.insert(0, ".")
  try:
      from octseg import config, splits as U, dataset as D
  except ImportError:
      import config, utils as U, dataset as D
  try:
      from octseg import losses as L          # after Step 4
  except ImportError:
      try:
          from octseg import train as L       # after Step 3
      except ImportError:
          import train as L                   # before Step 3
  import os, albumentations as A
  from transformers import SegformerImageProcessor
  files = sorted(os.listdir(config.IMG_DIR))
  tr, va, te = U.get_stratified_splits(files)
  print("split", hashlib.md5(",".join(sorted(tr)+sorted(va)+sorted(te)).encode()).hexdigest())
  p = SegformerImageProcessor.from_pretrained(config.MODEL_NAME)
  ti = [os.path.join(config.IMG_DIR, f) for f in files if "_".join(f.split("_")[:2]) in te][:3]
  tm = [x.replace(config.IMG_DIR, config.MASK_DIR) for x in ti]
  ds = D.OCTDataset(ti, tm, p, transform=A.Compose([A.Resize(*config.AUG_SIZE)]), cfg=config)
  print("item0", float(ds[0]["pixel_values"].double().sum()), int(ds[0]["labels"].sum()))
  g = torch.Generator().manual_seed(0); x = torch.randn(2, 4, 32, 32, generator=g); y = torch.randint(0, 4, (2, 32, 32), generator=g)
  print("focal", float(L.FocalLoss(alpha=torch.ones(4), gamma=3.0)(x, y)), "tversky", float(L.TverskyLoss(0.1, 0.9, 4)(x, y)))
  ```
  Run: `venv/bin/python /tmp/parity.py | tee /tmp/parity_before.txt`

### Step 1 — Repo hygiene (no code touched)

- [ ] Move the tracked root leftovers into frozen history:
  `mkdir -p experiments/legacy_root && git mv predictions.png experiments/legacy_root/ && git mv failures experiments/legacy_root/failures`
- [ ] `git rm update_git.sh`. Its `git add .` is the only path by which outputs get committed (B6).
- [ ] Create `.env.example` with one line: `DISCORD_WEBHOOK_URL=`
- [ ] Append to `.gitignore`:
  ```
  # project outputs
  /outputs/
  /eval_results_*/
  /hybrid_eval_results/
  /attention_results/
  .codegraph/
  # thesis build products (after Step 5)
  docs/thesis/*.aux
  docs/thesis/*.bbl
  docs/thesis/*.blg
  docs/thesis/*.fdb_latexmk
  docs/thesis/*.fls
  docs/thesis/*.lof
  docs/thesis/*.lot
  docs/thesis/*.log
  docs/thesis/*.out
  docs/thesis/*.toc
  docs/thesis/*.synctex.gz
  docs/thesis/OCT.pdf
  docs/thesis/CNN/
  docs/thesis/TRANSFORMERS/
  /analysis/external_roche/
  ```
  Leave `/dokumentacja` and `_help_data` in place until Steps 5–6 remove those folders.
- [ ] `git rm --cached .codegraph/.gitignore`
- **Verify:** `git status --short` shows only the moves/deletions above; `git check-ignore -v outputs/a eval_results_x/a hybrid_eval_results/a` prints three matches.

### Step 2 — Pin dependencies

- [ ] Rewrite `requirements.txt` with the versions observed in `venv/lib/python3.13/site-packages` (§1.7), adding `torch==2.11.0` and `requests==2.33.1`, and `pyyaml==6.0.3` once Step 8 lands. Drop `monai`, `einops`, `SimpleITK` from the direct list. Before dropping `SimpleITK`, check whether `medpy` requires it **[INFERENCE]** (`pip show medpy` → Requires). Add a comment that torch must be installed per platform (this host: `+rocm7.2`; the Windows lab machine is presumably CUDA, given `C:\Users\Student\…` in logs).
- [ ] On the Windows lab machine, run `pip freeze > requirements-lab-freeze.txt` and commit it. That environment produced all reported numbers, and its versions are unknown here.
- [ ] Create `requirements-dev.txt`: `pytest`, `ruff`.
- **Verify:** `venv/bin/python -m pip install --dry-run -r requirements.txt` resolves; `venv/bin/python -m pip check`.

### Step 3 — Create the package (moves + import rewrites only)

- [ ] Moves:
  ```
  mkdir -p octseg scripts splits && touch octseg/__init__.py
  git mv config.py               octseg/config.py
  git mv config_irf_expert.py    octseg/config_irf_expert.py   # replaced by configs/irf_expert.yaml in Step 8
  git mv dataset.py              octseg/dataset.py
  git mv utils.py                octseg/splits.py
  git mv train.py                octseg/train.py
  git mv train_expert.py         octseg/train_expert.py
  git mv eval.py                 octseg/evaluate.py
  git mv eval_hybrid.py          octseg/evaluate_hybrid.py
  git mv hybrid_inference.py     octseg/hybrid_inference.py
  git mv attention_visualizer.py octseg/attention_visualizer.py
  git mv main.py                 octseg/pipeline.py
  git mv run.sh                  scripts/run.sh
  git mv run.bat                 scripts/run.bat
  git mv test_patients.txt       splits/test_patients.txt
  ```
- [ ] Import rewrites (exact lines):

  | File | Old | New |
  |---|---|---|
  | `octseg/config_irf_expert.py:3` | `from config import *` | `from octseg.config import *` |
  | `octseg/dataset.py:8` | `import config as global_config` | `from octseg import config as global_config` |
  | `octseg/train.py:8,13,14` | `from dataset import OCTDataset` / `import config` / `from utils import get_stratified_splits` | `from octseg.dataset import OCTDataset` / `from octseg import config` / `from octseg.splits import get_stratified_splits` |
  | `octseg/train_expert.py:4-5` | `from train import train_model` / `import config_irf_expert as config` | `from octseg.train import train_model` / `from octseg import config_irf_expert as config` |
  | `octseg/evaluate.py:10,12,14` | `from dataset …` / `import config as global_config` / `from utils …` | `from octseg.dataset …` / `from octseg import config as global_config` / `from octseg.splits …` |
  | `octseg/hybrid_inference.py:5-6` | `import config` / `import config_irf_expert as expert_config` | `from octseg import config` / `from octseg import config_irf_expert as expert_config` |
  | `octseg/evaluate_hybrid.py:10-14` | `from hybrid_inference …` / `import config` / `from utils …` / `from dataset …` / `from eval import BoundaryPrecisionAnalyzer` | `from octseg.hybrid_inference …` / `from octseg import config` / `from octseg.splits …` / `from octseg.dataset …` / `from octseg.evaluate import BoundaryPrecisionAnalyzer` |
  | `octseg/attention_visualizer.py:8-9,47` | `from dataset …` / `import config` / `from utils …` | `from octseg.dataset …` / `from octseg import config` / `from octseg.splits …` |
  | `octseg/pipeline.py:11-14` | `from train …` / `from eval import evaluate_model` / `from attention_visualizer …` / `import config` | `from octseg.train …` / `from octseg.evaluate import evaluate_model` / `from octseg.attention_visualizer …` / `from octseg import config` |

- [ ] Anchor paths to the repo root in `octseg/config.py` so they no longer depend on the CWD:
  `ROOT = Path(__file__).resolve().parents[1]`; `DATA_DIR = Path(os.getenv("OCT_DATA_DIR", ROOT / "data_folder"))`; `IMG_DIR`/`MASK_DIR` as `str(DATA_DIR / …)`, kept as strings because `dataset.py:101-102` does `str.replace`; `SPLIT_DIR = ROOT / "splits"`; `load_dotenv(ROOT / ".env", override=True)`.
- [ ] Replace the CWD `"test_patients.txt"` reads (`evaluate.py:49-50`, `attention_visualizer.py:46-51`) with `config.SPLIT_DIR / "test_patients.txt"`.
- [ ] `octseg/pipeline.py:79-82`: the archive scan of `"."` would now copy nothing useful. Change it to `shutil.copytree(ROOT/"octseg", exp_dir/"src", ignore=shutil.ignore_patterns("__pycache__"))`, and copy `requirements.txt` + `splits/`. Also write `git rev-parse HEAD` and `git status --porcelain` to `exp_dir/git.txt` when git is available.
- [ ] `scripts/run.sh`: `cd "$(dirname "$0")/.." && source venv/bin/activate && python -m octseg.pipeline "$@"`. `scripts/run.bat`: `cd /d %~dp0..` then `call venv\Scripts\activate` and `python -m octseg.pipeline %*`.
- [ ] Add `pyproject.toml` with tool config only: `[tool.ruff] extend-exclude = ["experiments", "docs/thesis", "analysis", "venv"]`; `[tool.pytest.ini_options] testpaths = ["tests"]`.
- **Verify:**
  - `venv/bin/python -m compileall -q octseg`
  - `venv/bin/python -c "import octseg.pipeline, octseg.train_expert, octseg.evaluate_hybrid, octseg.attention_visualizer"`
  - `venv/bin/python -m octseg.evaluate --help && venv/bin/python -m octseg.evaluate_hybrid --help`
  - Grep for leftovers (expect no hits): `grep -rnE '^(import|from) (config|config_irf_expert|dataset|utils|train|eval|hybrid_inference|attention_visualizer)\b' octseg`
  - `venv/bin/python /tmp/parity.py | diff /tmp/parity_before.txt -` → no diff. The same script works before Step 3, after Step 3, and after Step 4 thanks to the import fallbacks.

### Step 4 — Extract the shared modules (pure moves, no logic change)

- [ ] `octseg/losses.py` ← `train.py:23-101` (three loss classes). In `train.py`, `from octseg.losses import BoundaryLoss, FocalLoss, TverskyLoss`; move `distance_transform_edt` (`train.py:16`) with them.
- [ ] `octseg/metrics.py` ← `BoundaryPrecisionAnalyzer` (`evaluate.py:22-40`), plus a `SegmentationMetrics` accumulator built from `evaluate.py:101-109,210-226,273-291`. `evaluate.py` and `evaluate_hybrid.py` both use it, which removes `evaluate_hybrid.py:83-125,176-183`. `evaluate_hybrid.py:14` then imports from `octseg.metrics`.
- [ ] `octseg/postprocess.py` ← `predict_logits(model, pixel_values, target_size, cfg)` (TTA, from `evaluate.py:131-158` / `hybrid_inference.py:40-69`), `sharpen_ped(probs)`, `apply_thresholds(probs, thresholds, irf_override)`, `clean_regions(mask, min_sizes, irf_open)`. `evaluate.py:160-200`, `hybrid_inference.py` and `train.py:266-278` call these. Train-time validation calls `clean_regions` with `irf_open=False` to preserve today's behavior (no opening at `train.py:270-276`).
- [ ] `hybrid_inference.py`: delete `segment` (`:71-192`, unused). Rename `segment_with_attention` to `segment(image_np, return_attention=False)` and rebuild it on `postprocess.*`. Update `evaluate_hybrid.py:101`.
- [ ] `octseg/viz.py` ← `save_predictions_grid(vis_data, cfg, path)` from `evaluate.py:228-271` (delete the copy at `evaluate_hybrid.py:132-174`), `select_vis_indices(mask_paths, classes, target_class)` from `evaluate.py:60-79` (delete `evaluate_hybrid.py:69-81` and `attention_visualizer.py:56-66`), and `plot_history` from `train.py:309-321`.
- [ ] `octseg/runs.py` ← `send_discord_notification`, `setup_logging`, `check_vram_diagnostics`, `archive_experiment_sources` (`pipeline.py:18-86`).
- [ ] `octseg/splits.py`: add `load_splits(split_dir) -> (train, val, test)` reading the three txt files, and route `train.py:151`, `evaluate.py:48-52`, `evaluate_hybrid.py:46`, `attention_visualizer.py:45-51` through it. Generate `splits/train_patients.txt` and `val_patients.txt` once, using `get_stratified_splits` on the current data (the test file already matches, §1.2 item 7).
- **Verify:** parity script gives no diff (it now picks up `octseg.losses`). Add a temporary parity case for `postprocess`: random `probs` with seed 0, comparing the old `evaluate.py` inline block (copy from tag `pre-reorg`) with `apply_thresholds`+`clean_regions`, using `np.array_equal`. Then `venv/bin/python -m pytest` (tests from §4 M3, if already written).

### Step 5 — Docs and thesis moves

- [ ] `mkdir -p docs/archive/reports`
- [ ] `git mv plan.md improvement_plan.md IRF_EXPERT_PLAN.md test19_prep.md summary.md Dokumentacja.md docs/archive/`
- [ ] `git mv PODSUMOWANIE_PRAC_PO_2_TYGODNIACH_TESTOW.odt PODSUMOWANIE_PRAC_PO_2_TYGODNIACH_TESTOW.pdf docs/archive/reports/`
- [ ] `git mv changelog.md CHANGELOG.md`
- [ ] Write `docs/METHODS.md`, `docs/EXPERIMENTS.md`, `docs/ROADMAP.md` and the new `README.md` per §5.
- [ ] Thesis (untracked today, so use plain `mv`): `mv dokumentacja docs/thesis`. Delete the `/dokumentacja` line from `.gitignore` (`:278`). Then `git add docs/thesis` (the Step 1 rules keep build products, `OCT.pdf` and `CNN/` out).
- **Verify:** `cd docs/thesis && latexmk -pdf -interaction=nonstopmode Maksymilian.tex` (latexmk and pdflatex exist at `/usr/bin`). All `\includegraphics` paths are relative to the thesis folder (`test2/metrics.png`, `test18/attention_maps/…`, `*.png`), so they resolve unchanged. Then `git status --short docs/thesis | grep -E '\.(aux|log|pth|synctex\.gz)$|OCT\.pdf'` should print nothing.

### Step 6 — Foreign helper code

- [ ] `mkdir -p analysis && mv _help_data analysis/external_roche`. It is untracked, so plain `mv`. Remove the `_help_data` line from `.gitignore` (`:276`); `/analysis/external_roche/` from Step 1 covers it.
- **Verify:** `git status --short` shows no `analysis/` entries.

### Step 7 — Outputs relocation and CLIs

- [ ] `pipeline.py:99-101`: `exp_dir = ROOT/"outputs"/"runs"/f"{ts}_{args.name}"`, with argparse `--name` (default `run`), `--config`, `--epochs`. Remove the root `best_model.pth` dance (`pipeline.py:107-110,122-129`) and pass `save_path=exp_dir/"best_model.pth"` directly.
- [ ] `train_expert.py`: same run-dir handling. `EXPERT_SAVE_PATH` is replaced by `<run_dir>/best_model.pth`, so `train_expert.py:17-19` no longer deletes a root file.
- [ ] `evaluate.py:324` default → `ROOT/"outputs"/"eval"/ts`; `evaluate_hybrid.py:210` default → `ROOT/"outputs"/"hybrid_eval"/ts`; `attention_visualizer.py:14` default → `ROOT/"outputs"/"attention"`.
- [ ] Promotion to history (documented in `docs/EXPERIMENTS.md`, no script): `rsync -a --exclude '*.pth' outputs/runs/<run>/ experiments/test19-<slug>/`, then add a row to the index table.
- **Verify:** `venv/bin/python -m octseg.pipeline --help`. To test the output paths without paying for training, run `venv/bin/python -m octseg.evaluate --model <any.pth> --output /tmp/evalcheck` with a checkpoint from the lab machine. The only full-pipeline smoke is `scripts/run.sh --name smoke --epochs 1`, which costs one full epoch over 56 patients. Afterwards `git status --short` must show nothing under `outputs/`.

---

## 4. Code quality improvements (after migration; each its own commit)

Effort: **S** ≤ 2 h, **M** ≈ half a day, **L** ≈ 1–2 days.

### High

| ID | Item | Evidence | Change | Effort |
|---|---|---|---|---|
| H1 | Fix expert config propagation (B1, B2) | `train.py:182-185`, `dataset.py:28,36`, `config.py:109-111`, `config_irf_expert.py:6,9` | Pass `cfg=cfg` to both `OCTDataset(...)` calls. Compute batch/accum from `cfg.MODEL_NAME` when the config is built, not at import. Decide 2D vs 2.5D for the expert once (record it in `ROADMAP.md`) and make `hybrid_inference.py:83-86` feed the same representation. Must land **before test19 runs**. | S |
| H2 | One post-processing/TTA path | §1.2 items 1–3 | Done structurally in Step 4. Then delete the getattr defaults that disagree with config (`hybrid_inference.py:175-177` vs `config.py:55`). | M |
| H3 | One metrics definition and a metrics file | §1.2 item 4, B3 | Make `metrics.py` own the HD95 policy explicitly: `hd95_missing="skip"` (default; reproduces the thesis numbers) or `"penalty100"`. Use it in validation, eval and hybrid eval. Write `metrics.yaml` (YAML, because `*.json` is ignored) into every run/eval dir. | M |
| H4 | Reproducibility | no seeding (§1.7); unpinned deps; split recomputed per call | Add `seed_everything(cfg.SEED)` (python/numpy/torch + `DataLoader(generator=…, worker_init_fn=…)` at `train.py:187-188`), frozen `splits/*.txt` (Step 4), pinned `requirements.txt` (Step 2), plus `config.yaml` and `git.txt` snapshots per run (Steps 3/8). Do not force `cudnn.deterministic` (AMP on a 12 GB GPU; a slower epoch is not worth it for a thesis). | S |
| H5 | Fail loudly on checkpoint problems (B4) | `eval.py:85-88`, `attention_visualizer.py:33-37`, `hybrid_inference.py:23,30` | Raise when the file is missing. Use `load_state_dict(..., strict=True)` and `torch.load(..., weights_only=True)` everywhere. | S |

### Medium

| ID | Item | Evidence | Change | Effort |
|---|---|---|---|---|
| M1 (= Step 8) | Config system | §1.5 | `octseg/config.py`: `@dataclass Config` holding today's values as defaults; `load_config(path)` merges `configs/base.yaml` ← `configs/<variant>.yaml` (`irf_expert.yaml` sets `base: base.yaml`). Delete `config_irf_expert.py` and the star import. Remove dead keys (`OPTIMIZER_TYPE`, `USE_SOFT_CRF`, `CRF_ITERATIONS`, `USE_FOCAL_TVERSKY`) and make `HYBRID_*` the only defaults for `evaluate_hybrid` argparse. Replace `getattr(cfg, …, default)` with attribute access. `DEVICE` and the Discord URL are resolved at runtime, not stored in YAML. | M |
| M2 | Uniform CLIs | `main.py` ignores args (`run.sh` passes `$1`); `eval.py:336` always uses the base config | Every `python -m octseg.<x>` takes `--config` and `--output`. `pipeline`/`train_expert` also take `--name`/`--epochs`. `evaluate` takes `--config configs/irf_expert.yaml` so it can evaluate the expert alone. | S |
| M3 | Tests (`tests/`, synthetic, CPU, < 10 s) | none exist | `test_splits.py`: patient-disjoint, every device in every split, deterministic, equals `splits/*.txt` (skips if `data_folder` is absent). `test_metrics.py`: IoU/Dice on hand-built masks incl. empty-class cases; HD95 skip vs penalty policy. `test_postprocess.py`: reverse-priority order, `irf_override=False` keeps SRF/PED, min-region drops small blobs, per-class min sizes. `test_dataset.py`: tmp-dir PNG volume to check 2.5D neighbor fallback at volume edges (`dataset.py:89-93`), `target_class` remap (`:114-117`), and **that `cfg.USE_25D=False` is honored** (regression for B1). `test_losses.py`: Tversky is 0 at perfect prediction; focal reduces to CE when `gamma=0`. | M |
| M4 | Logging | B5; `print` in `attention_visualizer.py:19,101,140`; three handler setups (`main.py:29-59`, `eval.py:293-310`, `train_expert.py:32-36`) | One `runs.setup_logging(run_dir)` on the root logger, called by each CLI `main()`; modules use `log = logging.getLogger(__name__)`. | S |
| M5 | Cheaper training step | `train.py:224` passes `labels=` so HF also computes an unused internal CE loss | Call `model(pixel_values=…)` only. Behavior unchanged; one less loss computation per step. | S |

### Low

| ID | Item | Evidence | Change | Effort |
|---|---|---|---|---|
| L1 | Dead code/imports | `main.py:7-8`; `attention_visualizer.py:22-24` (monai); `USE_MULTIMODAL` branch is dead under 2.5D (`dataset.py:99-108`) | `ruff check --select F401,F841 octseg`; remove the monai branch. Keep the multimodal branch only if a planned ablation needs it (it consumes `denoised_images`/`edge_map_images`). | S |
| L2 | Smaller figure blobs | ~10 MB PNGs at `dpi=200` (`eval.py:270`) | `dpi=120` for run artifacts; the thesis copies its own figures anyway. | S |
| L3 | Attention naming | `attention_visualizer.py:106-108,125` names blocks "stage" | Rename the output to `block{idx}` **only for new runs**. Don't regenerate thesis figures; instead fix the captions in `Maksymilian.tex:470-530` ("Stage 6" → "blok 6") if the supervisor cares. | S |
| L4 | Docstrings | `dokumentacja/plan_projektu.md` part I asks for Google-style docstrings | Add them to public functions in `losses/metrics/postprocess/dataset` only. | M |
| L5 | History rewrite | 101 MiB pack (§1.7) | **Do not** rewrite. The repo is shared with Eryk (`PODSUMOWANIE…pdf` p.1), and 101 MiB is tolerable. Revisit with `git filter-repo --strip-blobs-bigger-than 5M` only if hosting limits bite. | — |

---

## 5. Docs consolidation

| Source | Destination | Action |
|---|---|---|
| `README.md` | `README.md` | Rewrite to: purpose (`README.md:1-9`), setup (venv + pinned reqs + `.env.example`), data layout (`data_folder/cropped_images|cropped_masks`, filename pattern `utils.py:7`), commands (`python -m octseg.pipeline|train_expert|evaluate|evaluate_hybrid`), a headline result table (test11-gold and test18 from `README.md:45-53`), links to `docs/`. Remove the stale "Key Features" claims (Soft-CRF, retina masking; §1.8). |
| `README.md:22-90` (Experimental History) | `docs/EXPERIMENTS.md` | Move verbatim and prepend the index table from §1.6 (dir ↔ original `run_` timestamp ↔ backbone ↔ mIoU/mDice). Add the promotion rule from Step 7. |
| `Dokumentacja.md` (PL) + `summary.md` (EN) | `docs/METHODS.md` (Polish, thesis language) | Merge into one methods description; correct against code: LR/scheduler (`train.py:197-202`), TTA scales (`config.py:57`), Boundary Loss off (`config.py:39`), no retina masking, no CRF, thresholds (`config.py:66-70`), HD95 policy (B3). Originals go to `docs/archive/`. |
| `plan.md`, `improvement_plan.md`, `IRF_EXPERT_PLAN.md`, `test19_prep.md` | `docs/ROADMAP.md` (+ originals to `docs/archive/`) | Carry over only the open items. **Test 19:** expert training and hybrid eval; resolve the B1 2D/2.5D decision and the α/β conflict (0.05/0.95 in `IRF_EXPERT_PLAN.md:22` vs 0.3/0.7 in `config_irf_expert.py:22-23`). The planned merge logic (`IRF_EXPERT_PLAN.md:32-35`) and the implemented soft blend (`hybrid_inference.py:93-141`) differ; record which one the thesis reports. **Targets:** IRF IoU > 0.65 (`IRF_EXPERT_PLAN.md:38`), IRF Dice > 0.73, HD95 < 68, CPU inference < 500 ms/B-scan (`improvement_plan.md:29-31`), < 1 s/slice for the hybrid (`IRF_EXPERT_PLAN.md:40`). **Benchmark:** nnUNet (Eryk) vs SegFormer-B2 vs SwinUNETR table (`plan.md:23`). **Thesis sections still unchecked** (`plan.md:27-29`). **Post-thesis:** cross-device calibration, MedSAM-2 (`improvement_plan.md:22-24`). Drop the stale checklist items: test17 "ACTIVE" (`plan.md:16`) was abandoned per `README.md:61-72`, and failure analysis (`plan.md:20`) no longer exists. |
| `changelog.md` | `CHANGELOG.md` | Rename. Add an entry per migration step and per H/M fix. |
| `dokumentacja/plan_projektu.md` | `docs/thesis/plan_projektu.md` | Moves with the thesis (Step 5); it is the thesis outline. |
| `PODSUMOWANIE_PRAC_PO_2_TYGODNIACH_TESTOW.odt/.pdf` | `docs/archive/reports/` | Archive unchanged (dated progress report, 22.05). |
| `experiments/*/README.md`, `experiments/*/plan.md` | unchanged | Frozen snapshots copied by `main.py:81`; never edit. |

---

## 6. Risks and mitigations

| Risk | Detail | Mitigation |
|---|---|---|
| Breaking the Windows lab workflow | Runs are executed from `C:\Users\Student\Documents\python\oct-project` (`experiments/test18/experiment.log:1`) via `run.bat`. Moving to `python -m` and `scripts/` changes the invocation. | Step 3 updates `scripts/run.bat` with `cd /d %~dp0..`. Keep `data_folder` as the default name. `OCT_DATA_DIR` env var for other locations. Test on the lab machine before test19. |
| Silent metric drift during dedup | Four post-processing copies differ subtly (IRF opening only in eval/hybrid, not validation; different min-size fallbacks). | Step 0 parity snapshot + Step 4 `np.array_equal` parity; keep current semantics behind explicit parameters; behavior changes only in §4 commits, logged in `CHANGELOG.md`. |
| Thesis numbers vs new metric code | Thesis tables (`Maksymilian.tex:393-410`) use the skip-HD95 eval path (B3). | Default `hd95_missing="skip"`; any new policy is reported as a separate, labelled column. |
| Thesis file references | `\includegraphics` uses paths inside `dokumentacja/` (`test2/metrics.png`, `test18/attention_maps/att_idx*_stage*.png`); the text names `attention_visualizer.py` (`Maksymilian.tex:305`). | Move the folder as a unit (Step 5) and keep the module name. Figures in `docs/thesis/test*/` are copies, independent of `experiments/`, so experiments can stay put. |
| Thesis currently unversioned | `/dokumentacja` is ignored (`.gitignore:278`); loss of the laptop means loss of the thesis. | Step 5 tracks sources. Ignore rules keep the 21 MB `OCT.pdf` and 798 MB of CNN checkpoints out. Run `git status` before the first commit and confirm no `.pth`/`OCT.pdf` is staged. |
| Old snapshots look importable | `experiments/test*/` contain `config.py`, `train.py`, … with flat imports. | Excluded from ruff/pytest via `pyproject.toml`; never on `sys.path` because code runs as `python -m octseg.*` from the root. |
| Unreproducible past results | No SegFormer `.pth` exists on this host (§1.6); the lab venv versions are unknown. | Step 2 lab `pip freeze`; from test19 on, per-run `config.yaml` + `git.txt` + `metrics.yaml`; keep checkpoints of promoted runs outside git (lab disk) and record their path in `docs/EXPERIMENTS.md`. |
| Large files in git history | 101 MiB pack, ~10 MB PNGs, tracked `failures/`. | No rewrite (L5). Stop growth with B6 ignores, removal of `update_git.sh`, and L2. |
| `.gitignore` global patterns | `*.json`, `*.csv`, `conf/`, `outputs/`, `data/` are ignored (`.gitignore:180,185-186,212,215`). | Use `configs/` + YAML + `.txt` splits as designed; never name a tracked dir `conf/` or `data/`. |
| `.env` leakage | The only secret is `DISCORD_WEBHOOK_URL`; it has never been committed. | `.env.example` (Step 1). Loading the dotenv explicitly from the repo root (Step 3) avoids picking up a stray `.env` from the CWD. |
| Broken local venv | `venv/bin/python` → Python 3.14 without the packages. | Step 0 `venv --upgrade` with 3.13; if that fails, recreate (`/usr/bin/python3.13 -m venv venv && venv/bin/pip install -r requirements.txt` + platform torch). |

---

## 7. Order of work (summary)

1. Step 0–2 (tag, venv, hygiene, pins): ~1 h, no risk.
2. **H1** (expert cfg bug): before any test19 training.
3. Step 3–4 (package + dedup with parity checks): ~1 day.
4. H3/H4/H5 + M1 (Step 8) + M3 tests: ~1–2 days.
5. Step 5–7 (docs, thesis tracking, outputs/CLIs): ~half a day.
6. Low items as time allows; L5 never unless forced.
