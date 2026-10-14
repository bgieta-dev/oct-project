# Roadmap

Open items carried over from `docs/archive/{plan,improvement_plan,IRF_EXPERT_PLAN,test19_prep}.md` and the thesis
"okres wakacyjny" list (`docs/thesis/Maksymilian.tex`).

## Vacation plan status

Code is in `octseg/` (see `CHANGELOG.md` [Unreleased]); every lab run below is still to be done on the Windows /
RTX 3060 machine.

| item | code | lab run |
|---|---|---|
| Shared groundwork: `split_files`, `ARCH` / `DEVICE_OVERRIDE` / `WEIGHT_DECAY`, N-D Tversky, `slice_mean_iou`, `Peak training VRAM` log | done | — |
| Test19 base (SegFormer-B2) | done (old flat pipeline) | done: test mIoU 0.7029 (`experiments/test19`); recheck with the new package pending |
| Test19 IRF expert + hybrid (`replace` mode, `sweep_hybrid` selection on val) | done | pending |
| Inference time / VRAM benchmark (`octseg.benchmark`) | done | pending |
| Native 3D SwinUNETR (`configs/swin_unetr_3d.yaml`) | done | pending |
| RETFound-OCT + LoRA ablation (planned test20–22) | done | blocked: HF access to `YukunZhou/RETFound_mae_natureOCT` not granted yet (403) |
| More training data (`Maksymilian.tex`) | out of scope | — |

## Test 19: IRF expert + hybrid evaluation (next run)

The base model is test19 (`docs/EXPERIMENTS.md`): test mIoU 0.7029, IRF Dice 0.6958, IRF IoU 0.5335. Its checkpoint
is on the lab machine (`experiments\run_2026-06-18_22-22-31\`, unverified — locate `best_model.pth`).

1. Recheck the base with the new package:
   `python -m octseg.evaluate --model <base> --output outputs/eval/test19-base-recheck`. Accept when `Final mIoU` is
   within ±0.005 of 0.7029; otherwise (or if strict loading fails) retrain with
   `python -m octseg.pipeline --name test19-base`.
2. Train the expert: `python -m octseg.train_expert --name test19-expert`.
3. Select the merge variant on **val**:
   `python -m octseg.sweep_hybrid --base-weights <base> --expert-weights <expert> --split val --name test19-sweep`.
4. If the sweep logs a best variant, run its logged `evaluate_hybrid` command once on test with
   `--output outputs/hybrid_eval/test19-hybrid`. If it logs "hybrid not adopted", the reported model is the base.
5. Write the chosen values into `configs/hybrid.yaml` (`HYBRID_*`), record the decision under "Merge logic" below,
   promote the runs (see `docs/EXPERIMENTS.md`).

### Decisions recorded

* **Expert input = 2.5D** (t-1, t, t+1), same as the base model. Before the reorganisation `config_irf_expert.py`
  said "force 2D" but `train.py` did not pass the expert config to `OCTDataset`, so the expert silently trained on
  2.5D; `hybrid_inference.py` then fed it the middle slice replicated ×3. `IRF_EXPERT_PLAN.md` also asks for 2.5D.
  Now `configs/irf_expert.yaml` has `USE_25D: true`, the config is passed to the dataset (regression test in
  `tests/test_dataset.py`), and the hybrid feeds both models the same image (and refuses mismatching `USE_25D`).
* **Expert batch size** follows its backbone (mit-b0: 16 × 2) instead of inheriting the mit-b2 values.
* **Tversky α/β for the expert: 0.3 / 0.7** (as implemented). `IRF_EXPERT_PLAN.md` / `test19_prep.md` say
  0.05 / 0.95; the extreme setting is not run.
* **Merge logic.** The planned logic of `IRF_EXPERT_PLAN.md` (expert mask replaces base IRF) is implemented as
  `HYBRID_ENSEMBLE_MODE: replace`; its third step ("OR at expert confidence > 0.7") needs no code, because the
  expert threshold of 0.25 already includes every pixel above 0.7. The variant is chosen on the validation split by
  `octseg.sweep_hybrid` among 14 candidates (soft linear/confidence/max, hard, replace): highest IRF Dice among the
  variants that beat the base IRF Dice and keep SRF/PED Dice within 0.01 of the base. **Open:** the sweep has not
  run; record the chosen variant and its val/test numbers here after step 4.

### Targets

* IRF IoU > 0.65 (test19 base: 0.5335), IRF Dice > 0.73 (0.6958), mHD95 < 68 (73.75).
* Predicted IRF region count approaching GT (test19 base: 4.5 predicted vs 8.9 GT per slice).
* CPU inference < 500 ms per B-scan (base), < 1 s per slice for the hybrid (measure with `octseg.benchmark`).

## Lab runs after test19

* **Benchmark** (`outputs/benchmark/`, copy the `benchmark.yaml` files to `experiments/benchmarks/`):
  `python -m octseg.benchmark --config configs/base.yaml --model <base> --device cuda --name bench-segformer-gpu`,
  the same with `--device cpu --slices 20`, with `--config configs/hybrid.yaml --expert-config configs/irf_expert.yaml
  --expert-weights <expert>` for the hybrid, and with `--config configs/swin_unetr_3d.yaml --volumes 3` for 3D.
* **SwinUNETR 3D:** download `model_swinvit.pt` (MONAI release asset) to `outputs/pretrained/`, smoke run
  `python -m octseg.pipeline --config configs/swin_unetr_3d.yaml --name swin3d-smoke --epochs 2` (`Peak training
  VRAM` must be < 11000 MB; otherwise `SWIN_ROI: [32, 128, 128]`, then `SWIN_FEATURE_SIZE: 24` +
  `SWIN_PRETRAINED: null`), then the full run with `scripts\run.bat --config configs/swin_unetr_3d.yaml`.
* **RETFound ablation:** once Hugging Face grants access, `scripts\run_retfound_ablation.bat` (preflights the
  weights, then three consecutive `testN` runs). If the GPU drops off the bus ("GPU is lost"), run
  `scripts\recover_gpu.bat` (or reboot) and rerun the remaining configs.
* Push results with `scripts\git_update.bat "<message>"`.

## Benchmark and thesis

* Comparison table: nnUNet (Eryk) vs SegFormer-B2 vs SwinUNETR 3D (re-added as `ARCH: swin_unetr_3d`) vs
  RETFound + LoRA, plus inference time / VRAM per model.
* Thesis chapters still open: SegFormer description (self-attention, MLP head), 2.5D stacking and percentile
  normalisation methodology, quantitative results on RETOUCH.
* Thresholds: the reported test18/test19 snapshots use SRF/PED thresholds 0.40 and `MIN_REGION_SIZE` 50, the current
  `configs/base.yaml` 0.80 and 80 (see `docs/METHODS.md` §3); the thesis must state the values that produced its
  numbers.
* Optional: fix the figure captions "Stage 6/11" in `docs/thesis/Maksymilian.tex` to "blok 6/11" (they are encoder
  block indices, see `docs/METHODS.md` §5).

## Post-thesis ideas

* Cross-device calibration (fine-tuning on AROI or Spectralis-only subsets).
* MedSAM-2 zero-shot experiments for rare OCT pathologies.
* Run the lab machine `pip freeze > requirements-lab-freeze.txt` and commit it (versions that produced the
  reported numbers are unknown here).
