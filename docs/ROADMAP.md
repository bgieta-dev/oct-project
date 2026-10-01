# Roadmap

Open items carried over from `docs/archive/{plan,improvement_plan,IRF_EXPERT_PLAN,test19_prep}.md`.

## Test 19: IRF expert + hybrid evaluation (next run)

1. Train a fresh base model: `python -m octseg.pipeline --name test19-base` (no SegFormer checkpoint exists on the
   dev host; the lab machine holds the old ones).
2. Train the expert: `python -m octseg.train_expert --name test19-expert`.
3. Evaluate the ensemble:
   `python -m octseg.evaluate_hybrid --base-weights <base>/best_model.pth --expert-weights <expert>/best_model.pth`.
4. Promote the runs (see `docs/EXPERIMENTS.md`).

### Decisions recorded

* **Expert input = 2.5D** (t-1, t, t+1), same as the base model. Before the reorganisation `config_irf_expert.py`
  said "force 2D" but `train.py` did not pass the expert config to `OCTDataset`, so the expert silently trained on
  2.5D; `hybrid_inference.py` then fed it the middle slice replicated ×3. `IRF_EXPERT_PLAN.md` also asks for 2.5D.
  Now `configs/irf_expert.yaml` has `USE_25D: true`, the config is passed to the dataset (regression test in
  `tests/test_dataset.py`), and the hybrid feeds both models the same image (and refuses mismatching `USE_25D`).
* **Expert batch size** follows its backbone (mit-b0: 16 × 2) instead of inheriting the mit-b2 values.
* **Tversky α/β for the expert: 0.3 / 0.7** (as implemented). `IRF_EXPERT_PLAN.md` / `test19_prep.md` say
  0.05 / 0.95; if test19 is meant to use the extreme setting, change `TVERSKY_ALPHA/BETA` in `configs/irf_expert.yaml`.
* **Merge logic.** The planned logic (expert mask replaces base IRF, OR at expert confidence > 0.7) differs from the
  implemented soft blend (`HYBRID_ENSEMBLE_MODE: soft`, linear, expert weight 0.4, IRF threshold 0.25). **Open:**
  decide which variant the thesis reports and note it here after test19.

### Targets

* IRF IoU > 0.65 (was ~0.51), IRF Dice > 0.73, mHD95 < 68.
* Predicted IRF region count approaching GT (8.9 vs 4.3 before the expert).
* CPU inference < 500 ms per B-scan (base), < 1 s per slice for the hybrid.

## Benchmark and thesis

* Comparison table: nnUNet (Eryk) vs SegFormer-B2 vs SwinUNETR (SwinUNETR code was removed after test13/14).
* Thesis chapters still open: SegFormer description (self-attention, MLP head), 2.5D stacking and percentile
  normalisation methodology, quantitative results on RETOUCH.
* Optional: fix the figure captions "Stage 6/11" in `docs/thesis/Maksymilian.tex` to "blok 6/11" (they are encoder
  block indices, see `docs/METHODS.md` §5).

## Post-thesis ideas

* Cross-device calibration (fine-tuning on AROI or Spectralis-only subsets).
* MedSAM-2 zero-shot experiments for rare OCT pathologies.
* Failure-case extraction (top-k worst slices) is not in the current `evaluate.py`; the old implementation is in
  `experiments/test10/eval.py`.
* Run the lab machine `pip freeze > requirements-lab-freeze.txt` and commit it (versions that produced the
  reported numbers are unknown here).
