# Guardrails Implementation Plan: oct-project

## 1. Detected Stack
- **Repository**: `/home/bgieta/Coding/oct-project`
- **Domain**: OCT retinal fluid segmentation (IRF / SRF / PED) on the RETOUCH dataset (SegFormer, RETFound-OCT + LoRA, SwinUNETR 3D). Part of engineering thesis at Poznan University of Technology.
- **Language**: Python (47 files, ~6,546 LOC).
- **Core ML Dependencies**: PyTorch, Torchvision, Transformers, MONAI, TIMM, PEFT, Albumentations, MedPy, Scikit-learn, Scipy, OpenCV, Pillow.
- **Testing**: PyTest with synthetic fixtures in `tests/` (13 test files).
- **Runtime Environment**: Python 3.13.5 (`venv/` present, standardized in `AGENTS.md`).

## 2. Baseline Numbers
- **Tracked files**: 199 files.
- **Unit test suite**: 91 passed, 1 skipped (`tests/test_splits.py:46: data_folder not present`). 0 failures, 0 errors.
- **Lint status (clean blocking set)**: 0 errors today across entire repository.
- **Lint status (strict touched-file set)**: 183 findings across 12 files (51 UP006, 28 W293, 18 UP035, 16 I001, 14 UP045, 11 BLE001, 11 B905, 8 W291, 4 F401, 3 UP008, 3 C408, 3 C420, 2 RUF005, 2 SIM401, 2 RUF013, 2 RUF007, 1 SIM102, 1 E701, 1 UP007, 1 UP015, 1 RUF100).
- **Secrets**: 0 hardcoded secrets found in git history or tracked code.

## 3. Python Version Evaluation
- **Directive**: Python target 3.15 if possible, else newest supported by ML deps (3.14, 3.13, 3.12).
- **Evaluation Result**: Python 3.15 attempted. PyTorch and MONAI lack `cp315` binary wheels on PyPI (`uv pip install --dry-run --python 3.15` failed resolution with: `torch>=2.8.0 has no wheels with matching Python ABI tag cp315`). Python 3.13 is fully supported, has binary wheels for all project dependencies, is installed locally, and has 100% test pass (91/91).
- **Adopted Version**: Python 3.13 (`.python-version = 3.13`, `pyproject.toml` target-version = `py313`).

## 4. Deliverables & Proposed Files
| File | Action | Purpose |
|---|---|---|
| `AGENTS.md` | Edit | Append short "Guardrails" section (<150 lines total) |
| `.omp/rules/engineering-guardrails.md` | Create | Project-specific rule (tooling, Python 3.13, lint/test ratchets, hygiene) |
| `.omp/skills/verify/SKILL.md` | Create | Exact deterministic verification playbook for oct-project |
| `pyproject.toml` | Edit | Configure `[tool.ruff]` blocking rules and formatter |
| `.pre-commit-config.yaml` | Create | Local gate (hygiene, ruff blocking check, gitleaks secret scan) |
| `.github/workflows/ci.yml` | Create | CI workflow (permissions: read, ruff, gitleaks, CPU unit tests) |
| `docs/guardrails/SOURCES.md` | Create | Provenance table for every adopted practice and tool pin |
| `docs/guardrails/PLAN.md` | Create | This document |
| `.gitignore` | Edit | Append `.ruff_cache/` |
| `.python-version` | Create | Pin Python version to `3.13` |

## 5. Blocking Rules vs Touched-File Rules
- **Blocking Rules (0 errors today, enforced repo-wide & in pre-commit/CI)**:
  `E9`, `E711`, `E712`, `E713`, `E714`, `E721`, `E722`, `F` (ignore `F401`), `B` (ignore `B905`), `YTT`, `ASYNC`, `T10`, `EXE`, `PIE`, `PGH`, `PLE`, `W605`, `S103`, `S104`, `S105`, `S106`, `S107`, `S108`, `S602`, `S604`, `S605`, `S608`.
- **Touched-File Rules (enforced on modified files only; baseline 183 findings)**:
  `select = ["E", "F", "W", "B", "I", "UP", "SIM", "C4", "RUF", "S", "BLE"]`
  `ignore = ["E501", "S101", "S603", "S607", "COM812", "ISC001"]`

## 6. Risk Surfaces Found
1. **Model Checkpoints**: Large binary weight files (`*.pth`, `*.pt`) must never be committed; gitignored and verified by verify skill.
2. **External Gated Weights**: HuggingFace gated model `RETFound_mae_natureOCT` requires `HF_TOKEN`; must remain in `.env` (template in `.env.example`).
3. **Subprocess Calls**: 5 calls in `run_series.py` and `runs.py`; must use argument lists without `shell=True`.
4. **HTTP Calls**: Discord notifications in `runs.py` and HuggingFace downloads in `main.py`; explicit timeouts enforced.
5. **Auto-Sync**: Verified that `/home/bgieta/Coding/sync-projects.py` does NOT monitor `oct-project`.

## 7. Open Questions
None.
