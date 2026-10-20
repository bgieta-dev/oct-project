# Rule: Engineering Guardrails (Tooling, Python version, lint ratchet, change hygiene)

## 1. Tooling: uv only
- Python environment management MUST use `uv`: `uv venv --python 3.13 venv`, `uv pip install -r requirements.txt -r requirements-dev.txt`, `uv tool run pre-commit`.
- BANNED: bare `pip install`, `python -m pip`, `poetry`, `pipenv`, `sudo pip`, `--break-system-packages`. System Python is PEP 668 protected.
- Platform wheels (PyTorch): On GPU/CUDA/ROCm workstations, install the matching platform torch wheel before installing requirements. CI and CPU environments use `--extra-index-url https://download.pytorch.org/whl/cpu`.
- New dependencies: Verify PyPI presence, repository provenance, release age >30 days. Never add unvetted packages suggested by LLMs (slopsquatting defence).

## 2. Python version policy
- Project runtime & dev target: **Python 3.13** (`.python-version`).
- Python 3.15 evaluation: Attempted per global directive. Blocked because PyTorch and MONAI lack `cp315` binary wheels on PyPI. Adopted Python 3.13 as the newest fully supported version across PyTorch, MONAI, TIMM, Albumentations, and MedPy.
- Always pass `encoding="utf-8"` to file open/read/write operations.

## 3. Lint ratchet (ruff, pinned `0.17.1`)
- **Blocking set** in `pyproject.toml` (`[tool.ruff.lint]`): `E9`, `E711`, `E712`, `E713`, `E714`, `E721`, `E722`, `F` (ignore `F401`), `B` (ignore `B905`), `YTT`, `ASYNC`, `T10`, `EXE`, `PIE`, `PGH`, `PLE`, `W605`, `S103`, `S104`, `S105`, `S106`, `S107`, `S108`, `S602`, `S604`, `S605`, `S608`. Clean repo-wide (0 errors).
- **Touched-file rule**: Any file edited MUST also pass the strict ruleset:
  `ruff check --select E,F,W,B,I,UP,SIM,C4,RUF,S,BLE --ignore E501,S101,S603,S607,COM812,ISC001 <file>`
  Fix findings in files you touch; do not mass-rewrite untouched legacy files.
  Baseline (2026-10-20): 183 findings across 12 files (51 UP006, 28 W293, 18 UP035, 16 I001, 14 UP045, 11 BLE001, 11 B905, 8 W291, 4 F401, 3 UP008, 3 C408, 3 C420, 2 RUF005, 2 SIM401, 2 RUF013, 2 RUF007, 1 SIM102, 1 E701, 1 UP007, 1 UP015, 1 RUF100).
- Never add `# noqa` or ignore rules to mask findings without written rationale.

## 4. Test and evaluation ratchet
- Test suite baseline: **91 passed, 1 skipped** (`tests/test_splits.py:46: data_folder not present`).
- Test count MUST NOT decrease. Never delete, skip, or weaken tests.
- Unit tests run CPU-only on synthetic data: `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ./venv/bin/pytest -v`.
- BANNED: Executing multi-epoch GPU training or unmonitored evaluation runs in CI or automated guardrail checks.

## 5. Change hygiene & safety invariants
1. Check `git status --short` before editing; never touch files modified by another agent or user.
2. Search before write: reuse existing classes, helpers, and configurations.
3. External network calls (Discord webhooks, HuggingFace Hub downloads) MUST specify explicit bounded timeouts (`timeout=...`).
4. Secret hygiene: Plaintext API keys (`HF_TOKEN`, `DISCORD_WEBHOOK_URL`) are BANNED from git tracking and log files. Always use `.env` with `.env.example` templates.
5. Checkpoint hygiene: Model weights (`*.pth`, `*.pt`, `*.bin`, `*.safetensors`) MUST NEVER be committed to version control.
6. Verify before done: Run `skill://verify` or `.omp/skills/verify/SKILL.md` to confirm clean working tree, 0 blocking lint errors, and 100% test suite pass.

## 6. Enforcement layers
| Layer | Artifact | Enforcement |
|---|---|---|
| Steering | `AGENTS.md`, `.omp/rules/engineering-guardrails.md` | Agent guidance & negative constraints |
| Local gate | `.pre-commit-config.yaml` | Hygiene, blocking ruff check, gitleaks secret scan |
| CI gate | `.github/workflows/ci.yml` | GitHub Actions: blocking ruff, gitleaks, CPU unit tests |
| Ratchet | `pyproject.toml` | 0 blocking errors repo-wide; touched files clean under strict set |

Sources and verification dates: `docs/guardrails/SOURCES.md`.
