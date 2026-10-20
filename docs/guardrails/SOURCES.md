# Guardrail Sources

Every adopted practice, where it came from, when it was checked. Re-verify before changing a guardrail.

| Practice | Source | Checked | Applied in |
|---|---|---|---|
| Defence in depth: AGENTS.md (steering) + pre-commit (local) + CI (hard gate); CI is non-bypassable boundary | https://axonbuild.com/blog/guardrails-for-ai-coding-agents/ , https://yelno.ai/blog/pre-commit-hook-ai-coding-agent-guardrails | 2026-10-20 | `.omp/rules/engineering-guardrails.md` section 6, `.pre-commit-config.yaml`, `ci.yml` |
| Keep AGENTS.md lean (<150 lines), point to canonical files, list negative constraints | https://www.aihero.dev/a-complete-guide-to-agents-md , https://www.philschmid.de/writing-good-agents | 2026-10-20 | `AGENTS.md` |
| PyTorch CI with uv, CPU wheels via extra-index, synthetic tensor unit tests without live GPU training | https://github.com/astral-sh/uv/issues/10590 , https://adrianosantospb.medium.com/automating-ci-cd-with-pytorch-and-github-actions-on-a-gpu-runner-943f8c389585 | 2026-10-20 | `.github/workflows/ci.yml`, `rule section 4` |
| Gitleaks in pre-commit and CI; agents may use `--no-verify`, so CI repeats checks | https://blog.gitguardian.com/ai-agent-security-best-practices/ | 2026-10-20 | `.pre-commit-config.yaml`, `ci.yml` |
| Explicit ruff rule selection over ALL; avoid formatter conflict rules COM812/ISC001 | https://docs.astral.sh/ruff/linter/ , https://paths.grasp.study/modules/9fb968e1-738a-43c2-98c7-f3a5e7466eaf/lessons/f38c6986-8f60-4925-b4af-a372b7b94a2a | 2026-10-20 | `pyproject.toml`, rule section 3 |
| Pinned tool versions: astral-sh/setup-uv v10, actions/checkout v7, ruff-pre-commit v0.17.1, gitleaks v8.30.1, pre-commit-hooks v6.0.0 | GitHub releases API, PyPI JSON | 2026-10-20 | `ci.yml`, `.pre-commit-config.yaml`, `pyproject.toml` |
| Python 3.13 target: Python 3.15 attempted per directive; PyPI JSON check confirmed torch and monai lack cp315 wheels | https://pypi.org/pypi/torch/json , local dry-run resolution | 2026-10-20 | `.python-version`, `pyproject.toml`, rule section 2 |
