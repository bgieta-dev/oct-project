---
name: verify
description: Deterministic verification playbook for oct-project (clean tree, ruff blocking check, pre-commit hygiene, CPU pytest suite, secret scan).
alwaysApply: false
---

# Skill: verify

## Objective
Execute comprehensive, deterministic verification across `oct-project` (`/home/bgieta/Coding/oct-project`). Validate clean git working tree, zero blocking lint errors, pre-commit hygiene hooks, absence of tracked model checkpoints or plaintext secrets, and complete CPU unit test suite execution without executing unmonitored GPU training runs.

---

## Pre-Flight Checks

1. **Git Repository Status & Cleanliness**:
   Ensure no untracked secrets, uncommitted conflicts, or unintended edits:
   ```bash
   cd /home/bgieta/Coding/oct-project
   git status --short
   ```

2. **Checkpoint & Secret Leakage Audit**:
   Verify no binary model checkpoints or secrets are tracked in git:
   ```bash
   # Checkpoints must never be committed
   TRACKED_CKPTS=$(git ls-files "*.pth" "*.pt" "*.bin" "*.safetensors")
   if [ -n "$TRACKED_CKPTS" ]; then
       echo "FATAL: Tracked model checkpoints detected: $TRACKED_CKPTS"
       exit 1
   fi

   # Secrets must not be tracked
   TRACKED_SECRETS=$(git ls-files | grep -E '(\.env$|\.kdbx$|\.sqlite$|token|keys\.txt)' || true)
   if [ -n "$TRACKED_SECRETS" ]; then
       echo "FATAL: Tracked secret files detected: $TRACKED_SECRETS"
       exit 1
   fi
   ```

---

## Step 1: Linting & Blocking Rules Check

Verify that all blocking rules pass with 0 errors across the repository:

```bash
cd /home/bgieta/Coding/oct-project
ruff check .
```

---

## Step 2: Pre-Commit Hygiene & Secrets Gate

Run the local pre-commit gate hooks:

```bash
cd /home/bgieta/Coding/oct-project
uv tool run pre-commit run --all-files
```

---

## Step 3: Full PyTest CPU Test Suite Execution

Run the complete test suite on CPU using synthetic test fixtures (no GPU training):

```bash
cd /home/bgieta/Coding/oct-project
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ./venv/bin/pytest -v
```

Expected baseline: 91 passed, 1 skipped (`tests/test_splits.py:46: data_folder not present`).

---

## Acceptance Criteria

Verification is considered **PASSED** only when ALL of the following criteria are met:
1. Working tree is clean or contains only intended, verified changes.
2. `ruff check .` passes with zero errors.
3. `uv tool run pre-commit run --all-files` passes all hygiene, syntax, and secret detection hooks.
4. No binary checkpoints (`*.pth`, `*.pt`) or plaintext secrets (`.env`) are tracked in git.
5. PyTest passes all 91 unit tests with zero unexpected failures or errors.
