#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

CONFIG="${1:-configs/test21.yaml}"

echo "=================================================="
echo "Starting oct-project training pipeline"
echo "Config: ${CONFIG}"
echo "=================================================="

if [[ "$#" -gt 0 && "$1" == --* ]]; then
    python -m octseg.pipeline --promote "$@"
else
    python -m octseg.pipeline --config "${CONFIG}" --promote "${@:2}"
fi

echo "[*] Pipeline finished successfully. Running git_update.sh to push artifacts..."
./git_update.sh "feat: promote experiment run artifacts"

echo "[*] Done! All experiment results committed and pushed to GitHub."
