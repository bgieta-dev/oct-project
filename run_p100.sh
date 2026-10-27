#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

CONFIG="${1:-configs/sam_adamw.yaml}"

echo "=================================================="
echo "Starting oct-project training pipeline on Tesla P100"
echo "Config: ${CONFIG}"
echo "=================================================="

# Stop inference worker container to free 13.1 GB VRAM
if command -v docker &>/dev/null && docker ps --format "{{.Names}}" | grep -q "qwen-inference-p100"; then
    echo "[*] Stopping qwen-inference-p100 to free GPU VRAM..."
    docker stop qwen-inference-p100 || true
fi

# Run pipeline with auto-promotion to experiments/
python -m octseg.pipeline --config "${CONFIG}" --promote "${@:2}"

echo "[*] Pipeline completed successfully. Running git_update.sh to push experiment artifacts to GitHub..."
./git_update.sh "feat(p100): promote experiment run artifacts"

echo "[*] Done! All experiment results committed and pushed to GitHub."
