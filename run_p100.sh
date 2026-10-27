#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

CONFIG="${1:-configs/sam_adamw.yaml}"

echo "=================================================="
echo "Starting oct-project training pipeline on Tesla P100"
echo "Config: ${CONFIG}"
echo "=================================================="

# Check and stop inference worker container to free 13.1 GB VRAM
WAS_QWEN_RUNNING=0
if command -v docker &>/dev/null && docker ps --format "{{.Names}}" | grep -q "qwen-inference-p100"; then
    echo "[*] Temporarily stopping qwen-inference-p100 to free GPU VRAM..."
    docker stop qwen-inference-p100
    WAS_QWEN_RUNNING=1
fi

cleanup() {
    echo ""
    echo "[*] Pipeline finished or interrupted."
    if [ "${WAS_QWEN_RUNNING}" -eq 1 ]; then
        echo "[*] Restoring qwen-inference-p100 container..."
        docker start qwen-inference-p100 || true
    fi
}
trap cleanup EXIT INT TERM

# Run pipeline with auto-promotion to experiments/
python -m octseg.pipeline --config "${CONFIG}" --promote "${@:2}"

echo "[*] Pipeline completed successfully. Running git_update.sh to push experiment artifacts to GitHub..."
./git_update.sh "feat(p100): promote experiment run artifacts"

echo "[*] Done! All experiment results committed and pushed to GitHub."
