#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

BASE="${BASE:-outputs/runs/2026-10-27_14-57-10_test20/best_model.pth}"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    BASE="$1"
    shift
fi

echo "=================================================="
echo "Starting oct-project calibration sweep on Tesla P100"
echo "Base weights: ${BASE}"
echo "=================================================="

# Stop inference worker container to free 13.1 GB VRAM
if command -v docker &>/dev/null && docker ps --format "{{.Names}}" | grep -q "qwen-inference-p100"; then
    echo "[*] Stopping qwen-inference-p100 to free GPU VRAM..."
    docker stop qwen-inference-p100 || true
fi

# Run calibration sweep
python -m octseg.calibrate_postprocess --base-weights "$BASE" "$@"

echo "[*] Calibration sweep completed successfully. Running git_update.sh to push artifacts to GitHub..."
./git_update.sh "feat(calibrate): commit calibration sweep artifacts"

echo "[*] Done! Calibration results committed and pushed to GitHub."
