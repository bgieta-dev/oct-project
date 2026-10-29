#!/usr/bin/env bash
set -euo pipefail

P100_HOST="${P100_HOST:-bgieta@192.168.1.131}"
REMOTE_DIR="/home/bgieta/oct-project"
CONFIG="${1:-configs/test21.yaml}"

echo "=========================================================="
echo "Connecting to Tesla P100 host: ${P100_HOST}"
echo "Remote Directory: ${REMOTE_DIR}"
echo "Config Target: ${CONFIG}"
echo "=========================================================="

# Check SSH connectivity
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${P100_HOST}" "exit" 2>/dev/null; then
    echo "[!] Error: Cannot connect to ${P100_HOST} via SSH."
    exit 1
fi

# Check if training is already in progress
ALREADY_RUNNING=$(ssh "${P100_HOST}" "tmux list-sessions 2>/dev/null | grep 'oct_training' || true")

if [ -n "${ALREADY_RUNNING}" ]; then
    echo "[*] Training session 'oct_training' is ALREADY RUNNING on P100."
    echo ""
    echo "--- GPU Telemetry ---"
    ssh "${P100_HOST}" "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader"
    echo ""
    echo "--- Recent Training Log ---"
    ssh "${P100_HOST}" "tail -c 600 ${REMOTE_DIR}/train.log | tr '\r' '\n' | grep -E '(Epoch|mIoU|Loss|Best)' | tail -n 8"
    echo ""
    echo "To attach interactively, run:"
    echo "  ssh -t ${P100_HOST} 'tmux attach -t oct_training'"
    exit 0
fi

echo "[*] Pulling latest code and configs from GitHub on P100..."
ssh "${P100_HOST}" "cd ${REMOTE_DIR} && git pull origin main"

echo "[*] Launching training in detached tmux session 'oct_training'..."
ssh "${P100_HOST}" "tmux new-session -d -s oct_training 'cd ${REMOTE_DIR} && ./run_p100.sh ${CONFIG} 2>&1 | tee train.log'"

echo "[*] Waiting 6 seconds for startup initialization..."
sleep 6

echo ""
echo "--- Initial Startup Log ---"
ssh "${P100_HOST}" "tail -n 15 ${REMOTE_DIR}/train.log"
echo ""
echo "--- GPU Telemetry ---"
ssh "${P100_HOST}" "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader"

echo ""
echo "=========================================================="
echo "Training started successfully on Tesla P100!"
echo "Session: tmux attach -t oct_training"
echo "Log:     tail -f ${REMOTE_DIR}/train.log"
echo "=========================================================="
