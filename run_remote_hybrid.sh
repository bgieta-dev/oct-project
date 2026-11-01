#!/usr/bin/env bash
set -euo pipefail

P100_HOST="${P100_HOST:-bgieta@192.168.1.131}"
REMOTE_DIR="/home/bgieta/oct-project"

echo "=========================================================="
echo "Connecting to Tesla P100 host: ${P100_HOST}"
echo "Remote Directory: ${REMOTE_DIR}"
echo "Target: Turnkey Hybrid Pipeline (oct_hybrid)"
echo "=========================================================="

# Check SSH connectivity
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${P100_HOST}" "exit" 2>/dev/null; then
    echo "[!] Error: Cannot connect to ${P100_HOST} via SSH."
    exit 1
fi

# Check if hybrid pipeline session is already in progress
ALREADY_RUNNING=$(ssh "${P100_HOST}" "tmux list-sessions 2>/dev/null | grep 'oct_hybrid' || true")

if [ -n "${ALREADY_RUNNING}" ]; then
    echo "[*] Hybrid session 'oct_hybrid' is ALREADY RUNNING on P100."
    echo ""
    echo "--- GPU Telemetry ---"
    ssh "${P100_HOST}" "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader"
    echo ""
    echo "--- Recent Hybrid Log ---"
    ssh "${P100_HOST}" "tail -c 600 ${REMOTE_DIR}/hybrid.log 2>/dev/null | tr '\r' '\n' | grep -E '(Epoch|mIoU|Loss|Best|Step|Dice|Archiving)' | tail -n 8 || true"
    echo ""
    echo "To attach interactively, run:"
    echo "  ssh -t ${P100_HOST} 'tmux attach -t oct_hybrid'"
    exit 0
fi

echo "[*] Pulling latest code and configs from GitHub on P100..."
ssh "${P100_HOST}" "cd ${REMOTE_DIR} && git pull origin main"

echo "[*] Launching hybrid pipeline in detached tmux session 'oct_hybrid'..."
if [ $# -gt 0 ]; then
    ssh "${P100_HOST}" "tmux new-session -d -s oct_hybrid 'cd ${REMOTE_DIR} && ./run_hybrid_p100.sh $* 2>&1 | tee hybrid.log'"
else
    ssh "${P100_HOST}" "tmux new-session -d -s oct_hybrid 'cd ${REMOTE_DIR} && ./run_hybrid_p100.sh 2>&1 | tee hybrid.log'"
fi

echo "[*] Waiting 6 seconds for startup initialization..."
sleep 6

echo ""
echo "--- Initial Startup Log ---"
ssh "${P100_HOST}" "tail -n 15 ${REMOTE_DIR}/hybrid.log 2>/dev/null || true"
echo ""
echo "--- GPU Telemetry ---"
ssh "${P100_HOST}" "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader"

echo ""
echo "=========================================================="
echo "Hybrid pipeline started successfully on Tesla P100!"
echo "Session: ssh -t ${P100_HOST} 'tmux attach -t oct_hybrid'"
echo "Log:     ssh ${P100_HOST} 'tail -f ${REMOTE_DIR}/hybrid.log'"
echo "=========================================================="
