#!/usr/bin/env bash
# RETFound ablation: retfound_lora -> retfound_lora_nodetail -> retfound_frozen, one by one (testN, N+1, N+2).
# Extra args are forwarded, e.g.: scripts/run_retfound_ablation.sh --push   |   --epochs 1
cd "$(dirname "$0")/.." && source venv/bin/activate && python -m octseg.run_series \
    --configs configs/retfound_lora.yaml configs/retfound_lora_nodetail.yaml configs/retfound_frozen.yaml "$@"
