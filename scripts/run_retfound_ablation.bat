@echo off
REM RETFound ablation: retfound_lora (test20) -> retfound_lora_nodetail (test21) -> retfound_frozen (test22), one by one.
REM Extra args are forwarded, e.g.:  run_retfound_ablation.bat --push      or      run_retfound_ablation.bat --epochs 1
cd /d %~dp0..
call venv\Scripts\activate
python -m octseg.run_series --configs configs/retfound_lora.yaml configs/retfound_lora_nodetail.yaml configs/retfound_frozen.yaml %*
pause
