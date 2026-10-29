@echo off
setlocal enabledelayedexpansion

set "P100_HOST=bgieta@192.168.1.131"
set "REMOTE_DIR=/home/bgieta/oct-project"
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=configs/test21.yaml"

echo ==========================================================
echo Connecting to Tesla P100 host: %P100_HOST%
echo Remote Directory: %REMOTE_DIR%
echo Config Target: %CONFIG%
echo ==========================================================

REM Check SSH connectivity
ssh -o ConnectTimeout=5 -o BatchMode=yes %P100_HOST% "exit" 2>nul
if errorlevel 1 (
    echo [!] Error: Cannot connect to %P100_HOST% via SSH.
    echo Please verify host is reachable and SSH keys are configured.
    pause
    exit /b 1
)

REM Check if session is already running
for /f "tokens=*" %%a in ('ssh %P100_HOST% "tmux list-sessions 2>/dev/null | grep oct_training || true"') do set "ALREADY_RUNNING=%%a"

if not "%ALREADY_RUNNING%"=="" (
    echo [*] Training session 'oct_training' is ALREADY RUNNING on P100.
    echo.
    echo --- GPU Telemetry ---
    ssh %P100_HOST% "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader"
    echo.
    echo --- Recent Training Log ---
    ssh %P100_HOST% "tail -c 600 %REMOTE_DIR%/train.log | tr '\r' '\n' | grep -E '(Epoch|mIoU|Loss|Best)' | tail -n 8"
    echo.
    echo To attach interactively, run:
    echo   ssh -t %P100_HOST% "tmux attach -t oct_training"
    pause
    exit /b 0
)

echo [*] Pulling latest code and configs from GitHub on P100...
ssh %P100_HOST% "cd %REMOTE_DIR% && git pull origin main"

echo [*] Launching training in detached tmux session 'oct_training'...
ssh %P100_HOST% "tmux new-session -d -s oct_training 'cd %REMOTE_DIR% && ./run_p100.sh %CONFIG% 2>&1 | tee train.log'"

echo [*] Waiting 6 seconds for startup initialization...
timeout /t 6 /nobreak >nul

echo.
echo --- Initial Startup Log ---
ssh %P100_HOST% "tail -n 15 %REMOTE_DIR%/train.log"
echo.
echo --- GPU Telemetry ---
ssh %P100_HOST% "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader"

echo.
echo ==========================================================
echo Training started successfully on Tesla P100!
echo Session: ssh -t %P100_HOST% "tmux attach -t oct_training"
echo Log:     ssh %P100_HOST% "tail -f %REMOTE_DIR%/train.log"
echo ==========================================================
pause
