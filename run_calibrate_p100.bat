@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

set "BASE_WEIGHTS=outputs\runs\2026-10-27_14-57-10_test20\best_model.pth"
set "EXTRA_ARGS="

if not "%~1"=="" (
    set "ARG1=%~1"
    if "!ARG1:~0,2!"=="--" (
        set "EXTRA_ARGS=%*"
    ) else (
        set "BASE_WEIGHTS=%~1"
        shift
        :collect_args
        if "%~1"=="" goto run_calibrate
        set "EXTRA_ARGS=!EXTRA_ARGS! %1"
        shift
        goto collect_args
    )
)

:run_calibrate
echo ==================================================
echo Starting oct-project calibration sweep
echo Base weights: !BASE_WEIGHTS!
echo ==================================================

REM Stop inference worker container if docker is running
where docker >nul 2>nul
if %errorlevel% equ 0 (
    docker ps --format "{{.Names}}" 2>nul | findstr /i "qwen-inference-p100" >nul
    if %errorlevel% equ 0 (
        echo [*] Stopping qwen-inference-p100 to free GPU VRAM...
        docker stop qwen-inference-p100 >nul 2>&1
    )
)

python -m octseg.calibrate_postprocess --base-weights "!BASE_WEIGHTS!" !EXTRA_ARGS!
if errorlevel 1 (
    echo.
    echo [ERROR] Calibration sweep failed with exit code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo.
echo [*] Calibration finished successfully. Running git_update.bat to push artifacts...
if exist git_update.bat (
    call git_update.bat "feat(calibrate): commit calibration sweep artifacts"
) else if exist scripts\git_update.bat (
    call scripts\git_update.bat "feat(calibrate): commit calibration sweep artifacts"
)

echo.
echo [*] Done! All calibration results committed and pushed to GitHub.
pause
