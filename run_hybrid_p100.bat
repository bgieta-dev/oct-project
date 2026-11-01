@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

set "BASE=outputs\runs\2026-10-27_14-57-10_test20\best_model.pth"
set "EXTRA_ARGS="

if not "%~1"=="" (
    set "ARG1=%~1"
    if "!ARG1:~0,2!"=="--" (
        set "EXTRA_ARGS=%*"
    ) else (
        set "BASE=%~1"
        shift
        :collect_args
        if "%~1"=="" goto run_pipeline
        set "EXTRA_ARGS=!EXTRA_ARGS! %1"
        shift
        goto collect_args
    )
)

:run_pipeline
echo ==================================================
echo Starting oct-project hybrid pipeline
echo Base weights: !BASE!
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

python -m octseg.pipeline_hybrid --base-weights "!BASE!" --promote !EXTRA_ARGS!
if errorlevel 1 (
    echo.
    echo [ERROR] Hybrid pipeline failed with exit code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo.
echo [*] Pipeline finished successfully. Running git_update.bat to push artifacts...
if exist git_update.bat (
    call git_update.bat "feat(hybrid): promote test22 hybrid experiment run artifacts"
) else if exist scripts\git_update.bat (
    call scripts\git_update.bat "feat(hybrid): promote test22 hybrid experiment run artifacts"
)

echo.
echo [*] Done! All hybrid experiment results committed and pushed to GitHub.
pause
