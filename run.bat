@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

set "FIRST_ARG=%~1"

if "%FIRST_ARG%"=="" (
    echo =======================================================
    echo Starting oct-project training pipeline on Windows
    echo Config: configs/test21.yaml (default)
    echo =======================================================
    python -m octseg.pipeline --config configs/test21.yaml --promote
) else (
    if "%FIRST_ARG:~0,2%"=="--" (
        echo =======================================================
        echo Starting oct-project training pipeline on Windows
        echo Arguments: %*
        echo =======================================================
        python -m octseg.pipeline --promote %*
    ) else (
        set "CONFIG=%FIRST_ARG%"
        shift
        set "REST_ARGS="
        :collect_args
        if "%~1"=="" goto run_pipeline
        set "REST_ARGS=!REST_ARGS! %1"
        shift
        goto collect_args

        :run_pipeline
        echo =======================================================
        echo Starting oct-project training pipeline on Windows
        echo Config: !CONFIG!
        echo =======================================================
        python -m octseg.pipeline --config "!CONFIG!" --promote !REST_ARGS!
    )
)

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed with exit code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo.
echo [*] Pipeline finished successfully. Running git_update.bat to push artifacts...
call git_update.bat "feat(windows): promote experiment run artifacts"

echo.
echo [*] Done! All experiment results committed and pushed to GitHub.
pause
