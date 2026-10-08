@echo off
setlocal
cd /d %~dp0

echo ==========================================================
echo  NVIDIA GPU Recovery Launcher
echo ==========================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0recover_gpu.ps1" %*

if errorlevel 1 (
    echo.
    echo Recovery finished with non-zero exit code.
)

pause
