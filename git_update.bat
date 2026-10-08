@echo off
setlocal enabledelayedexpansion

cd /d %~dp0

for /f "tokens=*" %%i in ('git symbolic-ref --short HEAD 2^>nul') do set MAIN_BRANCH=%%i
if "%MAIN_BRANCH%"=="" set MAIN_BRANCH=main

git add .
git diff-index --quiet HEAD
if errorlevel 1 (
    set "COMMIT_MSG=%~1"
    if "!COMMIT_MSG!"=="" set "COMMIT_MSG=chore: sync main repo"
    git commit -m "!COMMIT_MSG!"
    git pull --rebase origin %MAIN_BRANCH%
    if errorlevel 1 (
        git rebase --abort 2>nul
        echo Failed to sync main repo. Manual intervention may be required.
        exit /b 1
    )
    git push origin %MAIN_BRANCH%
) else (
    echo No changes in main repo
)

echo All synced.
pause
