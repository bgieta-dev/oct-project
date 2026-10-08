@echo off
cd /d %~dp0..
call git_update.bat %*
