@echo off
chcp 65001 > nul
title TRELLIS.2 App
rem Runs in the default WSL distro; set TRELLIS_WSL_DISTRO to pick another (e.g. Ubuntu-22.04)
if defined TRELLIS_WSL_DISTRO (
  wsl.exe -d %TRELLIS_WSL_DISTRO% --cd "%~dp0." -- bash -lc "bash ./start_app_zh.sh"
) else (
  wsl.exe --cd "%~dp0." -- bash -lc "bash ./start_app_zh.sh"
)
pause
