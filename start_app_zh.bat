@echo off
chcp 65001 > nul
title TRELLIS.2 App
wsl.exe -d Ubuntu-22.04 -u loren -- bash -lc "bash /mnt/c/Users/ADMIN/Desktop/TRELLIS.2/start_app_zh.sh"
pause
