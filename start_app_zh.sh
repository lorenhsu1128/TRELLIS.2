#!/usr/bin/env bash
# Launch the Traditional Chinese TRELLIS.2 web app (app_zh.py) inside WSL, reachable from the LAN.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
cd "$(dirname "$(readlink -f "$0")")"
export APP_PORT="${APP_PORT:-7860}"
export PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False
# Fixed mmap threshold: large tensor buffers are mmapped and returned to the OS when freed
# (glibc otherwise raises the threshold dynamically and the heap fragments until OOM)
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
echo "============================================================"
echo " TRELLIS.2 圖片轉 3D 啟動中（首次載入模型約需 1 分鐘）"
echo " 本機：      http://localhost:${APP_PORT}"
for ip in $(hostname -I); do
  case "$ip" in 127.*|172.*|169.254.*) ;; *) echo " 區域網路：  http://${ip}:${APP_PORT}" ;; esac
done
echo " 關閉服務：  在此視窗按 Ctrl+C"
echo "============================================================"
exec python app_zh.py
