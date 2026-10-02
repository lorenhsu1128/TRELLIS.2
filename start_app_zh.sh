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
# GLB 壓縮 tab: Node.js deps of the vendored glb-shrink pipeline
if command -v node >/dev/null && [ ! -d tools/glb_shrink/node_modules ]; then
  echo "安裝 GLB 壓縮所需的 Node.js 套件…"
  (cd tools/glb_shrink && npm install --no-audit --no-fund --loglevel=error)
fi
echo "============================================================"
echo " TRELLIS.2 圖片轉 3D 啟動中（首次載入模型約需 1 分鐘）"
echo " 本機：      http://localhost:${APP_PORT}"
for ip in $(hostname -I); do
  case "$ip" in 127.*|172.*|169.254.*) ;; *) echo " 區域網路：  http://${ip}:${APP_PORT}" ;; esac
done
echo " 關閉服務：  在此視窗按 Ctrl+C（直接關閉視窗也會停止服務）"
echo "============================================================"
# Server output goes to a log file and this window only tails it: selecting text in a Windows
# console (QuickEdit) pauses its output, which would otherwise block the server mid-request.
LOG_DIR="$HOME/.cache/trellis2_app_zh"
LOG="$LOG_DIR/server.log"
mkdir -p "$LOG_DIR"
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
echo " 伺服器紀錄： $LOG"
python app_zh.py > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' INT TERM EXIT
tail -n +1 -F --pid="$SERVER_PID" "$LOG" 2>/dev/null
rc=0
wait "$SERVER_PID" || rc=$?
echo "伺服器已停止（結束代碼 $rc），詳細紀錄見 $LOG"
