#!/usr/bin/env bash
# 生产模式：uvicorn 的 lifespan 会阻塞直到 RAG 完全就绪后才开放端口
# 因此 /health 能通 == RAG 已就绪，无需二阶段等待
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uvicorn server:app --port 8000 &
BACK_PID=$!

echo "等待后端就绪（包含 RAG 模型加载，最多 180 秒）..."
for i in $(seq 1 180); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "后端就绪（${i}s），启动前端..."
    break
  fi
  [ "$i" -eq 180 ] && { echo "后端超时，退出"; kill "$BACK_PID"; exit 1; }
  sleep 1
done

streamlit run app_streamlit.py &
FRONT_PID=$!

cleanup() { kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true; }
trap cleanup EXIT
wait
