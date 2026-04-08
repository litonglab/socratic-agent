#!/usr/bin/env bash
# 开发模式：uvicorn 热重载 + Streamlit 自动刷新
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 开发时跳过索引重建，并启用后台预热（服务器立即响应，首次请求有冷启动，可接受）
export RAG_REBUILD_INDEX=0
export RAG_DEV_FAST_START=1

echo "[dev] 启动后端（--reload）..."
uvicorn server:app --reload --port 8000 &
BACK_PID=$!

echo "[dev] 等待后端就绪..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[dev] 后端就绪（${i}s），启动前端..."
    break
  fi
  [ "$i" -eq 30 ] && { echo "[dev] 后端超时，退出"; kill "$BACK_PID"; exit 1; }
  sleep 1
done

streamlit run app_streamlit.py &
FRONT_PID=$!

cleanup() { kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true; }
trap cleanup EXIT
wait
