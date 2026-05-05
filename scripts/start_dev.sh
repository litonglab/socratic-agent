#!/usr/bin/env bash
# 开发模式：uvicorn 热重载（后端）
# 前端 React (vite) 请单独在 frontend/ 目录运行：
#   cd frontend && npm install && npm run dev
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 开发时跳过索引重建，并启用后台预热（服务器立即响应，首次请求有冷启动，可接受）
export RAG_REBUILD_INDEX=0
export RAG_DEV_FAST_START=1

echo "[dev] 启动后端（--reload）..."
uvicorn server:app --reload --port 8000
