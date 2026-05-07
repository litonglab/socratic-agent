#!/usr/bin/env bash
# 开发模式：uvicorn 热重载（后端）
# 前端 React (vite) 请单独在 frontend/ 目录运行：
#   cd frontend && npm install && npm run dev
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]] && [[ ! -f .env ]]; then
  echo "[dev] 未检测到 DEEPSEEK_API_KEY，且仓库根目录没有 .env。" >&2
  echo "[dev] 请先: cp .env.example .env 并在 .env 里填写正确的 DEEPSEEK_API_KEY=sk-..." >&2
  echo "[dev] 或在同一终端先: export DEEPSEEK_API_KEY='sk-...' 再运行本脚本。" >&2
  exit 1
fi

# 与 start_wsl 一致：把 .env 导出到当前 Shell，uvicorn 子进程可直接继承（避免仅靠 Python load_dotenv 的时机差异）
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[dev] 虽已存在 .env，但未读到 DEEPSEEK_API_KEY。请检查变量名是否为 DEEPSEEK_API_KEY（勿写成 DEEPSEKK_API_KEY）。" >&2
  exit 1
fi

# 开发时跳过索引重建，并启用后台预热（服务器立即响应，首次请求有冷启动，可接受）
export RAG_REBUILD_INDEX=0
export RAG_DEV_FAST_START=1

echo "[dev] 启动后端（--reload）..."
uvicorn server:app --reload --port 8000
