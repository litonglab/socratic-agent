#!/usr/bin/env bash
# 生产模式：uvicorn 的 lifespan 会阻塞直到 RAG 完全就绪后才开放端口
# 因此 /health 能通 == RAG 已就绪
# 前端请在 frontend/ 目录用 `npm run build` 打包后由 Nginx / FastAPI 反代静态文件
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]] && [[ ! -f .env ]]; then
  echo "[server] 未检测到 DEEPSEEK_API_KEY，且仓库根目录没有 .env。" >&2
  echo "[server] 请先: cp .env.example .env 并在 .env 里填写正确的 DEEPSEEK_API_KEY=sk-..." >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[server] .env 中缺少 DEEPSEEK_API_KEY 或变量名错误（应为 DEEPSEEK_API_KEY）。" >&2
  exit 1
fi

uvicorn server:app --port 8000
