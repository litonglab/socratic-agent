#!/usr/bin/env bash
# 生产模式：uvicorn 的 lifespan 会阻塞直到 RAG 完全就绪后才开放端口
# 因此 /health 能通 == RAG 已就绪
# 前端请在 frontend/ 目录用 `npm run build` 打包后由 Nginx / FastAPI 反代静态文件
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uvicorn server:app --port 8000
