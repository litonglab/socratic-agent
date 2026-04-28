#!/usr/bin/env bash
# WSL 稳定启动脚本：
# - 自动加载 .venv 和 .env
# - 优先使用本地 embedding 模型目录
# - 默认禁用 reranker，避免二次下载和额外初始化负担
# - Streamlit 关闭 file watcher，规避 transformers/torchvision 懒加载问题
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 如果当前 shell 处于 conda 环境，先尽量退出，避免动态库和 PATH 污染。
if [ -n "${CONDA_PREFIX:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    # conda 是 shell function 时，deactivate 会直接修改当前 shell 环境。
    while [ -n "${CONDA_PREFIX:-}" ]; do
      conda deactivate >/dev/null 2>&1 || break
    done
  fi
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL
fi

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if [ -d "$ROOT_DIR/models/bge-m3" ] && [ -z "${EMBEDDING_MODEL_NAME:-}" ]; then
  export EMBEDDING_MODEL_NAME="$ROOT_DIR/models/bge-m3"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DISABLE_RERANKER="${DISABLE_RERANKER:-1}"
export RAG_REBUILD_INDEX="${RAG_REBUILD_INDEX:-0}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
unset LD_LIBRARY_PATH PYTHONPATH

BACK_HOST="${BACK_HOST:-0.0.0.0}"
BACK_PORT="${BACK_PORT:-8000}"
FRONT_HOST="${FRONT_HOST:-0.0.0.0}"
FRONT_PORT="${FRONT_PORT:-8501}"

echo "[wsl] root: $ROOT_DIR"
echo "[wsl] backend: http://$BACK_HOST:$BACK_PORT"
echo "[wsl] frontend: http://$FRONT_HOST:$FRONT_PORT"
echo "[wsl] embedding: ${EMBEDDING_MODEL_NAME:-BAAI/bge-m3}"
echo "[wsl] reranker disabled: ${DISABLE_RERANKER}"

cleanup() {
  local code=$?
  if [ -n "${FRONT_PID:-}" ]; then
    kill "$FRONT_PID" 2>/dev/null || true
  fi
  if [ -n "${BACK_PID:-}" ]; then
    kill "$BACK_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "[wsl] starting backend..."
uvicorn server:app --host "$BACK_HOST" --port "$BACK_PORT" &
BACK_PID=$!

echo "[wsl] waiting for backend ready..."
for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:${BACK_PORT}/health" >/dev/null 2>&1; then
    echo "[wsl] backend ready (${i}s), starting frontend..."
    break
  fi
  if [ "$i" -eq 300 ]; then
    echo "[wsl] backend timeout, exiting"
    exit 1
  fi
  sleep 1
done

streamlit run app_streamlit.py \
  --server.address "$FRONT_HOST" \
  --server.port "$FRONT_PORT" \
  --server.headless true \
  --server.fileWatcherType none &
FRONT_PID=$!

echo "[wsl] services started"
echo "[wsl] open in Windows browser: http://localhost:${FRONT_PORT}"

wait
