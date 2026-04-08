#!/usr/bin/env bash
# 一键将 RAG-Agent 注册为 systemd 服务（开机自启 + 崩溃自动重启）
# 用法：bash scripts/install_service.sh
set -euo pipefail

# ---------- 自动检测当前路径和用户 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CURRENT_USER="$(whoami)"
SERVICE_NAME="rag-agent"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=== RAG-Agent 服务安装 ==="
echo "  项目路径 : $ROOT_DIR"
echo "  运行用户 : $CURRENT_USER"
echo "  服务文件 : $SERVICE_DST"
echo ""

# ---------- 检查 systemd ----------
if ! command -v systemctl &>/dev/null; then
  echo "❌ 当前系统不支持 systemd，无法安装"
  exit 1
fi

# ---------- 检查 start_all.sh 可执行 ----------
chmod +x "$ROOT_DIR/scripts/start_all.sh"

# ---------- 生成服务文件（自动填入路径/用户） ----------
cat > /tmp/${SERVICE_NAME}.service <<EOF
[Unit]
Description=RAG Agent (FastAPI + Streamlit)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${ROOT_DIR}
ExecStart=/bin/bash ${ROOT_DIR}/scripts/start_all.sh

Restart=on-failure
RestartSec=5s
StartLimitBurst=3
StartLimitIntervalSec=60s

StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

# ---------- 安装 ----------
sudo cp /tmp/${SERVICE_NAME}.service "$SERVICE_DST"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo ""
echo "✅ 安装完成"
echo ""
echo "常用命令："
echo "  sudo systemctl status  $SERVICE_NAME    # 查看状态"
echo "  sudo systemctl restart $SERVICE_NAME    # 重启"
echo "  sudo systemctl stop    $SERVICE_NAME    # 停止"
echo "  journalctl -u $SERVICE_NAME -f          # 实时日志"
