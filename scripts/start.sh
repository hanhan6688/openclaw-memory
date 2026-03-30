#!/bin/bash
# OpenClaw Memory System 启动脚本
# ================================

# 项目根目录
PROJECT_DIR="/Users/apple/.openclaw/memory_system"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

# 检查 venv 是否存在
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ venv 不存在，请先运行 setup.sh"
    exit 1
fi

# 切换到项目目录
cd "$PROJECT_DIR"

# 启动 API 服务
echo "🚀 启动 OpenClaw Memory System..."
echo "   Python: $($VENV_PYTHON --version)"
echo "   端口: 8082"
echo ""

exec "$VENV_PYTHON" main.py api
