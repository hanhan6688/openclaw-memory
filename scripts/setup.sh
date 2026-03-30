#!/bin/bash
# OpenClaw Memory System 安装脚本
# ================================

PROJECT_DIR="/Users/apple/.openclaw/memory_system"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== OpenClaw Memory System 安装 ==="
echo ""

# 检查 Python 3.10
PYTHON_PATH="/opt/homebrew/bin/python3.10"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ 未找到 Python 3.10"
    echo "   请先安装: brew install python@3.10"
    exit 1
fi

echo "✅ Python: $($PYTHON_PATH --version)"

# 创建 venv
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  venv 已存在，删除重建..."
    rm -rf "$VENV_DIR"
fi

echo "📦 创建虚拟环境..."
$PYTHON_PATH -m venv "$VENV_DIR"

# 安装依赖
echo "📥 安装依赖..."
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -i https://mirrors.aliyun.com/pypi/simple/

echo ""
echo "✅ 安装完成!"
echo "   启动服务: $PROJECT_DIR/scripts/start.sh"
