#!/bin/bash
# OpenClaw Memory System 后台启动脚本
# ================================

PROJECT_DIR="/Users/apple/.openclaw/memory_system"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_FILE="/tmp/memory_api.log"

# 检查是否已运行
if pgrep -f "python.*main.py.*api" > /dev/null; then
    echo "⚠️  服务已在运行中"
    echo "   查看日志: tail -f $LOG_FILE"
    exit 0
fi

# 启动服务（自动包含实时同步）
cd "$PROJECT_DIR"
nohup "$VENV_PYTHON" main.py api > "$LOG_FILE" 2>&1 &

sleep 3

# 检查是否启动成功
if pgrep -f "python.*main.py.*api" > /dev/null; then
    echo "✅ 服务已启动"
    echo ""
    echo "   📡 API: http://localhost:8082"
    echo "   🔄 实时同步: 已启用"
    echo "   📊 日志: tail -f $LOG_FILE"
    echo ""
    echo "   健康检查: curl http://localhost:8082/health"
else
    echo "❌ 启动失败，查看日志: cat $LOG_FILE"
fi