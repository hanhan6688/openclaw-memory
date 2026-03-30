#!/bin/bash
# OpenClaw Memory System 停止脚本
# ================================

if pgrep -f "python.*main.py.*api" > /dev/null; then
    pkill -f "python.*main.py.*api"
    echo "✅ 服务已停止"
else
    echo "⚠️  服务未运行"
fi
