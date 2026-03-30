#!/bin/bash
# 记忆系统启动脚本 - 随 OpenClaw 启动

MEMORY_DIR="$HOME/.openclaw/memory_system"
LOG_FILE="$MEMORY_DIR/logs/api.log"
PID_FILE="$MEMORY_DIR/api.pid"

start() {
    # 检查是否已运行
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Memory API already running (PID: $PID)"
            return 0
        fi
    fi
    
    cd "$MEMORY_DIR"
    source venv/bin/activate
    
    # 启动 API
    nohup python main.py api --port 8082 >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Memory API started (PID: $(cat $PID_FILE))"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill $PID 2>/dev/null
        rm -f "$PID_FILE"
        echo "Memory API stopped"
    fi
}

case "$1" in
    start) start ;;
    stop) stop ;;
    restart) stop; sleep 1; start ;;
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "Running (PID: $PID)"
            else
                echo "Not running (stale PID file)"
            fi
        else
            echo "Not running"
        fi
        ;;
    *) echo "Usage: $0 {start|stop|restart|status}" ;;
esac
