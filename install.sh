#!/bin/bash
# OpenClaw Memory System 安装脚本
# ================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCLAW_DIR="${OPENCLAW_DIR:-$HOME/.openclaw}"
EXTENSIONS_DIR="$OPENCLAW_DIR/extensions"
MEMORY_SYSTEM_DIR="$OPENCLAW_DIR/memory_system"

echo ""
echo "=============================================="
echo "  OpenClaw Memory System 安装"
echo "=============================================="
echo ""

# 检查 Python
echo "🔍 检查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装 Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   ✅ Python: $PYTHON_VERSION"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未安装"
    exit 1
fi
echo "   ✅ pip: $(pip3 --version | awk '{print $2}')"

# 检查 Docker
echo ""
echo "🔍 检查 Docker..."
if ! command -v docker &> /dev/null; then
    echo "   ⚠️ Docker 未安装（Weaviate 需要 Docker）"
    echo "   安装方法: brew install docker 或访问 https://docs.docker.com/get-docker/"
else
    echo "   ✅ Docker: $(docker --version | awk '{print $3}' | tr -d ',')"
fi

# 检查 Ollama
echo ""
echo "🔍 检查 Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "   ⚠️ Ollama 未安装（用于嵌入和摘要）"
    echo "   安装方法: brew install ollama 或访问 https://ollama.ai"
else
    echo "   ✅ Ollama: $(ollama --version 2>/dev/null | awk '{print $3}' || echo 'installed')"
fi

echo ""
echo "📦 安装依赖..."

# 创建虚拟环境
if [ ! -d "$MEMORY_SYSTEM_DIR/venv" ]; then
    echo "   创建 Python 虚拟环境..."
    cd "$MEMORY_SYSTEM_DIR"
    python3 -m venv venv
fi

# 安装 Python 依赖
echo "   安装 Python 包..."
cd "$MEMORY_SYSTEM_DIR"
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
deactivate

echo "   ✅ Python 依赖安装完成"

# 安装 OpenClaw 插件
echo ""
echo "📦 安装 OpenClaw 插件..."

# 创建扩展目录
mkdir -p "$EXTENSIONS_DIR"

# 复制插件文件（如果插件目录不在当前目录）
if [ -d "$SCRIPT_DIR/../extensions/memory-weaviate" ]; then
    cp -r "$SCRIPT_DIR/../extensions/memory-weaviate" "$EXTENSIONS_DIR/"
    echo "   ✅ 插件已复制到 $EXTENSIONS_DIR/memory-weaviate"
elif [ -d "$OPENCLAW_DIR/extensions/memory-weaviate" ]; then
    echo "   ✅ 插件已存在于 $EXTENSIONS_DIR/memory-weaviate"
else
    echo "   ⚠️ 插件目录未找到，请手动复制"
fi

echo ""
echo "=============================================="
echo "  配置 OpenClaw"
echo "=============================================="
echo ""

# 检查 openclaw.json
OPENCLAW_CONFIG="$OPENCLAW_DIR/openclaw.json"
if [ -f "$OPENCLAW_CONFIG" ]; then
    # 检查是否已配置 memory-weaviate
    if grep -q '"memory-weaviate"' "$OPENCLAW_CONFIG"; then
        echo "   ✅ memory-weaviate 已在配置中"
    else
        echo "   添加 memory-weaviate 到配置..."
        # 使用 Python 来更新 JSON（更安全）
        python3 << 'EOF'
import json
import os

config_path = os.path.expanduser("~/.openclaw/openclaw.json")
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except:
    config = {}

# 确保 plugins 部分存在
if 'plugins' not in config:
    config['plugins'] = {}
if 'entries' not in config['plugins']:
    config['plugins']['entries'] = {}
if 'slots' not in config['plugins']:
    config['plugins']['slots'] = {}

# 添加 memory-weaviate 配置
config['plugins']['slots']['memory'] = 'memory-weaviate'
config['plugins']['entries']['memory-weaviate'] = {
    'enabled': True,
    'config': {
        'autoSync': True,
        'autoRecall': True,
        'autoCapture': True,
        'syncInterval': 30
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("   ✅ 配置已更新")
EOF
    fi
else
    echo "   ⚠️ OpenClaw 配置文件不存在，请先运行 openclaw"
fi

echo ""
echo "=============================================="
echo "  启动服务"
echo "=============================================="
echo ""

# 启动 Weaviate
echo "🐳 检查 Weaviate..."
if docker ps | grep -q weaviate; then
    echo "   ✅ Weaviate 已运行"
else
    echo "   启动 Weaviate..."
    docker run -d \
        --name weaviate \
        -p 8080:8080 \
        -p 50051:50051 \
        --restart unless-stopped \
        cr.weaviate.io/semitechnologies/weaviate:latest 2>/dev/null || \
    docker start weaviate 2>/dev/null || \
    echo "   ⚠️ 无法启动 Weaviate，请手动启动"
fi

# 检查 Ollama 模型
echo ""
echo "🦙 检查 Ollama 模型..."
if command -v ollama &> /dev/null; then
    # 检查嵌入模型
    if ! ollama list | grep -q "nomic-embed-text"; then
        echo "   下载 nomic-embed-text..."
        ollama pull nomic-embed-text
    else
        echo "   ✅ nomic-embed-text 已安装"
    fi
    
    # 检查对话模型
    if ! ollama list | grep -q "llama3.2:3b"; then
        echo "   下载 llama3.2:3b..."
        ollama pull llama3.2:3b
    else
        echo "   ✅ llama3.2:3b 已安装"
    fi
fi

echo ""
echo "=============================================="
echo "  ✅ 安装完成！"
echo "=============================================="
echo ""
echo "下一步："
echo ""
echo "1. 重启 OpenClaw Gateway:"
echo "   openclaw gateway restart"
echo ""
echo "2. 启动记忆系统后端:"
echo "   cd ~/.openclaw/memory_system"
echo "   source venv/bin/activate"
echo "   python main.py api"
echo ""
echo "3. 测试记忆功能:"
echo "   在对话中说: \"记住我喜欢用 Python\""
echo "   然后问: \"我喜欢什么语言？\""
echo ""
echo "4. CLI 命令:"
echo "   openclaw wmem status   # 查看状态"
echo "   openclaw wmem search <query>  # 搜索记忆"
echo ""