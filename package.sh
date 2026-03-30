#!/bin/bash
# OpenClaw Memory System 打包脚本
# 用于创建可分发的压缩包
# ================================

set -e

VERSION=${1:-"1.0.0"}
OUTPUT_DIR="$(pwd)"
TEMP_DIR=$(mktemp -d)
PACKAGE_NAME="openclaw-memory-weaviate-${VERSION}"

echo ""
echo "=============================================="
echo "  打包 OpenClaw Memory System v${VERSION}"
echo "=============================================="
echo ""

# 创建临时目录结构
mkdir -p "$TEMP_DIR/$PACKAGE_NAME"

# 复制 memory_system
echo "📦 复制 memory_system..."
cp -r "$HOME/.openclaw/memory_system" "$TEMP_DIR/$PACKAGE_NAME/"
rm -rf "$TEMP_DIR/$PACKAGE_NAME/memory_system/venv" 2>/dev/null || true
rm -rf "$TEMP_DIR/$PACKAGE_NAME/memory_system/__pycache__" 2>/dev/null || true
rm -rf "$TEMP_DIR/$PACKAGE_NAME/memory_system/.git" 2>/dev/null || true
rm -f "$TEMP_DIR/$PACKAGE_NAME/memory_system/*.pyc" 2>/dev/null || true

# 复制插件
echo "📦 复制 memory-weaviate 插件..."
mkdir -p "$TEMP_DIR/$PACKAGE_NAME/extensions"
cp -r "$HOME/.openclaw/extensions/memory-weaviate" "$TEMP_DIR/$PACKAGE_NAME/extensions/"

# 创建安装脚本
echo "📝 创建安装脚本..."
cat > "$TEMP_DIR/$PACKAGE_NAME/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# OpenClaw Memory System 安装脚本
# ================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCLAW_DIR="$HOME/.openclaw"
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
    echo "   ✅ Ollama 已安装"
fi

echo ""
echo "📦 安装文件..."

# 复制 memory_system
echo "   复制 memory_system..."
rm -rf "$MEMORY_SYSTEM_DIR" 2>/dev/null || true
cp -r "$SCRIPT_DIR/memory_system" "$MEMORY_SYSTEM_DIR/"

# 复制插件
echo "   复制 memory-weaviate 插件..."
mkdir -p "$EXTENSIONS_DIR"
cp -r "$SCRIPT_DIR/extensions/memory-weaviate" "$EXTENSIONS_DIR/"

# 创建虚拟环境
echo ""
echo "📦 创建 Python 虚拟环境..."
cd "$MEMORY_SYSTEM_DIR"
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
deactivate
echo "   ✅ 依赖安装完成"

# 更新 OpenClaw 配置
echo ""
echo "📝 配置 OpenClaw..."
python3 << 'EOF'
import json
import os

config_path = os.path.expanduser("~/.openclaw/openclaw.json")
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except:
    config = {}

if 'plugins' not in config:
    config['plugins'] = {}
if 'entries' not in config['plugins']:
    config['plugins']['entries'] = {}
if 'slots' not in config['plugins']:
    config['plugins']['slots'] = {}

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

# 启动 Weaviate
echo ""
echo "🐳 启动 Weaviate..."
if docker ps | grep -q weaviate; then
    echo "   ✅ Weaviate 已运行"
else
    docker run -d \
        --name weaviate \
        -p 8080:8080 \
        -p 50051:50051 \
        --restart unless-stopped \
        cr.weaviate.io/semitechnologies/weaviate:latest 2>/dev/null || \
    docker start weaviate 2>/dev/null || \
    echo "   ⚠️ 请手动启动 Weaviate"
fi

# 检查 Ollama 模型
echo ""
echo "🦙 检查 Ollama 模型..."
if command -v ollama &> /dev/null; then
    ollama pull nomic-embed-text 2>/dev/null || true
    ollama pull llama3.2:3b 2>/dev/null || true
    echo "   ✅ 模型已准备好"
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
echo "2. 启动记忆系统后端（可选，插件会自动启动）:"
echo "   cd ~/.openclaw/memory_system && source venv/bin/activate && python main.py api"
echo ""
INSTALL_EOF
chmod +x "$TEMP_DIR/$PACKAGE_NAME/install.sh"

# 创建快速启动脚本
echo "📝 创建启动脚本..."
cat > "$TEMP_DIR/$PACKAGE_NAME/start.sh" << 'START_EOF'
#!/bin/bash
# 快速启动脚本
# ================================

cd "$HOME/.openclaw/memory_system"
source venv/bin/activate
python main.py api
START_EOF
chmod +x "$TEMP_DIR/$PACKAGE_NAME/start.sh"

# 创建 README
echo "📝 创建 README..."
cat > "$TEMP_DIR/$PACKAGE_NAME/README.md" << 'README_EOF'
# OpenClaw Memory System

基于 Weaviate 的长期记忆系统，与 OpenClaw 无缝集成。

## 特性

- 🧠 **向量记忆存储** - 语义搜索，支持混合检索
- 🕸️ **知识图谱** - 自动提取实体关系
- 🔄 **实时同步** - 自动同步会话消息
- 🎯 **Auto-Recall** - 自动注入相关记忆
- 📤 **Auto-Capture** - 自动捕获重要信息

## 快速开始

### 1. 解压并安装

```bash
tar -xzf openclaw-memory-weaviate-*.tar.gz
cd openclaw-memory-weaviate-*
./install.sh
```

### 2. 重启 OpenClaw

```bash
openclaw gateway restart
```

### 3. 测试

在对话中说：
- "记住我喜欢用 Python"
- 然后问 "我喜欢什么编程语言？"

## 系统要求

- Python 3.10+
- Docker (Weaviate)
- Ollama (可选，用于嵌入和摘要)

## CLI 命令

```bash
# 查看状态
openclaw wmem status

# 搜索记忆
openclaw wmem search "查询内容"

# 同步会话
openclaw wmem sync
```

## 工具

- `memory_recall` - 搜索记忆
- `memory_store` - 存储信息
- `kg_search` - 搜索知识图谱
- `memory_sync` - 手动同步

## 配置

在 `~/.openclaw/openclaw.json` 中：

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-weaviate"
    },
    "entries": {
      "memory-weaviate": {
        "enabled": true,
        "config": {
          "autoSync": true,
          "autoRecall": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

## 手动启动后端

```bash
cd ~/.openclaw/memory_system
source venv/bin/activate
python main.py api
```

后端运行在 http://localhost:8082

## 架构

```
OpenClaw Gateway
└── memory-weaviate 插件
    ├── Tools (memory_recall, memory_store, kg_search, memory_sync)
    ├── Hooks (before_agent_start, agent_end)
    └── Python Backend (端口 8082)
        └── Weaviate (端口 8080)
```

## License

MIT
README_EOF

# 打包
echo ""
echo "📦 创建压缩包..."
cd "$TEMP_DIR"
tar -czf "$OUTPUT_DIR/$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"

# 清理
rm -rf "$TEMP_DIR"

echo ""
echo "=============================================="
echo "  ✅ 打包完成！"
echo "=============================================="
echo ""
echo "输出文件: $OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
echo ""
echo "分发方法:"
echo "1. 将压缩包发送给其他人"
echo "2. 解压后运行 ./install.sh"
echo "3. 重启 openclaw gateway restart"
echo ""