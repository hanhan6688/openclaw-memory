# OpenClaw Memory System

<div align="center">

**AI Agent 长期记忆系统**

基于 Weaviate 的向量记忆存储与智能检索

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## ✨ 特性

- 🧠 **向量记忆存储** - 基于 Weaviate 的高效向量存储与语义检索
- 🕸️ **知识图谱** - 自动从对话中提取实体关系，构建知识网络
- 🔄 **实时同步** - 监控 Agent 会话文件，实时同步记忆
- 📅 **时间查询** - 支持"上周说的那个广告商"等自然语言时间查询
- 🎯 **多 Agent 隔离** - 每个 Agent 独立的记忆空间
- 🎨 **Web UI** - 美观的可视化界面，支持日历选择和知识图谱展示
- 📤 **数据导出/导入** - 支持数据备份和迁移
- ⏰ **时间锚点衰减** - 先解析自然语言时间锚点，再结合时间衰减给近期记忆更高权重
- 🔍 **混合检索** - 向量检索 + BM25 关键词检索融合
- 🎯 **意图识别** - 自动识别查询意图，选择最佳检索策略
- 🖼️ **多模态支持** - 支持图片、文档、音频、视频存储

## 📦 快速开始

### 方式一：集成到 OpenClaw（推荐）

将本系统作为 OpenClaw 插件使用，实现无缝集成。

```bash
# 1. 解压下载的压缩包
tar -xzf openclaw-memory-weaviate-*.tar.gz
cd openclaw-memory-weaviate-*

# 2. 运行安装脚本
./install.sh

# 3. 重启 OpenClaw
openclaw gateway restart

# 4. 完成！记忆系统会自动启动
```

### 方式二：独立运行

作为独立服务运行，通过 API 访问。

### 1. 系统要求

- Python 3.10+
- Docker (用于 Weaviate)
- Ollama (可选，用于 AI 摘要)

### 2. 安装

```bash
# 解压项目
cd openclaw-memory-system

# 运行安装脚本
chmod +x install.sh
./install.sh
```

或手动安装：

```bash
# 创建虚拟环境 (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖 (使用国内镜像)
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 3. 启动 Weaviate

```bash
# Docker 方式 (推荐)
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest

# 或使用 docker-compose
docker-compose up -d
```

### 4. 启动 Ollama (可选但推荐)

```bash
# 安装 Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama
ollama serve

# 下载模型
ollama pull nomic-embed-text  # 嵌入模型 (必需)
ollama pull llama3.2:3b       # 聊天模型 (用于摘要)
```

### 5. 启动服务

```bash
# 使用启动脚本
./scripts/start_bg.sh   # 后台启动
./scripts/start.sh      # 前台启动
./scripts/stop.sh       # 停止服务

# 或手动启动
source venv/bin/activate
python main.py api
```

### 6. 访问 Web UI

打开浏览器访问: http://localhost:8082/index.html

## 📖 使用示例

### Python API

```python
from openclaw_memory.core.memory_store import MemoryStore
from openclaw_memory.core.knowledge_graph import KnowledgeGraph

# 创建记忆存储
store = MemoryStore("my_agent")

# 存储记忆
store.store(
    content="用户对抖音广告投放很感兴趣，预算约 5 万",
    metadata={
        "memory_type": "preference",
        "importance": 0.8
    }
)

# 检索记忆
results = store.search("广告投放预算")
for r in results:
    print(r['content'])

# 知识图谱
kg = KnowledgeGraph("my_agent")
kg.add_entity("抖音广告", "产品")
kg.add_relation("用户", "感兴趣", "抖音广告")
```

### 多模态存储

```python
from openclaw_memory.core.memory_store import MemoryStore
from openclaw_memory.core.multimodal import MultimodalMemoryStore

store = MemoryStore("my_agent")
mm_store = MultimodalMemoryStore(store)

# 存储图片（自动生成描述）
mm_store.remember_image("/path/to/image.jpg", importance=0.8)

# 存储文档
mm_store.remember_document("/path/to/document.pdf")

# 存储音频（自动转录）
mm_store.remember_audio("/path/to/audio.mp3")

# 存储视频（提取关键帧）
mm_store.remember_video("/path/to/video.mp4")

# 搜索图片
results = mm_store.search_similar_images("产品截图")
```

### REST API

```bash
# 搜索记忆
curl "http://localhost:8082/agents/main/memories/search?q=广告"

# 按时间范围查询
curl "http://localhost:8082/agents/main/memories?start=2024-01-01&end=2024-01-31"

# 获取知识图谱
curl "http://localhost:8082/agents/main/kg/graph"

# 上传多模态内容
curl -X POST "http://localhost:8082/agents/main/multimodal/upload" \
  -F "file=@/path/to/image.jpg" \
  -F "description=产品截图" \
  -F "importance=0.8"

# 搜索多模态内容
curl "http://localhost:8082/agents/main/multimodal/search?q=产品&type=image"

# 高级检索（混合模式）
curl "http://localhost:8082/agents/main/memories/search?q=广告&mode=hybrid"

# 带意图识别的检索
curl "http://localhost:8082/agents/main/memories/search/intent?q=好像说过什么来着"

# 英文时间锚点检索
curl "http://localhost:8082/agents/main/memories/search/intent?q=What did we decide 3 days ago?"

# 按时间范围检索
curl "http://localhost:8082/agents/main/memories/search/time?range=week&q=产品"

# 查看时间锚点衰减配置（接口名保持兼容）
curl "http://localhost:8082/agents/main/memories/time-decay"

# 导出数据
curl "http://localhost:8082/agents/main/export" -o backup.json

# 导入数据
curl -X POST "http://localhost:8082/agents/main/import" \
  -H "Content-Type: application/json" \
  -d @backup.json
```

### 高级检索功能

```bash
# 1. 混合检索（向量 + BM25）
curl "http://localhost:8082/agents/main/memories/search?q=产品需求&mode=hybrid"

# 2. 纯向量检索（语义相似）
curl "http://localhost:8082/agents/main/memories/search?q=产品需求&mode=vector"

# 3. 纯 BM25 检索（关键词精确匹配）
curl "http://localhost:8082/agents/main/memories/search?q=产品需求&mode=bm25"

# 4. 自动意图识别
# 模糊查询 -> 自动使用向量检索
curl "http://localhost:8082/agents/main/memories/search/intent?q=好像说过什么来着"
# 精确查询 -> 自动使用 BM25
curl "http://localhost:8082/agents/main/memories/search/intent?q=具体说了什么"
# 时间查询 -> 自动提取时间锚点并做时间过滤
curl "http://localhost:8082/agents/main/memories/search/intent?q=昨天的内容"
# 英文时间查询同样支持
curl "http://localhost:8082/agents/main/memories/search/intent?q=The roadmap from last month"
```

### 时间锚点返回字段

当查询中包含时间表达时，`search/intent` 返回中会额外包含：

- `reference_time`：解析出的参考时间
- `time_range`：解析出的过滤时间范围
- `anchor_type`：`point` 或 `range`
- `anchor_granularity`：`minute` / `hour` / `day` / `week` / `month` / `year`
- `cleaned_query`：去掉时间短语后的正文查询，用于更干净的语义检索

### 时间锚点衰减权重

系统会先识别时间锚点（如“昨天”“3天前”“last month”“2 hours ago”），
再在该锚点上下文内应用时间衰减，而不是对所有查询一律做简单的“越新越高”排序。

| 时间 | 权重 |
|------|------|
| 今天 | 1.20x |
| 昨天 | 0.90x |
| 3天前 | 0.73x |
| 7天前 | 0.53x |
| 14天前 | 0.28x |
| 30天前 | 0.08x |

## 🏗️ 项目结构

```
openclaw-memory-system/
├── openclaw_memory/          # 核心包
│   ├── core/                 # 核心模块
│   │   ├── memory_store.py   # 记忆存储
│   │   ├── knowledge_graph.py # 知识图谱
│   │   ├── weaviate_client.py # Weaviate 客户端
│   │   ├── embeddings.py     # 嵌入生成
│   │   ├── summarizer.py     # AI 摘要
│   │   ├── hybrid_retrieval.py # 高级检索
│   │   └── multimodal.py     # 多模态支持
│   ├── api/                  # API 服务
│   │   └── server.py         # Flask 服务器
│   └── sync/                 # 同步模块
│       └── realtime_sync.py  # 实时同步
├── ui/                       # Web UI
│   └── index.html            # 单页应用
├── config/                   # 配置文件
│   └── settings.yaml         # 配置模板
├── scripts/                  # 工具脚本
│   ├── setup.sh              # 安装脚本
│   ├── start.sh              # 启动脚本
│   ├── start_bg.sh           # 后台启动
│   └── stop.sh               # 停止脚本
├── main.py                   # 主入口
├── requirements.txt          # Python 依赖
├── docker-compose.yml        # Docker 编排
└── README.md                 # 本文件
```

## ⚙️ 配置

编辑 `config/settings.yaml`:

```yaml
weaviate:
  host: localhost
  port: 8080

ollama:
  base_url: http://localhost:11434
  embed_model: nomic-embed-text
  chat_model: llama3.2:3b

api:
  port: 8082
```

## 🔧 命令参考

```bash
python main.py [command] [options]

Commands:
  api         启动 API 服务器
  sync        启动实时同步服务
  check       检查系统状态

Options:
  --port, -p  API 端口 (默认: 8082)
  --agent, -a 指定 Agent
```

## 📊 数据存储

| 数据类型 | 存储位置 | 说明 |
|---------|---------|------|
| 会话记录 | `{agents_dir}/{agent}/sessions/*.jsonl` | JSONL 格式 |
| 向量记忆 | Weaviate (localhost:8080) | 向量数据库 |
| 知识图谱 | Weaviate (localhost:8080) | 实体关系网络 |

## 🚀 部署

### Docker Compose

```bash
docker-compose up -d
```

### 手动部署

1. 安装 Python 3.10
2. 创建 venv 并安装依赖
3. 启动 Weaviate
4. 启动服务

## 📊 性能

- 向量检索延迟: < 50ms (10万条记录)
- 内存占用: ~200MB (基础服务)
- 支持并发: 100+ QPS

## 🔄 数据迁移

```bash
# 导出数据
curl http://localhost:8082/agents/main/export -o backup.json

# 在新电脑上导入
curl -X POST http://localhost:8082/agents/main/import \
  -H "Content-Type: application/json" \
  -d @backup.json
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 License

MIT License

---

