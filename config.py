"""
记忆系统配置文件
支持多种向量数据库：Weaviate, PostgreSQL+pgvector, ChromaDB, Milvus

## 推荐模型配置（M1 芯片）

### 实体提取/对话模型（按推荐顺序）
1. qwen2.5:3b   - 2.0GB - 阿里通义千问，中文理解最强
2. llama3.2:3b  - 2.0GB - Meta，多语言支持好
3. qwen2.5:7b   - 4.7GB - 需要更多内存，效果更好
4. mistral:7b   - 4.1GB - 欧洲开源，性能优秀

### 嵌入模型
1. nomic-embed-text - 274MB - 英文为主，速度快
2. bge-m3           - 2.2GB - 中文最佳，多语言
3. bge-large        - 1.3GB - 中文嵌入

### 设置方法
export OLLAMA_CHAT_MODEL=llama3.1:8b
export OLLAMA_EMBED_MODEL=bge-m3

## 向量数据库选择

### Weaviate（默认）
- 专用向量数据库，功能丰富
- 安装：brew install weaviate
- 启动：weaviate

### PostgreSQL + pgvector（推荐）
- 一个数据库搞定所有，支持 SQL 查询
- 安装：brew install postgresql@16 pgvector
- 启动：brew services start postgresql@16
- 创建数据库：createdb openclaw_memory
- 启用扩展：psql -d openclaw_memory -c "CREATE EXTENSION vector;"

### ChromaDB（最轻量）
- 纯 Python，无需额外部署
- 安装：pip install chromadb

### Milvus（大规模）
- 高性能分布式向量数据库
- 需要 Docker 运行

### 切换方法
export VECTOR_STORE=pgvector  # 或 weaviate, chroma, milvus
"""

import os
import platform
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============== 向量数据库配置 ==============
# 支持：weaviate, pgvector, chroma, milvus
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE", "weaviate")

# ============== Weaviate 配置 ==============
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_URL = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"

# Weaviate 本地安装目录
WEAVIATE_DIR = PROJECT_ROOT / "weaviate"

# Weaviate 数据目录
WEAVIATE_DATA_DIR = DATA_DIR / "weaviate"
WEAVIATE_DATA_DIR.mkdir(exist_ok=True)

# ============== PostgreSQL + pgvector 配置 ==============
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "openclaw_memory")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# ============== ChromaDB 配置 ==============
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))

# ============== Milvus 配置 ==============
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB = os.getenv("MILVUS_DB", "default")

# ============== 嵌入模型配置 ==============
# 可选: minilm, ollama, modelscope, huggingface
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "minilm")

# MiniLM 配置 (默认使用 all-MiniLM-L6-v2)
MINILM_MODEL = os.getenv("MINILM_MODEL", "all-MiniLM-L6-v2")
MINILM_DIMENSION = 384

# Ollama 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 嵌入模型（推荐 nomic-embed-text 或 bge-m3）
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# 对话/提取模型（推荐 qwen2.5:3b 或 llama3.2:3b）
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

# HuggingFace / ModelScope 嵌入模型
# 推荐中文模型: bge-base-zh, bge-large-zh, text2vec-base-chinese, m3e-base
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "bge-base-zh")

# 模型缓存目录
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(DATA_DIR / "models"))

# ============== 检索配置 ==============
# 向量检索召回数量
RECALL_LIMIT = int(os.getenv("RECALL_LIMIT", "20"))
# 重排序后返回数量
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
# 时间衰减因子 (0-1, 越大衰减越快)
TIME_DECAY_FACTOR = float(os.getenv("TIME_DECAY_FACTOR", "0.1"))
# 重要性权重 (0-1)
IMPORTANCE_WEIGHT = float(os.getenv("IMPORTANCE_WEIGHT", "0.3"))

# OpenClaw Agent 目录 (通过环境变量配置，默认 ~/.openclaw/agents)
OPENCLAW_DIR = os.path.expanduser(os.getenv("OPENCLAW_DIR", "~/.openclaw"))
AGENTS_DIR = os.path.join(OPENCLAW_DIR, "agents")

# 记忆配置
SUMMARY_INTERVAL_HOURS = 6
CHECKPOINT_FILE = DATA_DIR / "checkpoints.json"

# 集合前缀
VECTOR_COLLECTION_PREFIX = "AgentMemory_"
KG_COLLECTION_PREFIX = "AgentKG_"

# 同步配置
REALTIME_POLL_INTERVAL = 30
REALTIME_MIN_MESSAGES = 5

# API 端口
API_PORT = 8082

# 知识图谱可视化端口
KG_VIS_PORT = 8081

# ============== 知识图谱配置 ==============
# 实体提取配置
KG_MIN_ENTITY_LENGTH = 2  # 实体名称最小长度
KG_MAX_ENTITY_LENGTH = 50  # 实体名称最大长度
KG_MIN_CONFIDENCE = 0.5    # 最小置信度阈值

# 关系提取配置
KG_RELATION_TYPES = [
    "IsA", "PartOf", "HasA",
    "WorksFor", "Manages", "CollaboratesWith", "Knows",
    "Uses", "DependsOn", "Implements", "IntegratesWith",
    "RelatedTo", "Causes", "HasProperty"
]

# ============== 向量维度 ==============
# MiniLM-L6-v2: 384
# nomic-embed-text: 768
# bge-m3: 1024
# 根据 EMBED_PROVIDER 自动选择
if EMBED_PROVIDER == "minilm":
    VECTOR_DIM = MINILM_DIMENSION
else:
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# 实体关系抽取配置
EXTRACTOR_BACKEND = os.getenv("EXTRACTOR_BACKEND", "decoder")  # decoder, encoder, hybrid
