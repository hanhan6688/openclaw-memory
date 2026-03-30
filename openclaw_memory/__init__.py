"""
OpenClaw Memory System - AI Agent 长期记忆系统
============================================

一个为 AI Agent 提供长期记忆能力的向量数据库系统。

特性:
- 向量记忆存储与语义检索
- 知识图谱自动构建
- 实时对话同步
- 时间范围查询
- 多 Agent 隔离

快速开始:
    from openclaw_memory import MemoryStore
    
    store = MemoryStore("my_agent")
    store.remember("用户喜欢抖音广告投放", importance=0.8)
    results = store.recall("广告投放")
"""

__version__ = "1.0.0"
__author__ = "OpenClaw Team"

from .core.memory_store import MemoryStore
from .core.knowledge_graph import KnowledgeGraph
from .core.weaviate_client import WeaviateClient
from .sync.realtime_sync import RealtimeSyncService

__all__ = [
    "MemoryStore",
    "KnowledgeGraph", 
    "WeaviateClient",
    "RealtimeSyncService",
]