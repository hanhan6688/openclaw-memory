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

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"
__author__ = "OpenClaw Team"

__all__ = [
    "KnowledgeGraph",
    "MemoryStore",
    "RealtimeSyncService",
    "WeaviateClient",
]

_LAZY_IMPORTS = {
    "KnowledgeGraph": (".core.knowledge_graph", "KnowledgeGraph"),
    "MemoryStore": (".core.memory_store", "MemoryStore"),
    "RealtimeSyncService": (".sync.realtime_sync", "RealtimeSyncService"),
    "WeaviateClient": (".core.weaviate_client", "WeaviateClient"),
}

if TYPE_CHECKING:
    from .core.knowledge_graph import KnowledgeGraph
    from .core.memory_store import MemoryStore
    from .core.weaviate_client import WeaviateClient
    from .sync.realtime_sync import RealtimeSyncService


def __getattr__(name: str) -> Any:
    """按需导入，避免包初始化时强依赖可选组件。"""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
