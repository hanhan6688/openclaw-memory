"""
向量存储抽象层
支持多种向量数据库：Weaviate, PostgreSQL+pgvector, ChromaDB, Milvus

统一接口设计：
- 一个数据库同时支持记忆表和知识图谱表
- 通过 agent_id 隔离不同 Agent 的数据
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


class VectorStore(ABC):
    """
    向量存储抽象基类
    
    所有向量数据库适配器都需要实现这个接口
    """
    
    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.user_id = user_id
        self._connected = False
    
    # ==================== 连接管理 ====================
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到向量数据库"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    # ==================== 集合/表管理 ====================
    
    @abstractmethod
    def create_collections(self) -> bool:
        """创建记忆和知识图谱集合/表"""
        pass
    
    @abstractmethod
    def delete_collections(self) -> bool:
        """删除集合/表"""
        pass
    
    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """检查集合/表是否存在"""
        pass
    
    # ==================== 记忆操作 ====================
    
    @abstractmethod
    def insert_memory(self, data: Dict, vector: List[float] = None) -> str:
        """
        插入记忆
        
        Args:
            data: 记忆数据
                - content: 文本内容
                - memory_type: 记忆类型
                - importance: 重要性
                - timestamp: 时间戳
                - raw_conversation: 原始对话
            vector: 嵌入向量（可选，某些数据库可自动生成）
        
        Returns:
            记录ID
        """
        pass
    
    @abstractmethod
    def get_memories(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """获取记忆列表"""
        pass
    
    @abstractmethod
    def search_memories(self, vector: List[float], limit: int = 10, 
                       filters: Dict = None) -> List[Dict]:
        """
        向量搜索记忆
        
        Args:
            vector: 查询向量
            limit: 返回数量
            filters: 过滤条件 (如 {"memory_type": "conversation"})
        """
        pass
    
    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """根据ID获取记忆"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass
    
    @abstractmethod
    def count_memories(self) -> int:
        """统计记忆数量"""
        pass
    
    # ==================== 知识图谱操作 ====================
    
    @abstractmethod
    def insert_kg(self, data: Dict, vector: List[float] = None) -> str:
        """
        插入知识图谱条目
        
        Args:
            data: 知识图谱数据
                - entity_name: 实体名称
                - entity_type: 实体类型
                - relation_type: 关系类型
                - target_entity: 目标实体
                - context: 上下文
                - confidence: 置信度
                - timestamp: 时间戳
            vector: 嵌入向量
        """
        pass
    
    @abstractmethod
    def get_kg(self, limit: int = 500) -> List[Dict]:
        """获取知识图谱列表"""
        pass
    
    @abstractmethod
    def search_kg(self, vector: List[float], limit: int = 10) -> List[Dict]:
        """向量搜索知识图谱"""
        pass
    
    @abstractmethod
    def get_kg_by_entity(self, entity_name: str) -> List[Dict]:
        """根据实体名获取相关记录"""
        pass
    
    @abstractmethod
    def delete_kg(self, kg_id: str) -> bool:
        """删除知识图谱条目"""
        pass
    
    @abstractmethod
    def count_kg(self) -> int:
        """统计知识图谱数量"""
        pass
    
    # ==================== 批量操作 ====================
    
    def batch_insert_memories(self, items: List[Dict], vectors: List[List[float]] = None) -> List[str]:
        """批量插入记忆"""
        ids = []
        for i, item in enumerate(items):
            vec = vectors[i] if vectors and i < len(vectors) else None
            ids.append(self.insert_memory(item, vec))
        return ids
    
    def batch_insert_kg(self, items: List[Dict], vectors: List[List[float]] = None) -> List[str]:
        """批量插入知识图谱"""
        ids = []
        for i, item in enumerate(items):
            vec = vectors[i] if vectors and i < len(vectors) else None
            ids.append(self.insert_kg(item, vec))
        return ids
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "connected": self._connected,
            "memory_count": self.count_memories(),
            "kg_count": self.count_kg()
        }


class VectorStoreFactory:
    """向量存储工厂"""
    
    _stores = {}  # 缓存已创建的实例
    
    @classmethod
    def create(cls, store_type: str, agent_id: str, user_id: str = "default", 
               config: Dict = None) -> VectorStore:
        """
        创建向量存储实例
        
        Args:
            store_type: 存储类型 (weaviate, pgvector, chroma, milvus)
            agent_id: Agent ID
            user_id: 用户 ID
            config: 额外配置
        """
        cache_key = f"{store_type}:{agent_id}:{user_id}"
        if cache_key in cls._stores:
            return cls._stores[cache_key]
        
        store_type = store_type.lower()
        
        if store_type == "weaviate":
            from .adapters.weaviate_adapter import WeaviateAdapter
            store = WeaviateAdapter(agent_id, user_id, config)
        elif store_type in ("pgvector", "postgres", "postgresql"):
            from .adapters.pgvector_adapter import PgVectorAdapter
            store = PgVectorAdapter(agent_id, user_id, config)
        elif store_type in ("chroma", "chromadb"):
            from .adapters.chroma_adapter import ChromaAdapter
            store = ChromaAdapter(agent_id, user_id, config)
        elif store_type == "milvus":
            from .adapters.milvus_adapter import MilvusAdapter
            store = MilvusAdapter(agent_id, user_id, config)
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")
        
        cls._stores[cache_key] = store
        return store
    
    @classmethod
    def get_supported_stores(cls) -> List[str]:
        """获取支持的存储类型"""
        return ["weaviate", "pgvector", "chroma", "milvus"]
    
    @classmethod
    def clear_cache(cls):
        """清除缓存"""
        cls._stores.clear()