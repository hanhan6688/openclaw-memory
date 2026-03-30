"""
知识图谱管理器 - 支持 NetworkX、NebulaGraph 和 Weaviate

提供统一的知识图谱接口，底层可切换不同的图数据库。
优先级：NetworkX（内存）> NebulaGraph（Docker）> Weaviate（兼容）
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from abc import ABC, abstractmethod

# 导入 NetworkX 客户端（优先，无需 Docker）
try:
    from .networkx_kg_client import NetworkXKGClient, get_nx_client
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# 尝试导入 NebulaGraph 客户端
try:
    from .nebula_kg_client import NebulaKGClient, get_nebula_client
    NEBULA_AVAILABLE = True
except ImportError:
    NEBULA_AVAILABLE = False

# 导入 Weaviate 客户端
from .weaviate_client import WeaviateClient


class KnowledgeGraphBackend(ABC):
    """知识图谱后端抽象基类"""
    
    @abstractmethod
    def add_entity(self, entity_id: str, name: str, entity_type: str, 
                   properties: Dict = None, confidence: float = 0.8) -> bool:
        """添加实体"""
        pass
    
    @abstractmethod
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                     confidence: float = 0.7, evidence: str = "") -> bool:
        """添加关系"""
        pass
    
    @abstractmethod
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """搜索实体"""
        pass
    
    @abstractmethod
    def get_neighbors(self, entity_id: str, relation_type: str = None, limit: int = 20) -> List[Dict]:
        """获取邻居"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """获取统计信息"""
        pass


class NebulaGraphBackend(KnowledgeGraphBackend):
    """NebulaGraph 后端"""
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.client: Optional[NebulaKGClient] = None
    
    def _ensure_client(self) -> bool:
        """确保客户端已连接"""
        if self.client and self.client._connected:
            return True
        
        if not NEBULA_AVAILABLE:
            return False
        
        self.client = get_nebula_client(self.agent_id)
        return self.client._connected
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   properties: Dict = None, confidence: float = 0.8) -> bool:
        if not self._ensure_client():
            return False
        return self.client.add_entity(entity_id, name, entity_type, properties, confidence)
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                     confidence: float = 0.7, evidence: str = "") -> bool:
        if not self._ensure_client():
            return False
        return self.client.add_relation(source_id, target_id, relation_type, confidence, evidence)
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        if not self._ensure_client():
            return []
        return self.client.search_entities(query, entity_type, limit)
    
    def get_neighbors(self, entity_id: str, relation_type: str = None, limit: int = 20) -> List[Dict]:
        if not self._ensure_client():
            return []
        return self.client.get_neighbors(entity_id, relation_type, limit)
    
    def get_stats(self) -> Dict:
        if not self._ensure_client():
            return {"connected": False}
        return self.client.get_stats()


class WeaviateKGBBackend(KnowledgeGraphBackend):
    """Weaviate 知识图谱后端（兼容旧版）"""
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.client: Optional[WeaviateClient] = None
    
    def _ensure_client(self) -> bool:
        """确保客户端已连接"""
        if self.client:
            return True
        
        try:
            self.client = WeaviateClient(self.agent_id)
            return self.client.client is not None
        except Exception:
            return False
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   properties: Dict = None, confidence: float = 0.8) -> bool:
        if not self._ensure_client():
            return False
        
        entity_data = {
            "name": name,
            "type": entity_type,
            "confidence": confidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(properties or {})
        }
        
        try:
            result = self.client.insert_kg_entity(entity_data)
            return result is not None
        except Exception:
            return False
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                     confidence: float = 0.7, evidence: str = "") -> bool:
        if not self._ensure_client():
            return False
        
        relation_data = {
            "source": source_id,
            "target": target_id,
            "relation": relation_type,
            "confidence": confidence,
            "evidence": evidence,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            result = self.client.insert_kg_relation(relation_data)
            return result is not None
        except Exception:
            return False
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        if not self._ensure_client():
            return []
        
        try:
            results = self.client.search_kg(query, limit=limit)
            if entity_type:
                results = [r for r in results if r.get("type") == entity_type]
            return results
        except Exception:
            return []
    
    def get_neighbors(self, entity_id: str, relation_type: str = None, limit: int = 20) -> List[Dict]:
        # Weaviate 的邻居查询实现
        return []
    
    def get_stats(self) -> Dict:
        if not self._ensure_client():
            return {"connected": False}
        
        try:
            entities = self.client.get_kg_entities(limit=10000)
            relations = self.client.get_kg_relations(limit=10000)
            return {
                "connected": True,
                "entities": len(entities),
                "relations": len(relations)
            }
        except Exception:
            return {"connected": False}


class NetworkXBackend(KnowledgeGraphBackend):
    """NetworkX 知识图谱后端（优先，无需 Docker）"""
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.client: Optional[NetworkXKGClient] = None
    
    def _ensure_client(self) -> bool:
        """确保客户端已连接"""
        if self.client:
            return True
        
        if not NETWORKX_AVAILABLE:
            return False
        
        self.client = get_nx_client(self.agent_id)
        return True
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   properties: Dict = None, confidence: float = 0.8) -> bool:
        if not self._ensure_client():
            return False
        return self.client.add_entity(entity_id, name, entity_type, properties, confidence)
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                     confidence: float = 0.7, evidence: str = "") -> bool:
        if not self._ensure_client():
            return False
        return self.client.add_relation(source_id, target_id, relation_type, confidence, evidence)
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        if not self._ensure_client():
            return []
        return self.client.search_entities(query, entity_type, limit)
    
    def get_neighbors(self, entity_id: str, relation_type: str = None, limit: int = 20) -> List[Dict]:
        if not self._ensure_client():
            return []
        return self.client.get_neighbors(entity_id, relation_type, limit)
    
    def get_stats(self) -> Dict:
        if not self._ensure_client():
            return {"connected": False}
        return self.client.get_stats()


class KnowledgeGraphManager:
    """
    知识图谱管理器
    
    自动选择可用的后端：
    1. 优先使用 NetworkX（内存图，无需 Docker）
    2. 尝试 NebulaGraph（如果可用）
    3. 回退到 Weaviate（兼容旧版）
    """
    
    def __init__(self, agent_id: str = "main", prefer_nebula: bool = False):
        self.agent_id = agent_id
        self.backend: Optional[KnowledgeGraphBackend] = None
        self._prefer_nebula = prefer_nebula
    
    def _init_backend(self) -> bool:
        """初始化后端"""
        if self.backend:
            return True
        
        # 优先使用 NetworkX（无需 Docker，立即可用）
        if NETWORKX_AVAILABLE:
            nx_backend = NetworkXBackend(self.agent_id)
            if nx_backend._ensure_client():
                self.backend = nx_backend
                print(f"✅ 知识图谱后端: NetworkX (内存图)")
                return True
        
        # 尝试 NebulaGraph（如果指定）
        if self._prefer_nebula and NEBULA_AVAILABLE:
            nebula = NebulaGraphBackend(self.agent_id)
            if nebula._ensure_client():
                self.backend = nebula
                print(f"✅ 知识图谱后端: NebulaGraph")
                return True
        
        # 回退到 Weaviate
        weaviate = WeaviateKGBBackend(self.agent_id)
        if weaviate._ensure_client():
            self.backend = weaviate
            print(f"✅ 知识图谱后端: Weaviate (回退)")
            return True
        
        print("⚠️ 知识图谱后端不可用")
        return False
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   properties: Dict = None, confidence: float = 0.8) -> bool:
        """添加实体"""
        if not self._init_backend():
            return False
        return self.backend.add_entity(entity_id, name, entity_type, properties, confidence)
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                     confidence: float = 0.7, evidence: str = "") -> bool:
        """添加关系"""
        if not self._init_backend():
            return False
        return self.backend.add_relation(source_id, target_id, relation_type, confidence, evidence)
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """搜索实体"""
        if not self._init_backend():
            return []
        return self.backend.search_entities(query, entity_type, limit)
    
    def get_neighbors(self, entity_id: str, relation_type: str = None, limit: int = 20) -> List[Dict]:
        """获取邻居"""
        if not self._init_backend():
            return []
        return self.backend.get_neighbors(entity_id, relation_type, limit)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._init_backend():
            return {"connected": False}
        return self.backend.get_stats()
    
    def extract_and_store(self, text: str, source: str = "chat") -> Dict:
        """
        从文本提取实体和关系并存储
        
        Returns:
            {
                "entities": [实体列表],
                "relations": [关系列表],
                "stored_entities": 存储的实体数,
                "stored_relations": 存储的关系数
            }
        """
        from .extraction.decoder_extractor import EnhancedDecoderExtractor
        
        extractor = EnhancedDecoderExtractor()
        result = extractor.extract(text)
        
        stored_entities = 0
        stored_relations = 0
        entity_id_map = {}
        
        # 存储实体
        for entity in result.entities:
            entity_id = f"{self.agent_id}_{entity.name}_{entity.type}"
            if self.add_entity(
                entity_id=entity_id,
                name=entity.name,
                entity_type=entity.type,
                properties=entity.metadata,
                confidence=entity.confidence
            ):
                stored_entities += 1
                entity_id_map[entity.name] = entity_id
        
        # 存储关系
        for relation in result.relations:
            source_id = entity_id_map.get(relation.source)
            target_id = entity_id_map.get(relation.target)
            
            if source_id and target_id:
                if self.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation.type,
                    confidence=relation.confidence,
                    evidence=relation.evidence or ""
                ):
                    stored_relations += 1
        
        return {
            "entities": [{"name": e.name, "type": e.type} for e in result.entities],
            "relations": [{"source": r.source, "target": r.target, "type": r.type} for r in result.relations],
            "stored_entities": stored_entities,
            "stored_relations": stored_relations
        }


# 工厂函数
_kg_managers: Dict[str, KnowledgeGraphManager] = {}


def get_knowledge_graph(agent_id: str = "main", prefer_nebula: bool = True) -> KnowledgeGraphManager:
    """获取知识图谱管理器"""
    if agent_id not in _kg_managers:
        _kg_managers[agent_id] = KnowledgeGraphManager(agent_id, prefer_nebula)
    return _kg_managers[agent_id]