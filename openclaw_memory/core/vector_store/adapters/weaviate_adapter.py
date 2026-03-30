"""
Weaviate 适配器
基于现有的 weaviate_client.py 重构
"""

import sys
import os
import uuid
from typing import List, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.vector_store import VectorStore

# Weaviate imports
try:
    import weaviate
    from weaviate.connect import ConnectionParams
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery
    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False

# 配置
from config import (
    WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_URL,
    VECTOR_COLLECTION_PREFIX, KG_COLLECTION_PREFIX
)


class WeaviateAdapter(VectorStore):
    """Weaviate 向量存储适配器"""
    
    def __init__(self, agent_id: str, user_id: str = "default", config: Dict = None):
        super().__init__(agent_id, user_id)
        self.config = config or {}
        
        # 集合名称（Weaviate 类名不允许 -，替换为 _）
        safe_agent_id = agent_id.replace('-', '_')
        self.memory_collection = f"{VECTOR_COLLECTION_PREFIX}{safe_agent_id}"
        self.kg_collection = f"{KG_COLLECTION_PREFIX}{safe_agent_id}"
        
        # 客户端
        self._client = None
    
    # ==================== 连接管理 ====================
    
    def connect(self) -> bool:
        """连接到 Weaviate"""
        if not HAS_WEAVIATE:
            raise ImportError("weaviate-client 未安装，请运行: pip install weaviate-client")
        
        if self._client is not None:
            return True
        
        try:
            host = self.config.get("host", WEAVIATE_HOST) or "localhost"
            port = self.config.get("port", WEAVIATE_PORT) or 8080
            
            self._client = weaviate.WeaviateClient(
                connection_params=ConnectionParams.from_params(
                    http_host=host,
                    http_port=port,
                    http_secure=False,
                    grpc_host=host,
                    grpc_port=50051,
                    grpc_secure=False
                )
            )
            self._client.connect()
            self._connected = True
            
            # 确保集合存在
            self.create_collections()
            
            return True
        except Exception as e:
            print(f"⚠️ Weaviate 连接失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> bool:
        """断开连接"""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._connected = False
        return True
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected and self._client is not None
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        if not self.is_connected():
            return {"status": "disconnected"}
        
        try:
            ready = self._client.is_ready()
            return {
                "status": "healthy" if ready else "unhealthy",
                "type": "weaviate",
                "collections": {
                    "memory": self.memory_collection,
                    "kg": self.kg_collection
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ==================== 集合管理 ====================
    
    def create_collections(self) -> bool:
        """创建集合"""
        if not self.is_connected():
            return False
        
        try:
            # 创建记忆集合
            if not self.collection_exists(self.memory_collection):
                self._client.collections.create(
                    name=self.memory_collection,
                    description=f"Agent {self.agent_id} 记忆存储",
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="memory_type", data_type=DataType.TEXT),
                        Property(name="importance", data_type=DataType.NUMBER),
                        Property(name="timestamp", data_type=DataType.DATE),
                        Property(name="agent_id", data_type=DataType.TEXT),
                        Property(name="raw_conversation", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=Configure.Vectorizer.none()
                )
            
            # 创建知识图谱集合
            if not self.collection_exists(self.kg_collection):
                self._client.collections.create(
                    name=self.kg_collection,
                    description=f"Agent {self.agent_id} 知识图谱",
                    properties=[
                        Property(name="entity_name", data_type=DataType.TEXT),
                        Property(name="entity_type", data_type=DataType.TEXT),
                        Property(name="relation_type", data_type=DataType.TEXT),
                        Property(name="target_entity", data_type=DataType.TEXT),
                        Property(name="context", data_type=DataType.TEXT),
                        Property(name="confidence", data_type=DataType.NUMBER),
                        Property(name="timestamp", data_type=DataType.DATE),
                    ],
                    vectorizer_config=Configure.Vectorizer.none()
                )
            
            return True
        except Exception as e:
            print(f"⚠️ 创建集合失败: {e}")
            return False
    
    def delete_collections(self) -> bool:
        """删除集合"""
        if not self.is_connected():
            return False
        
        try:
            if self.collection_exists(self.memory_collection):
                self._client.collections.delete(self.memory_collection)
            if self.collection_exists(self.kg_collection):
                self._client.collections.delete(self.kg_collection)
            return True
        except Exception as e:
            print(f"⚠️ 删除集合失败: {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """检查集合是否存在"""
        if not self.is_connected():
            return False
        try:
            self._client.collections.get(name)
            return True
        except Exception:
            return False
    
    # ==================== 记忆操作 ====================
    
    def insert_memory(self, data: Dict, vector: List[float] = None) -> str:
        """插入记忆"""
        if not self.is_connected():
            self.connect()
        
        obj_uuid = str(uuid.uuid4())
        
        properties = {
            "content": data.get("content", ""),
            "memory_type": data.get("memory_type", "conversation"),
            "importance": data.get("importance", 0.5),
            "timestamp": data.get("timestamp"),
            "agent_id": self.agent_id,
            "raw_conversation": data.get("raw_conversation", "")[:10000] if data.get("raw_conversation") else ""
        }
        
        # 处理时间戳
        if not properties["timestamp"]:
            from datetime import datetime
            properties["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        try:
            collection = self._client.collections.get(self.memory_collection)
            collection.data.insert(
                uuid=obj_uuid,
                properties=properties,
                vector=vector or [0.0] * 768
            )
            return obj_uuid
        except Exception as e:
            print(f"⚠️ 插入记忆失败: {e}")
            return ""
    
    def get_memories(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """获取记忆列表"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.memory_collection)
            results = collection.query.fetch_objects(limit=limit, offset=offset)
            
            memories = []
            for obj in results.objects:
                memories.append({
                    "id": str(obj.uuid),
                    **obj.properties
                })
            return memories
        except Exception as e:
            print(f"⚠️ 获取记忆失败: {e}")
            return []
    
    def search_memories(self, vector: List[float], limit: int = 10,
                       filters: Dict = None) -> List[Dict]:
        """向量搜索记忆"""
        if not self.is_connected():
            self.connect()

        try:
            collection = self._client.collections.get(self.memory_collection)
            results = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )

            memories = []
            for obj in results.objects:
                distance = None
                if obj.metadata:
                    distance = getattr(obj.metadata, 'distance', None)
                memories.append({
                    "id": str(obj.uuid),
                    "distance": distance,
                    **obj.properties
                })
            return memories
        except Exception as e:
            print(f"⚠️ 搜索记忆失败: {e}")
            return []
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """根据ID获取记忆"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.memory_collection)
            obj = collection.query.fetch_object_by_id(memory_id)
            if obj:
                return {"id": str(obj.uuid), **obj.properties}
            return None
        except Exception:
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.memory_collection)
            collection.data.delete_by_id(memory_id)
            return True
        except Exception:
            return False
    
    def count_memories(self) -> int:
        """统计记忆数量"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.memory_collection)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count
        except Exception:
            return 0
    
    # ==================== 知识图谱操作 ====================
    
    def insert_kg(self, data: Dict, vector: List[float] = None) -> str:
        """插入知识图谱"""
        if not self.is_connected():
            self.connect()
        
        obj_uuid = str(uuid.uuid4())
        
        properties = {
            "entity_name": data.get("entity_name", ""),
            "entity_type": data.get("entity_type", ""),
            "relation_type": data.get("relation_type", ""),
            "target_entity": data.get("target_entity", ""),
            "context": data.get("context", "")[:500] if data.get("context") else "",
            "confidence": data.get("confidence", 0.8),
            "timestamp": data.get("timestamp")
        }
        
        if not properties["timestamp"]:
            from datetime import datetime
            properties["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        try:
            collection = self._client.collections.get(self.kg_collection)
            collection.data.insert(
                uuid=obj_uuid,
                properties=properties,
                vector=vector or [0.0] * 768
            )
            return obj_uuid
        except Exception as e:
            print(f"⚠️ 插入知识图谱失败: {e}")
            return ""
    
    def get_kg(self, limit: int = 500) -> List[Dict]:
        """获取知识图谱列表"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=limit)
            
            items = []
            for obj in results.objects:
                items.append({
                    "id": str(obj.uuid),
                    **obj.properties
                })
            return items
        except Exception as e:
            print(f"⚠️ 获取知识图谱失败: {e}")
            return []
    
    def search_kg(self, vector: List[float], limit: int = 10) -> List[Dict]:
        """向量搜索知识图谱"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.kg_collection)
            results = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )
            
            items = []
            for obj in results.objects:
                items.append({
                    "id": str(obj.uuid),
                    **obj.properties
                })
            return items
        except Exception as e:
            print(f"⚠️ 搜索知识图谱失败: {e}")
            return []
    
    def get_kg_by_entity(self, entity_name: str) -> List[Dict]:
        """根据实体名获取相关记录"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=1000)
            
            items = []
            for obj in results.objects:
                if obj.properties.get("entity_name") == entity_name:
                    items.append({
                        "id": str(obj.uuid),
                        **obj.properties
                    })
            return items
        except Exception:
            return []
    
    def delete_kg(self, kg_id: str) -> bool:
        """删除知识图谱条目"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.kg_collection)
            collection.data.delete_by_id(kg_id)
            return True
        except Exception:
            return False
    
    def count_kg(self) -> int:
        """统计知识图谱数量"""
        if not self.is_connected():
            self.connect()
        
        try:
            collection = self._client.collections.get(self.kg_collection)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count
        except Exception:
            return 0
    
    # ==================== 扩展方法 ====================
    
    def get_all_entities(self, limit: int = 500) -> List[Dict]:
        """获取所有实体"""
        items = self.get_kg(limit)
        entities = []
        for item in items:
            if item.get("relation_type") == "is_entity":
                entities.append({
                    "id": item.get("id"),
                    "entity_name": item.get("entity_name"),
                    "entity_type": item.get("entity_type"),
                    "confidence": item.get("confidence")
                })
        return entities
    
    def get_all_relations(self, limit: int = 500) -> List[Dict]:
        """获取所有关系"""
        items = self.get_kg(limit)
        relations = []
        for item in items:
            rel_type = item.get("relation_type")
            if rel_type and rel_type != "is_entity":
                relations.append({
                    "id": item.get("id"),
                    "source": item.get("entity_name"),
                    "relation": rel_type,
                    "target": item.get("target_entity"),
                    "confidence": item.get("confidence")
                })
        return relations
    # ==================== MiniLM + 重排序检索 ====================


    def search_bm25(self, query: str, limit: int = 20) -> List[Dict]:
        """BM25 关键词检索
        
        Args:
            query: 查询文本
            limit: 返回数量
            
        Returns:
            检索结果列表
        """
        try:
            collection = self.client.collections.get(self.memory_collection)
            
            # BM25 检索，搜索 summary 和 content 字段
            response = collection.query.bm25(
                query=query,
                query_properties=["summary", "content"],
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "id": obj.uuid,
                    "content": obj.properties.get("content", ""),
                    "summary": obj.properties.get("summary", ""),
                    "importance": obj.properties.get("importance", 0.5),
                    "timestamp": obj.properties.get("timestamp"),
                    "keywords": obj.properties.get("keywords", []),
                    "_bm25_score": obj.metadata.score if obj.metadata else 0
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ BM25 检索失败: {e}")
            return []

    def search_hybrid(self, query: str, vector: List[float] = None, 
                      alpha: float = 0.5, limit: int = 20) -> List[Dict]:
        """混合检索（向量 + BM25）
        
        Args:
            query: 查询文本
            vector: 查询向量（可选，不提供则自动生成）
            alpha: 向量检索权重（0-1，1=纯向量，0=纯BM25）
            limit: 返回数量
            
        Returns:
            混合检索结果
        """
        try:
            collection = self.client.collections.get(self.memory_collection)
            
            # 如果没有提供向量，使用 Weaviate 内置的向量化
            if vector:
                response = collection.query.hybrid(
                    query=query,
                    vector=vector,
                    alpha=alpha,
                    limit=limit,
                    query_properties=["summary", "content"],
                    return_metadata=MetadataQuery(score=True, explain_score=True)
                )
            else:
                response = collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    limit=limit,
                    query_properties=["summary", "content"],
                    return_metadata=MetadataQuery(score=True, explain_score=True)
                )
            
            results = []
            for obj in response.objects:
                results.append({
                    "id": obj.uuid,
                    "content": obj.properties.get("content", ""),
                    "summary": obj.properties.get("summary", ""),
                    "importance": obj.properties.get("importance", 0.5),
                    "timestamp": obj.properties.get("timestamp"),
                    "keywords": obj.properties.get("keywords", []),
                    "_hybrid_score": obj.metadata.score if obj.metadata else 0
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ 混合检索失败: {e}")
            return []

    def search_with_rrf(self, query: str, vector: List[float],
                        vector_limit: int = 20, bm25_limit: int = 20,
                        k: int = 60) -> List[Dict]:
        """向量 + BM25 双路召回 + RRF 融合
        
        RRF (Reciprocal Rank Fusion) 公式:
        score = sum(1 / (k + rank))
        
        Args:
            query: 查询文本
            vector: 查询向量
            vector_limit: 向量检索数量
            bm25_limit: BM25 检索数量
            k: RRF 参数（默认60）
            
        Returns:
            RRF 融合后的结果
        """
        # 1. 向量检索
        vector_results = self.search_memories(vector, limit=vector_limit)
        
        # 2. BM25 检索
        bm25_results = self.search_bm25(query, limit=bm25_limit)
        
        # 3. RRF 融合
        rrf_scores = {}
        id_to_result = {}
        
        # 向量检索结果打分
        for rank, result in enumerate(vector_results, 1):
            rid = result.get("id")
            if rid:
                rrf_scores[rid] = rrf_scores.get(rid, 0) + 1 / (k + rank)
                id_to_result[rid] = result
                id_to_result[rid]["_vector_rank"] = rank
        
        # BM25 检索结果打分
        for rank, result in enumerate(bm25_results, 1):
            rid = result.get("id")
            if rid:
                rrf_scores[rid] = rrf_scores.get(rid, 0) + 1 / (k + rank)
                if rid not in id_to_result:
                    id_to_result[rid] = result
                id_to_result[rid]["_bm25_rank"] = rank
        
        # 4. 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for rid in sorted_ids:
            result = id_to_result[rid]
            result["_rrf_score"] = rrf_scores[rid]
            results.append(result)
        
        return results

    def search_with_rerank(
        self,
        query: str,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        rerank_top_k: int = 10,
        time_decay: float = 0.1,
        importance_weight: float = 0.3
    ) -> List[Dict]:
        """混合检索 + RRF 融合 + MiniLM 重排序
        
        流程:
        1. 向量检索（语义匹配）
        2. BM25 检索（关键词匹配）
        3. RRF 融合两路结果
        4. MiniLM 重排序

        Args:
            query: 查询文本
            vector_limit: 向量检索召回数量
            bm25_limit: BM25 检索召回数量
            rerank_top_k: 重排序后返回数量
            time_decay: 时间衰减因子
            importance_weight: 重要性权重

        Returns:
            重排序后的结果列表
        """
        from core.minilm_retriever import SimpleReranker
        from core.embeddings import OllamaEmbedding

        # 1. 生成查询向量
        ollama_embedder = OllamaEmbedding()
        query_vector = ollama_embedder.embed(query)

        # 2. 混合检索（向量 + BM25 + RRF 融合）
        candidates = self.search_with_rrf(
            query=query,
            vector=query_vector,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit
        )

        if not candidates:
            return []

        # 3. MiniLM 重排序
        reranker = SimpleReranker()
        reranked = reranker.rerank(
            query=query,
            results=candidates,
            top_k=rerank_top_k,
            time_decay_factor=time_decay,
            importance_weight=importance_weight
        )

        return reranked

    def search_by_time_with_rerank(
        self,
        query: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = 50,
        rerank_top_k: int = 10
    ) -> List[Dict]:
        """带时间过滤的检索 + 重排序
        
        Args:
            query: 查询文本
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 召回数量
            rerank_top_k: 返回数量
        """
        from core.minilm_retriever import MiniLMEmbedding, SimpleReranker
        
        # 获取时间范围内的记忆
        candidates = self.get_memories_by_time(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        
        if not candidates:
            return []
        
        # 重排序
        embedder = MiniLMEmbedding()
        reranker = SimpleReranker(embedder)
        
        return reranker.rerank(
            query=query,
            results=candidates,
            top_k=rerank_top_k
        )
