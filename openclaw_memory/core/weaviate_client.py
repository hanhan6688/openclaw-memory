"""
Weaviate 客户端 - 连接本地 weaviate
支持本地安装的 Weaviate 二进制文件和自动端口轮询
"""

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path
import sys
import os
import uuid
import subprocess
import time
import signal
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_URL,
    WEAVIATE_DATA_DIR, WEAVIATE_DIR,
    VECTOR_COLLECTION_PREFIX, KG_COLLECTION_PREFIX
)

# 全局 Weaviate 进程
_weaviate_process = None
_actual_weaviate_port = None  # 实际使用的端口


def find_available_port(start: int = 8080, end: int = 8099) -> Optional[int]:
    """查找可用端口"""
    import socket
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                result = s.connect_ex(("127.0.0.1", port))
                if result != 0:
                    return port
        except Exception:
            continue
    return None


def get_weaviate_port() -> int:
    """获取 Weaviate 端口（优先使用实际端口）"""
    global _actual_weaviate_port
    if _actual_weaviate_port:
        return _actual_weaviate_port
    return WEAVIATE_PORT


class WeaviateClient:
    """Weaviate 客户端"""

    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.user_id = user_id
        # Weaviate 类名不允许包含 -，替换为 _
        safe_agent_id = agent_id.replace('-', '_')
        self.memory_collection = f"{VECTOR_COLLECTION_PREFIX}{safe_agent_id}"
        self.kg_collection = f"{KG_COLLECTION_PREFIX}{safe_agent_id}"
        self._client = None
        self._connected = False  # 添加连接状态标志
    
    @property
    def client(self):
        if self._client is None:
            try:
                port = get_weaviate_port()
                # 确保 host 和 port 不为 None
                host = WEAVIATE_HOST or "localhost"
                port = port or 8080
                self._client = weaviate.WeaviateClient(
                    connection_params=ConnectionParams.from_params(
                        http_host=host,
                        http_port=port,
                        http_secure=False,
                        grpc_host=host, 
                        grpc_port=50051,  # Docker Weaviate 的 gRPC 端口
                        grpc_secure=False,
                    ),
                    skip_init_checks=True  # 跳过初始化检查以避免 gRPC 问题
                )
                self._client.connect()
                self._ensure_collections()
                self._update_schema()  # 更新 schema 添加新字段
                self._connected = True  # 设置连接状态
            except Exception as e:
                print(f"⚠️  Weaviate 连接失败: {e}")
                self._connected = False
                return None  # 返回None表示连接失败
        return self._client

    def _update_schema(self):
        """更新 schema，添加缺失的字段"""
        try:
            collection = self._client.collections.get(self.memory_collection)
            config = collection.config.get()
            existing_props = {p.name for p in config.properties}

            # 需要添加的新字段
            new_props = [
                ("session_id", DataType.TEXT),
                ("role", DataType.TEXT),
                ("quality", DataType.TEXT),
                ("source", DataType.TEXT),
                ("keywords", DataType.TEXT_ARRAY),
                ("tags", DataType.TEXT_ARRAY),
                ("summary_generated", DataType.BOOL),
                ("facts", DataType.TEXT_ARRAY),  # 新：原子化事实
            ]

            for prop_name, prop_type in new_props:
                if prop_name not in existing_props:
                    try:
                        collection.config.add_property(Property(name=prop_name, data_type=prop_type))
                        print(f"✅ 添加字段: {prop_name}")
                    except Exception as e:
                        print(f"⚠️ 添加字段 {prop_name} 失败: {e}")
        except Exception as e:
            print(f"⚠️ 更新 schema 失败: {e}")
    
    def _ensure_collections(self):
        if not self._client.collections.exists(self.memory_collection):
            self._client.collections.create(
                name=self.memory_collection,
                description=f"Agent {self.agent_id} 记忆",
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="memory_type", data_type=DataType.TEXT),
                    Property(name="importance", data_type=DataType.NUMBER),
                    Property(name="timestamp", data_type=DataType.DATE),
                    Property(name="agent_id", data_type=DataType.TEXT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="raw_conversation", data_type=DataType.TEXT),
                    # 新增字段
                    Property(name="session_id", data_type=DataType.TEXT),
                    Property(name="role", data_type=DataType.TEXT),
                    Property(name="quality", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                    Property(name="tags", data_type=DataType.TEXT_ARRAY),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
        
        if not self._client.collections.exists(self.kg_collection):
            self._client.collections.create(
                name=self.kg_collection,
                description=f"Agent {self.agent_id} 知识图谱",
                properties=[
                    Property(name="entity_name", data_type=DataType.TEXT),
                    Property(name="entity_type", data_type=DataType.TEXT),
                    Property(name="relation_type", data_type=DataType.TEXT),
                    Property(name="target_entity", data_type=DataType.TEXT),
                    Property(name="context", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                    Property(name="confidence", data_type=DataType.NUMBER),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
    
    def insert_memory(self, properties: dict, vector: List[float] = None, check_duplicate: bool = True) -> str:
        """插入记忆，支持去重检查
        
        Args:
            properties: 记忆属性
            vector: 向量
            check_duplicate: 是否检查重复（默认 True）
            
        Returns:
            记忆 ID，如果重复则返回 None
        """
        collection = self.client.collections.get(self.memory_collection)
        if "timestamp" not in properties:
            properties["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 去重检查：通过向量搜索检查是否已存在相似记忆
        if check_duplicate and vector:
            try:
                results = collection.query.near_vector(
                    near_vector=vector,
                    limit=5,  # 增加搜索范围
                    return_metadata=MetadataQuery(distance=True)
                )
                # 如果距离小于 0.15（相似度 > 85%），认为是重复
                for obj in results.objects:
                    if obj.metadata.distance < 0.15:
                        content = properties.get("content", "")[:100]
                        existing = str(obj.properties.get("content", ""))[:100]
                        # 内容也相似，跳过
                        if content and existing and (content in existing or existing in content):
                            print(f"⏭️ 跳过重复记忆: {content[:50]}...")
                            return None
            except Exception as e:
                print(f"⚠️ 去重检查失败: {e}")
        
        obj_uuid = str(uuid.uuid4())
        collection.data.insert(uuid=obj_uuid, properties=properties, vector=vector)
        return obj_uuid
    
    def update_memory(self, memory_id: str, properties: dict) -> bool:
        """更新记忆属性"""
        try:
            collection = self.client.collections.get(self.memory_collection)
            collection.data.update(uuid=memory_id, properties=properties)
            return True
        except Exception as e:
            print(f"更新记忆失败: {e}")
            return False

    def update_memory_vector(self, memory_id: str, vector: List[float]) -> bool:
        """更新记忆的向量"""
        try:
            collection = self.client.collections.get(self.memory_collection)
            # Weaviate 需要 properties 和 vector 一起更新
            # 先获取现有属性
            obj = collection.query.fetch_object_by_id(memory_id)
            if obj:
                collection.data.update(uuid=memory_id, properties=obj.properties, vector=vector)
                return True
            return False
        except Exception as e:
            print(f"更新向量失败: {e}")
            return False

    def insert_memory_batch(self, items: List[Dict]) -> int:
        collection = self.client.collections.get(self.memory_collection)
        count = 0
        with collection.batch.dynamic() as batch:
            for item in items:
                p = item.get("properties", {})
                if "timestamp" not in p:
                    p["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                batch.add_object(uuid=str(uuid.uuid4()), properties=p, vector=item.get("vector"))
                count += 1
        return count
    
    def search_memory(self, vector: List[float], limit: int = 10, sort_by_importance: bool = True) -> List[dict]:
        """向量检索记忆，支持按 importance 排序
        
        Args:
            vector: 查询向量
            limit: 返回数量
            sort_by_importance: 是否按重要性排序（高重要性优先）
        """
        # 检查连接状态
        if not self._connected:
            return []

        try:
            collection = self.client.collections.get(self.memory_collection)
            # 获取更多结果用于重排序
            fetch_limit = limit * 3 if sort_by_importance else limit
            results = collection.query.near_vector(
                near_vector=vector, limit=fetch_limit,
                return_metadata=MetadataQuery(distance=True)
            )
            
            memories = [
                {"_additional": {"id": str(o.uuid), "distance": o.metadata.distance},
                 "content": o.properties.get("content"),
                 "memory_type": o.properties.get("memory_type"),
                 "importance": o.properties.get("importance", 0.5),
                 "timestamp": str(o.properties.get("timestamp", "")),
                 "summary": o.properties.get("summary"),
                 "keywords": o.properties.get("keywords", []),
                 "raw_conversation": o.properties.get("raw_conversation")}
                for o in results.objects
            ]
            
            # 按 importance 排序（高优先），同时考虑向量相似度
            if sort_by_importance and memories:
                # 计算综合分数：importance * 0.4 + (1 - normalized_distance) * 0.6
                max_distance = max(m.get("_additional", {}).get("distance", 1) for m in memories) or 1
                
                def score_memory(m):
                    distance = m.get("_additional", {}).get("distance", 1)
                    importance = m.get("importance", 0.5)
                    # 归一化距离（距离越小越相似）
                    normalized_sim = 1 - (distance / max_distance) if max_distance > 0 else 0
                    # 综合分数：importance权重0.4，相似度权重0.6
                    return importance * 0.4 + normalized_sim * 0.6
                
                memories.sort(key=score_memory, reverse=True)
                memories = memories[:limit]
            
            return memories
        except Exception as e:
            print(f"⚠️ 搜索记忆时出错: {e}")
            return []

    def hybrid_search(self, query: str, vector: List[float] = None, 
                      alpha: float = 0.5, limit: int = 10) -> List[dict]:
        """
        Weaviate 原生混合检索 (向量 + BM25)
        
        Args:
            query: 查询文本 (用于 BM25)
            vector: 查询向量 (可选，会自动生成)
            alpha: 混合权重 (0 = 纯 BM25, 1 = 纯向量, 0.5 = 平衡)
            limit: 返回数量
        """
        if not self._connected:
            return []
        
        try:
            collection = self.client.collections.get(self.memory_collection)
            
            if vector:
                results = collection.query.hybrid(
                    query=query,
                    vector=vector,
                    alpha=alpha,
                    limit=limit,
                    return_metadata=MetadataQuery(score=True, explain_score=True)
                )
            else:
                results = collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    limit=limit,
                    return_metadata=MetadataQuery(score=True, explain_score=True)
                )
            
            memories = []
            for o in results.objects:
                memory = {
                    "_additional": {
                        "id": str(o.uuid),
                        "score": o.metadata.score if o.metadata else None,
                        "explain_score": o.metadata.explain_score if o.metadata else None
                    },
                    "content": o.properties.get("content"),
                    "summary": o.properties.get("summary"),
                    "importance": o.properties.get("importance", 0.5),
                    "timestamp": str(o.properties.get("timestamp", "")),
                    "keywords": o.properties.get("keywords", []),
                    "memory_type": o.properties.get("memory_type"),
                    "session_id": o.properties.get("session_id"),
                    "role": o.properties.get("role"),
                }
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            print(f"⚠️ 混合检索出错: {e}")
            return []

    def bm25_search(self, query: str, limit: int = 10) -> List[dict]:
        """
        Weaviate 原生 BM25 检索
        
        Args:
            query: 查询文本
            limit: 返回数量
        """
        if not self._connected:
            return []
        
        try:
            collection = self.client.collections.get(self.memory_collection)
            
            results = collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            
            memories = []
            for o in results.objects:
                memory = {
                    "_additional": {
                        "id": str(o.uuid),
                        "score": o.metadata.score if o.metadata else None
                    },
                    "content": o.properties.get("content"),
                    "summary": o.properties.get("summary"),
                    "importance": o.properties.get("importance", 0.5),
                    "timestamp": str(o.properties.get("timestamp", "")),
                    "keywords": o.properties.get("keywords", []),
                }
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            print(f"⚠️ BM25 检索出错: {e}")
            return []

    def get_memories(self, limit: int = 100) -> List[dict]:
        # 确保客户端已连接
        if self.client is None:
            return []
        
        try:
            collection = self.client.collections.get(self.memory_collection)
            results = collection.query.fetch_objects(limit=limit)
            return [
                {"_additional": {"id": str(o.uuid)},
                 "content": o.properties.get("content"),
                 "memory_type": o.properties.get("memory_type"),
                 "importance": o.properties.get("importance"),
                 "timestamp": str(o.properties.get("timestamp", "")),
                 "summary": o.properties.get("summary"),
                 "raw_conversation": o.properties.get("raw_conversation"),
                 "agent_id": o.properties.get("agent_id"),
                 "user_id": o.properties.get("user_id"),
                 # 新增字段
                 "session_id": o.properties.get("session_id"),
                 "role": o.properties.get("role"),
                 "source": o.properties.get("source"),
                 "quality": o.properties.get("quality"),
                 "keywords": o.properties.get("keywords"),
                 "tags": o.properties.get("tags"),
                 "summary_generated": o.properties.get("summary_generated")}
                for o in results.objects
            ]
        except Exception as e:
            print(f"⚠️ 获取记忆时出错: {e}")
            return []
    
    def get_memories_by_time(self, start_time: datetime, end_time: datetime, limit: int = 100) -> List[dict]:
        # 检查连接状态
        if self.client is None:
            return []
        
        try:
            collection = self.client.collections.get(self.memory_collection)
            results = collection.query.fetch_objects(
                filters=Filter.by_property("timestamp").greater_or_equal(start_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
                            & Filter.by_property("timestamp").less_or_equal(end_time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                limit=limit
            )
            return [
                {"_additional": {"id": str(o.uuid)},
                 "content": o.properties.get("content"),
                 "timestamp": str(o.properties.get("timestamp", "")),
                 "summary": o.properties.get("summary"),
                 "raw_conversation": o.properties.get("raw_conversation")}
                for o in results.objects
            ]
        except Exception as e:
            print(f"⚠️ 按时间获取记忆时出错: {e}")
            return []
    
    def insert_kg(self, properties: dict, vector: List[float] = None) -> str:
        collection = self.client.collections.get(self.kg_collection)
        if "timestamp" not in properties:
            properties["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        obj_uuid = str(uuid.uuid4())
        collection.data.insert(uuid=obj_uuid, properties=properties, vector=vector)
        return obj_uuid
    
    def get_kg(self, limit: int = 500) -> List[dict]:
        # 确保客户端已连接
        if self.client is None:
            return []
        
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=limit)
            return [
                {"_additional": {"id": str(o.uuid)},
                 "entity_name": o.properties.get("entity_name"),
                 "entity_type": o.properties.get("entity_type"),
                 "relation_type": o.properties.get("relation_type"),
                 "target_entity": o.properties.get("target_entity"),
                 "context": o.properties.get("context"),
                 "confidence": o.properties.get("confidence")}
                for o in results.objects
            ]
        except Exception as e:
            print(f"⚠️ 获取知识图谱时出错: {e}")
            return []
    


    def get_all_entities(self, limit: int = 500) -> List[dict]:
        """获取所有实体（relation_type 为 is_entity 的记录）"""
        if self.client is None:
            return []
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=limit)
            entities = []
            for o in results.objects:
                if o.properties.get("relation_type") == "is_entity":
                    entities.append({
                        "id": str(o.uuid),
                        "entity_name": o.properties.get("entity_name"),
                        "entity_type": o.properties.get("entity_type"),
                        "confidence": o.properties.get("confidence")
                    })
            return entities
        except Exception as e:
            print(f"⚠️ 获取实体时出错: {e}")
            return []

    def get_all_relations(self, limit: int = 500) -> List[dict]:
        """获取所有关系（relation_type 不为 is_entity 的记录）"""
        if self.client is None:
            return []
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=limit)
            relations = []
            for o in results.objects:
                rel_type = o.properties.get("relation_type")
                if rel_type and rel_type != "is_entity":
                    relations.append({
                        "id": str(o.uuid),
                        "source": o.properties.get("entity_name"),
                        "relation": rel_type,
                        "target": o.properties.get("target_entity"),
                        "confidence": o.properties.get("confidence")
                    })
            return relations
        except Exception as e:
            print(f"⚠️ 获取关系时出错: {e}")
            return []

    def get_kg_by_entity(self, entity_name: str) -> List[dict]:
        """获取与实体相关的所有记录"""
        if self.client is None:
            return []
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=1000)
            return [
                {"id": str(o.uuid), **o.properties}
                for o in results.objects
                if o.properties.get("entity_name") == entity_name
            ]
        except Exception as e:
            print(f"⚠️ 获取实体关系时出错: {e}")
            return []

    def search_kg_entities(self, vector: List[float], limit: int = 10) -> List[dict]:
        """向量搜索实体"""
        if self.client is None:
            return []
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.near_vector(
                near_vector=vector,
                limit=limit
            )
            return [
                {"id": str(o.uuid), **o.properties}
                for o in results.objects
            ]
        except Exception as e:
            print(f"⚠️ 搜索实体时出错: {e}")
            return []

    def delete_kg_by_id(self, uuid_str: str) -> bool:
        """根据UUID删除知识图谱条目"""
        if self.client is None:
            return False
        try:
            collection = self.client.collections.get(self.kg_collection)
            collection.data.delete_by_id(uuid_str)
            return True
        except Exception as e:
            print(f"⚠️ 删除知识图谱条目时出错: {e}")
            return False

    def delete_kg_by_entity(self, entity_name: str) -> int:
        """删除与实体相关的所有知识图谱条目"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.kg_collection)
            # 获取所有相关条目
            results = collection.query.fetch_objects(limit=1000)
            deleted_count = 0
            for obj in results.objects:
                if obj.properties.get("entity_name") == entity_name or obj.properties.get("target_entity") == entity_name:
                    try:
                        collection.data.delete_by_id(str(obj.uuid))
                        deleted_count += 1
                    except Exception:
                        pass
            return deleted_count
        except Exception as e:
            print(f"⚠️ 删除实体时出错: {e}")
            return 0

    def delete_kg_by_relation(self, source_entity: str, relation_type: str, target_entity: str) -> int:
        """删除特定关系"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=1000)
            deleted_count = 0
            for obj in results.objects:
                if (obj.properties.get("entity_name") == source_entity and
                    obj.properties.get("relation_type") == relation_type and
                    obj.properties.get("target_entity") == target_entity):
                    try:
                        collection.data.delete_by_id(str(obj.uuid))
                        deleted_count += 1
                    except Exception:
                        pass
            return deleted_count
        except Exception as e:
            print(f"⚠️ 删除关系时出错: {e}")
            return 0

    def delete_all_entities(self) -> int:
        """删除所有实体（没有 relation_type 的记录）"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=5000)
            deleted_count = 0
            for obj in results.objects:
                # 没有关系类型的记录是实体
                if not obj.properties.get("relation_type"):
                    try:
                        collection.data.delete_by_id(str(obj.uuid))
                        deleted_count += 1
                    except Exception:
                        pass
            return deleted_count
        except Exception as e:
            print(f"⚠️ 删除所有实体时出错: {e}")
            return 0

    def delete_all_relations(self) -> int:
        """删除所有关系（有 relation_type 的记录）"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=5000)
            deleted_count = 0
            for obj in results.objects:
                # 有关系类型的记录是关系
                if obj.properties.get("relation_type"):
                    try:
                        collection.data.delete_by_id(str(obj.uuid))
                        deleted_count += 1
                    except Exception:
                        pass
            return deleted_count
        except Exception as e:
            print(f"⚠️ 删除所有关系时出错: {e}")
            return 0

    def count_memories(self) -> int:
        """统计记忆数量"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.memory_collection)
            # 使用 estimate_object_count 快速获取数量
            try:
                return collection.aggregate().total_count
            except Exception:
                # 备用方法：手动遍历计数
                count = 0
                results = collection.query.fetch_objects(limit=100)
                count += len(results.objects)
                while results.objects and len(results.objects) == 100:
                    # 继续获取更多
                    results = collection.query.fetch_objects(limit=100)
                    count += len(results.objects)
                return count
        except Exception as e:
            print(f"⚠️ 统计记忆数量失败: {e}")
            return 0

    def count_entities(self) -> int:
        """统计实体数量"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=5000)
            entities = set()
            for obj in results.objects:
                if obj.properties.get("relation_type"):
                    # 关系记录，提取两端实体
                    entities.add(obj.properties.get("entity_name"))
                    entities.add(obj.properties.get("target_entity"))
                else:
                    # 实体记录
                    entities.add(obj.properties.get("entity_name"))
            return len(entities)
        except Exception:
            return 0

    def count_relations(self) -> int:
        """统计关系数量"""
        if self.client is None:
            return 0
        try:
            collection = self.client.collections.get(self.kg_collection)
            results = collection.query.fetch_objects(limit=5000)
            return sum(1 for obj in results.objects if obj.properties.get("relation_type"))
        except Exception:
            return 0


    def get_memory_by_id(self, uuid_str: str) -> Optional[dict]:
        """根据UUID获取记忆"""
        if self.client is None:
            return None
        try:
            collection = self.client.collections.get(self.memory_collection)
            result = collection.query.fetch_object_by_id(uuid_str)
            if result:
                return {
                    "id": str(result.uuid),
                    "content": result.properties.get("content"),
                    "summary": result.properties.get("summary"),
                    "keywords": result.properties.get("keywords", []),
                    "importance": result.properties.get("importance", 0.5),
                    "timestamp": str(result.properties.get("timestamp", "")),
                    "memory_type": result.properties.get("memory_type"),
                    "tags": result.properties.get("tags", []),
                    "source": result.properties.get("source")
                }
            return None
        except Exception as e:
            print(f"⚠️ 获取记忆时出错: {e}")
            return None

    def delete_memory_by_id(self, uuid_str: str) -> bool:
        """根据UUID删除记忆"""
        if self.client is None:
            return False
        try:
            collection = self.client.collections.get(self.memory_collection)
            collection.data.delete_by_id(uuid_str)
            return True
        except Exception as e:
            print(f"⚠️ 删除记忆时出错: {e}")
            return False

    def cleanup_low_importance_memories(self, threshold: float = 0.3, keep_recent: int = 100) -> Dict:
        """清理低 importance 的记忆，保留最近的高 importance 记忆
        
        Args:
            threshold: importance 阈值，低于此值的记忆将被清理
            keep_recent: 保留最近N条记忆（无论importance）
        
        Returns:
            清理统计信息
        """
        if self.client is None:
            return {"deleted": 0, "error": "Weaviate未连接"}
        
        try:
            collection = self.client.collections.get(self.memory_collection)
            
            # 获取所有记忆
            results = collection.query.fetch_objects(limit=1000)
            
            # 按 timestamp 排序
            memories = []
            for o in results.objects:
                memories.append({
                    "id": str(o.uuid),
                    "importance": o.properties.get("importance", 0.5),
                    "timestamp": str(o.properties.get("timestamp", "")),
                    "content": o.properties.get("content", "")[:50]
                })
            
            # 按时间排序（最新的在前）
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # 保护最近N条记忆
            protected_ids = set(m["id"] for m in memories[:keep_recent])
            
            # 找出需要删除的低 importance 记忆
            to_delete = []
            for m in memories:
                if m["id"] not in protected_ids and m.get("importance", 0.5) < threshold:
                    to_delete.append(m)
            
            # 执行删除
            deleted = 0
            for m in to_delete:
                try:
                    collection.data.delete_by_id(m["id"])
                    deleted += 1
                except Exception:
                    pass
            
            return {
                "deleted": deleted,
                "total_memories": len(memories),
                "protected": len(protected_ids),
                "threshold": threshold
            }
        except Exception as e:
            print(f"⚠️ 清理记忆时出错: {e}")
            return {"deleted": 0, "error": str(e)}

    def get_memories_by_importance(self, min_importance: float = 0.7, limit: int = 100) -> List[dict]:
        """获取高 importance 的记忆
        
        Args:
            min_importance: 最小 importance 阈值
            limit: 返回数量
        """
        if self.client is None:
            return []
        
        try:
            collection = self.client.collections.get(self.memory_collection)
            results = collection.query.fetch_objects(limit=500)
            
            memories = [
                {"_additional": {"id": str(o.uuid)},
                 "content": o.properties.get("content"),
                 "importance": o.properties.get("importance", 0.5),
                 "timestamp": str(o.properties.get("timestamp", "")),
                 "summary": o.properties.get("summary")}
                for o in results.objects
                if o.properties.get("importance", 0.5) >= min_importance
            ]
            
            # 按 importance 排序
            memories.sort(key=lambda x: x.get("importance", 0.5), reverse=True)
            return memories[:limit]
        except Exception as e:
            print(f"⚠️ 获取高重要性记忆时出错: {e}")
            return []

    def close(self):
        if self._client:
            try: self._client.close()
            except: pass
            self._client = None


WeaviateRestClient = WeaviateClient


def check_weaviate() -> bool:
    """检查 Weaviate 是否运行"""
    try:
        import requests
        r = requests.get(f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}/v1/.well-known/ready", timeout=5)
        if r.status_code == 200:
            print(f"✅ Weaviate 运行中 (http://localhost:{WEAVIATE_PORT})")
            return True
    except Exception:
        pass
    print(f"❌ Weaviate 未运行")
    return False


def get_weaviate_binary() -> Optional[str]:
    """获取 Weaviate 可执行文件路径"""
    # 优先使用本地安装
    local_bin = WEAVIATE_DIR / "weaviate"
    if local_bin.exists():
        return str(local_bin)
    
    # 回退到系统安装
    system_bin = shutil.which("weaviate")
    if system_bin:
        return system_bin
    
    return None


def ensure_weaviate_installed() -> bool:
    """确保 Weaviate 已安装"""
    binary = get_weaviate_binary()
    if binary:
        return True
    
    print("❌ Weaviate 未安装")
    print("\n安装方式:")
    print("  1. 本地安装 (推荐): python scripts/download_weaviate.py")
    print("  2. 系统安装: brew install weaviate")
    return False


def start_weaviate(background: bool = True, port: Optional[int] = None) -> bool:
    """启动 Weaviate（支持自动端口轮询）"""
    global _weaviate_process, _actual_weaviate_port

    # 检查是否已运行
    if check_weaviate():
        return True

    # 获取 Weaviate 二进制文件
    weaviate_bin = get_weaviate_binary()
    if not weaviate_bin:
        # 尝试自动下载
        print("🔄 Weaviate 未安装，尝试自动下载...")
        try:
            from scripts.download_weaviate import download_weaviate
            if not download_weaviate():
                return False
            weaviate_bin = get_weaviate_binary()
        except ImportError:
            print("❌ 请手动安装: python scripts/install_weaviate.py")
            return False

    # 端口轮询
    if port is None:
        port = find_available_port(8080, 8099)
        if not port:
            print("❌ 没有找到可用端口 (8080-8099)")
            return False

    print(f"🚀 启动 Weaviate...")
    print(f"   二进制: {weaviate_bin}")
    print(f"   数据目录: {WEAVIATE_DATA_DIR}")
    print(f"   端口: {port}")

    # 环境变量
    env = os.environ.copy()
    env["PERSISTENCE_DATA_PATH"] = str(WEAVIATE_DATA_DIR)
    env["AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"] = "true"
    env["QUERY_DEFAULTS_LIMIT"] = "100"
    env["DISABLE_MODULES"] = "text2vec-openai,text2vec-cohere,text2vec-palm"
    env["DEFAULT_VECTORIZER_MODULE"] = "none"

    try:
        if background:
            _weaviate_process = subprocess.Popen(
                [weaviate_bin, "--host", WEAVIATE_HOST, "--port", str(port)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _actual_weaviate_port = port

            # 等待启动
            for i in range(20):
                time.sleep(0.5)
                if check_weaviate():
                    print(f"✅ Weaviate 启动成功 (端口: {port})")
                    return True

            print("⚠️ Weaviate 启动超时")
            _actual_weaviate_port = None
            return False
        else:
            subprocess.run(
                [weaviate_bin, "--host", WEAVIATE_HOST, "--port", str(port)],
                env=env,
            )
            _actual_weaviate_port = port
            return True
    except FileNotFoundError:
        print("❌ Weaviate 可执行文件不存在")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        _actual_weaviate_port = None
        return False


def stop_weaviate():
    """停止 Weaviate"""
    global _weaviate_process
    if _weaviate_process:
        _weaviate_process.terminate()
        _weaviate_process = None
        print("✅ Weaviate 已停止")


# ==================== 电商模块扩展方法 ====================

class CommerceWeaviateClient(WeaviateClient):
    """电商模块专用 Weaviate 客户端"""
    
    def __init__(self, agent_id: str, collection_type: str = "default"):
        super().__init__(agent_id, "commerce")
        self.collection_type = collection_type
        self.user_profile_collection = f"UserProfile_{agent_id}"
        self.product_profile_collection = f"ProductProfile_{agent_id}"
        self.commerce_kg_collection = f"CommerceKG_{agent_id}"
    
    def insert_object(self, class_name: str, properties: dict, vector: List[float] = None) -> str:
        """插入对象到指定 collection"""
        collection = self.client.collections.get(class_name)
        obj_uuid = str(uuid.uuid4())
        collection.data.insert(uuid=obj_uuid, properties=properties, vector=vector)
        return obj_uuid
    
    def update_object(self, class_name: str, object_id: str, properties: dict, vector: List[float] = None):
        """更新对象"""
        collection = self.client.collections.get(class_name)
        collection.data.update(uuid=object_id, properties=properties, vector=vector)
    
    def delete_object(self, class_name: str, object_id: str):
        """删除对象"""
        collection = self.client.collections.get(class_name)
        collection.data.delete_by_id(object_id)
    
    def query_objects(self, class_name: str, where: dict = None, limit: int = 100) -> List[dict]:
        """查询对象"""
        collection = self.client.collections.get(class_name)
        
        if where:
            # 构建 filter
            filter_obj = self._build_filter(where)
            results = collection.query.fetch_objects(filters=filter_obj, limit=limit)
        else:
            results = collection.query.fetch_objects(limit=limit)
        
        return [
            {"_additional": {"id": str(o.uuid)}, **dict(o.properties)}
            for o in results.objects
        ]
    
    def vector_search(self, class_name: str, vector: List[float], limit: int = 10, filters: dict = None) -> List[dict]:
        """向量搜索"""
        collection = self.client.collections.get(class_name)
        
        if filters:
            filter_obj = self._build_filter(filters)
            results = collection.query.near_vector(
                near_vector=vector, limit=limit,
                filters=filter_obj,
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            results = collection.query.near_vector(
                near_vector=vector, limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )
        
        return [
            {"_additional": {"id": str(o.uuid), "distance": o.metadata.distance}, **dict(o.properties)}
            for o in results.objects
        ]
    
    def get_objects(self, class_name: str, limit: int = 100) -> List[dict]:
        """获取所有对象"""
        collection = self.client.collections.get(class_name)
        results = collection.query.fetch_objects(limit=limit)
        return [
            {"_additional": {"id": str(o.uuid)}, **dict(o.properties)}
            for o in results.objects
        ]
    
    def _build_filter(self, where: dict):
        """构建 Weaviate filter"""
        from weaviate.classes.query import Filter as WeaviateFilter
        
        operator = where.get("operator")
        
        if operator == "Equal":
            path = where.get("path", [])
            value = where.get("valueString") or where.get("valueNumber") or where.get("valueBoolean")
            if len(path) == 2:
                # 嵌套属性，如 ["basic", "age_range"]
                return WeaviateFilter.by_property(path[0]).contains_any([value])
            return WeaviateFilter.by_property(path[0]).equal(value)
        
        elif operator == "ContainsAny":
            path = where.get("path", [])
            values = where.get("valueText", [])
            return WeaviateFilter.by_property(path[0]).contains_any(values)
        
        elif operator == "And":
            operands = where.get("operands", [])
            filters = [self._build_filter(op) for op in operands]
            result = filters[0]
            for f in filters[1:]:
                result = result & f
            return result
        
        elif operator == "Or":
            operands = where.get("operands", [])
            filters = [self._build_filter(op) for op in operands]
            result = filters[0]
            for f in filters[1:]:
                result = result | f
            return result
        
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "start", "stop"])
    parser.add_argument("--foreground", action="store_true", help="前台运行")
    args = parser.parse_args()
    
    if args.action == "check":
        check_weaviate()
    elif args.action == "start":
        start_weaviate(background=not args.foreground)
    elif args.action == "stop":
        stop_weaviate()
