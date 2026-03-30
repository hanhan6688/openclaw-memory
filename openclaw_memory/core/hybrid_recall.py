"""
混合检索引擎 (Hybrid Recall Engine)

使用 Weaviate 的 BM25 + 向量搜索，手动融合结果
支持时间衰减和相关实体返回、中文分词预处理
"""

import math
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weaviate_client import WeaviateClient
from core.networkx_kg_client import get_nx_client
from core.embeddings import OllamaEmbedding


def tokenize_chinese_for_search(text: str) -> str:
    """
    中文分词预处理（用于 BM25 搜索）
    
    将中文按字符分割，英文按词分割
    """
    if not text:
        return ""
    
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        char = text[i]
        
        # 中文字符单独处理
        if '\u4e00' <= char <= '\u9fff':
            tokens.append(char)
            i += 1
        # 英文字母，收集整个单词
        elif char.isalpha():
            word_start = i
            while i < n and text[i].isalpha() and not ('\u4e00' <= text[i] <= '\u9fff'):
                i += 1
            tokens.append(text[word_start:i])
        # 数字，收集整个数字
        elif char.isdigit():
            num_start = i
            while i < n and text[i].isdigit():
                i += 1
            tokens.append(text[num_start:i])
        else:
            # 其他字符跳过
            i += 1
    
    return ' '.join(tokens)


class HybridRecallEngine:
    """
    混合检索引擎
    
    融合 BM25 和向量搜索结果
    """
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.vector_client = WeaviateClient(agent_id)
        self.kg_client = get_nx_client(agent_id)
        self.embedder = OllamaEmbedding()
        
        # 混合权重 (BM25 30% + 向量 60% + 时间 10%)
        self.bm25_weight = 0.30
        self.vector_weight = 0.60
        self.time_weight = 0.10
        
        # 时间衰减参数
        self.half_life_days = 7
    
    def recall(
        self,
        query: str,
        limit: int = 5,
        include_entities: bool = True
    ) -> Dict:
        """
        混合检索
        
        Args:
            query: 查询文本
            limit: 返回数量
            include_entities: 是否包含相关实体
        
        Returns:
            {
                "memories": [...],
                "entities": [...],
                "mode": "hybrid"
            }
        """
        results = {
            "memories": [],
            "entities": [],
            "mode": "hybrid"
        }
        
        if not self.vector_client.client:
            results["mode"] = "error"
            results["error"] = "Weaviate 未连接"
            return results
        
        try:
            collection = self.vector_client.client.collections.get(
                self.vector_client.memory_collection
            )
            
            # 1. BM25 搜索（使用中文分词预处理）
            bm25_results = {}
            try:
                # 对查询进行中文分词预处理
                tokenized_query = tokenize_chinese_for_search(query)
                
                bm25_objs = collection.query.bm25(
                    query=tokenized_query,
                    limit=limit * 2,
                    return_metadata=["score"]
                )
                
                # 归一化 BM25 分数
                scores = [obj.metadata.score for obj in bm25_objs.objects if obj.metadata and obj.metadata.score]
                max_score = max(scores) if scores else 1.0
                
                for obj in bm25_objs.objects:
                    obj_id = str(obj.uuid)
                    raw_score = obj.metadata.score if obj.metadata else 0
                    score = raw_score / max_score if max_score > 0 else 0.5
                    bm25_results[obj_id] = {
                        "id": obj_id,
                        "content": obj.properties.get("content", ""),
                        "summary": obj.properties.get("summary", ""),
                        "timestamp": obj.properties.get("timestamp", ""),
                        "importance": obj.properties.get("importance", 0.5),
                        "bm25_score": min(1.0, score),
                        "vector_score": 0
                    }
            except Exception as e:
                print(f"⚠️ BM25 搜索失败: {e}")
            
            # 2. 向量搜索
            try:
                query_vector = self.embedder.embed(query)
                vector_objs = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit * 2,
                    return_metadata=["distance"]
                )
                
                # 如果 BM25 没有结果，设置一个默认分数
                if not bm25_results:
                    default_bm25_score = 0.0
                else:
                    # 计算 BM25 平均分数作为默认值
                    avg_bm25 = sum(m["bm25_score"] for m in bm25_results.values()) / len(bm25_results)
                    default_bm25_score = avg_bm25 * 0.5  # 非匹配的记忆 BM25 分数减半
                
                for obj in vector_objs.objects:
                    obj_id = str(obj.uuid)
                    dist = obj.metadata.distance if obj.metadata else 0.5
                    vec_score = 1 - dist
                    
                    if obj_id in bm25_results:
                        # 同时出现在 BM25 和向量结果中，BM25 分数加成
                        bm25_results[obj_id]["vector_score"] = vec_score
                        bm25_results[obj_id]["bm25_boost"] = True
                    else:
                        # 只出现在向量结果中
                        bm25_results[obj_id] = {
                            "id": obj_id,
                            "content": obj.properties.get("content", ""),
                            "summary": obj.properties.get("summary", ""),
                            "timestamp": obj.properties.get("timestamp", ""),
                            "importance": obj.properties.get("importance", 0.5),
                            "bm25_score": default_bm25_score,
                            "vector_score": vec_score
                        }
            except Exception as e:
                print(f"⚠️ 向量搜索失败: {e}")
            
            # 3. 时间衰减 + 综合分数
            now = datetime.now(timezone.utc)
            
            for mem_id, mem in bm25_results.items():
                # 时间分数
                time_score = 0.5
                if mem.get("timestamp"):
                    try:
                        ts = datetime.fromisoformat(mem["timestamp"].replace("Z", "+00:00"))
                        days_ago = (now - ts.replace(tzinfo=None)).days
                        time_score = math.exp(-days_ago / self.half_life_days)
                        time_score = max(0.1, time_score)
                    except Exception:
                        pass
                
                # 综合分数
                final_score = (
                    mem["bm25_score"] * self.bm25_weight +
                    mem["vector_score"] * self.vector_weight +
                    time_score * self.time_weight
                )
                
                mem["time_score"] = time_score
                mem["final_score"] = final_score
            
            # 4. 排序
            sorted_results = sorted(
                bm25_results.values(),
                key=lambda x: x.get("final_score", 0),
                reverse=True
            )
            
            results["memories"] = sorted_results[:limit]
            
        except Exception as e:
            print(f"⚠️ 混合检索失败: {e}")
            results["mode"] = "error"
            results["error"] = str(e)
        
        # 5. 图谱实体
        if include_entities:
            try:
                entities = self.kg_client.search_entities(query, limit=5)
                results["entities"] = entities
            except Exception:
                pass
        
        return results


def get_recall_engine(agent_id: str = "main") -> HybridRecallEngine:
    """获取检索引擎"""
    return HybridRecallEngine(agent_id)