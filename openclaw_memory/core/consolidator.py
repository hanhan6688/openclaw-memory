"""
记忆整合器 (Memory Consolidator)

功能：
1. 合并相似记忆
2. 消除冗余信息
3. 提取共性，形成高层理解
4. 定期自动运行
"""

import os
import sys
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weaviate_client import WeaviateClient
from core.embeddings import OllamaEmbedding
from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


class MemoryConsolidator:
    """
    记忆整合器
    
    整合策略：
    1. 找出相似度 > 0.8 的记忆对
    2. 使用 LLM 合并内容
    3. 保留重要性更高的记忆
    4. 删除被合并的记忆
    """
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.client = WeaviateClient(agent_id)
        self.embedder = OllamaEmbedding()
        self.llm_url = OLLAMA_BASE_URL
        self.llm_model = OLLAMA_CHAT_MODEL
    
    def find_similar_memories(
        self,
        threshold: float = 0.85,
        limit: int = 100
    ) -> List[Dict]:
        """
        找出相似的记忆对
        
        Args:
            threshold: 相似度阈值
            limit: 检查的记忆数量
        
        Returns:
            相似记忆对列表
        """
        if not self.client.client:
            return []
        
        similar_pairs = []
        
        try:
            collection = self.client.client.collections.get(
                self.client.memory_collection
            )
            
            # 获取所有记忆
            results = collection.query.fetch_objects(limit=limit)
            
            memories = []
            for obj in results.objects:
                memories.append({
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "summary": obj.properties.get("summary", ""),
                    "importance": obj.properties.get("importance", 0.5),
                    "timestamp": obj.properties.get("timestamp", "")
                })
            
            # 两两比较
            for i, m1 in enumerate(memories):
                for m2 in memories[i+1:]:
                    # 计算相似度
                    sim = self._calculate_similarity(
                        m1["content"], m2["content"]
                    )
                    
                    if sim >= threshold:
                        similar_pairs.append({
                            "memory1": m1,
                            "memory2": m2,
                            "similarity": sim
                        })
            
            return similar_pairs
            
        except Exception as e:
            print(f"⚠️ 查找相似记忆失败: {e}")
            return []
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 生成向量
        v1 = self.embedder.embed(text1[:200])
        v2 = self.embedder.embed(text2[:200])
        
        # 余弦相似度
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def merge_memories(
        self,
        memory1: Dict,
        memory2: Dict
    ) -> Optional[Dict]:
        """
        合并两条记忆
        
        Returns:
            合并后的记忆内容
        """
        prompt = f"""请合并以下两条相似的记忆，保留重要信息，去除重复内容：

记忆1: {memory1['content'][:500]}
(重要性: {memory1.get('importance', 0.5)})

记忆2: {memory2['content'][:500]}
(重要性: {memory2.get('importance', 0.5)})

请输出合并后的记忆内容（不超过 300 字）:"""

        try:
            response = requests.post(
                f"{self.llm_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
        except Exception as e:
            print(f"⚠️ 合并记忆失败: {e}")
        
        return None
    
    def consolidate(
        self,
        threshold: float = 0.85,
        limit: int = 100,
        dry_run: bool = True
    ) -> Dict:
        """
        整合记忆
        
        Args:
            threshold: 相似度阈值
            limit: 检查的记忆数量
            dry_run: 是否只预览不执行
        
        Returns:
            整合结果
        """
        result = {
            "checked": 0,
            "similar_pairs": 0,
            "merged": 0,
            "deleted": 0,
            "details": []
        }
        
        # 找出相似记忆
        similar_pairs = self.find_similar_memories(threshold, limit)
        result["checked"] = limit
        result["similar_pairs"] = len(similar_pairs)
        
        if not similar_pairs:
            return result
        
        print(f"找到 {len(similar_pairs)} 对相似记忆")
        
        # 合并每对记忆
        for pair in similar_pairs:
            m1, m2 = pair["memory1"], pair["memory2"]
            
            print(f"\n相似度 {pair['similarity']:.2f}:")
            print(f"  记忆1: {m1['content'][:50]}...")
            print(f"  记忆2: {m2['content'][:50]}...")
            
            if dry_run:
                result["details"].append({
                    "similarity": pair["similarity"],
                    "memory1_id": m1["id"],
                    "memory2_id": m2["id"],
                    "action": "would_merge"
                })
                continue
            
            # 合并
            merged_content = self.merge_memories(m1, m2)
            
            if merged_content:
                # 更新重要性更高的记忆
                target = m1 if m1["importance"] >= m2["importance"] else m2
                to_delete = m2 if m1["importance"] >= m2["importance"] else m1
                
                # 更新记忆
                self.client.update_memory(target["id"], {
                    "content": merged_content,
                    "consolidated": True,
                    "consolidated_at": datetime.now(timezone.utc).isoformat()
                })
                
                # 删除另一条
                self.client.delete_memory(to_delete["id"])
                
                result["merged"] += 1
                result["deleted"] += 1
                
                print(f"  ✅ 已合并")
        
        return result


def run_consolidation(agent_id: str = "main", dry_run: bool = True):
    """运行记忆整合"""
    print(f"\n{'='*50}")
    print(f"记忆整合 - {agent_id}")
    print(f"模式: {'预览' if dry_run else '执行'}")
    print(f"{'='*50}\n")
    
    consolidator = MemoryConsolidator(agent_id)
    result = consolidator.consolidate(dry_run=dry_run)
    
    print(f"\n结果:")
    print(f"  检查: {result['checked']} 条")
    print(f"  相似: {result['similar_pairs']} 对")
    print(f"  合并: {result['merged']} 条")
    print(f"  删除: {result['deleted']} 条")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="记忆整合")
    parser.add_argument("--agent", default="main", help="Agent ID")
    parser.add_argument("--execute", action="store_true", help="执行合并（默认只预览）")
    args = parser.parse_args()
    
    run_consolidation(args.agent, dry_run=not args.execute)