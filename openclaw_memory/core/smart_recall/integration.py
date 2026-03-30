"""
智能召回器 - 集成模块

将智能召回策略集成到检索流程中

工作流程：
1. 用户查询进入
2. 分析是否需要召回
3. 决定召回层级和数量
4. 执行召回
5. 压缩结果
6. 返回

使用方式：
    from smart_recall import SmartRecaller
    
    recall = SmartRecaller("main")
    result = recall.smart_recall("上次讨论的项目怎么样")
"""

import sys
import os
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.smart_recall import (
    SmartRecallDecider,
    MemoryCompressor,
    RecallLevel,
    get_recall_decider,
    get_memory_compressor
)
from core.hybrid_recall import HybridRecallEngine, get_recall_engine
from core.networkx_kg_client import get_nx_client


# ============================================================================
# 智能召回器（集成层）
# ============================================================================

class SmartRecaller:
    """
    智能召回器
    
    集成召回决策、检索、压缩的完整流程
    
    原理：
    ┌─────────────────────────────────────────────────────────────┐
    │                     用户查询                                 │
    └───────────────────────┬─────────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Step 1: 关键词分析                                         │
    │  ─────────────────                                          │
    │  检查是否包含：                                              │
    │  - 时间词："上次"、"之前"、"昨天"                            │
    │  - 疑问词："什么"、"谁"、"怎样"                              │
    │  - 引用词："你说的"、"你提到"                                │
    │  - 实体词：人名、公司名、项目名                               │
    │  - 否定词："你好"、"帮我写"（这些不召回）                     │
    └───────────────────────┬─────────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Step 2: 召回决策                                           │
    │  ─────────────────                                          │
    │  根据匹配的关键词类别决定：                                   │
    │  - 是否召回                                                 │
    │  - 召回层级：标签/摘要/完整                                  │
    │  - 记忆数量：0-10条                                         │
    │  - 是否需要实体关系                                          │
    │  - 压缩比例：20%-100%                                       │
    └───────────────────────┬─────────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Step 3: 执行召回                                           │
    │  ─────────────────                                          │
    │  调用混合检索引擎：                                          │
    │  - BM25 关键词匹配 (30%)                                    │
    │  - 向量语义搜索 (60%)                                       │
    │  - 时间衰减 (10%)                                           │
    │  返回 Top-K 记忆                                            │
    └───────────────────────┬─────────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Step 4: 记忆压缩                                           │
    │  ─────────────────                                          │
    │  根据压缩比例处理：                                          │
    │  - 100%: 返回原文                                           │
    │  - 50%: 提取核心事实                                         │
    │  - 30%: 原子化事实提取（推荐）                                │
    │  - 20%: 只保留最核心                                         │
    │                                                             │
    │  压缩示例：                                                  │
    │  原文: "张三在字节跳动工作，使用 Python 和 Go 开发后端服务"   │
    │  压缩: "张三是字节跳动工程师 | 使用Python和Go开发"            │
    └───────────────────────┬─────────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Step 5: 实体关系召回（可选）                                │
    │  ─────────────────                                          │
    │  如果查询涉及"关系"、"合作"、"团队"等：                       │
    │  - 从知识图谱召回相关实体                                    │
    │  - 返回实体及其关系                                          │
    │  - 限制数量避免爆炸                                          │
    └───────────────────────┬─────────────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     返回结果                                 │
    │  {                                                          │
    │    "memories": [...],     // 压缩后的记忆                    │
    │    "entities": [...],     // 相关实体（可选）                │
    │    "decision": {...},     // 召回决策                        │
    │    "stats": {...}         // 统计信息                        │
    │  }                                                          │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.decider = get_recall_decider()
        self.compressor = get_memory_compressor()
        self.recall_engine = get_recall_engine(agent_id)
        self.kg_client = get_nx_client(agent_id)
    
    def smart_recall(
        self,
        query: str,
        force_recall: bool = False,
        max_tokens: int = 1000
    ) -> Dict:
        """
        智能召回主入口
        
        Args:
            query: 用户查询
            force_recall: 强制召回（忽略决策）
            max_tokens: 最大 token 数
        
        Returns:
            {
                "success": bool,
                "memories": List[Dict],      # 召回的记忆
                "entities": List[Dict],      # 相关实体
                "decision": Dict,            # 召回决策
                "stats": Dict                # 统计信息
            }
        """
        result = {
            "success": True,
            "memories": [],
            "entities": [],
            "decision": {},
            "stats": {
                "original_count": 0,
                "compressed_count": 0,
                "total_chars": 0
            }
        }
        
        # Step 1-2: 分析查询并决策
        decision = self.decider.get_recall_config(query)
        result["decision"] = decision
        
        # 如果不需要召回且不强制
        if not decision["should_recall"] and not force_recall:
            result["stats"]["reason"] = decision["reason"]
            return result
        
        # Step 3: 执行召回
        memory_limit = decision["memory_limit"]
        
        try:
            recall_result = self.recall_engine.recall(
                query=query,
                limit=memory_limit,
                include_entities=False  # 我们自己处理实体
            )
            
            raw_memories = recall_result.get("memories", [])
            result["stats"]["original_count"] = len(raw_memories)
            
            # Step 4: 记忆压缩
            compression_ratio = decision["compression_ratio"]
            
            # 根据层级决定压缩程度
            if decision["level"] == "tags":
                # 只提取标签/关键词
                compressed = self._extract_tags_only(raw_memories)
            elif decision["level"] == "summary":
                # 压缩摘要
                max_chars = int(max_tokens * 1.5 * compression_ratio)
                compressed = self.compressor.compress_batch(
                    raw_memories,
                    ratio=compression_ratio,
                    max_total_length=max_chars
                )
            else:
                # 完整内容，但也要限制长度
                max_chars = int(max_tokens * 1.5)
                compressed = self.compressor.compress_batch(
                    raw_memories,
                    ratio=0.7,  # 轻度压缩
                    max_total_length=max_chars
                )
            
            result["memories"] = compressed
            result["stats"]["compressed_count"] = len(compressed)
            result["stats"]["total_chars"] = sum(
                len(m.get("content", "")) for m in compressed
            )
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        # Step 5: 实体关系召回
        if decision["include_entities"]:
            try:
                entity_limit = decision["entity_limit"]
                entities = self._recall_entities(query, entity_limit)
                result["entities"] = entities
            except Exception as e:
                pass  # 实体召回失败不影响主流程
        
        return result
    
    def _extract_tags_only(self, memories: List[Dict]) -> List[Dict]:
        """只提取标签/关键词"""
        result = []
        for mem in memories:
            content = mem.get("summary", "") or mem.get("content", "")
            # 提取关键词
            keywords = self._extract_keywords(content)
            result.append({
                "id": mem.get("id"),
                "content": " | ".join(keywords[:5]),
                "score": mem.get("final_score", 0)
            })
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """简单关键词提取"""
        # 技术、项目、人名等关键词
        keywords = []
        
        # 技术关键词
        tech_pattern = r'\b(Python|Go|Java|React|Vue|Docker|Redis|PostgreSQL|MongoDB|Kubernetes|OpenAI|LLM|GPT|Claude)\b'
        keywords.extend(re.findall(tech_pattern, text, re.IGNORECASE))
        
        # 中文关键词（简单分词）
        # 提取 2-4 字的中文词
        cn_pattern = r'[\u4e00-\u9fa5]{2,4}'
        cn_words = re.findall(cn_pattern, text)
        keywords.extend(cn_words[:5])
        
        return list(set(keywords))
    
    def _recall_entities(self, query: str, limit: int) -> List[Dict]:
        """召回相关实体"""
        if not self.kg_client:
            return []
        
        try:
            # 从查询中提取可能的实体名
            entities = self.kg_client.search_entities(query, limit=limit)
            return entities
        except Exception:
            return []
    
    def quick_recall(self, query: str) -> str:
        """
        快速召回 - 返回压缩后的文本
        
        用于直接注入到 prompt 中
        """
        result = self.smart_recall(query)
        
        if not result["memories"]:
            return ""
        
        # 构建简洁的上下文
        lines = ["## 相关记忆"]
        for mem in result["memories"]:
            lines.append(f"- {mem.get('content', '')}")
        
        if result["entities"]:
            lines.append("\n## 相关实体")
            for ent in result["entities"][:3]:
                lines.append(f"- {ent.get('name', '')}: {ent.get('type', '')}")
        
        return "\n".join(lines)


# 导入 re 模块
import re


# ============================================================================
# 便捷函数
# ============================================================================

_smart_recaller = None


def get_smart_recaller(agent_id: str = "main") -> SmartRecaller:
    """获取智能召回器实例"""
    global _smart_recaller
    if _smart_recaller is None or _smart_recaller.agent_id != agent_id:
        _smart_recaller = SmartRecaller(agent_id)
    return _smart_recaller


def smart_recall(query: str, agent_id: str = "main") -> Dict:
    """便捷函数：执行智能召回"""
    return get_smart_recaller(agent_id).smart_recall(query)