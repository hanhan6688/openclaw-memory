"""
OpenClaw Memory System - 高级检索模块
================================

功能：
1. 时间衰减: 近期记忆权重更高
2. 混合检索: Weaviate 原生 hybrid (向量 + BM25)
3. 查询意图识别: 模糊匹配 vs 精确回忆

使用 Weaviate 原生混合检索:
- collection.query.hybrid(query, alpha=0.5)
- alpha=0: 纯 BM25
- alpha=1: 纯向量
- alpha=0.5: 平衡混合
"""

import re
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional
from collections import Counter


class TimeDecay:
    """
    时间衰减计算器
    
    近期记忆权重更高，随时间逐渐衰减
    
    公式: weight = base_weight × decay_factor^(days_ago)
    
    示例:
        今天: 1.2x
        昨天: 0.9x
        7天前: 0.5x
        30天前: 0.2x
    """
    
    def __init__(self, 
                 recent_boost: float = 1.2,  # 近期加成
                 decay_rate: float = 0.9,     # 衰减率
                 half_life_days: float = 7.0):  # 半衰期
        self.recent_boost = recent_boost
        self.decay_rate = decay_rate
        self.half_life_days = half_life_days
    
    def calculate_weight(self, timestamp: str, base_importance: float = 0.5) -> float:
        """
        计算时间衰减后的权重
        
        Args:
            timestamp: ISO 格式时间戳
            base_importance: 基础重要性 (0-1)
        
        Returns:
            衰减后的权重
        """
        try:
            # 解析时间戳
            if isinstance(timestamp, str):
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        memory_time = datetime.strptime(timestamp, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return base_importance
            else:
                memory_time = timestamp
            
            # 计算天数差
            now = datetime.now(timezone.utc)
            days_ago = (now - memory_time).total_seconds() / 86400
            
            # 近期加成（24小时内）
            if days_ago < 1:
                time_weight = self.recent_boost
            else:
                # 指数衰减
                time_weight = self.decay_rate ** (days_ago / self.half_life_days)
            
            # 综合权重 = 基础重要性 × 时间权重
            final_weight = base_importance * time_weight
            
            return min(final_weight, 1.0)
            
        except Exception:
            return base_importance
    
    def get_boost_factor(self, days_ago: float) -> float:
        """获取指定天数前的加成因子"""
        if days_ago < 1:
            return self.recent_boost
        return self.decay_rate ** (days_ago / self.half_life_days)


class QueryIntentRecognizer:
    """
    查询意图识别器
    
    识别用户是想"模糊匹配"还是"精确回忆"
    """
    
    # 模糊匹配关键词
    FUZZY_KEYWORDS = [
        "好像", "大概", "可能", "似乎", "类似", "差不多",
        "记得", "印象", "模糊", "应该", "什么来着"
    ]
    
    # 精确回忆关键词
    EXACT_KEYWORDS = [
        "具体", "精确", "原话", "确切", "一字不差",
        "原文", "完整", "详细", "准确"
    ]
    
    # 时间相关关键词
    TIME_KEYWORDS = [
        "今天", "昨天", "前天", "最近", "刚才",
        "上周", "上个月", "去年", "之前"
    ]
    
    def recognize(self, query: str) -> Dict:
        """
        识别查询意图
        
        Returns:
            {
                "intent": "fuzzy" | "exact" | "time" | "general",
                "confidence": 0.0-1.0,
                "suggested_mode": "vector" | "bm25" | "hybrid" | "time",
                "time_hint": "今天" | "最近" | None,
                "reason": "原因说明"
            }
        """
        # 检查引号（精确引用）
        if '"' in query or '"' in query or '「' in query:
            return {
                "intent": "exact",
                "confidence": 0.9,
                "suggested_mode": "bm25",
                "time_hint": None,
                "reason": "查询包含引号，可能是精确引用"
            }
        
        # 检查精确关键词
        if any(kw in query for kw in self.EXACT_KEYWORDS):
            return {
                "intent": "exact",
                "confidence": 0.8,
                "suggested_mode": "bm25",
                "time_hint": None,
                "reason": "包含精确关键词"
            }
        
        # 检查时间关键词
        time_keywords = [kw for kw in self.TIME_KEYWORDS if kw in query]
        if time_keywords:
            return {
                "intent": "time",
                "confidence": 0.85,
                "suggested_mode": "time",
                "time_hint": time_keywords[0],
                "reason": f"包含时间关键词: {time_keywords}"
            }
        
        # 检查模糊关键词
        if any(kw in query for kw in self.FUZZY_KEYWORDS):
            return {
                "intent": "fuzzy",
                "confidence": 0.7,
                "suggested_mode": "vector",
                "time_hint": None,
                "reason": "包含模糊关键词"
            }
        
        # 默认：混合模式
        return {
            "intent": "general",
            "confidence": 0.5,
            "suggested_mode": "hybrid",
            "time_hint": None,
            "reason": "通用查询，建议混合检索"
        }
    
    def extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """从查询中提取时间范围"""
        now = datetime.now(timezone.utc)
        
        if "今天" in query:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start, now)
        
        if "昨天" in query:
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = yesterday.replace(hour=23, minute=59, second=59)
            return (start, end)
        
        if "最近" in query or "这几天" in query:
            start = now - timedelta(days=7)
            return (start, now)
        
        if "上周" in query:
            days_since_monday = now.weekday()
            last_monday = now - timedelta(days=days_since_monday + 7)
            start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
            return (start, end)
        
        return None


class HybridRetriever:
    """
    混合检索器
    
    使用 Weaviate 原生混合检索 (向量 + BM25)
    """

    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.time_decay = TimeDecay()
        self.intent_recognizer = QueryIntentRecognizer()
        
        # alpha 参数: 0=纯BM25, 1=纯向量, 0.5=平衡
        self.default_alpha = 0.5

    def search(self, query: str, limit: int = 10, mode: str = "auto") -> List[Dict]:
        """
        混合检索

        Args:
            query: 查询文本
            limit: 返回数量
            mode: 检索模式
                - "auto": 自动识别意图
                - "vector": 纯向量检索 (alpha=1)
                - "bm25": 纯 BM25 检索 (alpha=0)
                - "hybrid": 混合检索 (alpha=0.5)
                - "time": 时间优先检索

        Returns:
            检索结果列表
        """
        # 意图识别
        intent_result = self.intent_recognizer.recognize(query)

        # 确定检索模式
        if mode == "auto":
            mode = intent_result["suggested_mode"]

        # 根据模式选择检索策略
        if mode == "vector":
            return self._vector_search(query, limit)
        elif mode == "bm25":
            return self._bm25_search(query, limit)
        elif mode == "time":
            time_range = self.intent_recognizer.extract_time_range(query)
            if time_range:
                return self._time_search(query, time_range, limit)
            else:
                return self._hybrid_search(query, limit)
        else:  # hybrid
            return self._hybrid_search(query, limit)

    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """纯向量检索 (alpha=1)"""
        return self._weaviate_hybrid_search(query, alpha=1.0, limit=limit)

    def _bm25_search(self, query: str, limit: int) -> List[Dict]:
        """纯 BM25 检索 (alpha=0)"""
        return self._weaviate_hybrid_search(query, alpha=0.0, limit=limit)

    def _hybrid_search(self, query: str, limit: int) -> List[Dict]:
        """混合检索 (alpha=0.5) - 使用 Weaviate 原生"""
        return self._weaviate_hybrid_search(query, alpha=self.default_alpha, limit=limit)
    
    def _weaviate_hybrid_search(self, query: str, alpha: float = 0.5, limit: int = 10) -> List[Dict]:
        """
        使用 Weaviate 原生混合检索
        
        Args:
            query: 查询文本
            alpha: 混合权重 (0=BM25, 1=向量, 0.5=平衡)
            limit: 返回数量
        """
        # 使用 Weaviate 原生混合检索
        results = self.memory_store.client.hybrid_search(query, alpha=alpha, limit=limit)
        
        # 应用时间衰减
        for r in results:
            r["time_weight"] = self.time_decay.calculate_weight(
                r.get("timestamp", ""),
                r.get("importance", 0.5)
            )
            # 设置检索模式标记
            if alpha == 1.0:
                r["search_mode"] = "vector"
            elif alpha == 0.0:
                r["search_mode"] = "bm25"
            else:
                r["search_mode"] = "hybrid"
                r["alpha"] = alpha
        
        return results

    def _time_search(self, query: str, time_range: Tuple[datetime, datetime], limit: int) -> List[Dict]:
        """时间优先检索"""
        start_time, end_time = time_range

        # 格式化时间
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        # 获取时间范围内的记忆
        memories = self.memory_store.get_by_time_range(start_str, end_str, limit * 2)

        # 如果有查询词，使用混合检索过滤
        if query:
            # 使用混合检索
            hybrid_results = self._hybrid_search(query, limit * 2)
            hybrid_ids = {r.get("id") for r in hybrid_results}
            memories = [m for m in memories if m.get("id") in hybrid_ids]

        # 应用时间衰减
        for m in memories:
            m["time_weight"] = self.time_decay.calculate_weight(
                m.get("timestamp", ""),
                m.get("importance", 0.5)
            )

        return memories[:limit]


# ==================== 便捷函数 ====================

def create_hybrid_retriever(memory_store) -> HybridRetriever:
    """创建混合检索器"""
    return HybridRetriever(memory_store)


# 示例用法
if __name__ == "__main__":
    # 测试时间衰减
    decay = TimeDecay()
    print("时间衰减测试:")
    print(f"  今天: {decay.get_boost_factor(0):.2f}x")
    print(f"  昨天: {decay.get_boost_factor(1):.2f}x")
    print(f"  7天前: {decay.get_boost_factor(7):.2f}x")
    print(f"  30天前: {decay.get_boost_factor(30):.2f}x")
    
    # 测试意图识别
    recognizer = QueryIntentRecognizer()
    test_queries = [
        "好像说过什么来着",
        "具体说了什么",
        "昨天的内容",
        "产品需求文档",
    ]
    
    print("\n意图识别测试:")
    for q in test_queries:
        result = recognizer.recognize(q)
        print(f"  '{q}' -> {result['intent']} ({result['suggested_mode']})")
