"""
智能召回策略模块

解决问题：
1. 什么时候触发记忆召回
2. 召回多少内容
3. 何时触发实体关系召回
4. 记忆压缩策略

策略：
1. 分层召回：标签 → 摘要 → 完整内容
2. 阈值触发：关键词/意图/相关性
3. 记忆压缩：原子化事实提取
4. 按需召回：只召回必要信息
"""

import re
import json
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RecallTrigger(Enum):
    """召回触发类型"""
    NEVER = "never"           # 不触发
    KEYWORD = "keyword"       # 关键词触发
    INTENT = "intent"         # 意图触发
    ALWAYS = "always"         # 始终触发


class RecallLevel(Enum):
    """召回层级"""
    TAGS_ONLY = "tags"        # 只返回标签/关键词
    SUMMARY = "summary"       # 返回压缩摘要
    FULL = "full"             # 返回完整内容


@dataclass
class RecallDecision:
    """召回决策结果"""
    should_recall: bool
    trigger: RecallTrigger
    level: RecallLevel
    memory_limit: int
    include_entities: bool
    entity_limit: int
    compression_ratio: float  # 0.0-1.0, 1.0 = 不压缩
    reason: str


# ============================================================================
# 触发关键词分类
# ============================================================================

# 需要召回记忆的关键词
MEMORY_TRIGGER_KEYWORDS = {
    # 时间相关
    "time": ["上次", "之前", "以前", "昨天", "前天", "上周", "上个月", "去年", 
             "刚才", "刚刚", "之前说的", "曾经", "一直以来", "还记得"],
    
    # 询问相关
    "query": ["什么", "谁", "哪里", "哪个", "怎样", "怎么", "为什么", "多少",
              "记得吗", "记得", "还记得", "说过", "提过", "讨论过", "聊过"],
    
    # 引用相关
    "reference": ["你说的", "你提到", "你告诉我", "你推荐", "你建议",
                  "我们讨论", "我们说过", "我们聊过", "之前那个"],
    
    # 任务相关
    "task": ["继续", "接着", "完成", "进度", "做到哪", "上次做"],
    
    # 人名/地名/项目名
    "entity": ["张三", "李四", "王五", "字节", "阿里", "腾讯", 
               "项目", "系统", "平台", "模块"]
}

# 不需要召回的关键词
NO_RECALL_KEYWORDS = [
    "你好", "谢谢", "好的", "收到", "明白", "了解", "知道", "OK", "ok",
    "今天天气", "几点", "什么时间", "帮我", "请", "麻烦",
    "写一个", "创建", "生成", "制作", "设计"
]

# 需要实体关系的关键词
ENTITY_TRIGGER_KEYWORDS = [
    "关系", "联系", "合作", "谁和谁", "哪些人", "相关", "关联",
    "团队", "组织", "公司", "部门", "负责", "参与", "成员",
    "技术栈", "用什么", "用了哪些", "涉及"
]


# ============================================================================
# 智能召回决策器
# ============================================================================

class SmartRecallDecider:
    """
    智能召回决策器
    
    根据查询内容决定：
    1. 是否需要召回记忆
    2. 召回层级
    3. 是否需要实体关系
    """
    
    def __init__(self):
        self.memory_keywords = MEMORY_TRIGGER_KEYWORDS
        self.no_recall_keywords = NO_RECALL_KEYWORDS
        self.entity_keywords = ENTITY_TRIGGER_KEYWORDS
    
    def decide(self, query: str, context: Dict = None) -> RecallDecision:
        """
        分析查询，决定召回策略
        
        Args:
            query: 用户查询
            context: 额外上下文（可选）
        
        Returns:
            RecallDecision: 召回决策
        """
        query_lower = query.lower()
        
        # 1. 检查是否明确不需要召回
        for kw in self.no_recall_keywords:
            if kw in query:
                return RecallDecision(
                    should_recall=False,
                    trigger=RecallTrigger.NEVER,
                    level=RecallLevel.TAGS_ONLY,
                    memory_limit=0,
                    include_entities=False,
                    entity_limit=0,
                    compression_ratio=1.0,
                    reason=f"包含不召回关键词: {kw}"
                )
        
        # 2. 检查触发关键词
        matched_categories = []
        for category, keywords in self.memory_keywords.items():
            for kw in keywords:
                if kw in query:
                    matched_categories.append(category)
                    break
        
        # 3. 决定召回层级
        if not matched_categories:
            # 没有匹配的关键词，不召回
            return RecallDecision(
                should_recall=False,
                trigger=RecallTrigger.NEVER,
                level=RecallLevel.TAGS_ONLY,
                memory_limit=0,
                include_entities=False,
                entity_limit=0,
                compression_ratio=1.0,
                reason="无触发关键词"
            )
        
        # 4. 根据匹配类别决定召回策略
        trigger = RecallTrigger.KEYWORD
        level = RecallLevel.SUMMARY  # 默认返回摘要
        memory_limit = 5
        compression_ratio = 0.3  # 压缩到30%
        
        # 时间相关 -> 需要更多记忆
        if "time" in matched_categories:
            memory_limit = 7
            level = RecallLevel.SUMMARY
            compression_ratio = 0.3
        
        # 查询相关 -> 需要精确信息
        if "query" in matched_categories:
            memory_limit = 5
            level = RecallLevel.FULL
            compression_ratio = 0.5
        
        # 引用相关 -> 需要精确匹配
        if "reference" in matched_categories:
            memory_limit = 3
            level = RecallLevel.FULL
            compression_ratio = 0.7
        
        # 任务相关 -> 需要上下文
        if "task" in matched_categories:
            memory_limit = 5
            level = RecallLevel.SUMMARY
            compression_ratio = 0.4
        
        # 5. 决定是否需要实体关系
        include_entities = False
        entity_limit = 0
        
        for kw in self.entity_keywords:
            if kw in query:
                include_entities = True
                entity_limit = 5
                break
        
        # 6. 如果查询长度很短，可能是简单问题，减少召回
        if len(query) < 10:
            memory_limit = min(memory_limit, 3)
        
        # 7. 如果查询包含多个问号，说明复杂，增加召回
        if query.count("?") > 1 or query.count("？") > 1:
            memory_limit = min(memory_limit + 2, 10)
        
        return RecallDecision(
            should_recall=True,
            trigger=trigger,
            level=level,
            memory_limit=memory_limit,
            include_entities=include_entities,
            entity_limit=entity_limit,
            compression_ratio=compression_ratio,
            reason=f"触发类别: {matched_categories}"
        )
    
    def get_recall_config(self, query: str) -> Dict:
        """
        获取召回配置（快捷方法）
        """
        decision = self.decide(query)
        return {
            "should_recall": decision.should_recall,
            "level": decision.level.value,
            "memory_limit": decision.memory_limit,
            "include_entities": decision.include_entities,
            "entity_limit": decision.entity_limit,
            "compression_ratio": decision.compression_ratio,
            "reason": decision.reason
        }


# ============================================================================
# 记忆压缩器
# ============================================================================

class MemoryCompressor:
    """
    记忆压缩器
    
    压缩策略：
    1. 原子化事实提取
    2. 去除冗余信息
    3. 保留关键实体
    """
    
    # 压缩模板
    COMPRESSION_TEMPLATES = {
        "identity": "{entity}是{value}",           # 张三是工程师
        "action": "{subject}{action}{object}",      # 张三开发了项目
        "attribute": "{entity}的{attr}是{value}",   # 项目的名称是X
        "relation": "{entity1}与{entity2}{relation}" # 张三与字节跳动合作
    }
    
    def compress(self, memory: Dict, ratio: float = 0.3) -> str:
        """
        压缩单条记忆
        
        Args:
            memory: 记忆对象
            ratio: 压缩比例 (0.0-1.0)
                - 1.0: 不压缩，返回原文
                - 0.5: 压缩到50%
                - 0.3: 压缩到30%（推荐）
                - 0.1: 极度压缩，只保留核心
        
        Returns:
            压缩后的文本
        """
        content = memory.get("summary", "") or memory.get("content", "")
        
        if ratio >= 1.0:
            return content
        
        # 如果已有摘要，直接使用
        if memory.get("summary") and len(memory["summary"]) < len(content) * ratio:
            return memory["summary"]
        
        # 根据压缩比例选择策略
        if ratio <= 0.2:
            return self._extract_core_fact(content)
        elif ratio <= 0.4:
            return self._extract_facts(content, max_facts=3)
        elif ratio <= 0.6:
            return self._extract_facts(content, max_facts=5)
        else:
            return self._truncate(content, max_length=int(len(content) * ratio))
    
    def _extract_core_fact(self, content: str) -> str:
        """提取核心事实（一条）"""
        # 提取主语 + 谓语 + 宾语
        patterns = [
            r'([^\s，。！？]+)(是|在|用|做|开发|负责|参与)([^\s，。！？]+)',
            r'([^\s，。！？]+)(说|告诉|问|讨论)([^\s，。！？]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return f"{match.group(1)}{match.group(2)}{match.group(3)}"
        
        # 降级：取第一句
        first_sentence = re.split(r'[。！？]', content)[0]
        return first_sentence[:50]
    
    def _extract_facts(self, content: str, max_facts: int = 3) -> str:
        """提取多个事实"""
        facts = []
        
        # 按句号分割
        sentences = re.split(r'[。！？\n]', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 提取事实
            fact = self._extract_core_fact(sentence)
            if fact and fact not in facts:
                facts.append(fact)
            
            if len(facts) >= max_facts:
                break
        
        return " | ".join(facts)
    
    def _truncate(self, content: str, max_length: int) -> str:
        """截断文本"""
        if len(content) <= max_length:
            return content
        
        # 尝试在句号处截断
        truncated = content[:max_length]
        last_period = truncated.rfind("。")
        
        if last_period > max_length * 0.7:
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def compress_batch(
        self, 
        memories: List[Dict], 
        ratio: float = 0.3,
        max_total_length: int = 500
    ) -> List[Dict]:
        """
        批量压缩记忆
        
        Args:
            memories: 记忆列表
            ratio: 压缩比例
            max_total_length: 总长度上限
        
        Returns:
            压缩后的记忆列表
        """
        compressed = []
        total_length = 0
        
        for mem in memories:
            compressed_content = self.compress(mem, ratio)
            
            # 检查总长度
            if total_length + len(compressed_content) > max_total_length:
                # 空间不足，使用更激进的压缩
                compressed_content = self.compress(mem, ratio * 0.5)
            
            if total_length + len(compressed_content) > max_total_length:
                break
            
            compressed.append({
                "id": mem.get("id"),
                "content": compressed_content,
                "score": mem.get("final_score", 0),
                "timestamp": mem.get("timestamp", "")
            })
            total_length += len(compressed_content)
        
        return compressed


# ============================================================================
# 全局实例
# ============================================================================

_decider = None
_compressor = None


def get_recall_decider() -> SmartRecallDecider:
    """获取召回决策器"""
    global _decider
    if _decider is None:
        _decider = SmartRecallDecider()
    return _decider


def get_memory_compressor() -> MemoryCompressor:
    """获取记忆压缩器"""
    global _compressor
    if _compressor is None:
        _compressor = MemoryCompressor()
    return _compressor


def analyze_recall_needs(query: str) -> Dict:
    """
    分析查询的召回需求（快捷函数）
    
    Returns:
        {
            "should_recall": bool,
            "level": "tags"/"summary"/"full",
            "memory_limit": int,
            "include_entities": bool,
            "entity_limit": int,
            "compression_ratio": float,
            "reason": str
        }
    """
    decider = get_recall_decider()
    return decider.get_recall_config(query)


def compress_memories(
    memories: List[Dict],
    query: str = None,
    max_tokens: int = 1000
) -> List[Dict]:
    """
    智能压缩记忆（快捷函数）
    
    Args:
        memories: 记忆列表
        query: 查询（用于确定压缩策略）
        max_tokens: 最大 token 数
    
    Returns:
        压缩后的记忆列表
    """
    compressor = get_memory_compressor()
    
    # 根据查询确定压缩比例
    ratio = 0.3  # 默认
    if query:
        decision = analyze_recall_needs(query)
        ratio = decision.get("compression_ratio", 0.3)
    
    # 估算字符上限 (约 1 token = 1.5 中文字符)
    max_chars = int(max_tokens * 1.5)
    
    return compressor.compress_batch(memories, ratio, max_chars)