"""
记忆提取器 - 借鉴 Mem0 的原子化事实提取
支持：事实提取、记忆合并、增量更新
"""

import requests
import re
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


# ============================================================================
# 提示词模板
# ============================================================================

FACT_EXTRACTION_PROMPT = """你是一个专业的信息提取器，从对话中提取原子化的事实。

## 规则
1. 每个事实必须是独立的、原子化的（一个事实只包含一个信息点）
2. 只提取用户消息中的信息，忽略助手回复
3. 保留具体数值：版本号、数量、金额、日期
4. 保留关键实体：人名、公司、项目、技术
5. 使用用户的原始语言

## 示例

输入: "Hi, my name is John. I am a software engineer at Google."
输出: {{"facts": ["User name is John", "User is a software engineer", "User works at Google"]}}

输入: "我们项目用的是 PostgreSQL 14.2，最大连接数 100。"
输出: {{"facts": ["项目使用 PostgreSQL 14.2", "最大连接数是 100"]}}

输入: "好的，我知道了。"
输出: {{"facts": []}}

## 对话内容
{conversation}

直接输出 JSON："""


MEMORY_UPDATE_PROMPT = """你是一个记忆管理器，负责合并新旧记忆。

## 操作类型
- ADD: 添加新事实
- UPDATE: 更新冲突的事实（新信息覆盖旧信息）
- DELETE: 删除过时的事实
- NONE: 无变化

## 规则
1. 相同主题的事实只保留最新/最完整的版本
2. 矛盾的事实用新事实替换旧事实
3. 补充性事实可以合并
4. 过时的信息标记为 DELETE

## 示例

旧记忆:
- ["User name is John", "User works at Google"]

新事实:
- ["User name is John Smith", "User is a senior engineer at Meta"]

结果:
{{
  "memory": [
    {{"text": "User name is John Smith", "event": "UPDATE", "old": "User name is John"}},
    {{"text": "User is a senior engineer at Meta", "event": "UPDATE", "old": "User works at Google"}}
  ]
}}

## 当前记忆
{existing_memory}

## 新提取的事实
{new_facts}

输出 JSON："""


MEMORY_DECAY_PROMPT = """评估以下事实的重要性，用于记忆衰减。

## 重要性标准 (0.0 - 1.0)
- 1.0: 核心身份信息（姓名、职业、联系方式）
- 0.9: 关键偏好和决策（技术栈选择、项目决定）
- 0.8: 重要事实（公司、项目名称、关键日期）
- 0.6: 一般信息（日常活动、临时计划）
- 0.4: 次要信息（闲聊内容）
- 0.2: 临时信息（很快过时的内容）

## 事实列表
{facts}

输出 JSON 格式:
{{"importance": [0.8, 0.6, ...]}}
"""


# ============================================================================
# 记忆提取器
# ============================================================================

class MemoryExtractor:
    """记忆提取器 - 原子化事实提取和合并"""
    
    def __init__(self, model: str = None):
        self.model = model or OLLAMA_CHAT_MODEL
        self.base_url = OLLAMA_BASE_URL
    
    def _call_llm(self, prompt: str, timeout: int = 30) -> Optional[str]:
        """调用 LLM"""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=timeout
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"⚠️ LLM 调用失败: {e}")
        return None
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON"""
        try:
            # 尝试直接解析
            return json.loads(text)
        except Exception:
            pass
        
        # 尝试提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass
        
        return None
    
    def extract_facts(self, conversation: str, role: str = "user") -> List[str]:
        """
        从对话中提取原子化事实
        
        Args:
            conversation: 对话内容
            role: 提取角色 ("user" 或 "assistant")
        
        Returns:
            事实列表
        """
        if not conversation or len(conversation.strip()) < 10:
            return []
        
        # 截断过长内容
        input_text = conversation[:2000] if len(conversation) > 2000 else conversation
        
        prompt = FACT_EXTRACTION_PROMPT.format(conversation=input_text)
        response = self._call_llm(prompt)
        
        if response:
            data = self._extract_json(response)
            if data and "facts" in data:
                facts = data["facts"]
                # 过滤空事实和过短事实
                return [f.strip() for f in facts if f and len(f.strip()) > 5]
        
        return []
    
    def merge_memories(
        self, 
        existing_facts: List[str], 
        new_facts: List[str]
    ) -> Tuple[List[str], List[dict]]:
        """
        合并新旧记忆
        
        Returns:
            (更新后的事实列表, 变更记录)
        """
        if not new_facts:
            return existing_facts, []
        
        if not existing_facts:
            return new_facts, [{"text": f, "event": "ADD"} for f in new_facts]
        
        prompt = MEMORY_UPDATE_PROMPT.format(
            existing_memory=json.dumps(existing_facts, ensure_ascii=False),
            new_facts=json.dumps(new_facts, ensure_ascii=False)
        )
        
        response = self._call_llm(prompt, timeout=45)
        changes = []
        result_facts = existing_facts.copy()
        
        if response:
            data = self._extract_json(response)
            if data and "memory" in data:
                for item in data["memory"]:
                    text = item.get("text", "")
                    event = item.get("event", "NONE")
                    old = item.get("old", "")
                    
                    if event == "ADD" and text:
                        result_facts.append(text)
                        changes.append({"text": text, "event": "ADD"})
                    
                    elif event == "UPDATE" and text and old:
                        if old in result_facts:
                            idx = result_facts.index(old)
                            result_facts[idx] = text
                            changes.append({"text": text, "event": "UPDATE", "old": old})
                    
                    elif event == "DELETE" and old:
                        if old in result_facts:
                            result_facts.remove(old)
                            changes.append({"text": old, "event": "DELETE"})
        
        else:
            # 降级：简单合并（添加新事实，跳过重复）
            for fact in new_facts:
                is_duplicate = any(
                    self._similarity(fact, existing) > 0.8 
                    for existing in existing_facts
                )
                if not is_duplicate:
                    result_facts.append(fact)
                    changes.append({"text": fact, "event": "ADD"})
        
        return result_facts, changes
    
    def assess_importance(self, facts: List[str]) -> List[float]:
        """评估事实重要性"""
        if not facts:
            return []
        
        prompt = MEMORY_DECAY_PROMPT.format(
            facts=json.dumps(facts, ensure_ascii=False)
        )
        
        response = self._call_llm(prompt, timeout=20)
        
        if response:
            data = self._extract_json(response)
            if data and "importance" in data:
                return [max(0.1, min(1.0, i)) for i in data["importance"]]
        
        # 降级：基于长度和关键词的简单评估
        result = []
        for fact in facts:
            score = 0.5
            # 包含数字
            if re.search(r'\d+', fact):
                score += 0.1
            # 包含关键实体
            if re.search(r'(公司|项目|版本|姓名|电话|邮箱)', fact):
                score += 0.2
            # 包含决策词
            if re.search(r'(决定|选择|使用|采用)', fact):
                score += 0.15
            result.append(min(1.0, score))
        
        return result
    
    def _similarity(self, text1: str, text2: str) -> float:
        """简单文本相似度（词重叠）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def process_conversation(
        self, 
        content: str, 
        role: str = "user",
        existing_facts: List[str] = None
    ) -> dict:
        """
        处理单条对话，提取并合并事实
        
        Returns:
            {
                "facts": ["事实1", "事实2"],
                "importance": [0.8, 0.6],
                "changes": [...],
                "summary": "合并后的摘要"
            }
        """
        # 1. 提取事实
        new_facts = self.extract_facts(content, role)
        
        if not new_facts:
            return {
                "facts": existing_facts or [],
                "importance": [],
                "changes": [],
                "summary": ""
            }
        
        # 2. 合并记忆
        if existing_facts:
            merged_facts, changes = self.merge_memories(existing_facts, new_facts)
        else:
            merged_facts = new_facts
            changes = [{"text": f, "event": "ADD"} for f in new_facts]
        
        # 3. 评估重要性
        importance = self.assess_importance(merged_facts)
        
        # 4. 生成摘要（用于向量检索）
        summary = self._generate_summary(merged_facts)
        
        return {
            "facts": merged_facts,
            "importance": importance,
            "changes": changes,
            "summary": summary
        }
    
    def _generate_summary(self, facts: List[str]) -> str:
        """将事实列表转换为摘要文本"""
        if not facts:
            return ""
        
        # 按重要性分组（如果有）
        # 简单方案：用分隔符连接
        return " | ".join(facts[:10])  # 最多 10 条事实


# ============================================================================
# 向后兼容的 Summarizer 包装器
# ============================================================================

class Summarizer:
    """摘要器 - 兼容旧接口"""
    
    def __init__(self, model: str = None):
        self.extractor = MemoryExtractor(model)
        self.model = model or OLLAMA_CHAT_MODEL
        self.base_url = OLLAMA_BASE_URL
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """生成摘要（兼容旧接口）"""
        facts = self.extractor.extract_facts(text)
        if facts:
            summary = " | ".join(facts[:5])
            return summary[:max_length] if len(summary) > max_length else summary
        return text[:max_length] if text else ""
    
    def clean_and_extract(self, text: str) -> dict:
        """清洗并提取信息（兼容旧接口）"""
        result = self.extractor.process_conversation(text)
        
        # 计算平均重要性
        importance = 0.5
        if result["importance"]:
            importance = sum(result["importance"]) / len(result["importance"])
        
        return {
            "summary": result["summary"],
            "importance": importance,
            "keywords": result["facts"][:5],
            "worth_remembering": len(result["facts"]) > 0
        }
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> list:
        """提取关键词"""
        facts = self.extractor.extract_facts(text)
        return facts[:max_keywords]
    
    def get_topic(self, text: str) -> str:
        """获取主题"""
        # 简单实现
        facts = self.extractor.extract_facts(text)
        if not facts:
            return "其他"
        
        text_lower = " ".join(facts).lower()
        
        if any(k in text_lower for k in ["代码", "编程", "技术", "bug", "开发"]):
            return "技术开发"
        if any(k in text_lower for k in ["项目", "进度", "任务", "团队"]):
            return "项目管理"
        if any(k in text_lower for k in ["产品", "功能", "需求", "用户"]):
            return "产品设计"
        if any(k in text_lower for k in ["错误", "问题", "修复", "调试"]):
            return "问题排查"
        
        return "其他"
    
    def get_brief(self, text: str) -> dict:
        """获取简介"""
        facts = self.extractor.extract_facts(text)
        topic = self.get_topic(text)
        
        return {
            "topic": topic,
            "keywords": facts[:4],
            "brief": f"[{topic}] {' | '.join(facts[:3])}"[:80]
        }


# 全局实例
_summarizer = None


def get_summarizer() -> Summarizer:
    """获取摘要器实例"""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer


def summarize(text: str, max_length: int = 100) -> str:
    """快捷函数"""
    return get_summarizer().summarize(text, max_length)