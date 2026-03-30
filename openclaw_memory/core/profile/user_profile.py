"""
用户画像构建器

功能：
1. 实体聚合 - 按类型分组
2. 行为分析 - 时间分布、话题分布
3. World Fact / Experience 区分
4. 偏好挖掘
"""

import re
import json
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.weaviate_client import WeaviateClient
from core.networkx_kg_client import get_nx_client


# ============================================================================
# 实体类型定义
# ============================================================================

ENTITY_TYPES = {
    "person": ["人物", "人名", "用户", "开发者", "工程师", "产品经理"],
    "organization": ["公司", "组织", "团队", "部门", "机构"],
    "technology": ["技术", "编程语言", "框架", "工具", "数据库", "库"],
    "project": ["项目", "产品", "系统", "服务", "平台"],
    "skill": ["技能", "能力", "专长"],
    "preference": ["偏好", "喜欢", "习惯", "风格"],
    "experience": ["经验", "经历", "工作经历", "项目经验"],
    "fact": ["事实", "信息", "数据", "配置", "版本"]
}


# ============================================================================
# World Fact / Experience 分类器
# ============================================================================

class MemoryClassifier:
    """记忆分类器 - 区分 World Fact 和 Experience"""
    
    # Fact 关键词
    FACT_PATTERNS = [
        r"是\s*\w+", r"使用\s*\w+", r"用\s*\w+", r"基于\s*\w+",
        r"版本\s*\d", r"\d+\.\d+", r"配置\s*\w+", r"参数\s*\w+",
        r"地址\s*\w+", r"端口\s*\d", r" IP ", r"URL",
        r"公司名|产品名|技术栈", r"数据库|框架|语言"
    ]
    
    # Experience 关键词
    EXPERIENCE_PATTERNS = [
        r"做过|做过\s*\w+", r"开发过|参与过|负责过",
        r"工作经验|项目经验", r"曾经|以前|之前",
        r"学习|研究|探索", r"解决|处理|修复",
        r"完成|实现|构建", r"遇到|碰到|发现"
    ]
    
    @classmethod
    def classify(cls, text: str) -> str:
        """
        分类记忆为 Fact 或 Experience
        
        Returns:
            "fact" | "experience" | "mixed"
        """
        fact_score = 0
        exp_score = 0
        
        for pattern in cls.FACT_PATTERNS:
            if re.search(pattern, text):
                fact_score += 1
        
        for pattern in cls.EXPERIENCE_PATTERNS:
            if re.search(pattern, text):
                exp_score += 1
        
        if fact_score > exp_score + 1:
            return "fact"
        elif exp_score > fact_score + 1:
            return "experience"
        elif fact_score > 0 and exp_score > 0:
            return "mixed"
        else:
            return "unknown"
    
    @classmethod
    def extract_type(cls, text: str, entity_type: str) -> str:
        """根据上下文判断实体子类型"""
        text_lower = text.lower()
        
        if entity_type == "technology":
            # 区分具体技术和概念
            if any(k in text_lower for k in ["python", "java", "go", "rust", "js", "typescript"]):
                return "language"
            elif any(k in text_lower for k in ["react", "vue", "angular", "django", "spring"]):
                return "framework"
            elif any(k in text_lower for k in ["postgresql", "mysql", "redis", "mongodb", "docker", "k8s"]):
                return "tool"
            return "concept"
        
        return entity_type


# ============================================================================
# 用户画像构建器
# ============================================================================

class UserProfile:
    """
    用户画像构建器
    
    功能：
    1. 实体聚合 - 按类型分组用户相关实体
    2. 行为分析 - 时间分布、话题分布
    3. World Fact / Experience 区分
    4. 偏好挖掘
    """
    
    def __init__(self, agent_id: str = "main"):
        self.agent_id = agent_id
        self.weaviate_client = WeaviateClient(agent_id)
        self.kg_client = get_nx_client(agent_id)
        self.classifier = MemoryClassifier()
    
    def _get_all_memories(self, limit: int = 500) -> List[Dict]:
        """获取所有记忆"""
        if not self.weaviate_client.client:
            return []
        
        try:
            collection = self.weaviate_client.client.collections.get(
                self.weaviate_client.memory_collection
            )
            results = collection.query.fetch_objects(limit=limit)
            
            memories = []
            for obj in results.objects:
                memories.append({
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "summary": obj.properties.get("summary", ""),
                    "timestamp": obj.properties.get("timestamp", ""),
                    "importance": obj.properties.get("importance", 0.5),
                    "entities": obj.properties.get("entities", [])
                })
            
            return memories
            
        except Exception as e:
            print(f"⚠️ 获取记忆失败: {e}")
            return []
    
    def build_profile(self, days: int = 30) -> Dict:
        """
        构建用户画像
        
        Args:
            days: 分析最近几天的记忆
        
        Returns:
            完整的用户画像
        """
        memories = self._get_all_memories()
        
        if not memories:
            return {
                "status": "no_data",
                "message": "没有记忆数据"
            }
        
        # 1. 分类记忆 (Fact vs Experience)
        classified = self._classify_memories(memories)
        
        # 2. 提取实体
        entities = self._extract_entities(memories)
        
        # 3. 行为分析
        behavior = self._analyze_behavior(memories)
        
        # 4. 偏好挖掘
        preferences = self._extract_preferences(memories)
        
        # 5. 生成摘要
        summary = self._generate_summary(classified, entities, behavior)
        
        return {
            "agent_id": self.agent_id,
            "total_memories": len(memories),
            "classification": {
                "facts": classified["facts"],
                "experiences": classified["experiences"],
                "mixed": classified["mixed"],
                "unknown": classified["unknown"]
            },
            "entities": entities,
            "behavior": behavior,
            "preferences": preferences,
            "summary": summary,
            "updated_at": datetime.now().isoformat()
        }
    
    def _classify_memories(self, memories: List[Dict]) -> Dict:
        """分类记忆"""
        result = {
            "facts": [],
            "experiences": [],
            "mixed": [],
            "unknown": []
        }
        
        for mem in memories:
            content = mem.get("summary") or mem.get("content", "")
            mem_type = self.classifier.classify(content)
            
            # 确保 mem_type 是有效的 key
            if mem_type not in result:
                mem_type = "unknown"
            
            result[mem_type].append({
                "id": mem.get("id"),
                "content": content[:200],
                "timestamp": mem.get("timestamp")
            })
        
        return result
    
    def _extract_entities(self, memories: List[Dict]) -> Dict:
        """提取并聚合实体"""
        entity_pool = defaultdict(list)
        
        # 从记忆和知识图谱中收集实体
        for mem in memories:
            # 从 summary 中提取
            summary = mem.get("summary", "") or mem.get("content", "")
            
            # 提取技术栈
            tech_keywords = [
                "Python", "JavaScript", "Go", "Rust", "Java", "TypeScript",
                "React", "Vue", "Angular", "Node.js", "Django", "Spring",
                "PostgreSQL", "MySQL", "Redis", "MongoDB", "Docker", "K8s",
                "Weaviate", "Ollama", "OpenAI", "LLM"
            ]
            
            for tech in tech_keywords:
                if tech in summary:
                    entity_pool["technology"].append(tech)
            
            # 提取项目关键词
            project_keywords = ["项目", "产品", "系统", "平台"]
            for kw in project_keywords:
                if kw in summary:
                    # 尝试提取项目名
                    matches = re.findall(r'[\w]+项目|[\w]+系统', summary)
                    entity_pool["project"].extend(matches)
            
            # 提取技能
            skill_keywords = ["技能", "擅长", "掌握", "使用"]
            for kw in skill_keywords:
                if kw in summary:
                    # 提取技能名
                    pattern = r'([A-Za-z]+|[\u4e00-\u9fa5]+)'
                    entity_pool["skill"].extend(re.findall(pattern, summary[:100]))
        
        # 也从知识图谱获取
        try:
            if self.kg_client:
                kg_entities = self.kg_client.get_all_entities(limit=100)
                for ent in kg_entities:
                    etype = ent.get("type", "unknown")
                    name = ent.get("name", "")
                    if name and etype != "unknown":
                        entity_pool[etype].append(name)
        except Exception:
            pass
        
        # 去重并统计频率
        result = {}
        for etype, names in entity_pool.items():
            counter = Counter(names)
            result[etype] = [
                {"name": name, "count": count}
                for name, count in counter.most_common(20)
            ]
        
        return result
    
    def _analyze_behavior(self, memories: List[Dict]) -> Dict:
        """行为分析"""
        # 时间分布
        time_distribution = defaultdict(int)
        topic_counter = Counter()
        
        for mem in memories:
            ts = mem.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour = dt.hour
                    weekday = dt.strftime("%A")
                    
                    # 工作时间分布
                    if 9 <= hour < 18:
                        time_distribution["work_hours"] += 1
                    elif 18 <= hour < 22:
                        time_distribution["evening"] += 1
                    else:
                        time_distribution["night"] += 1
                    
                    # 星期分布
                    time_distribution[f"weekday_{weekday}"] += 1
                except Exception:
                    pass
            
            # 话题提取
            content = mem.get("summary", "") or mem.get("content", "")
            topics = self._extract_topics(content)
            topic_counter.update(topics)
        
        # 计算会话数 - 处理 datetime 对象
        timestamps = []
        for m in memories:
            ts = m.get("timestamp", "")
            if hasattr(ts, 'isoformat'):
                ts = ts.isoformat()
            if ts:
                timestamps.append(str(ts)[:10])
        
        return {
            "time_distribution": dict(time_distribution),
            "top_topics": [
                {"topic": t, "count": c}
                for t, c in topic_counter.most_common(10)
            ],
            "total_sessions": len(set(timestamps))
        }
    
    def _extract_topics(self, content: str) -> List[str]:
        """提取话题"""
        topics = []
        
        topic_keywords = {
            "开发": ["开发", "代码", "编程", "实现", "功能"],
            "产品": ["产品", "需求", "设计", "用户", "体验"],
            "运维": ["部署", "运维", "监控", "服务器", "Docker"],
            "AI": ["AI", "LLM", "大模型", "GPT", "OpenAI", "Ollama"],
            "数据": ["数据库", "存储", "数据", "Redis", "PostgreSQL"],
            "团队": ["团队", "协作", "会议", "沟通", "成员"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in content for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_preferences(self, memories: List[Dict]) -> Dict:
        """偏好挖掘"""
        preferences = {
            "tech_stack": [],
            "work_style": [],
            "communication": []
        }
        
        # 从高重要性记忆提取偏好
        important_memories = [m for m in memories if m.get("importance", 0) > 0.6]
        
        tech_stack = []
        work_style = []
        
        for mem in important_memories[:50]:
            content = mem.get("summary", "") or mem.get("content", "")
            
            # 技术偏好
            techs = re.findall(r'(Python|JavaScript|Go|Rust|React|Vue|Node\.js|PostgreSQL|Redis|Docker)', content)
            tech_stack.extend(techs)
            
            # 工作风格
            if "快速" in content or "高效" in content:
                work_style.append("高效优先")
            if "简洁" in content or "简单" in content:
                work_style.append("简洁风格")
            if "自动化" in content:
                work_style.append("自动化")
            if "测试" in content:
                work_style.append("重视质量")
        
        # 统计
        if tech_stack:
            counter = Counter(tech_stack)
            preferences["tech_stack"] = [
                {"tech": t, "score": c}
                for t, c in counter.most_common(10)
            ]
        
        if work_style:
            counter = Counter(work_style)
            preferences["work_style"] = [
                {"style": s, "count": c}
                for s, c in counter.most_common(5)
            ]
        
        return preferences
    
    def _generate_summary(self, classified: Dict, entities: Dict, behavior: Dict) -> str:
        """生成画像摘要"""
        facts_count = len(classified.get("facts", []))
        exp_count = len(classified.get("experiences", []))
        
        # 主要技术栈
        techs = entities.get("technology", [])
        main_techs = [t["name"] for t in techs[:5]]
        
        # 主要话题
        topics = behavior.get("top_topics", [])
        main_topics = [t["topic"] for t in topics[:3]]
        
        summary = f"用户记忆系统包含 {facts_count} 条事实和 {exp_count} 条经验记录。"
        
        if main_techs:
            summary += f"主要技术栈: {', '.join(main_techs)}。"
        
        if main_topics:
            summary += f"近期关注: {', '.join(main_topics)}。"
        
        return summary
    
    def get_profile(self, days: int = 30) -> Dict:
        """获取用户画像（快捷方法）"""
        return self.build_profile(days)


# 全局实例
_profile_cache = {}


def get_user_profile(agent_id: str = "main") -> UserProfile:
    """获取用户画像实例"""
    if agent_id not in _profile_cache:
        _profile_cache[agent_id] = UserProfile(agent_id)
    return _profile_cache[agent_id]


def build_user_profile(agent_id: str = "main", days: int = 30) -> Dict:
    """快捷函数"""
    return get_user_profile(agent_id).build_profile(days)