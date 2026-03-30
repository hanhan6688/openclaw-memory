"""
记忆存储模块
支持：向量检索、时间衰减、重要性排序、实体自动提取
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from core.weaviate_client import WeaviateClient
from core.embeddings import OllamaEmbedding
from core.summarizer import Summarizer, get_summarizer


def extract_entities_enhanced(text: str) -> Dict[str, List[str]]:
    """
    增强的实体提取（基于规则 + 字典）
    
    返回按类型分组的实体字典
    """
    entities = {
        "technology": [],
        "person": [],
        "organization": [],
        "project": [],
        "location": [],
        "concept": []
    }
    
    # ===== 技术关键词 =====
    tech_keywords = [
        # 编程语言
        "Python", "JavaScript", "JS", "Go", "Golang", "Rust", "Java", "TypeScript", "TS",
        "C++", "C#", "PHP", "Ruby", "Swift", "Kotlin", "Scala", "R", "MATLAB",
        # 前端框架
        "React", "Vue", "Vue3", "Angular", "Svelte", "Next.js", "Nuxt.js", "Remix",
        # 后端框架
        "Node.js", "Django", "Flask", "FastAPI", "Spring", "SpringBoot", "Express",
        "Gin", "Echo", "Fiber", "Rails", "Laravel",
        # 数据库
        "PostgreSQL", "Postgres", "MySQL", "MariaDB", "Redis", "MongoDB", "SQLite",
        "Elasticsearch", "Weaviate", "Milvus", "Pinecone", "Qdrant",
        "Oracle", "SQLServer", "Cassandra", "DynamoDB",
        # 消息队列
        "Kafka", "RabbitMQ", "RocketMQ", "Redis", "NATS",
        # 容器和云
        "Docker", "Kubernetes", "K8s", "Docker Compose", "Helm",
        "AWS", "GCP", "Azure", "Aliyun", "腾讯云", "华为云",
        # AI/ML
        "OpenAI", "LLM", "GPT", "GPT-4", "Claude", "Gemini", "Llama",
        "Ollama", "LangChain", "LlamaIndex", "HuggingFace",
        "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "Pandas", "NumPy",
        # 工具
        "Git", "GitHub", "GitLab", "Bitbucket", "Jenkins", "Jira",
        "Nginx", "Apache", "HAProxy",
        # 操作系统
        "Mac", "macOS", "Linux", "Ubuntu", "CentOS", "Debian", "Windows",
        # 其他
        "REST", "GraphQL", "gRPC", "WebSocket", "HTTP", "HTTPS",
        "JSON", "YAML", "XML", "Protobuf",
        "JWT", "OAuth", "SAML", "SSO"
    ]
    
    # 技术关键词匹配
    text_lower = text.lower()
    for tech in tech_keywords:
        if tech.lower() in text_lower:
            if tech not in entities["technology"]:
                entities["technology"].append(tech)
    
    # ===== 人名识别 =====
    # 中文姓氏 + 名字模式
    surname_pattern = r'([张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董潘袁蔡蒋余于杜叶程魏苏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦傅方白邹孟熊秦邱江尹薛阎段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤][\u4e00-\u9fa5]{1,3})(?=(在|说|做|用|是|的|和|与|给|把|被|将|对|向|从|到|会|能|想|要))'
    person_matches = re.findall(surname_pattern, text)
    for match in person_matches:
        name = match[0] if isinstance(match, tuple) else match
        if 2 <= len(name) <= 4 and name not in entities["person"]:
            entities["person"].append(name)
    
    # 英文名字模式
    en_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    en_names = re.findall(en_name_pattern, text)
    common_words = {"The", "This", "That", "What", "When", "Where", "How", "Why", "If", "Then", "Else", "And", "But", "Or", "Not", "In", "On", "At", "To", "For", "With", "By", "From", "About", "Into", "Over", "After", "Before", "Between", "Under", "Above", "Below"}
    for name in en_names:
        if name not in common_words and len(name) > 2 and name not in entities["technology"]:
            if name not in entities["person"]:
                entities["person"].append(name)
    
    # ===== 组织/公司名 =====
    org_keywords = [
        "字节跳动", "阿里巴巴", "阿里", "腾讯", "百度", "华为", "小米", "京东",
        "美团", "滴滴", "拼多多", "快手", "网易", "新浪", "搜狐", "360",
        "OpenAI", "Google", "Meta", "Apple", "Microsoft", "Amazon", "Netflix",
        "Tesla", "Nvidia", "Intel", "AMD", "IBM", "Oracle"
    ]
    for org in org_keywords:
        if org in text:
            if org not in entities["organization"]:
                entities["organization"].append(org)
    
    # 公司后缀模式
    company_pattern = r'([\u4e00-\u9fa5]{2,8})(公司|集团|科技|网络|信息|数据)'
    company_matches = re.findall(company_pattern, text)
    for match in company_matches:
        company = match[0] + match[1] if isinstance(match, tuple) else match
        if company not in entities["organization"]:
            entities["organization"].append(company)
    
    # ===== 项目名 =====
    # 项目关键词后的名称
    project_pattern = r'([\u4e00-\u9fa5A-Za-z0-9_]+)(?=项目|系统|平台|服务|模块)'
    project_matches = re.findall(project_pattern, text)
    for proj in project_matches:
        if len(proj) >= 2 and proj not in entities["project"]:
            entities["project"].append(proj)
    
    # ===== 地点 =====
    location_keywords = [
        "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京",
        "苏州", "天津", "重庆", "长沙", "郑州", "青岛", "厦门",
        "美国", "中国", "日本", "韩国", "新加坡", "英国", "德国", "法国"
    ]
    for loc in location_keywords:
        if loc in text:
            if loc not in entities["location"]:
                entities["location"].append(loc)
    
    # 过滤空列表
    return {k: v for k, v in entities.items() if v}


def extract_entities_simple(text: str) -> List[str]:
    """
    简单实体提取（返回扁平列表，兼容旧接口）
    """
    result = extract_entities_enhanced(text)
    all_entities = []
    for entity_list in result.values():
        all_entities.extend(entity_list)
    return list(set(all_entities))


# ===== BM25 中文支持 =====

def tokenize_chinese(text: str) -> str:
    """
    中文分词预处理（用于 BM25 索引）
    
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


def preprocess_for_bm25(text: str) -> str:
    """
    BM25 预处理：将文本转换为空格分隔的词
    
    用于存储和搜索时保持一致的预处理
    """
    # 基础清理
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 中文分词
    return tokenize_chinese(text)


def clean_message_content(content: str) -> str:
    """
    清理消息内容

    移除代码块、URL、文件路径等噪声
    """
    if not content:
        return ""

    # 移除代码块
    content = re.sub(r'```[\s\S]*?```', '', content)
    # 移除行内代码
    content = re.sub(r'`[^`]+`', '', content)
    # 移除URL
    content = re.sub(r'https?://\S+', '', content)
    # 移除文件路径
    content = re.sub(r'/[\w/.-]+', '', content)
    # 移除多余空白
    content = re.sub(r'\s+', ' ', content)
    # 移除 markdown 标记
    content = re.sub(r'[#*_`>|-]+', '', content)

    return content.strip()


class MemoryStore:
    """记忆存储"""

    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.user_id = user_id
        self.client = WeaviateClient(agent_id, user_id)
        self.embedder = OllamaEmbedding()
        self.summarizer = get_summarizer()

    # ==================== 存储操作 ====================

    def store(self, content: str, metadata: Dict = None, generate_summary: bool = True) -> str:
        """
        存储记忆

        流程：
        1. 先存储 content（原文）
        2. summary 先用原文前150字作为临时值
        3. 后台异步生成真正的摘要，然后更新

        metadata 可包含：
        - summary: 外部提供的摘要（避免重复生成）
        - importance: 外部提供的重要性
        - quality: 质量等级
        - session_id: 会话ID
        - role: 消息角色
        - generate_summary: 是否后台生成摘要（默认 True）
        - facts: 原子化事实列表（新）
        """
        if not content or len(content.strip()) < 10:
            return None

        metadata = metadata or {}

        # 优先使用外部提供的摘要和重要性
        summary = metadata.get("summary")
        importance = metadata.get("importance", 0.5)
        keywords = metadata.get("keywords", [])
        facts = metadata.get("facts", [])  # 新：原子化事实

        # 如果没有外部摘要，先用原文前150字作为临时摘要
        temp_summary = summary or content[:150]

        # 基于临时摘要生成向量
        vector = self.embedder.embed(temp_summary)

        # 构建存储对象
        # 自动提取实体
        entities = metadata.get("entities", [])
        if not entities:
            entities = extract_entities_simple(content)
        
        memory_data = {
            "content": content,           # 原文（完整保存）
            "summary": temp_summary,      # 临时摘要（后续更新）
            "keywords": keywords,         # 关键词
            "facts": facts,               # 新：原子化事实
            "entities": entities,         # 新：自动提取的实体
            "importance": importance,
            "memory_type": metadata.get("memory_type", "conversation"),
            "timestamp": metadata.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "tags": metadata.get("tags", []),
            "source": metadata.get("source", "chat"),
            "session_id": metadata.get("session_id", ""),
            "role": metadata.get("role", ""),
            "quality": metadata.get("quality", "medium"),
            "summary_generated": False    # 标记摘要是否已生成
        }

        # 存储
        memory_id = self.client.insert_memory(memory_data, vector, check_duplicate=True)
        
        # 如果返回 None，表示重复
        if not memory_id:
            return None

        # 后台异步生成摘要
        if generate_summary and not summary and memory_id:
            self._generate_summary_async(memory_id, content)

        return memory_id

    def _generate_summary_async(self, memory_id: str, content: str):
        """后台异步生成摘要（使用原子化事实提取）"""
        import threading

        def generate():
            try:
                # 使用新的原子化事实提取
                result = self.summarizer.extractor.process_conversation(content)
                
                summary = result.get("summary", "")
                facts = result.get("facts", [])
                importance_list = result.get("importance", [])
                
                # 计算平均重要性
                importance = 0.5
                if importance_list:
                    importance = sum(importance_list) / len(importance_list)

                if not facts:
                    # 没有有价值的事实，删除
                    self.delete(memory_id)
                    return

                if summary:
                    # 更新摘要和事实
                    self.update(memory_id, {
                        "summary": summary,
                        "importance": importance,
                        "keywords": facts[:6],
                        "facts": facts,
                        "summary_generated": True
                    })
                    # 重新生成向量（基于事实）
                    vector = self.embedder.embed(summary)
                    self.client.update_memory_vector(memory_id, vector)
            except Exception as e:
                print(f"⚠️ 生成摘要失败 {memory_id}: {e}")

        # 启动后台线程
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def update_summary(self, memory_id: str, summary: str, importance: float = None, keywords: list = None) -> bool:
        """
        更新记忆的摘要

        Args:
            memory_id: 记忆ID
            summary: 新摘要
            importance: 新重要性
            keywords: 新关键词
        """
        updates = {
            "summary": summary,
            "summary_generated": True
        }
        if importance is not None:
            updates["importance"] = importance
        if keywords is not None:
            updates["keywords"] = keywords

        # 更新属性
        success = self.update(memory_id, updates)

        if success:
            # 重新生成向量
            vector = self.embedder.embed(summary)
            self.client.update_memory_vector(memory_id, vector)

        return success

    def store_batch(self, memories: List[Dict]) -> int:
        """批量存储记忆"""
        count = 0
        for memory in memories:
            content = memory.get("content", "")
            metadata = {k: v for k, v in memory.items() if k != "content"}
            if self.store(content, metadata):
                count += 1
        return count

    # ==================== 检索操作 ====================

    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """
        检索记忆

        流程：
        1. 向量检索 summary 字段
        2. 返回 top N 的原文 content 给 agent
        """
        if not query:
            return []

        # 向量检索
        query_vector = self.embedder.embed(query)
        results = self.client.search_memory(query_vector, limit * 2)

        # 处理结果
        memories = []
        for obj in results:
            memory = {
                "id": obj.get("_additional", {}).get("id"),
                "content": obj.get("content"),        # 原文（返回给 agent）
                "summary": obj.get("summary"),        # 摘要
                "keywords": obj.get("keywords", []),
                "importance": obj.get("importance", 0.5),
                "timestamp": obj.get("timestamp"),
                "distance": obj.get("_additional", {}).get("distance"),
            }
            memories.append(memory)

        # 按距离排序
        memories.sort(key=lambda x: x.get("distance", 1))

        return memories[:limit]

    def recall_interactive(self, query: str, limit: int = 5) -> List[Dict]:
        """智能查询 - 兼容旧接口"""
        return self.recall(query, limit)

    def recall_for_agent(self, query: str, limit: int = 5) -> Dict:
        """
        为 Agent 检索记忆

        返回格式：
        {
            "memories": [
                {
                    "id": "xxx",
                    "content": "原文内容",      # 给 agent 的完整信息
                    "summary": "摘要",          # 用于理解
                    "keywords": ["关键词"],
                    "importance": 0.8,
                    "timestamp": "2024-03-18"
                }
            ],
            "query": "查询内容",
            "count": 5
        }
        """
        memories = self.recall(query, limit)

        return {
            "memories": memories,
            "query": query,
            "count": len(memories)
        }

    def get_by_id(self, memory_id: str) -> Optional[Dict]:
        """根据 ID 获取记忆（回查原文）"""
        return self.client.get_memory_by_id(memory_id)

    def get_content(self, memory_id: str) -> Optional[str]:
        """获取记忆的原文内容"""
        memory = self.get_by_id(memory_id)
        if memory:
            return memory.get("content")
        return None

    # ==================== 时间范围查询 ====================

    def get_recent(self, limit: int = 10) -> List[Dict]:
        """获取最近记忆"""
        return self.client.get_recent_memories(limit)

    def get_by_time_range(self, start_time: str, end_time: str, limit: int = 100) -> List[Dict]:
        """按时间范围获取记忆"""
        return self.client.get_memories_by_time_range(start_time, end_time, limit)

    def get_conversation_history(self, limit: int = 100, start_time: str = None, end_time: str = None) -> List[Dict]:
        """获取对话历史"""
        if start_time and end_time:
            return self.get_by_time_range(start_time, end_time, limit)
        return self.get_recent(limit)

    # ==================== 更新和删除 ====================

    def update(self, memory_id: str, updates: Dict) -> bool:
        """更新记忆"""
        return self.client.update_memory(memory_id, updates)

    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        return self.client.delete_memory(memory_id)

    def clear_all(self) -> int:
        """清空所有记忆"""
        return self.client.clear_all_memories()

    def import_session_messages(self, messages: List[Dict], session_id: str) -> Dict:
        """
        导入会话消息到记忆

        Args:
            messages: 消息列表 [{"role": "user/assistant", "content": "..."}]
            session_id: 会话ID

        Returns:
            {"imported": int, "skipped": int}
        """
        imported = 0
        skipped = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # 只处理 user 和 assistant 消息
            if role not in ["user", "assistant"]:
                skipped += 1
                continue

            # content 已经被 SessionFileReader 清理过了
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                content = "\n".join(texts)

            if not content or len(content.strip()) < 10:
                skipped += 1
                continue

            # 跳过低质量消息
            if self._is_low_quality(content):
                skipped += 1
                continue

            # content 直接存储纯文本，不添加 role 前缀
            # role 单独存储在 role 字段

            # 存储记忆
            memory_data = {
                "session_id": session_id,
                "role": role,
                "timestamp": timestamp,
                "memory_type": "conversation",
                "source": "session_import"
            }

            try:
                memory_id = self.store(content, memory_data)
                if memory_id:
                    imported += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1

        return {"imported": imported, "skipped": skipped}

    def _is_low_quality(self, content: str) -> bool:
        """检查是否为低质量消息"""
        low_quality_patterns = [
            "好的", "明白", "收到", "了解", "嗯", "好的，", "好的。",
            "没问题", "可以的", "OK", "ok", "Ok"
        ]
        stripped = content.strip()
        if stripped in low_quality_patterns:
            return True
        if len(stripped) < 5:
            return True
        return False

    # ==================== 高级检索 ====================

    def search(self, query: str, limit: int = 10, mode: str = "hybrid") -> List[Dict]:
        """
        高级检索
        
        Args:
            query: 查询文本
            limit: 返回数量
            mode: 检索模式
                - "vector": 纯向量检索
                - "bm25": 纯关键词检索
                - "hybrid": 混合检索（默认）
                - "auto": 自动识别意图
        
        Returns:
            检索结果列表
        """
        from core.hybrid_retrieval import HybridRetriever
        
        retriever = HybridRetriever(self)
        return retriever.search(query, limit, mode)
    
    def search_with_intent(self, query: str, limit: int = 10) -> Dict:
        """
        带意图识别的检索
        
        Returns:
            {
                "intent": "fuzzy" | "exact" | "time" | "general",
                "mode": "vector" | "bm25" | "hybrid" | "time",
                "results": [...],
                "reason": "..."
            }
        """
        from core.hybrid_retrieval import HybridRetriever, QueryIntentRecognizer
        
        recognizer = QueryIntentRecognizer()
        intent_result = recognizer.recognize(query)
        
        retriever = HybridRetriever(self)
        results = retriever.search(query, limit, intent_result["suggested_mode"])
        
        return {
            "intent": intent_result["intent"],
            "mode": intent_result["suggested_mode"],
            "confidence": intent_result["confidence"],
            "reason": intent_result["reason"],
            "time_hint": intent_result.get("time_hint"),
            "results": results,
            "count": len(results)
        }
    
    def search_by_time(self, query: str, time_range: str = "recent", limit: int = 10) -> List[Dict]:
        """
        按时间范围检索
        
        Args:
            query: 查询文本
            time_range: 时间范围
                - "today": 今天
                - "yesterday": 昨天
                - "week": 最近一周
                - "month": 最近一个月
            limit: 返回数量
        """
        from core.hybrid_retrieval import HybridRetriever, QueryIntentRecognizer
        
        recognizer = QueryIntentRecognizer()
        time_query = f"{time_range} {query}" if query else time_range
        extracted_range = recognizer.extract_time_range(time_query)
        
        if not extracted_range:
            # 默认最近一周
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=7)
            extracted_range = (start, now)
        
        retriever = HybridRetriever(self)
        return retriever._time_search(query, extracted_range, limit)

    # ==================== 统计 ====================

    def count(self) -> int:
        """获取记忆数量"""
        return self.client.count_memories()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_memories": self.count(),
            "agent_id": self.agent_id,
            "user_id": self.user_id
        }


# 全局实例缓存
_stores = {}


def get_memory_store(agent_id: str, user_id: str = "default") -> MemoryStore:
    """获取记忆存储实例"""
    key = f"{agent_id}_{user_id}"
    if key not in _stores:
        _stores[key] = MemoryStore(agent_id, user_id)
    return _stores[key]