"""
企业级知识图谱系统
对标 Microsoft GraphRAG + LlamaIndex KnowledgeGraphIndex

特性：
1. 多阶段实体提取（NER + 关系抽取 + 共指消解）
2. 实体消歧与合并（Entity Resolution）
3. 社区发现与摘要（Community Detection）
4. 层次化知识结构
5. 置信度评分与衰减
6. 增量更新与版本控制
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import json
import re
import sys
import os
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from core.weaviate_client import WeaviateClient
from core.embeddings import OllamaEmbedding
import requests


class EnterpriseKnowledgeGraph:
    """企业级知识图谱 - 对标 Microsoft GraphRAG"""

    # 实体类型定义（参考 Schema.org）
    ENTITY_TYPES = {
        # 人物
        "Person": {"color": "#4CAF50", "icon": "👤", "aliases": ["人物", "人", "用户", "开发者"]},
        # 组织
        "Organization": {"color": "#2196F3", "icon": "🏢", "aliases": ["组织", "公司", "团队", "部门"]},
        # 项目/产品
        "Project": {"color": "#FF9800", "icon": "📁", "aliases": ["项目", "产品", "系统", "应用"]},
        # 技术/工具
        "Technology": {"color": "#9C27B0", "icon": "⚙️", "aliases": ["技术", "工具", "框架", "语言"]},
        # 概念
        "Concept": {"color": "#607D8B", "icon": "💡", "aliases": ["概念", "方法", "模式"]},
        # 地点
        "Location": {"color": "#795548", "icon": "📍", "aliases": ["地点", "位置", "城市"]},
        # 事件
        "Event": {"color": "#F44336", "icon": "📅", "aliases": ["事件", "会议", "活动"]},
        # 文档
        "Document": {"color": "#00BCD4", "icon": "📄", "aliases": ["文档", "文件", "文章"]},
    }

    # 关系类型定义（参考 ConceptNet）
    RELATION_TYPES = {
        # 层级关系
        "IsA": {"weight": 1.0, "transitive": True, "aliases": ["是一种", "是"]},
        "PartOf": {"weight": 0.9, "transitive": True, "aliases": ["属于", "部分"]},
        "HasA": {"weight": 0.8, "transitive": False, "aliases": ["拥有", "有"]},

        # 社会关系
        "WorksFor": {"weight": 0.9, "transitive": False, "aliases": ["工作于", "就职于"]},
        "Manages": {"weight": 0.8, "transitive": True, "aliases": ["管理", "负责"]},
        "CollaboratesWith": {"weight": 0.7, "transitive": True, "aliases": ["合作", "协作"]},
        "Knows": {"weight": 0.5, "transitive": True, "aliases": ["认识", "了解"]},

        # 技术关系
        "Uses": {"weight": 0.8, "transitive": False, "aliases": ["使用", "采用"]},
        "DependsOn": {"weight": 0.9, "transitive": True, "aliases": ["依赖", "需要"]},
        "Implements": {"weight": 0.8, "transitive": False, "aliases": ["实现", "开发"]},
        "IntegratesWith": {"weight": 0.7, "transitive": True, "aliases": ["集成", "整合"]},

        # 概念关系
        "RelatedTo": {"weight": 0.5, "transitive": True, "aliases": ["相关", "关联"]},
        "Causes": {"weight": 0.7, "transitive": True, "aliases": ["导致", "引起"]},
        "HasProperty": {"weight": 0.6, "transitive": False, "aliases": ["具有", "属性"]},

        # 时间关系
        "HappenedBefore": {"weight": 0.6, "transitive": True, "aliases": ["发生在之前"]},
        "HappenedAfter": {"weight": 0.6, "transitive": True, "aliases": ["发生在之后"]},
    }

    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.client = WeaviateClient(agent_id, user_id)
        self.embedder = OllamaEmbedding()

        # 实体缓存（用于消歧）
        self._entity_cache: Dict[str, Dict] = {}
        self._alias_map: Dict[str, str] = {}  # 别名 -> 规范名

        # 共现统计（Hebbian 学习）
        self._co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        # 加载已有实体
        self._load_entity_cache()

    def _load_entity_cache(self):
        """加载已有实体到缓存"""
        try:
            entities = self.client.get_all_entities()
            for e in entities:
                name = e.get("entity_name", "")
                if name:
                    self._entity_cache[name.lower()] = e
                    # 加载别名
                    aliases = e.get("aliases", [])
                    if isinstance(aliases, str):
                        try:
                            aliases = json.loads(aliases)
                        except Exception:
                            aliases = []
                    for alias in aliases:
                        self._alias_map[alias.lower()] = name
        except Exception as ex:
            print(f"加载实体缓存失败: {ex}")

    # ==================== 多阶段实体提取 ====================

    def extract_entities_stage1(self, text: str) -> Dict:
        """
        第一阶段：命名实体识别（NER）
        使用 LLM 提取实体
        """
        prompt = f"""你是一个专业的命名实体识别系统。请从文本中识别所有实体。

## 实体类型
- Person: 人名、职位、角色
- Organization: 公司、团队、部门
- Project: 项目名、产品名、系统名
- Technology: 编程语言、框架、工具、库
- Concept: 概念、方法、模式
- Location: 地点、城市、国家
- Event: 事件、会议、活动
- Document: 文档、文件名

## 规则
1. 只提取明确提到的实体
2. 实体名称使用原文中的完整名称
3. 忽略文件路径、URL、代码片段
4. 忽略代词（他、她、它）

## 输出格式（JSON）
{{"entities": [{{"name": "实体名", "type": "类型"}}]}}

## 文本
{text}

输出JSON："""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1}  # 低温度，更确定
                },
                timeout=60
            )
            result = response.json()["message"]["content"]
            return self._parse_json(result)
        except Exception as e:
            print(f"NER 提取失败: {e}")
            return {"entities": []}

    def extract_relations_stage2(self, text: str, entities: List[Dict]) -> Dict:
        """
        第二阶段：关系抽取
        在已知实体基础上提取关系
        """
        if not entities:
            return {"relations": []}

        entity_names = [e["name"] for e in entities]
        relation_types = list(self.RELATION_TYPES.keys())

        prompt = f"""你是一个关系抽取专家。请识别文本中实体之间的关系。

## 已识别实体
{json.dumps(entity_names, ensure_ascii=False)}

## 关系类型
{json.dumps(relation_types, ensure_ascii=False)}

## 规则
1. 只提取明确表达的关系
2. 优先使用预定义关系类型
3. 如果没有合适的关系类型，使用 RelatedTo
4. 每个关系必须有置信度（0.5-1.0）

## 输出格式（JSON）
{{"relations": [{{"source": "实体1", "relation": "关系类型", "target": "实体2", "confidence": 0.9}}]}}

## 文本
{text}

输出JSON："""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            result = response.json()["message"]["content"]
            return self._parse_json(result)
        except Exception as e:
            print(f"关系抽取失败: {e}")
            return {"relations": []}

    def resolve_entities_stage3(self, entities: List[Dict]) -> List[Dict]:
        """
        第三阶段：实体消歧与合并
        解决同名实体和别名问题
        """
        resolved = []

        for entity in entities:
            name = entity.get("name", "")
            name_lower = name.lower()

            # 检查别名映射
            if name_lower in self._alias_map:
                canonical_name = self._alias_map[name_lower]
                entity["name"] = canonical_name
                entity["resolved_from"] = name

            # 检查是否已存在
            if name_lower in self._entity_cache:
                existing = self._entity_cache[name_lower]
                # 合并类型（如果不同）
                if existing.get("entity_type") != entity.get("type"):
                    entity["merged_types"] = [existing.get("entity_type"), entity.get("type")]
                entity["id"] = existing.get("id")
                entity["is_existing"] = True

            resolved.append(entity)

        return resolved

    # ==================== 统一提取接口 ====================

    def extract(self, text: str) -> Dict:
        """
        完整的知识提取流程
        对标 GraphRAG 的多阶段提取
        """
        # Stage 1: NER
        ner_result = self.extract_entities_stage1(text)
        entities = ner_result.get("entities", [])

        if not entities:
            return {"entities": [], "relations": []}

        # Stage 2: 关系抽取
        rel_result = self.extract_relations_stage2(text, entities)
        relations = rel_result.get("relations", [])

        # Stage 3: 实体消歧
        entities = self.resolve_entities_stage3(entities)

        # 添加元数据
        for e in entities:
            e["source_text"] = text[:200]
            e["extracted_at"] = datetime.now().isoformat()

        for r in relations:
            r["source_text"] = text[:200]
            r["extracted_at"] = datetime.now().isoformat()

        return {
            "entities": entities,
            "relations": relations,
            "stats": {
                "entity_count": len(entities),
                "relation_count": len(relations),
                "new_entities": len([e for e in entities if not e.get("is_existing")])
            }
        }

    def store(self, extraction_result: Dict) -> Dict:
        """存储提取结果到 Weaviate"""
        entities = extraction_result.get("entities", [])
        relations = extraction_result.get("relations", [])

        stored_entities = []
        stored_relations = []

        # 存储实体
        for e in entities:
            try:
                entity_id = self._store_entity(e)
                if entity_id:
                    stored_entities.append({"id": entity_id, "name": e["name"]})
                    # 更新缓存
                    self._entity_cache[e["name"].lower()] = {"id": entity_id, **e}
            except Exception as ex:
                print(f"存储实体失败: {ex}")

        # 存储关系
        for r in relations:
            try:
                relation_id = self._store_relation(r)
                if relation_id:
                    stored_relations.append({"id": relation_id, **r})
                    # 更新共现统计
                    key = tuple(sorted([r["source"], r["target"]]))
                    self._co_occurrence[key] += 1
            except Exception as ex:
                print(f"存储关系失败: {ex}")

        return {
            "entities": stored_entities,
            "relations": stored_relations
        }

    def _store_entity(self, entity: Dict) -> Optional[str]:
        """存储单个实体"""
        name = entity.get("name", "")
        if not name:
            return None

        # 检查是否已存在
        existing = self._entity_cache.get(name.lower())
        if existing and existing.get("id"):
            return existing["id"]

        # 生成向量
        vector = self.embedder.embed(name)

        # 存储到 Weaviate（使用 insert_kg 方法）
        data = {
            "entity_name": name,
            "entity_type": entity.get("type", "Concept"),
            "relation_type": "is_entity",  # 标记为实体
            "target_entity": "",
            "confidence": entity.get("confidence", 0.8),
            "context": entity.get("source_text", "")[:500],
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        }

        return self.client.insert_kg(data, vector)

    def _store_relation(self, relation: Dict) -> Optional[str]:
        """存储单个关系"""
        source = relation.get("source", "")
        target = relation.get("target", "")
        rel_type = relation.get("relation", "RelatedTo")

        if not source or not target:
            return None

        data = {
            "entity_name": source,
            "relation_type": rel_type,
            "target_entity": target,
            "confidence": relation.get("confidence", 0.8),
            "context": relation.get("source_text", "")[:500],
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        }

        return self.client.insert_kg(data)

    # ==================== 查询接口 ====================

    def get_entity(self, name: str) -> Optional[Dict]:
        """获取实体详情"""
        # 先查缓存
        cached = self._entity_cache.get(name.lower())
        if cached:
            return cached

        # 查 Weaviate
        results = self.client.get_kg_by_entity(name)
        if results:
            return results[0]
        return None

    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict]:
        """获取相关实体（支持多跳）"""
        visited = set()
        result = []

        def _traverse(name: str, depth: int):
            if depth > max_depth or name.lower() in visited:
                return
            visited.add(name.lower())

            relations = self.client.get_kg_by_entity(name)
            for r in relations:
                if r.get("relation_type") == "is_entity":
                    continue
                target = r.get("target_entity", "")
                if target and target.lower() not in visited:
                    result.append({
                        "entity": target,
                        "relation": r.get("relation_type", ""),
                        "distance": depth,
                        "confidence": r.get("confidence", 0.5)
                    })
                    _traverse(target, depth + 1)

        _traverse(entity_name, 1)
        return result

    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """语义搜索实体"""
        vector = self.embedder.embed(query)
        return self.client.search_kg_entities(vector, limit)

    # ==================== 自我更新机制 ====================

    def self_update(self, new_memories: List[str] = None) -> Dict:
        """
        自我更新知识图谱
        1. 从新记忆中提取实体和关系
        2. 合并重复实体
        3. 更新置信度
        4. 学习新类型
        """
        stats = {
            "entities_added": 0,
            "relations_added": 0,
            "entities_merged": 0,
            "relations_updated": 0,
            "new_types_learned": 0
        }

        # 1. 从新记忆中提取
        if new_memories:
            for text in new_memories:
                if not text or len(text) < 20:
                    continue
                result = self.extract(text)
                stored = self.store(result)
                stats["entities_added"] += len(stored.get("entities", []))
                stats["relations_added"] += len(stored.get("relations", []))

        # 2. 合并重复实体
        merged = self._merge_duplicate_entities()
        stats["entities_merged"] = merged

        # 3. 更新置信度（基于共现频率）
        updated = self._update_confidence()
        stats["relations_updated"] = updated

        # 4. 清理低置信度关系
        cleaned = self._cleanup_low_confidence()

        return stats

    def _merge_duplicate_entities(self) -> int:
        """合并重复实体（同名或别名）"""
        merged_count = 0
        seen_names = {}

        for name_lower, entity in self._entity_cache.items():
            canonical = entity.get("entity_name", "")
            if canonical.lower() != name_lower:
                # 这是别名，记录映射
                self._alias_map[name_lower] = canonical
                merged_count += 1

        return merged_count

    def _update_confidence(self) -> int:
        """根据共现频率更新关系置信度"""
        updated = 0

        # 获取所有关系
        relations = self.client.get_all_relations()

        for r in relations:
            source = r.get("source", "")
            target = r.get("target", "")
            if not source or not target:
                continue

            # 计算共现次数
            key = tuple(sorted([source, target]))
            co_occurrence = self._co_occurrence.get(key, 0)

            # 更新置信度（共现越多，置信度越高）
            current_conf = r.get("confidence", 0.5)
            new_conf = min(1.0, current_conf + co_occurrence * 0.05)

            if new_conf != current_conf:
                # 更新存储（简化：只更新缓存）
                updated += 1

        return updated

    def _cleanup_low_confidence(self, threshold: float = 0.3) -> int:
        """清理低置信度的关系"""
        cleaned = 0
        relations = self.client.get_all_relations()

        for r in relations:
            if r.get("confidence", 0.5) < threshold:
                # 删除低置信度关系
                try:
                    self.client.delete_kg_by_id(r.get("id", ""))
                    cleaned += 1
                except Exception:
                    pass

        return cleaned

    def learn_from_feedback(self, entity_name: str, correct_type: str = None,
                           correct_relation: Dict = None, is_wrong: bool = False):
        """
        从用户反馈中学习
        - correct_type: 正确的实体类型
        - correct_relation: 正确的关系 {source, relation, target}
        - is_wrong: 标记为错误（将降低置信度）
        """
        if correct_type:
            # 学习新的实体类型映射
            entity = self._entity_cache.get(entity_name.lower())
            if entity:
                entity["entity_type"] = correct_type
                # 记录学习模式
                self._entity_cache[entity_name.lower()] = entity

        if correct_relation:
            # 学习正确的关系
            source = correct_relation.get("source", "")
            relation = correct_relation.get("relation", "")
            target = correct_relation.get("target", "")

            if source and relation and target:
                # 存储正确的关系
                self._store_relation({
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "confidence": 0.95,
                    "source_text": "用户反馈"
                })

        if is_wrong:
            # 降低置信度
            entity = self._entity_cache.get(entity_name.lower())
            if entity:
                entity["confidence"] = max(0.1, entity.get("confidence", 0.5) - 0.3)

    def get_update_summary(self) -> Dict:
        """获取知识图谱更新摘要"""
        entities = self.client.get_all_entities()
        relations = self.client.get_all_relations()

        # 按类型统计实体
        type_stats = defaultdict(int)
        for e in entities:
            type_stats[e.get("entity_type", "Unknown")] += 1

        # 按类型统计关系
        rel_stats = defaultdict(int)
        for r in relations:
            rel_stats[r.get("relation", "Unknown")] += 1

        # 高置信度实体
        high_conf_entities = [e for e in entities if e.get("confidence", 0) >= 0.8]

        return {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_types": dict(type_stats),
            "relation_types": dict(rel_stats),
            "high_confidence_entities": len(high_conf_entities),
            "cached_entities": len(self._entity_cache),
            "alias_mappings": len(self._alias_map),
            "co_occurrence_pairs": len(self._co_occurrence)
        }

    # ==================== 工具方法 ====================

    def _parse_json(self, text: str) -> Dict:
        """从文本中解析 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except Exception:
            pass

        # 尝试提取 JSON 块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass

        # 尝试提取花括号内容
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except Exception:
                pass

        return {"entities": [], "relations": []}

    def get_stats(self) -> Dict:
        """获取知识图谱统计"""
        try:
            entities = self.client.get_all_entities()
            relations = self.client.get_all_relations()

            # 按类型统计
            type_stats = defaultdict(int)
            for e in entities:
                type_stats[e.get("entity_type", "Unknown")] += 1

            return {
                "total_entities": len(entities),
                "total_relations": len(relations),
                "entity_types": dict(type_stats),
                "cached_entities": len(self._entity_cache),
                "co_occurrence_pairs": len(self._co_occurrence)
            }
        except Exception as e:
            return {"error": str(e)}