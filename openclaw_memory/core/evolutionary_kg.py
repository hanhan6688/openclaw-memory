"""
自我进化知识图谱系统
支持：
1. 动态规则学习 - 从用户反馈中学习
2. Hebbian学习 - 频繁共现实体增强连接
3. 置信度衰减 - 长时间未访问的关系衰减
4. 关系推理 - 从现有关系推理新关系
5. 矛盾检测与解决
6. 实体消歧与合并
7. 四步提取流程 - NER → 关系预判 → 关系分类 → 结构化转换
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from core.weaviate_client import WeaviateClient
from core.embeddings import OllamaEmbedding
import requests


class EvolutionaryKnowledgeGraph:
    """自我进化知识图谱"""

    # 预定义关系类型（可扩展）
    DEFAULT_RELATION_TYPES = {
        # 人物关系
        "合作": {"category": "social", "opposite": "竞争", "transitive": True},
        "竞争": {"category": "social", "opposite": "合作", "transitive": False},
        "认识": {"category": "social", "opposite": None, "transitive": True},
        "雇佣": {"category": "social", "opposite": "被雇佣", "transitive": False},
        "管理": {"category": "social", "opposite": "被管理", "transitive": True},
        "任职于": {"category": "social", "opposite": None, "transitive": False},
        "拥有资产": {"category": "social", "opposite": None, "transitive": False},
        "亲属": {"category": "social", "opposite": None, "transitive": True},

        # 技术关系
        "使用": {"category": "technical", "opposite": None, "transitive": False},
        "依赖": {"category": "technical", "opposite": "独立", "transitive": True},
        "包含": {"category": "technical", "opposite": "属于", "transitive": True},
        "实现": {"category": "technical", "opposite": None, "transitive": False},
        "调用": {"category": "technical", "opposite": "被调用", "transitive": False},
        "开发": {"category": "technical", "opposite": None, "transitive": False},

        # 概念关系
        "是一种": {"category": "conceptual", "opposite": None, "transitive": True},
        "相关": {"category": "conceptual", "opposite": None, "transitive": True},
        "属于": {"category": "conceptual", "opposite": "包含", "transitive": True},

        # 空间关系
        "位于": {"category": "spatial", "opposite": None, "transitive": False},

        # 业务关系
        "收购": {"category": "business", "opposite": "被收购", "transitive": False},
        "投资": {"category": "business", "opposite": "被投资", "transitive": False},
    }

    # 预定义实体类型
    DEFAULT_ENTITY_TYPES = {
        "人物": {"color": "#4CAF50", "icon": "👤", "patterns": [r'^[\u4e00-\u9fa5]{2,4}$']},
        "组织": {"color": "#2196F3", "icon": "🏢", "patterns": [r'.*公司$', r'.*集团$', r'.*团队$']},
        "项目": {"color": "#FF9800", "icon": "📁", "patterns": [r'.*项目$', r'.*系统$', r'.*应用$']},
        "技术": {"color": "#9C27B0", "icon": "⚙️", "patterns": [r'Python', r'Java', r'React', r'API', r'[A-Z][a-z]+']},
        "概念": {"color": "#607D8B", "icon": "💡", "patterns": []},
        "地点": {"color": "#795548", "icon": "📍", "patterns": [r'.*[省市县村镇]$']},
        "事件": {"color": "#F44336", "icon": "📅", "patterns": []},
        "产品": {"color": "#00BCD4", "icon": "📦", "patterns": []},
    }

    def __init__(self, agent_id: str, user_id: str = "default"):
        self.agent_id = agent_id
        self.client = WeaviateClient(agent_id, user_id)
        self.embedder = OllamaEmbedding()

        # 动态学习的规则
        self.learned_relation_types = {}
        self.learned_entity_types = {}
        self.entity_aliases = {}
        self.extraction_patterns = []

        # Hebbian学习参数
        self.co_occurrence = defaultdict(int)
        self.access_history = defaultdict(list)

        # 加载学习到的规则
        self._load_learned_rules()

    def _load_learned_rules(self):
        """从存储加载学习到的规则"""
        try:
            rules = self.client.get_kg_by_entity("__rules__")
            for rule in rules:
                if rule.get("relation_type") == "learned_relation":
                    self.learned_relation_types[rule.get("target_entity")] = json.loads(rule.get("context", "{}"))
                elif rule.get("relation_type") == "learned_entity":
                    self.learned_entity_types[rule.get("target_entity")] = json.loads(rule.get("context", "{}"))
                elif rule.get("relation_type") == "extraction_pattern":
                    self.extraction_patterns.append(json.loads(rule.get("context", "{}")))
        except Exception:
            pass

    def _save_learned_rule(self, rule_type: str, name: str, data: dict):
        """保存学习到的规则"""
        vector = self.embedder.embed(f"{rule_type} {name}")
        self.client.insert_kg({
            "entity_name": "__rules__",
            "entity_type": "system",
            "relation_type": rule_type,
            "target_entity": name,
            "context": json.dumps(data, ensure_ascii=False),
            "source": "learned",
            "confidence": 1.0,
            "access_count": 0
        }, vector)

    @property
    def all_relation_types(self) -> Dict:
        """获取所有关系类型（预定义+学习）"""
        return {**self.DEFAULT_RELATION_TYPES, **self.learned_relation_types}

    @property
    def all_entity_types(self) -> Dict:
        """获取所有实体类型（预定义+学习）"""
        return {**self.DEFAULT_ENTITY_TYPES, **self.learned_entity_types}

    def learn_entity_type(self, type_name: str, color: str = "#999999", icon: str = "📌"):
        """学习新的实体类型"""
        self.learned_entity_types[type_name] = {"color": color, "icon": icon}
        self._save_learned_rule("learned_entity", type_name, {"color": color, "icon": icon})
        return {"status": "learned", "type": type_name}

    def learn_relation_type(self, relation_name: str, category: str = "custom",
                            opposite: str = None, transitive: bool = False):
        """学习新的关系类型"""
        self.learned_relation_types[relation_name] = {
            "category": category,
            "opposite": opposite,
            "transitive": transitive
        }
        self._save_learned_rule("learned_relation", relation_name, {
            "category": category,
            "opposite": opposite,
            "transitive": transitive
        })
        return {"status": "learned", "relation": relation_name}

    # ==================== 四步提取流程 ====================

    def extract_with_context(self, text: str, context: Dict = None) -> Dict:
        """
        四步提取流程：
        1. 实体识别（NER）- 找实体
        2. 关系预判 - 筛选有效实体对
        3. 关系分类 - 确定关系类型
        4. 结构化转换 - 输出最终格式
        """
        context = context or {}

        # 步骤1：实体识别
        entities = self._step1_ner(text)
        if not entities:
            return {"entities": [], "relations": [], "inferred_relations": []}

        # 步骤2：关系预判
        entity_pairs = self._step2_relation_prediction(text, entities)

        # 步骤3：关系分类
        relations = self._step3_relation_classification(text, entity_pairs)

        # 步骤4：结构化转换
        result = self._step4_structural_conversion(entities, relations)

        return result

    def _step1_ner(self, text: str) -> List[Dict]:
        """
        步骤1：实体识别（NER）
        目标：从文本中定位所有有意义的实体，并标注类型
        """
        entity_types = list(self.all_entity_types.keys())

        prompt = f"""你是一个实体识别专家。请从文本中识别所有有意义的实体。

## 实体类型
{json.dumps(entity_types, ensure_ascii=False)}

## 识别规则
1. 人物：人名、职位、角色（如：张三、CEO、产品经理）
2. 组织：公司、团队、部门（如：阿里巴巴、技术部）
3. 项目：项目名、系统名、产品名（如：记忆系统、电商平台）
4. 技术：编程语言、框架、工具（如：Python、React、Docker）
5. 地点：地理位置（如：北京、杭州）
6. 概念：抽象概念、方法论
7. 事件：会议、活动、事故（如：产品发布会、系统故障）
8. 产品：具体产品（如：iPhone、微信）

## 跳过
- 文件路径、URL、命令行
- 代码片段、错误信息
- 无意义的停用词

## 输出格式（JSON数组）
[
  {{"name": "实体名", "type": "类型", "confidence": 0.9, "evidence": "原文中的位置或上下文"}}
]

## 文本
{text}

只输出JSON数组，不要其他内容："""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=60
            )
            result = response.json()["message"]["content"]

            # 解析JSON
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                entities = json.loads(json_match.group())
                # 验证和规范化
                return self._validate_entities(entities)
        except Exception as e:
            print(f"NER失败: {e}")

        return []

    def _step2_relation_prediction(self, text: str, entities: List[Dict]) -> List[Tuple]:
        """
        步骤2：关系预判
        目标：排除无关联的实体对，减少后续计算量
        """
        if len(entities) < 2:
            return []

        entity_names = [e["name"] for e in entities]

        prompt = f"""你是一个关系预判专家。判断哪些实体对之间可能存在关系。

## 实体列表
{json.dumps(entity_names, ensure_ascii=False)}

## 原文
{text}

## 任务
1. 分析实体在原文中的共现和语义关联
2. 只保留可能存在关系的实体对
3. 排除无直接关联的实体对（如：金额和地点通常无直接关系）

## 输出格式（JSON数组）
[
  {{"source": "实体A", "target": "实体B", "has_relation": true, "reason": "简要原因"}}
]

只输出可能存在关系的实体对，不要其他内容："""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=60
            )
            result = response.json()["message"]["content"]

            # 解析JSON
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                pairs = json.loads(json_match.group())
                # 转换为元组列表
                return [(p["source"], p["target"]) for p in pairs if p.get("has_relation", True)]
        except Exception as e:
            print(f"关系预判失败: {e}")
            # 降级：返回所有可能的实体对
            return [(entities[i]["name"], entities[j]["name"])
                    for i in range(len(entities))
                    for j in range(i+1, len(entities))]

        return []

    def _step3_relation_classification(self, text: str, entity_pairs: List[Tuple]) -> List[Dict]:
        """
        步骤3：关系分类
        目标：给有效实体对标注具体关系类型
        """
        if not entity_pairs:
            return []

        relation_types = list(self.all_relation_types.keys())

        prompt = f"""你是一个关系分类专家。为实体对标注具体的关系类型。

## 关系类型词典
{json.dumps(relation_types, ensure_ascii=False)}

## 关系类型说明
- 合作/竞争：人物或组织之间的合作关系
- 雇佣/任职于：人物与组织的工作关系
- 拥有资产：人物拥有的资产信息
- 亲属：人物之间的亲属关系
- 使用/开发：人物或组织使用/开发技术或产品
- 位于：实体与地点的关系
- 收购/投资：商业交易关系
- 包含/属于：整体与部分的关系
- 是一种：类型继承关系
- 相关：其他语义关联

## 原文
{text}

## 实体对
{json.dumps([{"source": p[0], "target": p[1]} for p in entity_pairs], ensure_ascii=False)}

## 输出格式（JSON数组）
[
  {{
    "source": "源实体",
    "relation": "关系类型",
    "target": "目标实体",
    "confidence": 0.8,
    "evidence": "原文依据"
  }}
]

如果实体对之间没有明确关系，可以跳过。只输出JSON数组："""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=60
            )
            result = response.json()["message"]["content"]

            # 解析JSON
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                relations = json.loads(json_match.group())
                # 验证关系类型
                return [r for r in relations if self._validate_relation(r)]
        except Exception as e:
            print(f"关系分类失败: {e}")

        return []

    def _step4_structural_conversion(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """
        步骤4：结构化转换
        目标：转换为知识图谱格式
        """
        # 实体去重和规范化
        entity_map = {}
        for entity in entities:
            name = entity.get("name", "").strip()
            normalized = self.entity_aliases.get(name, name)
            entity["name"] = normalized
            entity_map[name] = normalized

            # 验证实体类型
            if entity.get("type") not in self.all_entity_types:
                entity["type"] = self._infer_entity_type(entity["name"], entity.get("type"))

        # 规范化关系中的实体名
        for relation in relations:
            relation["source"] = entity_map.get(relation.get("source"), relation.get("source"))
            relation["target"] = entity_map.get(relation.get("target"), relation.get("target"))

        return {
            "entities": entities,
            "relations": relations,
            "inferred_relations": []
        }

    # ==================== 验证方法 ====================

    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """验证和规范化实体列表"""
        valid_entities = []
        seen_names = set()

        for entity in entities:
            name = entity.get("name", "").strip()
            if not name or name in seen_names:
                continue

            # 检查别名
            normalized = self.entity_aliases.get(name, name)

            # 验证类型
            entity_type = entity.get("type", "概念")
            if entity_type not in self.all_entity_types:
                entity_type = self._infer_entity_type(normalized, entity_type)

            valid_entities.append({
                "name": normalized,
                "type": entity_type,
                "confidence": entity.get("confidence", 0.8),
                "evidence": entity.get("evidence", "")
            })
            seen_names.add(name)

        return valid_entities

    def _validate_relation(self, relation: Dict) -> bool:
        """验证关系是否有效"""
        if not relation.get("source") or not relation.get("target"):
            return False

        relation_type = relation.get("relation", "")
        # 允许预定义关系或自定义关系
        if relation_type in self.all_relation_types:
            return True

        # 如果不在预定义中，也允许（可能是新关系）
        return len(relation_type) > 0

    def _infer_entity_type(self, name: str, suggested_type: str = None) -> str:
        """基于规则推断实体类型"""
        # 检查预定义模式
        for entity_type, config in self.all_entity_types.items():
            patterns = config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, name):
                    return entity_type

        # 基于特征推断
        if re.search(r'[省市县村镇路]', name):
            return "地点"
        if re.search(r'公司|集团|团队|组织|部门', name):
            return "组织"
        if re.search(r'项目|系统|应用|平台', name):
            return "项目"
        if re.search(r'\d+.*[美元|元|万|亿]', name):
            return "金额"
        if re.search(r'\d{4}年|\d+月|\d+日', name):
            return "时间"
        if re.search(r'^[\u4e00-\u9fa5]{2,4}$', name):
            return "人物"

        return suggested_type or "概念"

    # ==================== 存储方法 ====================

    def store_with_learning(self, extraction_result: Dict, source: str = None) -> Dict:
        """存储提取结果并更新学习状态"""
        stored = {"entities": [], "relations": [], "inferred": []}

        entities = extraction_result.get("entities", [])
        entity_names = [e["name"] for e in entities]

        # 存储实体
        for entity in entities:
            entity_id = self._store_entity(
                entity["name"],
                entity.get("type", "概念"),
                source=source,
                confidence=entity.get("confidence", 1.0)
            )
            stored["entities"].append({"name": entity["name"], "id": entity_id})

        # 更新共现计数（Hebbian学习）
        for i, name_a in enumerate(entity_names):
            for name_b in entity_names[i+1:]:
                key = tuple(sorted([name_a, name_b]))
                self.co_occurrence[key] += 1

        # 存储关系
        for relation in extraction_result.get("relations", []):
            relation_id = self._store_relation(
                relation["source"],
                relation["relation"],
                relation["target"],
                context=relation.get("evidence"),
                source=source,
                confidence=relation.get("confidence", 1.0)
            )
            stored["relations"].append({
                "source": relation["source"],
                "relation": relation["relation"],
                "target": relation["target"],
                "id": relation_id
            })

        # 存储推理关系
        for relation in extraction_result.get("inferred_relations", []):
            relation_id = self._store_relation(
                relation["source"],
                relation["relation"],
                relation["target"],
                context=f"推理: {relation.get('reason', '')}",
                source=f"inferred:{source}",
                confidence=relation.get("confidence", 0.5)
            )
            stored["inferred"].append({
                "source": relation["source"],
                "relation": relation["relation"],
                "target": relation["target"],
                "id": relation_id
            })

        return stored

    def _store_entity(self, name: str, entity_type: str, source: str = None, confidence: float = 1.0) -> str:
        """存储实体"""
        vector = self.embedder.embed(f"{name} {entity_type}")
        return self.client.insert_kg({
            "entity_name": name,
            "entity_type": entity_type,
            "relation_type": "is_a",
            "target_entity": entity_type,
            "context": f"{name} 是一个 {entity_type}",
            "source": source,
            "confidence": confidence,
            "access_count": 0
        }, vector)

    def _store_relation(self, source_entity: str, relation_type: str, target_entity: str,
                        context: str = None, source: str = None, confidence: float = 1.0) -> str:
        """存储关系"""
        relation_text = f"{source_entity} {relation_type} {target_entity}"
        vector = self.embedder.embed(relation_text)
        return self.client.insert_kg({
            "entity_name": source_entity,
            "entity_type": "entity",
            "relation_type": relation_type,
            "target_entity": target_entity,
            "context": context or relation_text,
            "source": source,
            "confidence": confidence,
            "access_count": 0
        }, vector)

    # ==================== 其他方法 ====================

    def hebbian_reinforce(self, entity_names: List[str]):
        """Hebbian强化"""
        for i, name_a in enumerate(entity_names):
            for name_b in entity_names[i+1:]:
                key = tuple(sorted([name_a, name_b]))
                self.co_occurrence[key] += 1

                if self.co_occurrence[key] >= 3:
                    self._ensure_relation(name_a, "相关", name_b, strength=self.co_occurrence[key])

    def _ensure_relation(self, source: str, relation: str, target: str, strength: int = 1):
        """确保关系存在"""
        existing = self.client.get_kg_by_entity(source)
        for e in existing:
            if e.get("target_entity") == target and e.get("relation_type") == relation:
                return
        self._store_relation(source, relation, target, context=f"自动关联(共现{strength}次)", confidence=min(0.9, strength * 0.1))

    def infer_new_relations(self) -> List[Dict]:
        """推理新关系"""
        relations = self.get_all_relations()
        inferred = []

        for r1 in relations:
            if not self.all_relation_types.get(r1["relation"], {}).get("transitive", False):
                continue

            for r2 in relations:
                if r1["target"] == r2["source"] and r1["relation"] == r2["relation"]:
                    exists = any(
                        r["source"] == r1["source"] and r["target"] == r2["target"]
                        for r in relations
                    )
                    if not exists:
                        inferred.append({
                            "source": r1["source"],
                            "relation": r1["relation"],
                            "target": r2["target"],
                            "reason": f"传递性推理: {r1['source']}->{r1['target']}->{r2['target']}",
                            "confidence": min(r1.get("confidence", 0.5), r2.get("confidence", 0.5)) * 0.8
                        })

        return inferred

    def detect_contradictions(self) -> List[Dict]:
        """检测矛盾关系"""
        relations = self.get_all_relations()
        contradictions = []

        for r1 in relations:
            opposite = self.all_relation_types.get(r1["relation"], {}).get("opposite")
            if not opposite:
                continue

            for r2 in relations:
                if (r1["source"] == r2["source"] and
                    r1["target"] == r2["target"] and
                    r2["relation"] == opposite):
                    contradictions.append({
                        "entity_a": r1["source"],
                        "entity_b": r1["target"],
                        "relation_1": r1,
                        "relation_2": r2,
                        "type": "direct_contradiction"
                    })

        return contradictions

    def resolve_contradiction(self, contradiction: Dict, keep: str = "higher_confidence"):
        """解决矛盾"""
        r1 = contradiction["relation_1"]
        r2 = contradiction["relation_2"]

        if keep == "higher_confidence":
            to_remove = r1 if r1.get("confidence", 0) < r2.get("confidence", 0) else r2
        elif keep == "newer":
            to_remove = r1
        else:
            to_remove = r2

        self.client.delete_kg_by_entity(to_remove["source"])

    def merge_entities(self, entity_a: str, entity_b: str, canonical: str):
        """合并实体"""
        other = entity_b if canonical == entity_a else entity_a
        self.entity_aliases[other] = canonical

        relations = self.client.get_kg_by_entity(entity_a) + self.client.get_kg_by_entity(entity_b)

        for r in relations:
            if r.get("entity_name") in [entity_a, entity_b]:
                r["entity_name"] = canonical
            if r.get("target_entity") in [entity_a, entity_b]:
                r["target_entity"] = canonical

        self.client.delete_kg_by_entity(entity_a)
        self.client.delete_kg_by_entity(entity_b)

    def learn_from_feedback(self, entity: str, relation: str, target: str,
                            feedback: str, correct: bool):
        """从用户反馈学习"""
        if correct:
            self._store_relation(entity, relation, target, confidence=1.0)
            self.extraction_patterns.append({
                "pattern": f"{entity} {relation} {target}",
                "feedback": "correct",
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.client.delete_kg_by_entity(entity)
            self.extraction_patterns.append({
                "pattern": f"NOT: {entity} {relation} {target}",
                "feedback": "incorrect",
                "timestamp": datetime.now().isoformat()
            })

    def get_all_entities(self) -> List[Dict]:
        """获取所有实体"""
        all_kg = self.client.get_kg(limit=1000)
        entities = {}

        for obj in all_kg:
            name = obj.get("entity_name")
            if name and name != "__rules__" and name not in entities:
                entities[name] = {
                    "name": name,
                    "type": obj.get("entity_type"),
                    "access_count": obj.get("access_count", 0),
                    "color": self.all_entity_types.get(obj.get("entity_type"), {}).get("color", "#999"),
                    "icon": self.all_entity_types.get(obj.get("entity_type"), {}).get("icon", "📌")
                }

        return list(entities.values())

    def get_all_relations(self) -> List[Dict]:
        """获取所有关系"""
        all_kg = self.client.get_kg(limit=1000)
        relations = []
        seen = set()

        for obj in all_kg:
            if obj.get("relation_type") == "is_a":
                continue

            key = (obj.get("entity_name"), obj.get("relation_type"), obj.get("target_entity"))
            if key not in seen:
                seen.add(key)
                relations.append({
                    "source": obj.get("entity_name"),
                    "relation": obj.get("relation_type"),
                    "target": obj.get("target_entity"),
                    "context": obj.get("context"),
                    "confidence": obj.get("confidence"),
                    "category": self.all_relation_types.get(obj.get("relation_type"), {}).get("category", "custom")
                })

        return relations

    def get_graph_data(self) -> Dict:
        """获取知识图谱可视化数据"""
        entities = self.get_all_entities()
        relations = self.get_all_relations()

        connection_count = defaultdict(int)
        for r in relations:
            connection_count[r["source"]] += 1
            connection_count[r["target"]] += 1

        nodes = [
            {
                "id": e["name"],
                "label": e["name"],
                "group": e.get("type", "概念"),
                "color": e.get("color", "#999"),
                "icon": e.get("icon", "📌"),
                "size": 10 + connection_count.get(e["name"], 0) * 3
            }
            for e in entities
        ]

        links = [
            {
                "source": r["source"],
                "target": r["target"],
                "relation": r["relation"],
                "category": r.get("category", "custom"),
                "value": r.get("confidence", 1.0)
            }
            for r in relations
            if r["source"] and r["target"]
        ]

        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "entity_types": self.all_entity_types,
                "relation_types": {k: v["category"] for k, v in self.all_relation_types.items()},
                "co_occurrence_count": len(self.co_occurrence)
            }
        }

    def close(self):
        """关闭连接"""
        self.client.close()