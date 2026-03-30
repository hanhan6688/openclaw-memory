"""
增强版实体关系抽取器
借鉴 GraphRAG、Mem0、LlamaIndex 的最佳实践

优化点：
1. 联合抽取 - NER + RE 一步完成，减少 LLM 调用
2. CoT 提示 - 思维链引导模型推理
3. 置信度校准 - 多维度验证
4. 去重和冲突检测
5. 增量更新支持
"""

import json
import re
import time
import requests
import sys
import os
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base import (
    EntityRelationExtractor, 
    ExtractionResult, 
    Entity, 
    Relation,
    ExtractorBackend
)
from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


# ============================================================================
# 提示词模板 - 借鉴 GraphRAG
# ============================================================================

UNIFIED_EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从文本中提取实体和关系。

## 任务
分析以下文本，识别所有有意义的实体及其之间的关系。

## 实体类型（优先级排序）
1. **人物** - 人名、职位、角色（如：张三、CEO、用户）
2. **组织** - 公司、团队、部门（如：腾讯、研发部）
3. **项目** - 项目名、系统名、产品名（如：微信、电商平台）
4. **技术** - 编程语言、框架、工具、概念（如：Python、React、微服务）
5. **产品** - 具体产品、服务（如：iPhone、云服务）
6. **地点** - 地理位置（如：深圳、总部）
7. **事件** - 会议、活动、事故（如：发布会、迭代会议）
8. **概念** - 抽象概念、方法论（如：敏捷开发、用户体验）

## 关系类型
| 类型 | 说明 | 示例 |
|------|------|------|
| 合作 | 协作关系 | 张三 合作 李四 |
| 任职于 | 工作关系 | 张三 任职于 腾讯 |
| 管理 | 管理关系 | 李四 管理 研发部 |
| 开发 | 开发关系 | 团队A 开发 电商平台 |
| 使用 | 技术依赖 | 项目 使用 React |
| 包含 | 组成关系 | 平台 包含 支付模块 |
| 依赖 | 依赖关系 | 模块A 依赖 模块B |
| 是一种 | 分类关系 | Vue 是一种 框架 |
| 位于 | 位置关系 | 公司 位于 深圳 |
| 相关 | 相关关系 | 概念A 相关 概念B |

## 提取规则
1. **实体**：必须是文本中明确提及的，不要推断
2. **关系**：必须有文本证据支持
3. **置信度**：
   - 0.9+ 直接陈述
   - 0.7-0.9 明确暗示
   - 0.5-0.7 可能推断
4. **跳过**：
   - 文件路径、URL、命令
   - 代码片段、错误信息
   - 单字词、停用词
   - 过于泛化的词（如：用户、系统、问题）

## 输出格式
```json
{{
  "analysis": "简要分析文本内容...",
  "entities": [
    {{"name": "实体名", "type": "类型", "confidence": 0.9, "evidence": "原文证据"}}
  ],
  "relations": [
    {{"source": "实体A", "relation": "关系类型", "target": "实体B", "confidence": 0.8, "evidence": "原文证据"}}
  ]
}}
```

## 文本
{text}

请先分析，再输出 JSON："""


INCREMENTAL_EXTRACTION_PROMPT = """你是一个知识图谱更新专家。请分析新文本，提取需要更新的实体和关系。

## 已有实体
{existing_entities}

## 已有关系
{existing_relations}

## 新文本
{text}

## 任务
1. 识别新实体（不在已有列表中的）
2. 识别新关系（或需要更新的关系）
3. 检测矛盾（新信息与旧信息冲突）

## 输出格式
```json
{{
  "new_entities": [...],
  "new_relations": [...],
  "updated_relations": [
    {{"old": "旧关系", "new": "新关系", "reason": "更新原因"}}
  ],
  "contradictions": [
    {{"old": "旧信息", "new": "新信息", "description": "矛盾描述"}}
  ]
}}
```"""


# ============================================================================
# 增强版抽取器
# ============================================================================

class EnhancedDecoderExtractor(EntityRelationExtractor):
    """
    增强版 Decoder 抽取器
    
    优化：
    1. 联合抽取（一次 LLM 调用完成 NER + RE）
    2. CoT 提示（思维链）
    3. 多轮验证
    4. 增量更新
    """
    
    def __init__(self,
                 entity_types: List[str] = None,
                 relation_types: List[str] = None,
                 model: str = None,
                 base_url: str = None,
                 timeout: int = 60,
                 enable_verification: bool = True):
        super().__init__(entity_types, relation_types)
        self.model = model or OLLAMA_CHAT_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.timeout = timeout
        self.enable_verification = enable_verification
        
        # 缓存
        self._entity_cache: Dict[str, Entity] = {}
        self._relation_cache: Set[Tuple[str, str, str]] = set()
    
    @property
    def backend(self) -> ExtractorBackend:
        return ExtractorBackend.DECODER
    
    def extract(self, text: str) -> ExtractionResult:
        """联合抽取实体和关系"""
        start_time = time.time()
        
        # 截断过长文本
        input_text = text[:2000] if len(text) > 2000 else text
        
        # 联合抽取
        prompt = UNIFIED_EXTRACTION_PROMPT.format(text=input_text)
        response = self._call_llm(prompt)
        
        if response:
            result = self._parse_unified_response(response, text)
            
            # 验证
            if self.enable_verification and result.entities:
                result = self._verify_extraction(result, text)
            
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        
        return ExtractionResult(
            entities=[],
            relations=[],
            latency_ms=(time.time() - start_time) * 1000,
            backend=self.backend_name
        )
    
    def extract_incremental(self, text: str, 
                           existing_entities: List[str] = None,
                           existing_relations: List[Tuple] = None) -> ExtractionResult:
        """增量抽取"""
        start_time = time.time()
        
        existing_entities = existing_entities or []
        existing_relations = existing_relations or []
        
        prompt = INCREMENTAL_EXTRACTION_PROMPT.format(
            existing_entities=json.dumps(existing_entities, ensure_ascii=False),
            existing_relations=json.dumps([f"{s}-{r}-{t}" for s, r, t in existing_relations], ensure_ascii=False),
            text=text[:1500]
        )
        
        response = self._call_llm(prompt)
        
        if response:
            result = self._parse_incremental_response(response, text)
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        
        return ExtractionResult(backend=self.backend_name)
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """批量抽取"""
        return [self.extract(text) for text in texts]
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM"""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 低温度，更确定
                        "top_p": 0.9
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
        
        except Exception as e:
            print(f"⚠️ LLM 调用失败: {e}")
        
        return None
    
    def _parse_unified_response(self, response: str, original_text: str) -> ExtractionResult:
        """解析联合抽取结果"""
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if not json_match:
            return ExtractionResult(backend=self.backend_name, raw_output=response)
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ExtractionResult(backend=self.backend_name, raw_output=response)
        
        entities = []
        entity_names = set()
        
        # 解析实体
        for e in data.get("entities", []):
            name = e.get("name", "").strip()
            
            # 去重和过滤
            if not name or len(name) < 2 or name in entity_names:
                continue
            
            # 过滤无效实体
            if self._is_invalid_entity(name):
                continue
            
            # 类型验证
            entity_type = e.get("type", "概念")
            if entity_type not in self.entity_types:
                entity_type = self._infer_entity_type(name)
            
            entity = Entity(
                name=name,
                type=entity_type,
                confidence=min(1.0, max(0.1, e.get("confidence", 0.8))),
                metadata={"evidence": e.get("evidence", "")}
            )
            
            entities.append(entity)
            entity_names.add(name)
        
        relations = []
        relation_keys = set()
        
        # 解析关系
        for r in data.get("relations", []):
            source = r.get("source", "").strip()
            target = r.get("target", "").strip()
            
            # 验证实体存在
            if source not in entity_names or target not in entity_names:
                continue
            
            rel_type = r.get("relation", "相关")
            key = (source, rel_type, target)
            
            # 去重
            if key in relation_keys:
                continue
            
            relation = Relation(
                source=source,
                target=target,
                type=rel_type,
                confidence=min(1.0, max(0.1, r.get("confidence", 0.7))),
                evidence=r.get("evidence", "")
            )
            
            relations.append(relation)
            relation_keys.add(key)
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            backend=self.backend_name,
            raw_output=response
        )
    
    def _parse_incremental_response(self, response: str, original_text: str) -> ExtractionResult:
        """解析增量抽取结果"""
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if not json_match:
            return ExtractionResult(backend=self.backend_name)
        
        try:
            data = json.loads(json_match.group())
        except Exception:
            return ExtractionResult(backend=self.backend_name)
        
        entities = []
        for e in data.get("new_entities", []):
            name = e.get("name", "").strip()
            if name and len(name) >= 2:
                entities.append(Entity(
                    name=name,
                    type=e.get("type", "概念"),
                    confidence=e.get("confidence", 0.8)
                ))
        
        relations = []
        for r in data.get("new_relations", []):
            source = r.get("source", "").strip()
            target = r.get("target", "").strip()
            if source and target:
                relations.append(Relation(
                    source=source,
                    target=target,
                    type=r.get("relation", "相关"),
                    confidence=r.get("confidence", 0.7),
                    evidence=r.get("evidence", "")
                ))
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            backend=self.backend_name
        )
    
    def _verify_extraction(self, result: ExtractionResult, original_text: str) -> ExtractionResult:
        """验证抽取结果"""
        if not result.entities:
            return result
        
        # 验证实体是否在原文中
        verified_entities = []
        text_lower = original_text.lower()
        
        for entity in result.entities:
            if entity.name.lower() in text_lower:
                verified_entities.append(entity)
            elif entity.confidence >= 0.9:
                # 高置信度实体即使不在原文也保留（可能是同义词）
                verified_entities.append(entity)
        
        # 验证关系实体存在
        entity_names = {e.name for e in verified_entities}
        verified_relations = []
        
        for relation in result.relations:
            if relation.source in entity_names and relation.target in entity_names:
                verified_relations.append(relation)
        
        return ExtractionResult(
            entities=verified_entities,
            relations=verified_relations,
            backend=self.backend_name,
            raw_output=result.raw_output
        )
    
    def _is_invalid_entity(self, name: str) -> bool:
        """检查是否是无效实体"""
        # 时间词
        time_words = [
            "今天", "明天", "后天", "昨天", "前天",
            "上个月", "下个月", "这个月", "去年", "今年", "明年",
            "上周", "下周", "这周", "周末", "工作日",
            "早上", "中午", "下午", "晚上", "凌晨",
            "一点", "两点", "三点", "现在", "之后", "之前"
        ]
        if name in time_words:
            return True
        
        # 数量词
        if re.match(r'^\d+(\.\d+)?(个|件|条|次|天|小时|分钟|秒)?$', name):
            return True
        
        # 过于泛化的词
        generic_words = [
            "用户", "系统", "问题", "功能", "数据", "内容", "信息",
            "方法", "方式", "东西", "地方", "情况", "方面", "部分",
            "结果", "原因", "过程", "步骤", "要求", "需要"
        ]
        if name in generic_words:
            return True
        
        # 单个字符
        if len(name) < 2:
            return True
        
        # 纯数字
        if name.isdigit():
            return True
        
        return False
    
    def _infer_entity_type(self, name: str) -> str:
        """推断实体类型"""
        patterns = {
            "人物": [
                r'^[\u4e00-\u9fa5]{2,4}$',  # 2-4个中文字
                r'.*(先生|女士|总|经理|工程师|架构师)$'
            ],
            "组织": [
                r'.*(公司|集团|团队|部门|实验室|研究院)$',
                r'^(腾讯|阿里|字节|华为|百度|小米|美团|滴滴|京东)$'
            ],
            "项目": [
                r'.*(项目|系统|平台|应用|App|服务)$',
                r'^(微信|支付宝|抖音|淘宝|京东|美团)$'
            ],
            "技术": [
                r'^(Python|Java|JavaScript|Go|Rust|C\+\+|React|Vue|Angular|Spring)$',
                r'^(Docker|K8s|Kubernetes|MySQL|Redis|MongoDB|PostgreSQL)$',
                r'.*(框架|库|语言|数据库|中间件)$'
            ],
            "产品": [
                r'^(iPhone|iPad|Mac|Android|Windows)$',
                r'.*(手机|电脑|平板|设备)$'
            ],
            "地点": [
                r'.*[省市县村镇区]$',
                r'^(北京|上海|广州|深圳|杭州|成都|武汉|西安)$'
            ],
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, name, re.IGNORECASE):
                    return entity_type
        
        return "概念"
    
    def get_entity_co_occurrence(self, texts: List[str]) -> Dict[Tuple[str, str], int]:
        """获取实体共现统计"""
        co_occurrence = defaultdict(int)
        
        for text in texts:
            result = self.extract(text)
            entity_names = [e.name for e in result.entities]
            
            # 统计共现
            for i, e1 in enumerate(entity_names):
                for e2 in entity_names[i+1:]:
                    key = tuple(sorted([e1, e2]))
                    co_occurrence[key] += 1
        
        return dict(co_occurrence)
    
    def merge_results(self, results: List[ExtractionResult]) -> ExtractionResult:
        """合并多个抽取结果"""
        all_entities = {}
        all_relations = {}
        
        for result in results:
            # 合并实体（保留高置信度）
            for entity in result.entities:
                if entity.name not in all_entities or entity.confidence > all_entities[entity.name].confidence:
                    all_entities[entity.name] = entity
            
            # 合并关系
            for relation in result.relations:
                key = (relation.source, relation.type, relation.target)
                if key not in all_relations or relation.confidence > all_relations[key].confidence:
                    all_relations[key] = relation
        
        return ExtractionResult(
            entities=list(all_entities.values()),
            relations=list(all_relations.values()),
            backend=self.backend_name
        )


# ============================================================================
# 替换原来的 DecoderExtractor
# ============================================================================

# 保持向后兼容
DecoderExtractor = EnhancedDecoderExtractor