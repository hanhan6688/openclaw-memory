"""
实体关系抽取抽象基类
定义统一接口，支持多种后端实现
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class ExtractorBackend(Enum):
    """抽取器后端类型"""
    DECODER = "decoder"      # LLM (Llama, GPT, etc.)
    ENCODER = "encoder"      # BERT, RoBERTa, DeBERTa
    HYBRID = "hybrid"        # 混合模式


@dataclass
class Entity:
    """实体"""
    name: str                          # 实体名称
    type: str                          # 实体类型
    start: int = -1                    # 起始位置
    end: int = -1                      # 结束位置
    confidence: float = 1.0            # 置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class Relation:
    """关系"""
    source: str                        # 源实体
    target: str                        # 目标实体
    type: str                          # 关系类型
    confidence: float = 1.0            # 置信度
    evidence: str = ""                 # 证据文本
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata
        }


@dataclass
class ExtractionResult:
    """抽取结果"""
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    latency_ms: float = 0.0            # 耗时（毫秒）
    backend: str = "unknown"           # 使用的后端
    raw_output: Any = None             # 原始输出（调试用）
    
    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "latency_ms": self.latency_ms,
            "backend": self.backend
        }
    
    @property
    def entity_names(self) -> List[str]:
        """获取所有实体名称"""
        return [e.name for e in self.entities]
    
    @property
    def relation_triples(self) -> List[tuple]:
        """获取所有关系三元组"""
        return [(r.source, r.type, r.target) for r in self.relations]


class EntityRelationExtractor(ABC):
    """实体关系抽取器抽象基类"""
    
    # 默认实体类型
    DEFAULT_ENTITY_TYPES = [
        "人物", "组织", "项目", "技术", "概念", "地点", "事件", "产品"
    ]
    
    # 默认关系类型
    DEFAULT_RELATION_TYPES = [
        "合作", "竞争", "认识", "雇佣", "管理", "任职于", "拥有资产", "亲属",
        "使用", "依赖", "包含", "实现", "调用", "开发", "是一种", "相关",
        "属于", "位于", "收购", "投资"
    ]
    
    def __init__(self, 
                 entity_types: List[str] = None,
                 relation_types: List[str] = None):
        """
        初始化抽取器
        
        Args:
            entity_types: 实体类型列表
            relation_types: 关系类型列表
        """
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self.relation_types = relation_types or self.DEFAULT_RELATION_TYPES
    
    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """
        抽取实体和关系
        
        Args:
            text: 输入文本
            
        Returns:
            ExtractionResult: 抽取结果
        """
        pass
    
    @abstractmethod
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """
        批量抽取
        
        Args:
            texts: 文本列表
            
        Returns:
            抽取结果列表
        """
        pass
    
    def extract_entities(self, text: str) -> List[Entity]:
        """只抽取实体"""
        result = self.extract(text)
        return result.entities
    
    def extract_relations(self, text: str) -> List[Relation]:
        """只抽取关系"""
        result = self.extract(text)
        return result.relations
    
    @property
    @abstractmethod
    def backend(self) -> ExtractorBackend:
        """返回后端类型"""
        pass
    
    @property
    def backend_name(self) -> str:
        """返回后端名称"""
        return self.backend.value


class ExtractorFactory:
    """抽取器工厂"""
    
    _instances = {}
    
    @classmethod
    def get_extractor(cls, 
                      backend: str = "decoder",
                      entity_types: List[str] = None,
                      relation_types: List[str] = None,
                      **kwargs) -> EntityRelationExtractor:
        """
        获取抽取器实例
        
        Args:
            backend: 后端类型 "decoder", "encoder", "hybrid"
            entity_types: 实体类型
            relation_types: 关系类型
            **kwargs: 额外参数
            
        Returns:
            抽取器实例
        """
        from .decoder_extractor import DecoderExtractor
        from .encoder_extractor import EncoderExtractor
        from .hybrid_extractor import HybridExtractor
        
        key = f"{backend}_{id(entity_types)}_{id(relation_types)}"
        
        if key not in cls._instances:
            if backend == "decoder":
                cls._instances[key] = DecoderExtractor(
                    entity_types=entity_types,
                    relation_types=relation_types,
                    **kwargs
                )
            elif backend == "encoder":
                cls._instances[key] = EncoderExtractor(
                    entity_types=entity_types,
                    relation_types=relation_types,
                    **kwargs
                )
            elif backend == "hybrid":
                cls._instances[key] = HybridExtractor(
                    entity_types=entity_types,
                    relation_types=relation_types,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
        
        return cls._instances[key]
