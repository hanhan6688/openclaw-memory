"""
混合抽取器
结合 Encoder（快速）和 Decoder（准确）的优势

策略：
- 简单/高频场景 → Encoder（毫秒级）
- 复杂/低频场景 → Decoder（秒级）
- 不确定时 → 两者都用，取置信度高的结果
"""

import time
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

from .base import (
    EntityRelationExtractor, 
    ExtractionResult, 
    Entity, 
    Relation,
    ExtractorBackend
)
from .decoder_extractor import DecoderExtractor
from .encoder_extractor import EncoderExtractor


@dataclass
class HybridConfig:
    """混合抽取器配置"""
    # 默认后端
    default_backend: str = "decoder"
    
    # 置信度阈值
    confidence_threshold: float = 0.8
    
    # 文本长度阈值（短文本用 Encoder）
    short_text_threshold: int = 100
    
    # 是否启用 Encoder（需要训练后才能启用）
    encoder_enabled: bool = False
    
    # Encoder 模型路径
    encoder_ner_model: str = None
    encoder_re_model: str = None
    
    # 自定义路由函数
    router: Callable[[str], str] = None


class HybridExtractor(EntityRelationExtractor):
    """
    混合抽取器
    
    智能路由：
    1. 根据文本特征选择后端
    2. 根据置信度决定是否需要第二个后端验证
    3. 合并结果
    """
    
    def __init__(self,
                 entity_types: List[str] = None,
                 relation_types: List[str] = None,
                 config: HybridConfig = None):
        super().__init__(entity_types, relation_types)
        self.config = config or HybridConfig()
        
        # 初始化两个后端
        self._decoder = DecoderExtractor(
            entity_types=entity_types,
            relation_types=relation_types
        )
        
        self._encoder = None  # 延迟初始化
        self._initialized = False
    
    @property
    def backend(self) -> ExtractorBackend:
        return ExtractorBackend.HYBRID
    
    def _lazy_init(self):
        """延迟初始化 Encoder"""
        if self._initialized:
            return
        
        if self.config.encoder_enabled:
            try:
                from .encoder_extractor import EncoderConfig
                encoder_config = EncoderConfig(
                    ner_model_path=self.config.encoder_ner_model,
                    re_model_path=self.config.encoder_re_model
                )
                self._encoder = EncoderExtractor(
                    entity_types=self.entity_types,
                    relation_types=self.relation_types,
                    config=encoder_config
                )
            except Exception as e:
                print(f"⚠️ Encoder 初始化失败: {e}")
                self._encoder = None
        
        self._initialized = True
    
    def extract(self, text: str) -> ExtractionResult:
        """抽取实体和关系"""
        start_time = time.time()
        
        self._lazy_init()
        
        # 路由决策
        backend = self._route(text)
        
        if backend == "encoder" and self._encoder is not None:
            result = self._encoder.extract(text)
            result.backend = "encoder"
            
            # 如果置信度低，用 Decoder 验证
            if self._needs_verification(result):
                decoder_result = self._decoder.extract(text)
                result = self._merge_results(result, decoder_result)
        else:
            result = self._decoder.extract(text)
            result.backend = "decoder"
        
        result.latency_ms = (time.time() - start_time) * 1000
        return result
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """批量抽取"""
        self._lazy_init()
        
        # 分组：Encoder 处理 vs Decoder 处理
        encoder_texts = []
        decoder_texts = []
        
        for text in texts:
            backend = self._route(text)
            if backend == "encoder" and self._encoder is not None:
                encoder_texts.append(text)
            else:
                decoder_texts.append(text)
        
        results = []
        
        # Encoder 批量处理
        if encoder_texts and self._encoder:
            encoder_results = self._encoder.extract_batch(encoder_texts)
            for r in encoder_results:
                r.backend = "encoder"
            results.extend(encoder_results)
        
        # Decoder 处理
        if decoder_texts:
            decoder_results = self._decoder.extract_batch(decoder_texts)
            for r in decoder_results:
                r.backend = "decoder"
            results.extend(decoder_results)
        
        return results
    
    def _route(self, text: str) -> str:
        """
        路由决策
        
        Returns:
            "encoder" or "decoder"
        """
        # 自定义路由
        if self.config.router:
            return self.config.router(text)
        
        # Encoder 未启用
        if not self.config.encoder_enabled or self._encoder is None:
            return "decoder"
        
        # 规则路由
        text_len = len(text)
        
        # 短文本 → Encoder
        if text_len < self.config.short_text_threshold:
            return "encoder"
        
        # 包含复杂句式 → Decoder
        complex_patterns = [
            "虽然", "但是", "如果", "那么", "因为", "所以",
            "不仅", "而且", "要么", "要么", "一方面", "另一方面"
        ]
        for pattern in complex_patterns:
            if pattern in text:
                return "decoder"
        
        # 默认
        return self.config.default_backend
    
    def _needs_verification(self, result: ExtractionResult) -> bool:
        """判断是否需要 Decoder 验证"""
        # 实体置信度低
        for entity in result.entities:
            if entity.confidence < self.config.confidence_threshold:
                return True
        
        # 关系置信度低
        for relation in result.relations:
            if relation.confidence < self.config.confidence_threshold:
                return True
        
        # 实体数量异常
        if len(result.entities) == 0:
            return True
        
        return False
    
    def _merge_results(self, encoder_result: ExtractionResult, 
                       decoder_result: ExtractionResult) -> ExtractionResult:
        """合并两个结果"""
        # 实体合并（去重）
        entity_names = set()
        entities = []
        
        for e in encoder_result.entities:
            if e.name not in entity_names:
                entity_names.add(e.name)
                entities.append(e)
        
        for e in decoder_result.entities:
            if e.name not in entity_names:
                entity_names.add(e.name)
                entities.append(e)
            else:
                # 更新置信度
                for existing in entities:
                    if existing.name == e.name and e.confidence > existing.confidence:
                        existing.confidence = e.confidence
                        existing.type = e.type
        
        # 关系合并
        relation_keys = set()
        relations = []
        
        for r in encoder_result.relations:
            key = (r.source, r.type, r.target)
            if key not in relation_keys:
                relation_keys.add(key)
                relations.append(r)
        
        for r in decoder_result.relations:
            key = (r.source, r.type, r.target)
            if key not in relation_keys:
                relation_keys.add(key)
                relations.append(r)
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            backend="hybrid",
            latency_ms=encoder_result.latency_ms + decoder_result.latency_ms
        )
    
    def enable_encoder(self, ner_model_path: str = None, re_model_path: str = None):
        """启用 Encoder 后端"""
        self.config.encoder_enabled = True
        self.config.encoder_ner_model = ner_model_path
        self.config.encoder_re_model = re_model_path
        self._initialized = False
        self._lazy_init()
    
    def disable_encoder(self):
        """禁用 Encoder 后端"""
        self.config.encoder_enabled = False
        self._encoder = None
