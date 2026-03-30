"""
实体关系抽取模块
支持多种后端：Decoder (LLM)、Encoder (BERT)、混合模式
"""

from .base import EntityRelationExtractor, ExtractionResult
from .decoder_extractor import DecoderExtractor
from .encoder_extractor import EncoderExtractor
from .hybrid_extractor import HybridExtractor

__all__ = [
    'EntityRelationExtractor',
    'ExtractionResult',
    'DecoderExtractor',
    'EncoderExtractor', 
    'HybridExtractor'
]
