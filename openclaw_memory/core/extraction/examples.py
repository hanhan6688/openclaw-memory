"""
实体关系抽取使用示例
"""

from typing import List
from .base import ExtractorFactory, ExtractionResult


def example_basic_usage():
    """基本使用示例"""
    
    # 1. 使用默认后端（Decoder）
    extractor = ExtractorFactory.get_extractor()
    
    text = "张三在阿里巴巴工作，阿里巴巴位于杭州"
    result = extractor.extract(text)
    
    print("=== 实体 ===")
    for entity in result.entities:
        print(f"  {entity.name} ({entity.type})")
    
    print("=== 关系 ===")
    for relation in result.relations:
        print(f"  {relation.source} --[{relation.type}]--> {relation.target}")
    
    print(f"耗时: {result.latency_ms:.2f}ms")


def example_custom_entity_types():
    """自定义实体类型示例"""
    
    # 电商场景的实体类型
    entity_types = ["产品", "供应商", "客户", "订单", "物流", "平台"]
    relation_types = ["供应", "采购", "销售", "发货", "存储"]
    
    extractor = ExtractorFactory.get_extractor(
        backend="decoder",
        entity_types=entity_types,
        relation_types=relation_types
    )
    
    text = "产品A由供应商B供应，在淘宝平台销售给客户C"
    result = extractor.extract(text)
    
    print(result.to_dict())


def example_batch_extraction():
    """批量抽取示例"""
    
    extractor = ExtractorFactory.get_extractor()
    
    texts = [
        "张三在阿里巴巴工作",
        "李四开发了微信小程序",
        "王五投资了字节跳动"
    ]
    
    results = extractor.extract_batch(texts)
    
    for text, result in zip(texts, results):
        print(f"文本: {text}")
        print(f"实体: {result.entity_names}")
        print(f"关系: {result.relation_triples}")
        print()


def example_switch_backend():
    """切换后端示例"""
    
    # 当前使用 Decoder
    decoder = ExtractorFactory.get_extractor(backend="decoder")
    
    # 未来切换到 Encoder（需要训练后）
    # encoder = ExtractorFactory.get_extractor(backend="encoder")
    
    # 或者使用混合模式
    # hybrid = ExtractorFactory.get_extractor(backend="hybrid")
    
    print("当前后端: decoder")
    print("切换方式: 修改 EXTRACTOR_BACKEND 环境变量或配置")


def example_enable_encoder():
    """启用 Encoder 示例（未来使用）"""
    
    from .hybrid_extractor import HybridExtractor, HybridConfig
    
    config = HybridConfig(
        encoder_enabled=True,
        encoder_ner_model="/path/to/ner/model",
        encoder_re_model="/path/to/re/model"
    )
    
    extractor = HybridExtractor(config=config)
    
    # 或者动态启用
    # extractor.enable_encoder(
    #     ner_model_path="/path/to/ner/model",
    #     re_model_path="/path/to/re/model"
    # )


if __name__ == "__main__":
    example_basic_usage()
