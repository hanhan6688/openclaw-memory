"""
Encoder 抽取器
使用 BERT/RoBERTa/DeBERTa 进行实体关系抽取

特点：
- 推理速度快（毫秒级）
- 准确率高
- 需要训练数据

使用方式：
1. 准备标注数据
2. 训练模型（或使用预训练模型）
3. 启用 encoder 后端
"""

import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import (
    EntityRelationExtractor, 
    ExtractionResult, 
    Entity, 
    Relation,
    ExtractorBackend
)

# 检查是否有 transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers 未安装，Encoder 抽取器不可用")
    print("   安装命令: pip install transformers torch")


@dataclass
class EncoderConfig:
    """Encoder 配置"""
    model_name: str = "hfl/chinese-roberta-wwm-ext"  # 默认中文模型
    max_length: int = 512
    batch_size: int = 8
    device: str = "cpu"  # "cpu" or "cuda"
    
    # NER 模型路径（训练后）
    ner_model_path: Optional[str] = None
    
    # 关系抽取模型路径
    re_model_path: Optional[str] = None


class EncoderExtractor(EntityRelationExtractor):
    """
    基于 Encoder (BERT) 的实体关系抽取器
    
    优势：
    - 推理速度快（毫秒级）
    - 批量处理效率高
    - 准确率高（有训练数据时）
    
    劣势：
    - 需要标注数据训练
    - 泛化能力不如大模型
    """
    
    def __init__(self,
                 entity_types: List[str] = None,
                 relation_types: List[str] = None,
                 config: EncoderConfig = None):
        """
        初始化
        
        Args:
            entity_types: 实体类型列表
            relation_types: 关系类型列表
            config: Encoder 配置
        """
        super().__init__(entity_types, relation_types)
        self.config = config or EncoderConfig()
        
        self._ner_model = None
        self._ner_tokenizer = None
        self._re_model = None
        self._initialized = False
    
    @property
    def backend(self) -> ExtractorBackend:
        return ExtractorBackend.ENCODER
    
    def _lazy_init(self):
        """延迟初始化模型"""
        if self._initialized:
            return
        
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers 未安装，无法使用 Encoder 抽取器")
        
        # 加载 NER 模型
        if self.config.ner_model_path:
            self._ner_tokenizer = AutoTokenizer.from_pretrained(self.config.ner_model_path)
            self._ner_model = AutoModelForTokenClassification.from_pretrained(
                self.config.ner_model_path
            )
            self._ner_model.to(self.config.device)
            self._ner_model.eval()
        else:
            print("⚠️ NER 模型未训练，使用规则匹配")
        
        self._initialized = True
    
    def extract(self, text: str) -> ExtractionResult:
        """抽取实体和关系"""
        start_time = time.time()
        
        self._lazy_init()
        
        # 如果模型已加载，使用模型抽取
        if self._ner_model is not None:
            entities = self._extract_with_model(text)
        else:
            # 降级：使用规则匹配
            entities = self._extract_with_rules(text)
        
        # 关系抽取
        relations = self._extract_relations(text, entities)
        
        latency = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            latency_ms=latency,
            backend=self.backend_name
        )
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """批量抽取（Encoder 的优势）"""
        results = []
        
        self._lazy_init()
        
        # 批量处理
        if self._ner_model is not None and HAS_TRANSFORMERS:
            # 使用模型批量处理
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_results = self._extract_batch_with_model(batch)
                results.extend(batch_results)
        else:
            # 降级：逐个处理
            for text in texts:
                results.append(self.extract(text))
        
        return results
    
    def _extract_with_model(self, text: str) -> List[Entity]:
        """使用模型抽取实体"""
        if not HAS_TRANSFORMERS or self._ner_model is None:
            return []
        
        import torch
        
        # Tokenize
        inputs = self._ner_tokenizer(
            text, 
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self._ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode
        entities = self._decode_ner_predictions(text, inputs, predictions)
        
        return entities
    
    def _extract_batch_with_model(self, texts: List[str]) -> List[ExtractionResult]:
        """批量抽取"""
        results = []
        
        for text in texts:
            result = self.extract(text)
            results.append(result)
        
        return results
    
    def _decode_ner_predictions(self, text: str, inputs, predictions) -> List[Entity]:
        """解码 NER 预测结果"""
        # TODO: 根据 BIO 标签解码
        # 这里需要根据实际的标签体系实现
        return []
    
    def _extract_with_rules(self, text: str) -> List[Entity]:
        """规则匹配（降级方案）"""
        entities = []
        
        # 简单的模式匹配
        patterns = {
            "组织": [r'([\u4e00-\u9fa5]+公司)', r'([\u4e00-\u9fa5]+集团)'],
            "项目": [r'([\u4e00-\u9fa5]+项目)', r'([\u4e00-\u9fa5]+系统)'],
            "技术": [r'(Python|Java|React|Vue|Docker|Kubernetes)'],
            "地点": [r'([\u4e00-\u9fa5]+[省市县])'],
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text)
                for match in matches:
                    name = match.group(1)
                    if name and len(name) >= 2:
                        entities.append(Entity(
                            name=name,
                            type=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7
                        ))
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """抽取关系"""
        # TODO: 实现关系抽取
        # 可以使用:
        # 1. 规则模板
        # 2. 分类模型
        # 3. 联合抽取模型
        return []
    
    def train_ner(self, train_data: List[Dict], output_dir: str):
        """
        训练 NER 模型
        
        Args:
            train_data: 训练数据
                [{"text": "...", "entities": [{"start": 0, "end": 2, "type": "人物"}]}]
            output_dir: 输出目录
        """
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers 未安装")
        
        # TODO: 实现训练逻辑
        # 1. 数据预处理
        # 2. 标签映射
        # 3. 训练循环
        # 4. 保存模型
        print(f"训练 NER 模型，数据量: {len(train_data)}")
        print(f"输出目录: {output_dir}")
    
    def train_re(self, train_data: List[Dict], output_dir: str):
        """
        训练关系抽取模型
        
        Args:
            train_data: 训练数据
                [{"text": "...", "relations": [{"source": "A", "target": "B", "type": "合作"}]}]
            output_dir: 输出目录
        """
        # TODO: 实现训练逻辑
        print(f"训练 RE 模型，数据量: {len(train_data)}")
        print(f"输出目录: {output_dir}")


# 预训练模型推荐
RECOMMENDED_MODELS = {
    "中文NER": [
        "hfl/chinese-roberta-wwm-ext-large",
        "hfl/chinese-roberta-wwm-ext",
        "bert-base-chinese",
        "uer/roberta-base-chinese-extractive-qa",
    ],
    "中文关系抽取": [
        "hfl/chinese-roberta-wwm-ext",
        "bert-base-chinese",
    ],
    "英文NER": [
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dslim/bert-base-NER",
    ],
}
