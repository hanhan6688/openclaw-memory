# 实体关系抽取模块

支持多种后端的实体关系抽取，可灵活切换。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    EntityRelationExtractor                   │
│                    (抽象基类)                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ DecoderExtractor│  │ EncoderExtractor│  │HybridExtract│ │
│  │   (LLM)         │  │   (BERT)        │  │   (混合)    │ │
│  │                 │  │                 │  │             │ │
│  │ • 零样本能力强  │  │ • 推理速度快    │  │ • 智能路由  │ │
│  │ • 灵活可调整    │  │ • 准确率高      │  │ • 最佳效果  │ │
│  │ • 秒级延迟      │  │ • 毫秒级延迟    │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 使用方式

### 1. 基本使用

```python
from openclaw_memory.core.extraction import ExtractorFactory

# 获取抽取器（默认使用 Decoder）
extractor = ExtractorFactory.get_extractor()

# 抽取实体和关系
text = "张三在阿里巴巴工作，阿里巴巴位于杭州"
result = extractor.extract(text)

# 查看结果
print(result.entities)    # 实体列表
print(result.relations)   # 关系列表
```

### 2. 自定义实体类型

```python
# 电商场景
entity_types = ["产品", "供应商", "客户", "订单", "物流", "平台"]
relation_types = ["供应", "采购", "销售", "发货", "存储"]

extractor = ExtractorFactory.get_extractor(
    backend="decoder",
    entity_types=entity_types,
    relation_types=relation_types
)
```

### 3. 切换后端

```python
# 方式1: 环境变量
# export EXTRACTOR_BACKEND=encoder

# 方式2: 代码指定
extractor = ExtractorFactory.get_extractor(backend="encoder")
```

### 4. 启用 Encoder（需要训练后）

```python
from openclaw_memory.core.extraction import HybridExtractor, HybridConfig

config = HybridConfig(
    encoder_enabled=True,
    encoder_ner_model="/path/to/ner/model"
)

extractor = HybridExtractor(config=config)
```

## 后端对比

| 特性 | Decoder | Encoder | Hybrid |
|------|---------|---------|--------|
| 零样本能力 | ✅ 强 | ❌ 需训练 | ✅ |
| 推理速度 | 秒级 | 毫秒级 | 智能路由 |
| 准确率 | 70-80% | 90%+ (有训练数据) | 最佳 |
| 灵活性 | ✅ 高 | ❌ 低 | ✅ |
| 部署复杂度 | 低 | 中 | 中 |

## 切换时机

| 场景 | 推荐后端 |
|------|---------|
| 验证期，实体类型经常变 | Decoder |
| 数据量 < 100/天 | Decoder |
| 数据量 > 1000/天 | Encoder |
| 需要毫秒级响应 | Encoder |
| 追求最佳效果 | Hybrid |

## 训练 Encoder

```python
from openclaw_memory.core.extraction import EncoderExtractor

extractor = EncoderExtractor()

# 准备训练数据
train_data = [
    {
        "text": "张三在阿里巴巴工作",
        "entities": [
            {"start": 0, "end": 2, "type": "人物"},
            {"start": 3, "end": 7, "type": "组织"}
        ]
    },
    # ...
]

# 训练
extractor.train_ner(train_data, output_dir="/path/to/model")
```

## 推荐模型

### 中文 NER
- `hfl/chinese-roberta-wwm-ext-large`
- `hfl/chinese-roberta-wwm-ext`
- `bert-base-chinese`

### 英文 NER
- `dbmdz/bert-large-cased-finetuned-conll03-english`
- `dslim/bert-base-NER`
