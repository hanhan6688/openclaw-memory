"""
AI 处理器 - 统一的 AI 服务接口
"""

import os
import sys
from typing import Dict, List, Optional

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embeddings import OllamaEmbedding
from summarizer import Summarizer


class AIProcessor:
    """AI 处理器"""
    
    # 模型配置（使用 config 中的配置）
    SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"))
    NER_MODEL = os.getenv("NER_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"))
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    
    def __init__(self):
        self.embedder = OllamaEmbedding()
        self.summarizer = None  # 延迟初始化
    
    def check_health(self) -> Dict:
        """检查 AI 服务健康状态"""
        try:
            # 检查 Ollama
            models = self.embedder.list_models()
            return {
                "available": len(models) > 0,
                "models": models,
                "error": None
            }
        except Exception as e:
            return {
                "available": False,
                "models": [],
                "error": str(e)
            }
    
    def embed(self, text: str) -> List[float]:
        """生成嵌入向量"""
        return self.embedder.embed(text)
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """生成摘要"""
        if self.summarizer is None:
            self.summarizer = Summarizer()
        return self.summarizer.summarize(text, max_length)
    
    def extract_entities(self, text: str) -> Dict:
        """提取实体"""
        # 简单实现，可以后续扩展
        from embeddings import OllamaChat
        chat = OllamaChat()
        return chat.extract_entities(text)


# 全局实例
ai_processor = AIProcessor()