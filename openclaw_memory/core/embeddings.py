"""
Ollama Embedding 服务
使用 nomic-embed-text 进行向量嵌入
"""

import requests
import json
import time
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL


class OllamaEmbedding:
    """Ollama 嵌入服务"""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or OLLAMA_EMBED_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.dimension = 768  # nomic-embed-text 维度

    def list_models(self) -> List[str]:
        """列出可用的 Ollama 模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"⚠️ 获取模型列表失败: {e}")
        return []

    def embed(self, text: str, max_retries: int = 3) -> List[float]:
        """生成文本嵌入向量，带重试机制"""
        import re
        
        # 清理文本：移除控制字符和特殊字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 截断过长的文本（nomic-embed-text 最大 2048 tokens，约 1500 字符）
        if len(text) > 1500:
            text = text[:1500]

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=120
                )
                if response.status_code != 200:
                    print(f"⚠️ Ollama 返回 {response.status_code}: {response.text[:200]}")
                    raise Exception(f"Ollama error: {response.status_code}")
                return response.json()["embedding"]
            except Exception as e:
                last_error = e
                print(f"⚠️ Embedding 尝试 {attempt+1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 递增等待时间
                    continue

        # 如果所有重试都失败，返回一个零向量作为后备
        print(f"⚠️ Embedding 最终失败，使用零向量: {last_error}")
        return [0.0] * self.dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        return [self.embed(text) for text in texts]


class OllamaChat:
    """Ollama 聊天服务"""
    
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or OLLAMA_CHAT_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
    
    def chat(self, prompt: str, system: str = None) -> str:
        """发送聊天请求"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    def summarize(self, text: str) -> str:
        """摘要文本"""
        system = """你是一个专业的对话摘要助手。请将用户提供的对话内容进行简洁的摘要，
保留关键信息、实体、时间和重要事件。摘要应该简洁明了，便于后续检索。"""
        
        prompt = f"""请对以下对话进行摘要，提取关键信息：

对话内容：
{text}

请输出简洁的摘要（100-200字）："""
        
        return self.chat(prompt, system)
    
    def extract_entities(self, text: str) -> dict:
        """提取实体和关系"""
        system = """你是一个实体关系提取专家。请从文本中提取实体和它们之间的关系。
输出格式为 JSON：
{
  "entities": [{"name": "实体名", "type": "类型"}],
  "relations": [{"source": "源实体", "relation": "关系类型", "target": "目标实体"}]
}"""
        
        prompt = f"""请从以下文本中提取实体和关系：

文本：
{text}

请以 JSON 格式输出："""
        
        response = self.chat(prompt, system)
        
        # 尝试解析 JSON
        try:
            # 尝试找到 JSON 部分
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        return {"entities": [], "relations": []}
    
    def parse_time_expression(self, expression: str) -> dict:
        """解析时间表达式"""
        from datetime import datetime, timedelta
        
        system = """你是一个时间表达式解析专家。请解析用户的时间表达式，返回具体的日期范围。
输出格式为 JSON：
{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "interpreted": "你的理解"
}"""
        
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""请解析以下时间表达式，今天是 {today}：

时间表达式："{expression}"

请以 JSON 格式输出日期范围："""
        
        response = self.chat(prompt, system)
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        # 默认返回最近7天
        return {
            "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end_date": today,
            "interpreted": "默认最近7天"
        }


if __name__ == "__main__":
    # 测试嵌入
    print("🧪 测试 Ollama Embedding...")
    embedder = OllamaEmbedding()
    
    test_text = "这是一个测试文本，用于验证嵌入功能"
    vector = embedder.embed(test_text)
    print(f"✅ 嵌入向量维度: {len(vector)}")
    
    # 测试聊天
    print("\n🧪 测试 Ollama Chat...")
    chat = OllamaChat()
    
    summary = chat.summarize("用户: 你好，我是张三\n助手: 你好张三，有什么可以帮助你的？\n用户: 我想了解一下抖音广告投放\n助手: 好的，抖音广告投放有很多种方式...")
    print(f"✅ 摘要: {summary}")
    
    entities = chat.extract_entities("张三正在和李四合作开发一个新项目，他们使用Python和Weaviate构建记忆系统")
    print(f"✅ 实体: {entities}")