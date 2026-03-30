"""
MiniLM 嵌入 + 简单重排序
使用 all-MiniLM-L6-v2 进行向量嵌入和检索重排序
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import os
import sys

# 在导入 sentence_transformers 之前设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 延迟导入，避免启动时加载模型
_model = None
_reranker = None

def get_model():
    """延迟加载嵌入模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print("🔄 加载 all-MiniLM-L6-v2 模型（离线模式）...")
        try:
            # 尝试离线加载
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ 模型加载完成（离线模式）")
        except Exception as e:
            # 离线失败，尝试在线下载
            print(f"⚠️ 离线加载失败: {e}")
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_HUB_OFFLINE', None)
            print("🔄 尝试在线下载...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ 模型加载完成（在线模式）")
    return _model


class MiniLMEmbedding:
    """all-MiniLM-L6-v2 嵌入服务
    
    特点：
    - 维度：384
    - 速度快，适合实时检索
    - 多语言支持良好
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.dimension = 384  # MiniLM-L6-v2 的向量维度
        self._model = None
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._model = get_model()
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 清理文本
        text = self._clean_text(text)
        
        # 截断过长文本（MiniLM 最大 256 tokens，约 500 字符）
        if len(text) > 500:
            text = text[:500]
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"⚠️ Embedding 失败: {e}")
            return [0.0] * self.dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        # 清理和截断
        cleaned_texts = []
        for text in texts:
            text = self._clean_text(text)
            if len(text) > 500:
                text = text[:500]
            cleaned_texts.append(text)
        
        try:
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"⚠️ Batch embedding 失败: {e}")
            return [[0.0] * self.dimension for _ in texts]
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class SimpleReranker:
    """简单重排序器
    
    策略：
    1. 相似度重排序：基于查询和结果的语义相似度
    2. 时间衰减：较新的记忆权重更高
    3. 重要性加权：高重要性的记忆权重更高
    """
    
    def __init__(self, embedding_model=None):
        # 不在初始化时加载模型，延迟到需要时加载
        self._embedding_model = embedding_model
        self._model_loaded = embedding_model is not None
    
    @property
    def embedding_model(self):
        """延迟加载嵌入模型"""
        if not self._model_loaded:
            self._embedding_model = MiniLMEmbedding()
            self._model_loaded = True
        return self._embedding_model
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict], 
        top_k: int = 10,
        time_decay_factor: float = 0.1,
        importance_weight: float = 0.3
    ) -> List[Dict]:
        """重排序检索结果
        
        Args:
            query: 查询文本
            results: 原始检索结果
            top_k: 返回前 k 个结果
            time_decay_factor: 时间衰减因子（0-1，越大衰减越快）
            importance_weight: 重要性权重（0-1）
        
        Returns:
            重排序后的结果列表
        """
        if not results:
            return []
        
        # 生成查询向量
        query_vector = np.array(self.embedding_model.embed(query))
        
        # 计算每个结果的综合得分
        scored_results = []
        now = np.datetime64('now')
        
        for result in results:
            # 1. 计算语义相似度
            content = result.get('content', '') or result.get('summary', '')
            if not content:
                continue
            
            content_vector = np.array(self.embedding_model.embed(content))
            semantic_score = self._cosine_similarity(query_vector, content_vector)
            
            # 2. 计算时间衰减
            time_score = self._time_score(result.get('timestamp'), now, time_decay_factor)
            
            # 3. 获取重要性
            importance = result.get('importance', 0.5)
            
            # 4. 综合得分
            # 公式：semantic * (1 - importance_weight) + importance * importance_weight + time_decay
            final_score = (
                semantic_score * (1 - importance_weight) +
                importance * importance_weight +
                time_score * 0.2  # 时间作为辅助因素
            )
            
            # 保存原始相似度用于调试
            result['_semantic_score'] = float(semantic_score)
            result['_time_score'] = float(time_score)
            result['_rerank_score'] = float(final_score)
            
            scored_results.append((final_score, result))
        
        # 按得分降序排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 返回 top_k 结果
        return [r for _, r in scored_results[:top_k]]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _time_score(self, timestamp, now, decay_factor: float) -> float:
        """计算时间衰减得分
        
        越新的记忆得分越高
        """
        if not timestamp:
            return 0.5  # 没有时间戳，给中等分数
        
        try:
            # 解析时间戳
            if isinstance(timestamp, str):
                # 尝试多种格式
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S.%f']:
                    try:
                        ts = np.datetime64(timestamp[:19] if 'T' in timestamp else timestamp)
                        break
                    except Exception:
                        continue
                else:
                    return 0.5
            else:
                ts = np.datetime64(timestamp)
            
            # 计算天数差
            days_diff = (now - ts).astype('timedelta64[D]').astype(float)
            
            # 指数衰减：score = exp(-decay_factor * days / 30)
            # 30天为一个周期
            score = np.exp(-decay_factor * days_diff / 30)
            return float(min(1.0, max(0.0, score)))
            
        except Exception as e:
            return 0.5


class HybridRetriever:
    """混合检索器
    
    结合向量检索 + 重排序
    """
    
    def __init__(
        self,
        weaviate_adapter,
        embedding_model: MiniLMEmbedding = None,
        reranker: SimpleReranker = None
    ):
        self.weaviate = weaviate_adapter
        self.embedding_model = embedding_model or MiniLMEmbedding()
        self.reranker = reranker or SimpleReranker(self.embedding_model)
    
    def search(
        self,
        query: str,
        agent_id: str,
        limit: int = 20,
        rerank_top_k: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """执行检索 + 重排序
        
        Args:
            query: 查询文本
            agent_id: Agent ID
            limit: 向量检索返回数量（召回数量）
            rerank_top_k: 重排序后返回数量
            filters: 额外过滤条件
        
        Returns:
            重排序后的结果列表
        """
        # 1. 生成查询向量
        query_vector = self.embedding_model.embed(query)
        
        # 2. 向量检索（召回更多候选）
        candidates = self.weaviate.search_memories(
            vector=query_vector,
            limit=limit
        )
        
        if not candidates:
            return []
        
        # 3. 重排序
        reranked = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=rerank_top_k
        )
        
        return reranked
    
    def search_with_time_filter(
        self,
        query: str,
        agent_id: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = 20,
        rerank_top_k: int = 10
    ) -> List[Dict]:
        """带时间过滤的检索
        
        Args:
            query: 查询文本
            agent_id: Agent ID
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 召回数量
            rerank_top_k: 返回数量
        """
        # 生成查询向量
        query_vector = self.embedding_model.embed(query)
        
        # 时间范围检索
        candidates = self.weaviate.get_memories_by_time(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if not candidates:
            return []
        
        # 重排序
        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=rerank_top_k
        )


# 便捷函数
def create_embedder() -> MiniLMEmbedding:
    """创建嵌入器实例"""
    return MiniLMEmbedding()


def create_reranker(embedding_model: MiniLMEmbedding = None) -> SimpleReranker:
    """创建重排序器实例"""
    return SimpleReranker(embedding_model)


# 测试
if __name__ == "__main__":
    print("🧪 测试 MiniLM Embedding + Reranker...")
    
    # 测试嵌入
    embedder = MiniLMEmbedding()
    
    test_texts = [
        "抖音广告投放策略",
        "如何提高视频播放量",
        "用户增长数据分析"
    ]
    
    print("\n📝 测试嵌入...")
    for text in test_texts:
        vector = embedder.embed(text)
        print(f"  '{text[:20]}...' -> 向量维度: {len(vector)}")
    
    # 测试重排序
    print("\n📊 测试重排序...")
    reranker = SimpleReranker(embedder)
    
    query = "抖音广告"
    results = [
        {"content": "抖音广告投放需要关注ROI", "importance": 0.8, "timestamp": "2026-03-16T10:00:00"},
        {"content": "视频内容创作技巧", "importance": 0.6, "timestamp": "2026-03-15T10:00:00"},
        {"content": "抖音直播带货数据分析", "importance": 0.7, "timestamp": "2026-03-14T10:00:00"},
    ]
    
    reranked = reranker.rerank(query, results, top_k=3)
    
    print(f"\n查询: '{query}'")
    print("重排序结果:")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. {r['content'][:30]}... (得分: {r['_rerank_score']:.3f})")
    
    print("\n✅ 测试完成")