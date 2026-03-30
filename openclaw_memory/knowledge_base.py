#!/usr/bin/env python3
"""
知识库模块 - 文档上传、嵌入、检索
===============================

支持文档格式: PDF, TXT, MD, DOCX
"""

import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import weaviate
from weaviate.exceptions import WeaviateBaseError
import requests

# 文档解析库
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

try:
    import markdown
except ImportError:
    markdown = None


class DocumentChunker:
    """文档分块器"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分成多个块"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        text = text.strip()
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在句子边界分割
            if end < len(text):
                # 寻找最后一个句号、逗号或换行
                for sep in ['。', '！', '？', '，', '、', '\n', '. ', '? ', '! ']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start < 0:
                start = 0
        
        return chunks


class DocumentParser:
    """文档解析器"""
    
    @staticmethod
    def parse_file(file_path: str) -> str:
        """根据文件类型解析文本"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return DocumentParser._parse_pdf(file_path)
        elif ext in ['.txt', '.text']:
            return DocumentParser._parse_txt(file_path)
        elif ext in ['.md', '.markdown']:
            return DocumentParser._parse_md(file_path)
        elif ext in ['.docx', '.doc']:
            return DocumentParser._parse_docx(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")
    
    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """解析 PDF"""
        if not PyPDF2:
            raise ImportError("PyPDF2 未安装")
        
        text_parts = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    
    @staticmethod
    def _parse_txt(file_path: str) -> str:
        """解析 TXT"""
        # 尝试多种编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'big5']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # 最后尝试二进制读取
        with open(file_path, 'rb') as f:
            return f.decode('utf-8', errors='ignore')
    
    @staticmethod
    def _parse_md(file_path: str) -> str:
        """解析 Markdown"""
        if markdown:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            # 转换为纯文本（去掉 Markdown 语法）
            html = markdown.markdown(md_content)
            # 简单去除 HTML 标签
            import re
            text = re.sub(r'<[^>]+>', '', html)
            return text
        else:
            # 没有 markdown 库，直接读文本
            return DocumentParser._parse_txt(file_path)
    
    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """解析 DOCX"""
        if not docx:
            raise ImportError("python-docx 未安装")
        
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return '\n\n'.join(paragraphs)


class KnowledgeBase:
    """知识库管理器"""
    
    COLLECTION_NAME = "KnowledgeBase"
    
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text:latest",
        agent_id: str = None  # 支持 Agent 隔离
    ):
        self.weaviate_url = weaviate_url
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.agent_id = agent_id  # 当前 agent
        self.chunker = DocumentChunker()
        
        # 初始化 Weaviate 客户端 (v4 API)
        # 从 URL 中提取 host 和 port
        host = weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080
        
        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            headers={"X-OpenAI-Api-Key": "dummy"}
        )
        
        # 确保 Collection 存在
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保 KnowledgeBase Collection 存在"""
        try:
            from weaviate.classes.config import Property, DataType, Configure
            
            if not self.client.collections.exists(self.COLLECTION_NAME):
                self.client.collections.create(
                    name=self.COLLECTION_NAME,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="file_type", data_type=DataType.TEXT),
                        Property(name="document_id", data_type=DataType.TEXT),
                        Property(name="timestamp", data_type=DataType.TEXT),
                        Property(name="agent_id", data_type=DataType.TEXT),  # Agent 隔离
                    ],
                    vectorizer_config=Configure.Vectorizer.none()
                )
                print(f"[KnowledgeBase] 创建 Collection: {self.COLLECTION_NAME}")
            else:
                print(f"[KnowledgeBase] Collection 已存在: {self.COLLECTION_NAME}")
        except Exception as e:
            print(f"[KnowledgeBase] Collection 检查失败: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embed_model,
                    "prompt": text
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                print(f"[KnowledgeBase] 嵌入失败: {response.text}")
                return None
        except Exception as e:
            print(f"[KnowledgeBase] 嵌入请求失败: {e}")
            return None
    
    def _generate_doc_id(self, filename: str) -> str:
        """生成文档 ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"kb_{timestamp}_{name_hash}"
    
    def add_document(
        self,
        file_path: str,
        title: Optional[str] = None,
        agent_id: str = None
    ) -> Dict[str, Any]:
        """添加文档到知识库
        
        Args:
            file_path: 文件路径
            title: 文档标题
            agent_id: Agent ID (可选，默认使用 self.agent_id)
        """
        # 使用传入的 agent_id 或默认的
        effective_agent_id = agent_id or self.agent_id or "default"
        
        # 解析文档
        try:
            content = DocumentParser.parse_file(file_path)
        except Exception as e:
            return {"success": False, "error": f"文档解析失败: {e}"}
        
        if not content.strip():
            return {"success": False, "error": "文档内容为空"}
        
        # 分块
        chunks = self.chunker.chunk_text(content)
        if not chunks:
            return {"success": False, "error": "分块失败"}
        
        # 生成文档 ID
        filename = Path(file_path).name
        document_id = self._generate_doc_id(filename)
        file_type = Path(file_path).suffix.lower().lstrip('.')
        
        if not title:
            title = Path(file_path).stem
        
        # 存储到 Weaviate
        collection = self.client.collections.get(self.COLLECTION_NAME)
        success_count = 0
        
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                # 获取嵌入
                embedding = self._get_embedding(chunk)
                if not embedding:
                    continue
                
                # 构建对象
                obj = {
                    "title": title,
                    "content": chunk,
                    "chunk_index": i,
                    "source": filename,
                    "file_type": file_type,
                    "document_id": document_id,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": effective_agent_id  # Agent 隔离
                }
                
                try:
                    batch.add_object(
                        properties=obj,
                        vector=embedding
                    )
                    success_count += 1
                except Exception as e:
                    print(f"[KnowledgeBase] 添加 chunk 失败: {e}")
        
        # 清理临时文件
        try:
            os.remove(file_path)
        except Exception:
            pass
        
        return {
            "success": True,
            "document_id": document_id,
            "title": title,
            "chunks": len(chunks),
            "stored": success_count
        }
    
    def search(
        self,
        query: str,
        limit: int = 5,
        mode: str = "hybrid",  # "vector" | "bm25" | "hybrid"
        agent_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        搜索知识库 - 支持多路召回
        
        Args:
            query: 查询内容
            limit: 返回数量
            mode: 搜索模式
                - "vector": 向量相似度
                - "bm25": BM25 关键词搜索  
                - "hybrid": 混合搜索 (向量 + BM25)
            agent_id: Agent ID (可选，默认使用 self.agent_id)
        """
        from weaviate.classes.query import Filter
        
        # 使用传入的 agent_id 或默认的
        effective_agent_id = agent_id or self.agent_id
        
        try:
            collection = self.client.collections.get(self.COLLECTION_NAME)
            
            # 构建过滤条件
            filters = None
            if effective_agent_id:
                filters = Filter.by_property("agent_id").equal(effective_agent_id)
            
            # 1. 向量搜索
            vector_results = {}
            embedding = self._get_embedding(query)
            if embedding and mode in ["vector", "hybrid"]:
                try:
                    vec_resp = collection.query.near_vector(
                        near_vector=embedding,
                        limit=limit * 2,  # 多取一些用于融合
                        filters=filters
                    )
                    if vec_resp.objects:
                        for obj in vec_resp.objects:
                            props = obj.properties
                            doc_id = props.get("document_id", "unknown")
                            # 归一化分数 (Weaviate 返回 distance，越小越相似)
                            score = 1.0 / (1.0 + getattr(obj, 'distance', 0))
                            if doc_id not in vector_results:
                                vector_results[doc_id] = {
                                    "document_id": doc_id,
                                    "title": props.get("title"),
                                    "source": props.get("source"),
                                    "chunks": [],
                                    "vector_score": 0
                                }
                            vector_results[doc_id]["chunks"].append({
                                "content": props.get("content"),
                                "chunk_index": props.get("chunk_index", 0),
                            })
                            vector_results[doc_id]["vector_score"] = max(
                                vector_results[doc_id]["vector_score"], score
                            )
                except Exception as e:
                    print(f"[KnowledgeBase] 向量搜索失败: {e}")
            
            # 2. BM25 搜索 (简化版 - Weaviate v4 不返回原始分数)
            bm25_results = {}
            if mode in ["bm25", "hybrid"]:
                try:
                    bm25_resp = collection.query.bm25(
                        query=query,
                        limit=limit * 2,
                        filters=filters
                    )
                    if bm25_resp.objects:
                        # 按返回顺序赋予分数 (越前越高)
                        for rank, obj in enumerate(bm25_resp.objects):
                            props = obj.properties
                            doc_id = props.get("document_id", "unknown")
                            # 用排名作为简化分数
                            score = 1.0 / (rank + 1)
                            if doc_id not in bm25_results:
                                bm25_results[doc_id] = {
                                    "document_id": doc_id,
                                    "title": props.get("title"),
                                    "source": props.get("source"),
                                    "chunks": [],
                                    "bm25_score": 0
                                }
                            bm25_results[doc_id]["chunks"].append({
                                "content": props.get("content"),
                                "chunk_index": props.get("chunk_index", 0),
                            })
                            bm25_results[doc_id]["bm25_score"] = max(
                                bm25_results[doc_id]["bm25_score"], score
                            )
                except Exception as e:
                    print(f"[KnowledgeBase] BM25 搜索失败: {e}")
            
            # 3. 混合融合 (RRF - Reciprocal Rank Fusion)
            if mode == "hybrid":
                # 归一化并融合分数
                all_doc_ids = set(vector_results.keys()) | set(bm25_results.keys())
                fused_results = {}
                
                # 归一化向量分数到 [0, 1]
                max_vec_score = max([v.get("vector_score", 0) for v in vector_results.values()], default=1)
                if max_vec_score > 0:
                    for doc_id in vector_results:
                        vector_results[doc_id]["vector_score"] /= max_vec_score
                
                # 归一化 BM25 分数到 [0, 1]  
                max_bm25_score = max([v.get("bm25_score", 0) for v in bm25_results.values()], default=1)
                if max_bm25_score > 0:
                    for doc_id in bm25_results:
                        bm25_results[doc_id]["bm25_score"] /= max_bm25_score
                
                # 融合 (向量 60% + BM25 40%)
                for doc_id in all_doc_ids:
                    vec_s = vector_results.get(doc_id, {}).get("vector_score", 0)
                    bm25_s = bm25_results.get(doc_id, {}).get("bm25_score", 0)
                    fused_score = 0.6 * vec_s + 0.4 * bm25_s
                    
                    # 合并 chunks
                    vec_chunks = vector_results.get(doc_id, {}).get("chunks", [])
                    bm25_chunks = bm25_results.get(doc_id, {}).get("chunks", [])
                    all_chunks = {c["chunk_index"]: c for c in vec_chunks + bm25_chunks}
                    
                    fused_results[doc_id] = {
                        "document_id": doc_id,
                        "title": vector_results.get(doc_id, {}).get("title") or bm25_results.get(doc_id, {}).get("title"),
                        "source": vector_results.get(doc_id, {}).get("source") or bm25_results.get(doc_id, {}).get("source"),
                        "chunks": list(all_chunks.values()),
                        "vector_score": vec_s,
                        "bm25_score": bm25_s,
                        "fused_score": fused_score
                    }
                
                # 按融合分数排序
                sorted_docs = sorted(fused_results.values(), key=lambda x: x.get("fused_score", 0), reverse=True)
                return sorted_docs[:limit]
            
            elif mode == "vector":
                sorted_docs = sorted(vector_results.values(), key=lambda x: x.get("vector_score", 0), reverse=True)
                return sorted_docs[:limit]
            
            else:  # bm25
                sorted_docs = sorted(bm25_results.values(), key=lambda x: x.get("bm25_score", 0), reverse=True)
                return sorted_docs[:limit]
        
        except Exception as e:
            print(f"[KnowledgeBase] 搜索失败: {e}")
            return []
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档"""
        try:
            collection = self.client.collections.get(self.COLLECTION_NAME)
            
            # 获取所有不同的 document_id
            results = collection.query.fetch_objects(limit=1000)
            
            docs = {}
            if results.objects:
                for obj in results.objects:
                    props = obj.properties
                    doc_id = props.get("document_id", "unknown")
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "document_id": doc_id,
                            "title": props.get("title"),
                            "source": props.get("source"),
                            "file_type": props.get("file_type"),
                            "timestamp": props.get("timestamp"),
                            "chunk_count": 0
                        }
                    docs[doc_id]["chunk_count"] += 1
            
            # 按时间排序
            sorted_docs = sorted(
                docs.values(),
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            return sorted_docs
        
        except Exception as e:
            print(f"[KnowledgeBase] 列表失败: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            collection = self.client.collections.get(self.COLLECTION_NAME)
            
            # 删除该 document_id 的所有 chunk
            result = collection.data.delete_many(
                where={
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueText": document_id
                }
            )
            
            return result.get("successful", 0) > 0
        
        except Exception as e:
            print(f"[KnowledgeBase] 删除失败: {e}")
            return False


# 单例
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base(agent_id: str = None) -> KnowledgeBase:
    """获取知识库实例
    
    Args:
        agent_id: Agent ID，用于隔离不同 Agent 的知识库
    """
    global _kb_instance
    # 如果传入了 agent_id，创建新的实例（每个 agent 独立）
    if agent_id:
        return KnowledgeBase(agent_id=agent_id)
    
    # 否则返回全局单例
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance