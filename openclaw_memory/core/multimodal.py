"""
OpenClaw Memory System - 多模态存储
================================

支持图片、文档、音频、视频等多模态内容的存储和检索

架构:
┌─────────────────────────────────────────────────────────────────┐
│                     多模态存储流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  图片 ──► llava/moondream ──► 描述文本 + 图片向量 ──► 存储      │
│  文档 ──► 文本提取 ──► 文本向量 ──► 存储                         │
│  音频 ──► whisper ──► 转录文本 + 音频向量 ──► 存储              │
│  视频 ──► 关键帧提取 ──► 帧描述 + 视频向量 ──► 存储             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

import os
import base64
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class MultimodalProcessor:
    """多模态内容处理器"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.supported_types = {
            # 图片
            "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
            # 文档
            "application/pdf", "text/plain", "text/markdown", "text/html",
            "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            # 音频
            "audio/mpeg", "audio/wav", "audio/ogg", "audio/m4a", "audio/webm",
            # 视频
            "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
        }
        
        # 多模态模型配置
        self.vision_model = "llava"  # 图片理解
        self.audio_model = "whisper"  # 音频转文字
        self.embed_model = "nomic-embed-text"  # 文本嵌入
    
    def detect_type(self, file_path: str) -> Tuple[str, str]:
        """
        检测文件类型
        
        Returns:
            (category, mime_type) - 类别和 MIME 类型
        """
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type is None:
            # 根据扩展名判断
            ext = Path(file_path).suffix.lower()
            ext_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp",
                ".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown",
                ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/m4a",
                ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
            }
            mime_type = ext_map.get(ext, "application/octet-stream")
        
        # 分类
        if mime_type.startswith("image/"):
            category = "image"
        elif mime_type.startswith("audio/"):
            category = "audio"
        elif mime_type.startswith("video/"):
            category = "video"
        elif mime_type in ["application/pdf", "text/plain", "text/markdown"]:
            category = "document"
        else:
            category = "other"
        
        return category, mime_type
    
    def process_image(self, file_path: str) -> Dict:
        """
        处理图片 - 使用视觉模型生成描述
        
        Returns:
            {
                "description": "图片描述",
                "tags": ["标签1", "标签2"],
                "thumbnail": "base64缩略图",
                "metadata": {...}
            }
        """
        import requests
        
        # 读取图片并编码
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # 调用 Ollama 视觉模型
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": "请详细描述这张图片的内容，包括：1. 主要物体和场景 2. 颜色和构图 3. 可能的上下文或故事。用中文回答。",
                    "images": [image_data],
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "")
            else:
                description = f"[图片处理失败: {response.status_code}]"
        except Exception as e:
            # 如果视觉模型不可用，使用占位描述
            description = f"[图片: {Path(file_path).name}]"
        
        # 生成缩略图（可选）
        thumbnail = self._generate_thumbnail(file_path)
        
        # 提取元数据
        metadata = self._extract_image_metadata(file_path)
        
        return {
            "description": description,
            "tags": self._extract_tags_from_description(description),
            "thumbnail": thumbnail,
            "metadata": metadata
        }
    
    def process_document(self, file_path: str) -> Dict:
        """
        处理文档 - 提取文本内容
        
        Returns:
            {
                "text": "文档文本",
                "summary": "摘要",
                "metadata": {...}
            }
        """
        text = ""
        mime_type, _ = self.detect_type(file_path)
        
        if mime_type == "application/pdf":
            text = self._extract_pdf_text(file_path)
        elif mime_type in ["text/plain", "text/markdown"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif mime_type == "text/html":
            text = self._extract_html_text(file_path)
        else:
            text = f"[不支持的文档格式: {mime_type}]"
        
        # 生成摘要
        summary = self._generate_summary(text[:5000])  # 限制长度
        
        return {
            "text": text,
            "summary": summary,
            "metadata": {
                "file_name": Path(file_path).name,
                "file_size": os.path.getsize(file_path),
                "mime_type": mime_type
            }
        }
    
    def process_audio(self, file_path: str) -> Dict:
        """
        处理音频 - 使用 Whisper 转录
        
        Returns:
            {
                "transcription": "转录文本",
                "duration": 时长(秒),
                "metadata": {...}
            }
        """
        import requests
        
        # 方法1: 使用 Ollama Whisper（如果可用）
        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
            
            # 调用 Whisper API（需要单独部署或使用 OpenAI API）
            # 这里使用简化方案：调用本地 Whisper
            transcription = self._transcribe_with_whisper(file_path)
        except Exception as e:
            transcription = f"[音频转录失败: {e}]"
        
        return {
            "transcription": transcription,
            "metadata": {
                "file_name": Path(file_path).name,
                "file_size": os.path.getsize(file_path),
            }
        }
    
    def process_video(self, file_path: str) -> Dict:
        """
        处理视频 - 提取关键帧并生成描述
        
        Returns:
            {
                "frames": [{"time": 0, "description": "..."}],
                "summary": "视频摘要",
                "duration": 时长(秒),
                "metadata": {...}
            }
        """
        # 提取关键帧
        frames = self._extract_key_frames(file_path)
        
        # 对每个关键帧生成描述
        frame_descriptions = []
        for frame in frames:
            desc = self.process_image(frame["path"])
            frame_descriptions.append({
                "time": frame["time"],
                "description": desc["description"]
            })
        
        # 生成视频摘要
        all_descriptions = "\n".join([f"第{i+1}帧: {f['description']}" 
                                       for i, f in enumerate(frame_descriptions)])
        summary = self._generate_summary(all_descriptions)
        
        return {
            "frames": frame_descriptions,
            "summary": summary,
            "metadata": {
                "file_name": Path(file_path).name,
                "file_size": os.path.getsize(file_path),
            }
        }
    
    def store_multimodal(self, file_path: str, memory_store, metadata: Dict = None) -> str:
        """
        存储多模态内容到记忆系统
        
        Args:
            file_path: 文件路径
            memory_store: MemoryStore 实例
            metadata: 额外元数据
        
        Returns:
            memory_id
        """
        category, mime_type = self.detect_type(file_path)
        
        # 根据类型处理
        if category == "image":
            result = self.process_image(file_path)
            content = result["description"]
            extra_metadata = {
                "thumbnail": result["thumbnail"],
                "tags": result["tags"],
                "image_metadata": result["metadata"]
            }
        elif category == "document":
            result = self.process_document(file_path)
            content = result["text"]
            extra_metadata = {
                "summary": result["summary"],
                "document_metadata": result["metadata"]
            }
        elif category == "audio":
            result = self.process_audio(file_path)
            content = result["transcription"]
            extra_metadata = {
                "audio_metadata": result["metadata"]
            }
        elif category == "video":
            result = self.process_video(file_path)
            content = result["summary"]
            extra_metadata = {
                "frames": result["frames"],
                "video_metadata": result["metadata"]
            }
        else:
            content = f"[不支持的文件类型: {mime_type}]"
            extra_metadata = {}
        
        # 合并元数据
        full_metadata = {
            "memory_type": "multimodal",
            "content_type": category,
            "mime_type": mime_type,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_hash": self._calculate_hash(file_path),
            **(metadata or {}),
            **extra_metadata
        }
        
        # 存储到记忆系统
        memory_id = memory_store.store(content, full_metadata)
        
        return memory_id
    
    # ==================== 辅助方法 ====================
    
    def _generate_thumbnail(self, file_path: str, max_size: int = 200) -> str:
        """生成缩略图（base64）"""
        try:
            from PIL import Image
            import io
            
            with Image.open(file_path) as img:
                # 调整大小
                img.thumbnail((max_size, max_size))
                
                # 转换为 base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=70)
                return base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            return ""
    
    def _extract_image_metadata(self, file_path: str) -> Dict:
        """提取图片元数据"""
        try:
            from PIL import Image
            
            with Image.open(file_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode
                }
        except Exception:
            return {}
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """提取 PDF 文本"""
        try:
            import PyPDF2
            
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception:
            return f"[PDF 文本提取失败]"
    
    def _extract_html_text(self, file_path: str) -> str:
        """提取 HTML 文本"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n")
        except Exception:
            return ""
    
    def _transcribe_with_whisper(self, file_path: str) -> str:
        """使用 Whisper 转录音频"""
        try:
            import whisper
            
            model = whisper.load_model("base")
            result = model.transcribe(file_path)
            return result["text"]
        except Exception as e:
            return f"[转录失败: {e}]"
    
    def _extract_key_frames(self, file_path: str, num_frames: int = 5) -> List[Dict]:
        """提取视频关键帧"""
        try:
            import cv2
            
            frames = []
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = total_frames // num_frames
            
            for i in range(num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                ret, frame = cap.read()
                if ret:
                    # 保存帧
                    frame_path = f"/tmp/frame_{i}.jpg"
                    cv2.imwrite(frame_path, frame)
                    frames.append({
                        "time": i * interval / 30,  # 假设 30fps
                        "path": frame_path
                    })
            
            cap.release()
            return frames
        except Exception:
            return []
    
    def _generate_summary(self, text: str) -> str:
        """生成摘要"""
        if len(text) < 100:
            return text
        
        # 简单截取前 200 字作为摘要
        return text[:200] + "..." if len(text) > 200 else text
    
    def _extract_tags_from_description(self, description: str) -> List[str]:
        """从描述中提取标签"""
        # 简单的关键词提取
        keywords = []
        stop_words = {"的", "是", "在", "有", "和", "了", "这", "那", "一", "个"}
        
        for word in description.split():
            if len(word) > 1 and word not in stop_words:
                keywords.append(word)
        
        return list(set(keywords))[:10]
    
    def _calculate_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class MultimodalMemoryStore:
    """
    多模态记忆存储
    
    扩展 MemoryStore，支持多模态内容
    """
    
    def __init__(self, memory_store, processor: MultimodalProcessor = None):
        self.memory_store = memory_store
        self.processor = processor or MultimodalProcessor()
    
    def remember_image(self, file_path: str, description: str = None, 
                       tags: List[str] = None, importance: float = 0.5) -> str:
        """记住一张图片"""
        result = self.processor.process_image(file_path)
        
        content = description or result["description"]
        metadata = {
            "memory_type": "image",
            "content_type": "image",
            "file_path": file_path,
            "thumbnail": result["thumbnail"],
            "tags": tags or result["tags"],
            "importance": importance,
            "image_metadata": result["metadata"]
        }
        
        return self.memory_store.store(content, metadata)
    
    def remember_document(self, file_path: str, summary: str = None,
                          importance: float = 0.5) -> str:
        """记住一个文档"""
        result = self.processor.process_document(file_path)
        
        content = result["text"]
        metadata = {
            "memory_type": "document",
            "content_type": "document",
            "file_path": file_path,
            "summary": summary or result["summary"],
            "importance": importance,
            "document_metadata": result["metadata"]
        }
        
        return self.memory_store.store(content, metadata)
    
    def remember_audio(self, file_path: str, transcription: str = None,
                       importance: float = 0.5) -> str:
        """记住一段音频"""
        result = self.processor.process_audio(file_path)
        
        content = transcription or result["transcription"]
        metadata = {
            "memory_type": "audio",
            "content_type": "audio",
            "file_path": file_path,
            "importance": importance,
            "audio_metadata": result["metadata"]
        }
        
        return self.memory_store.store(content, metadata)
    
    def remember_video(self, file_path: str, summary: str = None,
                       importance: float = 0.5) -> str:
        """记住一段视频"""
        result = self.processor.process_video(file_path)
        
        content = summary or result["summary"]
        metadata = {
            "memory_type": "video",
            "content_type": "video",
            "file_path": file_path,
            "frames": result["frames"],
            "importance": importance,
            "video_metadata": result["metadata"]
        }
        
        return self.memory_store.store(content, metadata)
    
    def search_similar_images(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索相似图片"""
        results = self.memory_store.search(query, limit=limit)
        return [r for r in results if r.get("content_type") == "image"]
    
    def search_by_content_type(self, content_type: str, query: str = None, 
                                limit: int = 10) -> List[Dict]:
        """按内容类型搜索"""
        if query:
            results = self.memory_store.search(query, limit=limit * 2)
            return [r for r in results if r.get("content_type") == content_type][:limit]
        else:
            # 获取所有该类型的记忆
            all_memories = self.memory_store.get_recent(limit=1000)
            return [m for m in all_memories if m.get("content_type") == content_type][:limit]