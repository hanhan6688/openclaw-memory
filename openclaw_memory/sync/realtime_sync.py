"""
实时记忆同步服务 V2 - 优雅的文件监控和实时同步
特性:
1. 使用 watchdog 监控文件变化，实时同步
2. 智能批量处理，避免频繁写入
3. 支持 SSE 实时推送更新到前端
4. 自动重连和错误恢复
5. 自动提取实体关系到知识图谱
"""

import json
import time
import threading
import sys
import os
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from queue import Queue
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROJECT_ROOT
from memory_store import MemoryStore, clean_message_content
from knowledge_graph import KnowledgeGraph
from embeddings import OllamaEmbedding

# 导入知识图谱管理器（支持 NetworkX）
try:
    from kg_manager import KnowledgeGraphManager, NETWORKX_AVAILABLE
    KG_MANAGER_AVAILABLE = True
except ImportError:
    KG_MANAGER_AVAILABLE = False
    NETWORKX_AVAILABLE = False

# 尝试导入 watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog 未安装，使用轮询模式。安装: pip install watchdog")


@dataclass
class SessionState:
    """会话状态追踪"""
    last_message_id: str = ""
    last_timestamp: datetime = None
    last_file_size: int = 0
    last_file_hash: str = ""
    pending_messages: List[Dict] = field(default_factory=list)
    last_sync_time: datetime = None


@dataclass
class SyncEvent:
    """同步事件"""
    event_type: str  # 'new_message', 'sync_complete', 'error'
    agent_name: str
    session_id: str
    data: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SessionFileReader:
    """OpenClaw 会话文件读取器 - 优化版"""

    def __init__(self, agents_dir: str = None):
        if agents_dir is None:
            openclaw_dir = os.path.expanduser(os.getenv("OPENCLAW_DIR", "~/.openclaw"))
            agents_dir = os.path.join(openclaw_dir, "agents")
        self.agents_dir = Path(agents_dir)

    def list_agents(self) -> List[str]:
        """列出所有 Agent"""
        if not self.agents_dir.exists():
            return []
        return [d.name for d in self.agents_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    def get_sessions_dir(self, agent_name: str) -> Path:
        """获取 Agent 的 sessions 目录"""
        return self.agents_dir / agent_name / "sessions"

    def list_session_files(self, agent_name: str) -> List[Path]:
        """列出所有会话文件（包括 reset 备份文件）"""
        sessions_dir = self.get_sessions_dir(agent_name)
        if not sessions_dir.exists():
            return []
        
        files = []
        # 读取 .jsonl 文件
        for f in sessions_dir.glob("*.jsonl"):
            if not any(x in f.name for x in ['.deleted.', '.tmp']):
                files.append(f)
        
        # 读取 .reset. 备份文件（历史会话）
        for f in sessions_dir.glob("*.jsonl.reset.*"):
            if '.deleted.' not in f.name:
                files.append(f)
        
        return files

    def list_sessions(self, agent_name: str) -> List[Dict]:
        """列出 Agent 的所有会话 (兼容 API)"""
        files = self.list_session_files(agent_name)
        sessions = []
        seen_ids = set()  # 避免重复
        
        for f in files:
            # 从文件名提取 session_id
            # 例如: 868d6fd8-8a56-4a9c-b4b9-4fa5bc6f0c05.jsonl.reset.2026-03-16T02-33-30.069Z
            # 提取: 868d6fd8-8a56-4a9c-b4b9-4fa5bc6f0c05
            name = f.name
            if '.jsonl.reset.' in name:
                # reset 文件，提取原始 session_id
                session_id = name.split('.jsonl.reset.')[0]
                # 添加时间戳后缀以区分
                reset_time = name.split('.jsonl.reset.')[1]
                unique_id = f"{session_id}_reset_{reset_time[:10]}"  # 只取日期部分
            else:
                session_id = f.stem
                unique_id = session_id
            
            if unique_id in seen_ids:
                continue
            seen_ids.add(unique_id)
            
            stat = f.stat()
            sessions.append({
                'id': unique_id,
                'session_id': session_id,
                'file_path': str(f),
                'updated_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'title': f.stem[:20],
                'is_reset': '.reset.' in name
            })
        
        return sorted(sessions, key=lambda x: x['updated_at'], reverse=True)


    def read_sessions_json(self, agent_name: str) -> Dict:
        """读取 sessions.json 文件，返回当前活跃会话信息"""
        sessions_dir = self.get_sessions_dir(agent_name)
        sessions_json = sessions_dir / "sessions.json"
        
        if not sessions_json.exists():
            return {}
        
        try:
            with open(sessions_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"⚠️ 读取 sessions.json 失败: {e}")
            return {}

    def get_active_session_file(self, agent_name: str) -> Optional[Path]:
        """获取当前活跃的会话文件路径
        
        比较 sessions.json 和 .jsonl 文件的前5行，决定使用哪个同步
        """
        sessions_data = self.read_sessions_json(agent_name)
        sessions_dir = self.get_sessions_dir(agent_name)
        
        # 从 sessions.json 获取当前活跃会话
        active_session_key = f"agent:{agent_name}:{agent_name}"
        active_session = sessions_data.get(active_session_key, {})
        session_file_path = active_session.get('sessionFile', '')
        
        if session_file_path:
            session_file = Path(session_file_path)
            if session_file.exists():
                return session_file
        
        # 如果 sessions.json 中没有找到，查找最新的 .jsonl 文件
        jsonl_files = list(sessions_dir.glob("*.jsonl"))
        if jsonl_files:
            # 按修改时间排序，返回最新的
            return max(jsonl_files, key=lambda f: f.stat().st_mtime)
        
        return None

    def compare_and_sync(self, agent_name: str) -> Dict:
        """比较 sessions.json 和 .jsonl 文件，决定同步策略
        
        返回:
            - source: 'sessions_json' 或 'jsonl'
            - session_file: 要同步的文件路径
            - reason: 选择原因
        """
        sessions_data = self.read_sessions_json(agent_name)
        sessions_dir = self.get_sessions_dir(agent_name)
        
        # 获取 sessions.json 中的活跃会话
        active_session_key = f"agent:{agent_name}:{agent_name}"
        active_session = sessions_data.get(active_session_key, {})
        json_session_id = active_session.get('sessionId', '')
        json_updated_at = active_session.get('updatedAt', 0)
        
        # 获取最新的 .jsonl 文件
        jsonl_files = [f for f in sessions_dir.glob("*.jsonl") 
                       if not any(x in f.name for x in ['.deleted.', '.tmp', '.lock'])]
        
        if not jsonl_files:
            return {
                'source': 'none',
                'session_file': None,
                'reason': '没有找到会话文件'
            }
        
        # 找到最新的 .jsonl 文件
        latest_jsonl = max(jsonl_files, key=lambda f: f.stat().st_mtime)
        jsonl_mtime = latest_jsonl.stat().st_mtime * 1000  # 转换为毫秒
        
        # 比较更新时间
        if json_session_id and json_updated_at >= jsonl_mtime:
            # sessions.json 更新，使用其中指定的文件
            session_file = Path(active_session.get('sessionFile', ''))
            if session_file.exists():
                return {
                    'source': 'sessions_json',
                    'session_file': str(session_file),
                    'reason': f'sessions.json 更新 (sessionId: {json_session_id[:8]}...)'
                }
        
        # 使用最新的 .jsonl 文件
        return {
            'source': 'jsonl',
            'session_file': str(latest_jsonl),
            'reason': f'jsonl 文件更新 ({latest_jsonl.name[:20]}...)'
        }
    def read_session_messages(self, file_path) -> List[Dict]:
        """读取会话消息 (兼容 API)"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        messages, _ = self.read_new_messages(file_path)
        return messages

    def read_new_messages(self, file_path: Path, last_message_id: str = "") -> tuple:
        """读取新消息，返回 (messages, last_id)"""
        messages = []
        last_id = last_message_id
        found_last = not last_message_id  # 如果没有 last_id，从头开始读

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        
                        # 只处理消息类型
                        if obj.get('type') != 'message':
                            continue

                        msg_id = obj.get('id', '')
                        
                        # 跳过已处理的消息
                        if not found_last:
                            if msg_id == last_message_id:
                                found_last = True
                            continue

                        msg = obj.get('message', {})
                        role = msg.get('role', '')
                        content_parts = msg.get('content', [])

                        # 只同步 user 和 assistant 消息
                        if role not in ("user", "assistant"):
                            continue
                        timestamp = obj.get('timestamp', '')

                        # 提取文本内容
                        text_content = self._extract_text_content(content_parts)
                        
                        if role and text_content:
                            # 清理内容
                            cleaned_content = clean_message_content(text_content)
                            if cleaned_content:
                                messages.append({
                                    'id': msg_id,
                                    'role': role,
                                    'content': cleaned_content,
                                    'timestamp': timestamp
                                })
                                last_id = msg_id

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"⚠️ 读取文件失败 {file_path}: {e}")

        return messages, last_id

    def _extract_text_content(self, content_parts: List) -> str:
        """提取文本内容，清理元数据"""
        if isinstance(content_parts, str):
            return self._clean_metadata(content_parts)

        texts = []
        for part in content_parts:
            if part.get('type') == 'text':
                text = part.get('text', '')
                # 跳过系统消息
                if not text.startswith('A new session was started'):
                    # 清理元数据
                    cleaned = self._clean_metadata(text)
                    if cleaned:
                        texts.append(cleaned)
            # 跳过 toolCall 和 toolResult，只保留纯文本对话

        return '\n'.join(texts)

    def _clean_metadata(self, text: str) -> str:
        """清理消息中的元数据，只保留实际对话内容"""
        if not text:
            return ""

        import re

        # 1. 移除 "Sender (untrusted metadata): ```json{...}``` [时间]" 格式
        pattern1 = r'Sender \(untrusted metadata\):\s*```json\s*\{[^}]*\}\s*```\s*'
        text = re.sub(pattern1, '', text)

        # 2. 移除 "Sender (untrusted metadata):" 前缀
        pattern2 = r'^Sender \(untrusted metadata\):\s*'
        text = re.sub(pattern2, '', text)

        # 3. 移除时间戳 [Wed 20260318 09:48 GMT+8]
        pattern3 = r'^\[[^\]]+\]\s*'
        text = re.sub(pattern3, '', text)

        # 4. 再次移除时间戳（可能在换行后）
        text = re.sub(r'\n\[[^\]]+\]\s*', '\n', text)

        return text.strip()

    def get_file_hash(self, file_path: Path) -> str:
        """获取文件哈希（用于检测变化）"""
        try:
            # 只读取最后 10KB 来计算哈希（更快）
            with open(file_path, 'rb') as f:
                f.seek(0, 2)  # 移动到文件末尾
                size = f.tell()
                f.seek(max(0, size - 10240))  # 读取最后 10KB
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class FileWatchHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """文件变化处理器"""

    def __init__(self, callback):
        self.callback = callback
        self._debounce = {}  # 防抖

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def _handle_event(self, file_path: str):
        """处理文件事件（带防抖）"""
        if not file_path.endswith('.jsonl'):
            return

        # 防抖：同一文件 1 秒内只处理一次
        now = time.time()
        last_time = self._debounce.get(file_path, 0)
        if now - last_time < 1.0:
            return
        self._debounce[file_path] = now

        # 提取 agent 名称
        path = Path(file_path)
        try:
            parts = path.parts
            if "agents" in parts:
                idx = parts.index("agents")
                if idx + 1 < len(parts):
                    agent_name = parts[idx + 1]
                    self.callback(agent_name, file_path)
        except Exception:
            pass


class RealtimeSyncService:
    """实时同步服务 - 主服务类"""

    # 配置
    CONFIG = {
        'batch_size': 3,           # 批量处理消息数
        'batch_timeout': 30,       # 批量超时（秒）
        'poll_interval': 5,        # 轮询间隔（秒）
        'use_watchdog': True,      # 优先使用 watchdog
    }

    def __init__(self, config: Dict = None):
        self.config = {**self.CONFIG, **(config or {})}
        self.reader = SessionFileReader()
        
        # 状态
        self.session_states: Dict[str, SessionState] = {}
        self.memory_stores: Dict[str, MemoryStore] = {}
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
        
        # 持久化状态文件
        self.state_file = Path.home() / ".openclaw" / "memory_system" / "sync_state.json"
        self._load_state()
        
        # 事件队列（用于 SSE）
        self.event_queue = Queue()
        self.event_listeners: List = []  # SSE 监听器
        
        # 控制
        self.is_running = False
        self._stop_event = threading.Event()
        self._observer = None
        self._lock = threading.Lock()
    
    def _load_state(self):
        """加载持久化状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.get('session_states', {}).items():
                        state = SessionState()
                        state.last_message_id = value.get('last_message_id', '')
                        state.last_file_hash = value.get('last_file_hash', '')
                        self.session_states[key] = state
                print(f"📂 加载同步状态: {len(self.session_states)} 个会话")
            except Exception as e:
                print(f"⚠️ 加载状态失败: {e}")
    
    def _save_state(self):
        """保存持久化状态"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'session_states': {
                    key: {
                        'last_message_id': state.last_message_id,
                        'last_file_hash': state.last_file_hash
                    }
                    for key, state in self.session_states.items()
                }
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ 保存状态失败: {e}")

    def get_memory_store(self, agent_name: str) -> MemoryStore:
        """获取或创建 MemoryStore"""
        if agent_name not in self.memory_stores:
            self.memory_stores[agent_name] = MemoryStore(agent_name)
        return self.memory_stores[agent_name]

    def get_knowledge_graph(self, agent_name: str) -> KnowledgeGraph:
        """获取或创建 KnowledgeGraph"""
        if agent_name not in self.knowledge_graphs:
            self.knowledge_graphs[agent_name] = KnowledgeGraph(agent_name)
        return self.knowledge_graphs[agent_name]

    def start(self, background: bool = False):
        """启动服务
        
        Args:
            background: 是否在后台运行（不阻塞主线程）
        """
        print("\n" + "="*50)
        print("🚀 实时记忆同步服务 V2")
        print("="*50)

        self.is_running = True
        self._stop_event.clear()

        # 初始同步
        print("📦 执行初始同步...")
        self._initial_sync()

        # 启动监控
        if self.config['use_watchdog'] and WATCHDOG_AVAILABLE:
            print("👁️ 使用 watchdog 监控文件变化")
            self._start_watchdog()
        else:
            print(f"⏰ 使用轮询模式 (间隔: {self.config['poll_interval']}秒)")

        # 启动处理线程
        self._processor_thread = threading.Thread(
            target=self._process_loop, 
            daemon=True,
            name="SyncProcessor"
        )
        self._processor_thread.start()

        print("="*50 + "\n")
        print("✅ 服务已启动，等待文件变化...")

        # 如果不是后台模式，阻塞主线程
        if not background:
            try:
                while self.is_running:
                    self._stop_event.wait(timeout=1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """停止服务"""
        print("\n🛑 停止同步服务...")
        self.is_running = False
        self._stop_event.set()

        # 处理剩余消息
        self._flush_all_pending()

        # 保存状态
        self._save_state()

        # 停止 watchdog
        if self._observer:
            self._observer.stop()
            self._observer.join()

        # 关闭连接
        self._close_all()

        print("✅ 服务已停止")

    def _initial_sync(self):
        """初始同步所有会话"""
        for agent_name in self.reader.list_agents():
            for file_path in self.reader.list_session_files(agent_name):
                self._sync_session_file(agent_name, file_path, initial=True)

    def _start_watchdog(self):
        """启动 watchdog 监控"""
        handler = FileWatchHandler(self._on_file_changed)
        self._observer = Observer()

        for agent_name in self.reader.list_agents():
            sessions_dir = self.reader.get_sessions_dir(agent_name)
            if sessions_dir.exists():
                self._observer.schedule(handler, str(sessions_dir), recursive=False)
                print(f"   📂 监控: {agent_name}/sessions/")

        self._observer.start()

    def _on_file_changed(self, agent_name: str, file_path: str):
        """文件变化回调"""
        if not self.is_running:
            return

        # 放入队列异步处理
        self.event_queue.put(('file_change', agent_name, file_path))

    def _process_loop(self):
        """处理循环"""
        last_flush = time.time()

        while self.is_running:
            try:
                # 处理事件队列
                try:
                    event = self.event_queue.get(timeout=1.0)
                    if event[0] == 'file_change':
                        _, agent_name, file_path = event
                        self._sync_session_file(agent_name, file_path)
                except Exception:
                    pass

                # 定期刷新待处理消息
                if time.time() - last_flush >= self.config['batch_timeout']:
                    self._flush_all_pending()
                    last_flush = time.time()

            except Exception as e:
                print(f"❌ 处理错误: {e}")
                time.sleep(1)

    def _sync_session_file(self, agent_name: str, file_path: str, initial: bool = False):
        """同步会话文件"""
        path = Path(file_path)
        session_id = path.stem
        state_key = f"{agent_name}:{session_id}"

        with self._lock:
            state = self.session_states.get(state_key)
            if not state:
                state = SessionState()
                self.session_states[state_key] = state

            # 检查文件是否有变化（initial 模式下强制跳过哈希检查）
            current_hash = self.reader.get_file_hash(path)
            if not initial and current_hash and current_hash == state.last_file_hash:
                return  # 无变化

            # 读取新消息
            messages, last_id = self.reader.read_new_messages(path, state.last_message_id)

            if not messages:
                return

            # 更新状态
            state.last_message_id = last_id
            state.last_file_hash = current_hash
            state.pending_messages.extend(messages)
            
            # 保存状态
            self._save_state()

            print(f"📨 {agent_name}/{session_id[:8]}: +{len(messages)} 条消息")

            # 发送事件
            self._emit_event(SyncEvent(
                event_type='new_message',
                agent_name=agent_name,
                session_id=session_id,
                data={'count': len(messages), 'initial': initial}
            ))

            # 检查是否需要刷新
            if len(state.pending_messages) >= self.config['batch_size']:
                self._flush_pending(state_key)

    def _flush_pending(self, state_key: str):
        """刷新待处理消息（带智能过滤和去重）"""
        state = self.session_states.get(state_key)
        if not state or not state.pending_messages:
            return

        agent_name, session_id = state_key.split(':', 1)
        messages = state.pending_messages
        state.pending_messages = []
        state.last_sync_time = datetime.now()

        try:
            store = self.get_memory_store(agent_name)
            saved_count = 0
            skipped_count = 0
            duplicate_count = 0

            # 用于本批次去重
            seen_content = set()

            # 收集需要提取实体关系的内容
            kg_extract_contents = []

            for msg in messages:
                content = msg.get('content', '')
                role = msg.get('role', 'user')

                # 批次内去重
                content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                if content_hash in seen_content:
                    duplicate_count += 1
                    continue
                seen_content.add(content_hash)

                quality_info = self._assess_quality(content, role)
                if not quality_info['should_save']:
                    skipped_count += 1
                    continue

                summary = self._generate_summary(content)
                memory_data = {
                    'summary': summary,
                    'importance': quality_info['score'],
                    'memory_type': 'conversation',
                    'session_id': session_id,
                    'role': role,
                    'quality': quality_info['quality']
                }

                memory_id = store.store(content, memory_data)
                if memory_id:
                    saved_count += 1
                    # 收集需要提取实体关系的内容
                    if role == 'user' and len(content) > 20:
                        kg_extract_contents.append(content)
                else:
                    # 返回 None 表示重复
                    duplicate_count += 1

            if saved_count > 0 or skipped_count > 0 or duplicate_count > 0:
                print(f"✅ {agent_name}/{session_id[:8]}: 保存 {saved_count} 条, 跳过 {skipped_count} 条, 重复 {duplicate_count} 条")

            # 自动提取实体关系到知识图谱
            if kg_extract_contents and KG_MANAGER_AVAILABLE:
                self._extract_kg_async(agent_name, kg_extract_contents)

            self._emit_event(SyncEvent(
                event_type='sync_complete',
                agent_name=agent_name,
                session_id=session_id,
                data={'saved': saved_count, 'skipped': skipped_count, 'duplicates': duplicate_count}
            ))

        except Exception as e:
            print(f"❌ 存储失败: {e}")
            state.pending_messages = messages + state.pending_messages

    def _extract_kg_async(self, agent_name: str, contents: List[str]):
        """异步提取实体关系到知识图谱"""
        def extract():
            try:
                from extraction.decoder_extractor import EnhancedDecoderExtractor
                
                kg_manager = KnowledgeGraphManager(agent_name, prefer_nebula=False)
                extractor = EnhancedDecoderExtractor()
                
                total_entities = 0
                total_relations = 0
                
                for content in contents[:5]:  # 限制每次最多处理 5 条
                    try:
                        result = extractor.extract(content)
                        
                        # 存储实体
                        for entity in result.entities:
                            entity_id = f"{agent_name}_{entity.name}_{entity.type}"
                            kg_manager.add_entity(
                                entity_id=entity_id,
                                name=entity.name,
                                entity_type=entity.type,
                                properties=entity.metadata,
                                confidence=entity.confidence
                            )
                            total_entities += 1
                        
                        # 存储关系
                        entity_id_map = {}
                        for entity in result.entities:
                            entity_id_map[entity.name] = f"{agent_name}_{entity.name}_{entity.type}"
                        
                        for relation in result.relations:
                            source_id = entity_id_map.get(relation.source)
                            target_id = entity_id_map.get(relation.target)
                            if source_id and target_id:
                                kg_manager.add_relation(
                                    source_id=source_id,
                                    target_id=target_id,
                                    relation_type=relation.type,
                                    confidence=relation.confidence,
                                    evidence=relation.evidence or ""
                                )
                                total_relations += 1
                    except Exception as e:
                        pass
                
                if total_entities > 0 or total_relations > 0:
                    print(f"🕸️ {agent_name}: 提取 {total_entities} 实体, {total_relations} 关系")
                    
            except Exception as e:
                print(f"⚠️ KG 提取失败: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=extract, daemon=True)
        thread.start()

    def _assess_quality(self, content: str, role: str) -> dict:
        """评估消息质量"""
        import re
        if not content or len(content.strip()) < 10:
            return {'quality': 'low', 'score': 0.1, 'should_save': False}
        content = content.strip()
        low_patterns = [r'^好的?[，。！]?$', r'^明白[，。！]?$', r'^收到[，。！]?$', r'^可以[，。！]?$']
        for p in low_patterns:
            if re.match(p, content, re.IGNORECASE):
                return {'quality': 'low', 'score': 0.2, 'should_save': False}
        score = 0.5
        if len(content) > 100: score += 0.1
        if len(content) > 300: score += 0.1
        if role == 'user': score += 0.1
        high_indicators = [r'\d{4}[-/年]', r'\d+[%％]', r'https?://', r'项目|任务|需求|问题']
        score += sum(0.05 for p in high_indicators if re.search(p, content))
        score = min(1.0, max(0.1, score))
        return {'quality': 'high' if score >= 0.7 else 'medium' if score >= 0.4 else 'low', 'score': score, 'should_save': score >= 0.4}

    def _generate_summary(self, content: str) -> str:
        """生成摘要"""
        import re
        cleaned = re.sub(r'```[\s\S]*?```', '', content)
        cleaned = re.sub(r'https?://\S+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned[:150] + '...' if len(cleaned) > 150 else cleaned

    def _flush_all_pending(self):
        """刷新所有待处理消息"""
        for state_key in list(self.session_states.keys()):
            self._flush_pending(state_key)

    def _emit_event(self, event: SyncEvent):
        """发送事件到监听器"""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception:
                pass

    def add_listener(self, callback):
        """添加事件监听器"""
        self.event_listeners.append(callback)

    def remove_listener(self, callback):
        """移除事件监听器"""
        if callback in self.event_listeners:
            self.event_listeners.remove(callback)

    def _close_all(self):
        """关闭所有连接"""
        for store in self.memory_stores.values():
            try:
                store.close()
            except Exception:
                pass
        for kg in self.knowledge_graphs.values():
            try:
                kg.close()
            except Exception:
                pass

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'agents': len(self.memory_stores),
            'sessions': len(self.session_states),
            'pending_messages': sum(
                len(s.pending_messages) for s in self.session_states.values()
            )
        }


# 全局实例
_sync_service: Optional[RealtimeSyncService] = None

def get_sync_service() -> RealtimeSyncService:
    """获取全局同步服务实例"""
    global _sync_service
    if _sync_service is None:
        _sync_service = RealtimeSyncService()
    return _sync_service


if __name__ == "__main__":
    # 测试
    service = RealtimeSyncService()
    
    def on_event(event):
        print(f"🔔 事件: {event.event_type} - {event.agent_name}/{event.session_id[:8]}")
    
    service.add_listener(on_event)
    
    try:
        service.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()