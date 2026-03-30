"""
后台任务调度器

支持：
1. 定时同步会话到记忆
2. 定时提取知识图谱实体
3. 懒加载检查更新
4. 阈值触发批量处理
5. 文件变化实时同步（watchdog）
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from config import PROJECT_ROOT, AGENTS_DIR
from realtime_sync import SessionFileReader
from memory_store import MemoryStore

# 尝试导入 watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog 未安装，使用轮询模式。安装: pip install watchdog")


class FileChangeHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """文件变化处理器"""
    
    def __init__(self, callback):
        self.callback = callback
        self._debounce = {}  # 防抖
        self._last_sync = {}  # 记录上次同步时间
    
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
        
        # 防抖：同一文件 2 秒内只处理一次
        now = time.time()
        last_time = self._debounce.get(file_path, 0)
        if now - last_time < 2.0:
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
                    print(f"📁 检测到文件变化: {agent_name}/{path.name}")
                    self.callback(agent_name, file_path)
        except Exception as e:
            print(f"⚠️ 处理文件事件失败: {e}")


class BackgroundScheduler:
    """后台任务调度器"""
    
    def __init__(self, check_interval: int = 300):
        """
        Args:
            check_interval: 检查间隔（秒），默认5分钟
        """
        self.check_interval = check_interval
        self.session_reader = SessionFileReader()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # watchdog
        self._observer = None
        self._file_handler = None
        
        # 记录上次同步时间
        self.last_sync_times: Dict[str, datetime] = {}
        
        # 配置
        self.config = {
            # 同步间隔（小时）
            "sync_interval_hours": 1,
            
            # 实体提取间隔（小时）
            "extract_interval_hours": 6,
            
            # 新消息阈值（超过此数量触发同步）
            "new_messages_threshold": 50,
            
            # 夜间处理时间（小时，24小时制）
            "night_process_hour": 3,  # 凌晨3点
            
            # 是否启用懒加载检查
            "lazy_check_enabled": True,
            
            # 是否启用 watchdog 实时同步
            "watchdog_enabled": True,
        }
        
        # 统计
        self.stats = {
            "sync_count": 0,
            "extract_count": 0,
            "file_change_count": 0,
            "last_run": None,
            "errors": []
        }
    
    def start(self):
        """启动后台任务"""
        if self.running:
            return
        
        self.running = True
        
        # 启动 watchdog 文件监控
        if self.config["watchdog_enabled"] and WATCHDOG_AVAILABLE:
            self._start_watchdog()
        
        # 启动定时任务线程
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        print(f"✅ 后台任务调度器已启动，检查间隔: {self.check_interval}秒")
        if self.config["watchdog_enabled"] and WATCHDOG_AVAILABLE:
            print("👁️ 文件变化监控已启用（实时同步）")
    
    def stop(self):
        """停止后台任务"""
        self.running = False
        
        # 停止 watchdog
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        
        if self.thread:
            self.thread.join(timeout=5)
        
        print("⏹️ 后台任务调度器已停止")
    
    def _start_watchdog(self):
        """启动 watchdog 文件监控"""
        handler = FileChangeHandler(self._on_file_changed)
        self._observer = Observer()
        
        # 监控所有 agent 的 sessions 目录
        monitored_count = 0
        for agent_name in self.session_reader.list_agents():
            sessions_dir = self.session_reader.get_sessions_dir(agent_name)
            if sessions_dir and sessions_dir.exists():
                self._observer.schedule(handler, str(sessions_dir), recursive=False)
                print(f"   📂 监控: {agent_name}/sessions/")
                monitored_count += 1
        
        if monitored_count > 0:
            self._observer.start()
            self._file_handler = handler
        else:
            print("⚠️ 没有找到需要监控的目录")
    
    def _on_file_changed(self, agent_name: str, file_path: str):
        """文件变化回调"""
        if not self.running:
            return
        
        try:
            # 同步单个会话文件
            self._sync_session_file(agent_name, file_path)
            self.stats["file_change_count"] += 1
        except Exception as e:
            print(f"⚠️ 文件同步失败: {e}")
    
    def _sync_session_file(self, agent_name: str, file_path: str):
        """同步单个会话文件"""
        try:
            store = MemoryStore(agent_name)
            
            # 读取会话消息
            messages = self.session_reader.read_session_messages(file_path)
            
            if not messages:
                return
            
            # 提取 session_id
            path = Path(file_path)
            filename = path.stem
            if '.reset.' in filename:
                session_id = filename.split('.reset.')[0]
            else:
                session_id = filename
            
            # 导入记忆
            result = store.import_session_messages(messages, session_id)
            imported = result.get("imported", 0)
            
            if imported > 0:
                print(f"✅ [{agent_name}] 实时同步: {imported} 条新记忆")
            
            self.last_sync_times[f"{agent_name}_sync"] = datetime.now()
            
        except Exception as e:
            print(f"❌ [{agent_name}] 会话同步失败: {e}")
            raise
    
    def _run_loop(self):
        """主循环"""
        while self.running:
            try:
                self._check_and_run_tasks()
            except Exception as e:
                print(f"⚠️ 后台任务错误: {e}")
                self.stats["errors"].append({
                    "time": datetime.now().isoformat(),
                    "error": str(e)
                })
            
            time.sleep(self.check_interval)
    
    def _check_and_run_tasks(self):
        """检查并执行任务"""
        now = datetime.now()
        
        for agent_name in self.session_reader.list_agents():
            # 检查是否需要同步
            if self._should_sync(agent_name, now):
                print(f"🔄 [{agent_name}] 定时同步...")
                self._sync_agent(agent_name)
            
            # 检查是否需要提取实体
            if self._should_extract(agent_name, now):
                print(f"🧠 [{agent_name}] 开始提取实体...")
                self._extract_entities(agent_name)
        
        self.stats["last_run"] = now.isoformat()
    
    def _should_sync(self, agent_name: str, now: datetime) -> bool:
        """判断是否需要同步"""
        last_sync = self.last_sync_times.get(f"{agent_name}_sync")
        
        # 从未同步过
        if not last_sync:
            return True
        
        # 超过同步间隔
        interval = timedelta(hours=self.config["sync_interval_hours"])
        if now - last_sync > interval:
            return True
        
        return False
    
    def _should_extract(self, agent_name: str, now: datetime) -> bool:
        """判断是否需要提取实体"""
        last_extract = self.last_sync_times.get(f"{agent_name}_extract")
        
        # 从未提取过
        if not last_extract:
            return False  # 默认不同步时自动提取
        
        # 超过提取间隔
        interval = timedelta(hours=self.config["extract_interval_hours"])
        if now - last_extract > interval:
            return True
        
        # 夜间处理
        if now.hour == self.config["night_process_hour"]:
            return True
        
        return False
    
    def _sync_agent(self, agent_name: str):
        """同步单个Agent"""
        try:
            store = MemoryStore(agent_name)
            sessions = self.session_reader.list_sessions(agent_name)
            
            total_imported = 0
            for session in sessions:
                messages = self.session_reader.read_session_messages(session["file_path"])
                result = store.import_session_messages(messages, session["session_id"])
                total_imported += result.get("imported", 0)
            
            self.last_sync_times[f"{agent_name}_sync"] = datetime.now()
            self.stats["sync_count"] += 1
            
            print(f"✅ [{agent_name}] 同步完成: {total_imported} 条记忆")
            
        except Exception as e:
            print(f"❌ [{agent_name}] 同步失败: {e}")
            raise
    
    def _extract_entities(self, agent_name: str):
        """提取实体"""
        try:
            # 延迟导入避免循环依赖
            from core.evolutionary_kg import EvolutionaryKnowledgeGraph
            
            store = MemoryStore(agent_name)
            memories = store.client.get_memories(limit=100)
            
            if not memories:
                return
            
            ekg = EvolutionaryKnowledgeGraph(agent_name)
            
            total_entities = 0
            total_relations = 0
            
            for memory in memories:
                content = memory.get("content", "")
                if len(content) < 50:
                    continue
                
                try:
                    result = ekg.extract_with_context(content)
                    stored = ekg.store_with_learning(result)
                    total_entities += len(stored.get("entities", []))
                    total_relations += len(stored.get("relations", []))
                except Exception:
                    continue
            
            self.last_sync_times[f"{agent_name}_extract"] = datetime.now()
            self.stats["extract_count"] += 1
            
            print(f"✅ [{agent_name}] 提取完成: {total_entities} 实体, {total_relations} 关系")
            
        except Exception as e:
            print(f"❌ [{agent_name}] 提取失败: {e}")
            raise
    
    def trigger_sync(self, agent_name: str = None):
        """手动触发同步"""
        if agent_name:
            self._sync_agent(agent_name)
        else:
            for agent in self.session_reader.list_agents():
                self._sync_agent(agent)
    
    def trigger_extract(self, agent_name: str = None):
        """手动触发提取"""
        if agent_name:
            self._extract_entities(agent_name)
        else:
            for agent in self.session_reader.list_agents():
                self._extract_entities(agent)
    
    def get_status(self) -> dict:
        """获取状态"""
        return {
            "running": self.running,
            "check_interval": self.check_interval,
            "config": self.config,
            "watchdog_enabled": self.config["watchdog_enabled"] and WATCHDOG_AVAILABLE,
            "last_sync_times": {
                k: v.isoformat() if v else None
                for k, v in self.last_sync_times.items()
            },
            "stats": self.stats
        }
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"📝 配置更新: {key} = {value}")


# 全局实例
scheduler: Optional[BackgroundScheduler] = None


def get_scheduler() -> BackgroundScheduler:
    """获取调度器实例"""
    global scheduler
    if scheduler is None:
        scheduler = BackgroundScheduler()
    return scheduler


def start_scheduler():
    """启动调度器"""
    global scheduler
    scheduler = get_scheduler()
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    # 测试
    print("🧪 测试后台调度器...")
    
    s = BackgroundScheduler(check_interval=60)
    s.config["sync_interval_hours"] = 0.01  # 测试：36秒
    s.config["extract_interval_hours"] = 0.02  # 测试：72秒
    
    print(f"配置: {s.config}")
    print(f"状态: {s.get_status()}")
    
    # 运行一次检查
    s._check_and_run_tasks()