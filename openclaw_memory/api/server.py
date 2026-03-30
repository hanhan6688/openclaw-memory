"""
增强版 API 服务器 - 支持 OpenClaw Agent 集成、实时查询、知识图谱可视化
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from config import PROJECT_ROOT, API_PORT, KG_VIS_PORT
from core.ai_processor import AIProcessor, ai_processor
from memory_store import MemoryStore
from knowledge_graph import KnowledgeGraph
from realtime_sync import SessionFileReader as OpenClawSessionReader, RealtimeSyncService

# 导入新的知识图谱管理器（支持 NetworkX、NebulaGraph）
NETWORKX_AVAILABLE = False
KG_MANAGER_AVAILABLE = False

try:
    from openclaw_memory.core.kg_manager import KnowledgeGraphManager, get_kg_manager, NETWORKX_AVAILABLE
    KG_MANAGER_AVAILABLE = True
    print(f"[DEBUG] 导入成功: NETWORKX_AVAILABLE = {NETWORKX_AVAILABLE}")
except ImportError as e:
    print(f"[DEBUG] openclaw_memory.core.kg_manager 导入失败: {e}")
    try:
        from core.kg_manager import KnowledgeGraphManager, get_kg_manager, NETWORKX_AVAILABLE
        KG_MANAGER_AVAILABLE = True
        print(f"[DEBUG] core.kg_manager 导入成功: NETWORKX_AVAILABLE = {NETWORKX_AVAILABLE}")
    except ImportError as e2:
        print(f"[DEBUG] core.kg_manager 导入失败: {e2}")

# 创建 Flask 应用
app = Flask(__name__, static_folder=str(PROJECT_ROOT / "ui"))
CORS(app)

# 全局实时同步服务
sync_service = None

# 注册电商模块路由
try:
    from commerce.routes import register_commerce_routes
    register_commerce_routes(app)
except ImportError as e:
    print(f"⚠️ 电商模块加载失败: {e}")

# 全局实例
session_reader = OpenClawSessionReader()
memory_stores: dict = {}
knowledge_graphs: dict = {}
_kg_managers: dict = {}  # 新的知识图谱管理器缓存

# 后台调度器
from core.scheduler import get_scheduler, start_scheduler

# 用户画像模块
try:
    from core.profile import get_user_profile
    PROFILE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 用户画像模块加载失败: {e}")
    PROFILE_AVAILABLE = False


def get_memory_store(agent_id: str) -> MemoryStore:
    """获取 MemoryStore 实例"""
    if agent_id not in memory_stores:
        memory_stores[agent_id] = MemoryStore(agent_id)
    return memory_stores[agent_id]


def get_knowledge_graph(agent_id: str) -> KnowledgeGraph:
    """获取 KnowledgeGraph 实例（兼容旧版 Weaviate）"""
    if agent_id not in knowledge_graphs:
        knowledge_graphs[agent_id] = KnowledgeGraph(agent_id)
    return knowledge_graphs[agent_id]


def get_kg_manager(agent_id: str = "main"):
    """获取知识图谱管理器（支持 NebulaGraph 自动切换）"""
    if not KG_MANAGER_AVAILABLE:
        return None
    
    if agent_id not in _kg_managers:
        _kg_managers[agent_id] = KnowledgeGraphManager(agent_id, prefer_nebula=True)
    return _kg_managers[agent_id]


# ==================== 健康检查 ====================

@app.route('/health')
def health():
    """健康检查"""
    ai_health = ai_processor.check_health()
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "ai": ai_health["available"],
        "ai_models": ai_health["models"],
    })


@app.route('/health/ai')
def health_ai():
    """AI 健康检查"""
    health = ai_processor.check_health()
    return jsonify({
        "available": health["available"],
        "models": health["models"],
        "configured_models": {
            "summary": AIProcessor.SUMMARY_MODEL,
            "ner": AIProcessor.NER_MODEL,
            "embed": AIProcessor.EMBED_MODEL,
        }
    })


# ==================== 调度器管理 ====================

@app.route('/scheduler/status')
def scheduler_status():
    """获取调度器状态"""
    scheduler = get_scheduler()
    return jsonify(scheduler.get_status())


@app.route('/scheduler/config', methods=['GET', 'POST'])
def scheduler_config():
    """获取或更新调度器配置"""
    scheduler = get_scheduler()

    if request.method == 'POST':
        data = request.json
        scheduler.update_config(**data)
        return jsonify({"status": "updated", "config": scheduler.config})

    return jsonify(scheduler.config)


@app.route('/scheduler/trigger', methods=['POST'])
def scheduler_trigger():
    """手动触发任务"""
    data = request.json or {}
    task_type = data.get("type", "sync")  # sync 或 extract
    agent_name = data.get("agent")

    scheduler = get_scheduler()

    try:
        if task_type == "sync":
            scheduler.trigger_sync(agent_name)
        elif task_type == "extract":
            scheduler.trigger_extract(agent_name)
        else:
            return jsonify({"error": f"Unknown task type: {task_type}"}), 400

        return jsonify({"status": "triggered", "type": task_type, "agent": agent_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Agent 管理 ====================

@app.route('/agents')
def list_agents():
    """列出所有 Agent"""
    agents = session_reader.list_agents()
    result = []

    for agent_name in agents:
        sessions = session_reader.list_sessions(agent_name)
        last_active = None
        
        # 解析updated_at时间
        for s in sessions:
            updated = s.get("updated_at")
            if updated:
                try:
                    if isinstance(updated, str):
                        ts = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    else:
                        ts = updated
                    if last_active is None or ts > last_active:
                        last_active = ts
                except Exception:
                    pass

        result.append({
            "name": agent_name,
            "session_count": len(sessions),
            "last_active": last_active.isoformat() if last_active else None
        })

    return jsonify(result)


@app.route('/agents/<agent_name>/stats')
def agent_stats(agent_name):
    """获取 Agent 统计"""
    store = get_memory_store(agent_name)
    kg = get_knowledge_graph(agent_name)

    # 确保连接 - 使用 client 属性触发懒加载
    try:
        _ = store.client.client
        _ = kg.client.client
    except Exception:
        pass

    # 检查Weaviate连接状态
    if store.client.client is None or kg.client.client is None:
        return jsonify({
            "total_memories": 0,
            "total_sessions": 0,
            "total_entities": 0,
            "total_relations": 0,
            "earliest_memory": None,
            "latest_memory": None,
            "warning": "Weaviate未连接，无法检索统计数据"
        })

    # 获取记忆统计
    memories = store.client.get_memories(limit=1000)
    kg_data = kg.client.get_kg(limit=1000)

    sessions = set()
    timestamps = []

    for m in memories:
        ts = m.get("timestamp")
        if ts:
            timestamps.append(datetime.fromisoformat(ts.replace("Z", "")))

    # 统计实体和关系
    entities = set()
    relations = 0
    for k in kg_data:
        # 判断是关系还是实体
        if k.get("relation_type") and k.get("target_entity"):
            # 这是关系
            relations += 1
            # 同时记录实体
            entities.add(k.get("entity_name"))
            entities.add(k.get("target_entity"))
        else:
            # 这是实体
            entities.add(k.get("entity_name"))

    return jsonify({
        "total_memories": len(memories),
        "total_sessions": len(sessions),
        "total_entities": len(entities),
        "total_relations": relations,
        "earliest_memory": min(timestamps).isoformat() if timestamps else None,
        "latest_memory": max(timestamps).isoformat() if timestamps else None,
    })


# ==================== 记忆检索 ====================

@app.route('/search', methods=['POST'])
def search_memories():
    """
    搜索记忆
    
    返回 top5 的摘要和原文给 agent
    """
    data = request.json
    query = data.get("query", "")
    agent_name = data.get("agentName", "default")
    limit = data.get("limit", 5)  # 默认返回 top5

    if not query:
        return jsonify({"error": "Query required"}), 400

    store = get_memory_store(agent_name)

    # 使用智能检索
    results = store.recall(query, limit=limit)

    # 返回给 agent 的格式
    return jsonify({
        "memories": [
            {
                "id": m.get("id"),
                "content": m.get("content"),      # 原文（给 agent）
                "summary": m.get("summary"),      # 摘要
                "keywords": m.get("keywords", []),
                "importance": m.get("importance"),
                "timestamp": m.get("timestamp"),
                "score": 1 - (m.get("distance", 0.5))  # 相似度分数
            }
            for m in results
        ],
        "query": query,
        "count": len(results)
    })


@app.route('/memories/<memory_id>', methods=['GET'])
def get_memory_content(memory_id):
    """
    获取记忆原文（回查）
    
    当 agent 需要查看某条记忆的完整内容时使用
    """
    agent_name = request.args.get("agentName", "default")
    store = get_memory_store(agent_name)
    
    memory = store.get_by_id(memory_id)
    
    if not memory:
        return jsonify({"error": "Memory not found"}), 404
    
    return jsonify({
        "id": memory.get("id"),
        "content": memory.get("content"),      # 完整原文
        "summary": memory.get("summary"),
        "keywords": memory.get("keywords", []),
        "importance": memory.get("importance"),
        "timestamp": memory.get("timestamp"),
        "source": memory.get("source"),
        "tags": memory.get("tags", [])
    })


@app.route('/recall', methods=['POST'])
def recall_for_agent():
    """
    为 Agent 检索记忆
    
    专门给聊天 agent 使用的接口，返回格式更友好
    """
    data = request.json
    query = data.get("query", "")
    agent_name = data.get("agentName", "default")
    limit = data.get("limit", 5)

    if not query:
        return jsonify({"error": "Query required"}), 400

    store = get_memory_store(agent_name)
    
    # 使用 recall_for_agent 方法
    result = store.recall_for_agent(query, limit)

    return jsonify(result)


# ==================== 高级检索 API ====================

@app.route('/agents/<agent_name>/memories/search', methods=['GET'])
def advanced_search(agent_name):
    """
    高级检索 - 混合检索引擎
    
    Query Parameters:
        q: 查询文本
        mode: 检索模式 (vector|bm25|hybrid|auto)
        limit: 返回数量
        include_entities: 是否包含相关实体 (default: true)
    
    Returns:
        {
            "results": [...],
            "entities": [...],
            "mode": "hybrid",
            "count": 10
        }
    """
    query = request.args.get("q", "")
    mode = request.args.get("mode", "hybrid")
    limit = int(request.args.get("limit", 5))  # 默认返回前5条
    include_entities = request.args.get("include_entities", "true").lower() == "true"
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    try:
        # 使用混合检索引擎
        from core.hybrid_recall import get_recall_engine
        
        engine = get_recall_engine(agent_name)
        result = engine.recall(
            query=query,
            limit=limit,
            include_entities=include_entities
        )
        
        return jsonify({
            "success": True,
            "query": query,
            "mode": "hybrid",
            "count": len(result["memories"]),
            "results": result["memories"],
            "entities": result["entities"]
        })
    except Exception as e:
        # 降级到基础检索
        store = get_memory_store(agent_name)
        results = store.search(query, limit=limit, mode="vector")
        
        return jsonify({
            "success": True,
            "query": query,
            "mode": "vector (fallback)",
            "count": len(results),
            "results": results,
            "entities": [],
            "warning": f"混合检索失败，降级到向量搜索: {str(e)}"
        })


@app.route('/agents/<agent_name>/memories/search/intent', methods=['GET'])
def search_with_intent(agent_name):
    """
    带意图识别的检索
    
    自动识别查询意图，选择最佳检索策略
    
    Query Parameters:
        q: 查询文本
        limit: 返回数量
    
    Returns:
        {
            "intent": "fuzzy|exact|time|general",
            "mode": "vector|bm25|hybrid|time",
            "confidence": 0.8,
            "reason": "包含模糊关键词",
            "results": [...]
        }
    """
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 5))
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    store = get_memory_store(agent_name)
    
    try:
        result = store.search_with_intent(query, limit=limit)
        
        return jsonify({
            "success": True,
            "query": query,
            "intent": result["intent"],
            "mode": result["mode"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "time_hint": result.get("time_hint"),
            "count": result["count"],
            "results": result["results"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/memories/search/time', methods=['GET'])
def search_by_time(agent_name):
    """
    按时间范围检索
    
    Query Parameters:
        q: 查询文本（可选）
        range: 时间范围 (today|yesterday|week|month)
        limit: 返回数量
    
    Returns:
        {
            "results": [...],
            "time_range": "week"
        }
    """
    query = request.args.get("q", "")
    time_range = request.args.get("range", "week")
    limit = int(request.args.get("limit", 5))
    
    store = get_memory_store(agent_name)
    
    try:
        results = store.search_by_time(query, time_range=time_range, limit=limit)
        
        return jsonify({
            "success": True,
            "query": query,
            "time_range": time_range,
            "count": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/memories/time-decay', methods=['GET'])
def get_time_decay_info(agent_name):
    """
    获取时间衰减信息
    
    展示不同时间段的记忆权重
    """
    from core.hybrid_retrieval import TimeDecay
    
    decay = TimeDecay()
    
    return jsonify({
        "success": True,
        "config": {
            "recent_boost": decay.recent_boost,
            "decay_rate": decay.decay_rate,
            "half_life_days": decay.half_life_days
        },
        "weights": {
            "今天": f"{decay.get_boost_factor(0):.2f}x",
            "昨天": f"{decay.get_boost_factor(1):.2f}x",
            "3天前": f"{decay.get_boost_factor(3):.2f}x",
            "7天前": f"{decay.get_boost_factor(7):.2f}x",
            "14天前": f"{decay.get_boost_factor(14):.2f}x",
            "30天前": f"{decay.get_boost_factor(30):.2f}x"
        }
    })


@app.route('/agents/<agent_name>/memories/recent')
def recent_memories(agent_name):
    """获取最近记忆"""
    limit = request.args.get("limit", 5, type=int)
    store = get_memory_store(agent_name)
    
    # 确保客户端已连接
    if store.client.client is None:
        return jsonify([])
    
    results = store.client.get_memories(limit=limit)

    return jsonify([
        {
            "id": m.get("_additional", {}).get("id"),
            "content": m.get("content"),
            "summary": m.get("summary"),
            "timestamp": m.get("timestamp"),
            "importance": m.get("importance"),
            "entities": m.get("entities", []),
        }
        for m in results
    ])


@app.route('/agents/<agent_name>/memories/time')
def memories_by_time(agent_name):
    """按时间范围获取记忆"""
    range_str = request.args.get("range", "week")
    start_param = request.args.get("start")
    end_param = request.args.get("end")
    include_raw = request.args.get("includeRaw", "true") == "true"

    store = get_memory_store(agent_name)

    # 检查Weaviate连接状态
    if store.client.client is None:
        return jsonify({
            "memories": [],
            "rawMessages": None,
            "timeRange": {
                "start": (datetime.now() - timedelta(days=7)).isoformat(),
                "end": datetime.now().isoformat(),
            },
            "warning": "Weaviate未连接，无法检索记忆数据"
        })

    # 解析时间范围
    now = datetime.now()
    
    # 支持具体的日期参数
    if start_param:
        try:
            start = datetime.fromisoformat(start_param)
            # 设置为当天的开始时间
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            start = now - timedelta(days=7)
    else:
        if range_str == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif range_str == "yesterday":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif range_str == "week":
            start = now - timedelta(weeks=1)
        elif range_str == "lastweek":
            start = now - timedelta(weeks=2)
            now = now - timedelta(weeks=1)
        elif range_str == "month":
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(days=7)

    # 支持结束日期参数
    if end_param:
        try:
            end = datetime.fromisoformat(end_param)
            # 设置为当天的结束时间
            end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
            now = end  # 使用 end 作为上限
        except Exception:
            pass

    # 获取并过滤
    all_memories = store.client.get_memories(limit=500)

    results = []
    raw_messages = []

    for m in all_memories:
        ts_str = m.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                if start <= ts <= now:
                    results.append({
                        "id": m.get("_additional", {}).get("id"),
                        "content": m.get("content"),
                        "summary": m.get("summary"),
                        "timestamp": ts_str,
                        "importance": m.get("importance"),
                        "entities": m.get("entities", []),
                    })

                    if include_raw and m.get("raw_conversation"):
                        try:
                            raw_messages.extend(json.loads(m["raw_conversation"]))
                        except Exception:
                            pass
            except Exception:
                pass

    # 按时间戳倒序排序（最新的在前）
    results.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    
    return jsonify({
        "memories": results,
        "rawMessages": raw_messages if raw_messages else None,
        "timeRange": {
            "start": start.isoformat(),
            "end": now.isoformat(),
        }
    })


@app.route('/memories/<memory_id>')
def get_memory(memory_id):
    """获取单个记忆详情"""
    # 需要从所有 agent 中查找
    for agent_name in session_reader.list_agents():
        try:
            store = get_memory_store(agent_name)
            memories = store.client.get_memories(limit=1000)
            
            for m in memories:
                if m.get("_additional", {}).get("id") == memory_id:
                    raw_messages = None
                    if m.get("raw_conversation"):
                        try:
                            raw_messages = json.loads(m["raw_conversation"])
                        except Exception:
                            pass
                    
                    return jsonify({
                        "memory": {
                            "id": memory_id,
                            "content": m.get("content"),
                            "summary": m.get("summary"),
                            "timestamp": m.get("timestamp"),
                            "importance": m.get("importance"),
                            "entities": m.get("entities", []),
                        },
                        "rawMessages": raw_messages
                    })
        except Exception:
            pass
    
    return jsonify({"error": "Memory not found"}), 404


# ==================== 会话列表 ====================

@app.route('/agents/<agent_name>/sessions')
def list_sessions(agent_name):
    """获取 Agent 的会话列表（类似 ChatGPT 左侧历史记录）"""
    sessions = session_reader.list_sessions(agent_name)

    result = []
    for session in sessions:
        # 使用 unique_id 作为前端显示的 id
        unique_id = session.get("id", session.get("session_id", ""))
        session_id = session.get("session_id", "")
        file_path = session.get("file_path", "")
        is_reset = session.get("is_reset", False)

        # 获取会话基本信息
        messages = session_reader.read_session_messages(file_path)

        # 获取第一条用户消息作为标题
        title = "新对话"
        for msg in messages:
            if msg.get("role") == "user":
                title = msg.get("content", "")[:50]
                if len(msg.get("content", "")) > 50:
                    title += "..."
                break

        # 如果是 reset 文件，添加日期标记
        if is_reset:
            # 从文件名提取日期
            if '.reset.' in file_path:
                reset_date = file_path.split('.reset.')[-1][:10]
                title = f"[{reset_date}] {title}"

        # 处理updated_at - 可能是字符串或datetime
        updated_at = session.get("updated_at")
        if updated_at:
            if isinstance(updated_at, str):
                updated_at_str = updated_at
            else:
                updated_at_str = updated_at.isoformat()
        else:
            updated_at_str = None

        result.append({
            "id": unique_id,  # 使用 unique_id
            "session_id": session_id,  # 原始 session_id
            "title": title,
            "updatedAt": updated_at_str,
            "messageCount": len(messages),
            "filePath": file_path,
            "isReset": is_reset,
        })

    # 按更新时间倒序
    result.sort(key=lambda x: x.get("updatedAt") or "", reverse=True)

    return jsonify(result)


@app.route('/agents/<agent_name>/sessions/sync', methods=['POST'])
def sync_session_to_memory(agent_name):
    """手动同步会话到记忆系统，并从原始消息提取实体"""
    session_id = request.json.get("sessionId")
    extract_entities = request.json.get("extractEntities", True)  # 默认提取实体

    if session_id:
        # 同步单个会话
        sessions = session_reader.list_sessions(agent_name)
        # 支持通过 unique_id 或 session_id 查找
        session = next((s for s in sessions if s.get("id") == session_id or s["session_id"] == session_id), None)

        if not session:
            return jsonify({"error": "Session not found"}), 404

        messages = session_reader.read_session_messages(session["file_path"])
        store = get_memory_store(agent_name)

        result = store.import_session_messages(messages, session["session_id"])

        # 从原始消息提取实体
        entities_extracted = 0
        relations_extracted = 0
        if extract_entities and messages:
            extraction_result = extract_entities_from_messages(agent_name, messages)
            entities_extracted = extraction_result.get("entities", 0)
            relations_extracted = extraction_result.get("relations", 0)

        return jsonify({
            **result,
            "entitiesExtracted": entities_extracted,
            "relationsExtracted": relations_extracted
        })
    else:
        # 同步所有会话
        sessions = session_reader.list_sessions(agent_name)
        store = get_memory_store(agent_name)

        total_imported = 0
        total_entities = 0
        total_relations = 0

        for session in sessions:
            messages = session_reader.read_session_messages(session["file_path"])
            result = store.import_session_messages(messages, session["session_id"])
            total_imported += result.get("imported", 0)

            # 从原始消息提取实体
            if extract_entities and messages:
                extraction_result = extract_entities_from_messages(agent_name, messages)
                total_entities += extraction_result.get("entities", 0)
                total_relations += extraction_result.get("relations", 0)

        return jsonify({
            "status": "success",
            "sessionsProcessed": len(sessions),
            "totalImported": total_imported,
            "entitiesExtracted": total_entities,
            "relationsExtracted": total_relations
        })


def extract_entities_from_messages(agent_name: str, messages: list, batch_size: int = 5) -> dict:
    """从 user 和 assistant 的对话中提取实体和关系"""
    from core.evolutionary_kg import EvolutionaryKnowledgeGraph

    try:
        ekg = get_evolutionary_kg(agent_name)
    except Exception:
        return {"entities": 0, "relations": 0}

    # 过滤消息，只保留 user 和 assistant 的有效对话
    def should_skip_message(msg):
        """跳过工具调用等非对话消息"""
        content = msg.get("content", "")
        if not content:
            return True
        if isinstance(content, list):
            # 检查是否只有 toolCall 或 toolResult
            has_text = any(c.get("type") == "text" for c in content if isinstance(c, dict))
            if not has_text:
                return True
        return False

    def extract_text_content(content):
        """提取文本内容"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text", ""))
            return " ".join(texts)
        return ""

    filtered_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ["user", "assistant"]:
            continue
        if should_skip_message(msg):
            continue

        # 提取文本内容
        content = extract_text_content(msg.get("content", ""))
        if content and len(content) > 10:
            filtered_messages.append({"role": role, "content": content})

    if not filtered_messages:
        return {"entities": 0, "relations": 0}

    # 合并消息为文本块（user-assistant 对话对）
    text_blocks = []
    current_block = []

    for msg in filtered_messages:
        role = msg["role"]
        content = msg["content"][:500]  # 限制单条消息长度

        current_block.append(f"{role}: {content}")

        # 每个 user-assistant 对作为一个块
        if role == "assistant" and current_block:
            text_blocks.append("\n".join(current_block))
            current_block = []

    # 保存最后一批
    if current_block:
        text_blocks.append("\n".join(current_block))

    total_entities = 0
    total_relations = 0

    for text in text_blocks:
        if len(text) < 30:  # 跳过太短的文本
            continue

        try:
            result = ekg.extract_with_context(text)
            stored = ekg.store_with_learning(result)
            total_entities += len(stored.get("entities", []))
            total_relations += len(stored.get("relations", []))
        except Exception as e:
            print(f"实体提取错误: {e}")
            continue

    return {
        "entities": total_entities,
        "relations": total_relations
    }


@app.route('/sync/all', methods=['POST'])
def sync_all_agents():
    """同步所有 Agent 的会话到记忆系统"""
    agents = session_reader.list_agents()

    results = {}
    for agent_name in agents:
        sessions = session_reader.list_sessions(agent_name)
        store = get_memory_store(agent_name)

        total_imported = 0
        for session in sessions:
            messages = session_reader.read_session_messages(session["file_path"])
            result = store.import_session_messages(messages, session["session_id"])
            total_imported += result.get("imported", 0)

        results[agent_name] = {
            "sessions": len(sessions),
            "imported": total_imported
        }

    return jsonify(results)


@app.route('/agents/<agent_name>/kg/clear', methods=['DELETE'])
def clear_knowledge_graph(agent_name):
    """
    清空指定 Agent 的知识图谱

    删除所有实体和关系，可以重新构建
    """
    store = get_memory_store(agent_name)

    if store.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 500

    try:
        # 获取删除前的统计
        before_entities = store.client.count_entities()
        before_relations = store.client.count_relations()

        # 删除所有实体
        entities_deleted = store.client.delete_all_entities()

        # 删除所有关系
        relations_deleted = store.client.delete_all_relations()

        return jsonify({
            "success": True,
            "before": {
                "entities": before_entities,
                "relations": before_relations
            },
            "deleted": {
                "entities": entities_deleted,
                "relations": relations_deleted
            },
            "message": f"已清空知识图谱：删除 {entities_deleted} 个实体，{relations_deleted} 条关系"
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/agents/<agent_name>/kg/extract-from-memories', methods=['POST'])
def extract_from_memories(agent_name):
    """从现有记忆中提取实体和关系（后台任务）"""
    limit = request.json.get("limit", 50)  # 处理的记忆数量
    batch_size = request.json.get("batchSize", 5)  # 每批处理的消息数

    store = get_memory_store(agent_name)

    # 检查Weaviate连接
    if store.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 500

    # 获取记忆
    memories = store.client.get_memories(limit=limit)

    if not memories:
        return jsonify({
            "status": "no_memories",
            "message": "没有记忆可处理"
        })

    # 批量提取
    total_entities = 0
    total_relations = 0
    processed = 0

    for memory in memories:
        content = memory.get("content", "")
        if not content or len(content) < 50:
            continue

        try:
            extraction_result = extract_entities_from_messages(
                agent_name,
                [{"role": "user", "content": content}],
                batch_size=1
            )
            total_entities += extraction_result.get("entities", 0)
            total_relations += extraction_result.get("relations", 0)
            processed += 1
        except Exception as e:
            print(f"处理记忆错误: {e}")
            continue

    return jsonify({
        "status": "success",
        "memoriesProcessed": processed,
        "entitiesExtracted": total_entities,
        "relationsExtracted": total_relations
    })


# ==================== Workspace 解析 ====================

from core.workspace_parser import WorkspaceParser

workspace_parser = WorkspaceParser()


@app.route('/agents/<agent_name>/workspace/files')
def list_workspace_files(agent_name):
    """列出 workspace 中的文件"""
    summary = workspace_parser.get_file_summary(agent_name)
    return jsonify({
        "agent": agent_name,
        "workspace_path": workspace_parser.get_workspace_path(agent_name),
        "files": summary,
        "total": len(summary)
    })


@app.route('/agents/<agent_name>/workspace/parse', methods=['POST'])
def parse_workspace(agent_name):
    """解析 workspace 文件并提取实体"""
    max_files = request.json.get("maxFiles", 10)
    store_to_kg = request.json.get("storeToKg", True)

    result = workspace_parser.parse_workspace(agent_name, max_files)

    if not result["workspace_found"]:
        return jsonify({
            "status": "not_found",
            "message": f"Agent {agent_name} 的 workspace 未找到"
        })

    # 存储到知识图谱
    if store_to_kg and result["entities"]:
        try:
            from core.evolutionary_kg import EvolutionaryKnowledgeGraph
            ekg = get_evolutionary_kg(agent_name)

            stored_entities = 0
            stored_relations = 0

            for entity in result["entities"]:
                try:
                    ekg._store_entity(
                        entity["name"],
                        entity["type"],
                        source=f"workspace:{entity.get('file', 'unknown')}"
                    )
                    stored_entities += 1
                except Exception:
                    pass

            for relation in result["relations"]:
                try:
                    ekg._store_relation(
                        relation["source"],
                        relation["relation"],
                        relation["target"],
                        source=f"workspace:{relation.get('file', 'unknown')}"
                    )
                    stored_relations += 1
                except Exception:
                    pass

            result["stored_entities"] = stored_entities
            result["stored_relations"] = stored_relations

        except Exception as e:
            result["store_error"] = str(e)

    result["status"] = "success"
    return jsonify(result)


# ==================== Agent 配置文件 API ====================

# 配置文件定义
CONFIG_FILES = {
    "USER.md": {
        "title": "用户画像",
        "description": "我服务的这个人是谁？记录用户的姓名、偏好、项目、习惯等信息，帮助 Agent 更好地理解和帮助用户。",
        "icon": "👤"
    },
    "SOUL.md": {
        "title": "人格价值观",
        "description": "我是谁，我怎么做人？定义 Agent 的核心原则、边界、风格和持续学习的方式。",
        "icon": "🧠"
    },
    "IDENTITY.md": {
        "title": "身份标识",
        "description": "我叫啥，长啥样？定义 Agent 的名称、形象、头像等身份信息。",
        "icon": "🎭"
    }
}


@app.route('/agents/<agent_name>/config/files')
def get_config_files(agent_name):
    """获取 agent 的配置文件列表和内容"""
    workspace_path = workspace_parser.get_workspace_path(agent_name)

    if not workspace_path:
        return jsonify({
            "error": "workspace_not_found",
            "message": f"Agent {agent_name} 的 workspace 未找到"
        }), 404

    files = []
    for filename, meta in CONFIG_FILES.items():
        file_path = os.path.join(workspace_path, filename)
        file_info = {
            "filename": filename,
            "title": meta["title"],
            "description": meta["description"],
            "icon": meta["icon"],
            "exists": os.path.exists(file_path)
        }

        if file_info["exists"]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_info["content"] = f.read()
                file_info["size"] = os.path.getsize(file_path)
                file_info["modified"] = datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).isoformat()
            except Exception as e:
                file_info["error"] = str(e)
                file_info["content"] = ""
        else:
            file_info["content"] = ""
            file_info["size"] = 0

        files.append(file_info)

    return jsonify({
        "agent": agent_name,
        "workspace_path": workspace_path,
        "files": files
    })


@app.route('/agents/<agent_name>/config/files/<filename>', methods=['GET'])
def get_config_file(agent_name, filename):
    """获取单个配置文件内容"""
    if filename not in CONFIG_FILES:
        return jsonify({"error": "invalid_file", "message": f"不支持的配置文件: {filename}"}), 400

    workspace_path = workspace_parser.get_workspace_path(agent_name)
    if not workspace_path:
        return jsonify({"error": "workspace_not_found"}), 404

    file_path = os.path.join(workspace_path, filename)

    if not os.path.exists(file_path):
        return jsonify({
            "filename": filename,
            "content": "",
            "exists": False,
            "title": CONFIG_FILES[filename]["title"],
            "description": CONFIG_FILES[filename]["description"],
            "icon": CONFIG_FILES[filename]["icon"]
        })

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({
            "filename": filename,
            "content": content,
            "exists": True,
            "title": CONFIG_FILES[filename]["title"],
            "description": CONFIG_FILES[filename]["description"],
            "icon": CONFIG_FILES[filename]["icon"],
            "size": os.path.getsize(file_path),
            "modified": datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/config/files/<filename>', methods=['POST'])
def save_config_file(agent_name, filename):
    """保存配置文件"""
    if filename not in CONFIG_FILES:
        return jsonify({"error": "invalid_file", "message": f"不支持的配置文件: {filename}"}), 400

    workspace_path = workspace_parser.get_workspace_path(agent_name)
    if not workspace_path:
        return jsonify({"error": "workspace_not_found"}), 404

    # 确保 workspace 目录存在
    os.makedirs(workspace_path, exist_ok=True)

    file_path = os.path.join(workspace_path, filename)
    content = request.json.get("content", "")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return jsonify({
            "success": True,
            "filename": filename,
            "size": len(content),
            "message": f"{CONFIG_FILES[filename]['title']} 保存成功"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/config/files/<filename>', methods=['DELETE'])
def delete_config_file(agent_name, filename):
    """删除配置文件（重置为空）"""
    if filename not in CONFIG_FILES:
        return jsonify({"error": "invalid_file"}), 400

    workspace_path = workspace_parser.get_workspace_path(agent_name)
    if not workspace_path:
        return jsonify({"error": "workspace_not_found"}), 404

    file_path = os.path.join(workspace_path, filename)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({"success": True, "message": "文件已删除"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "message": "文件不存在"})


# ==================== 会话回放 ====================

@app.route('/agents/<agent_name>/sessions/<session_id>/messages')
def session_messages(agent_name, session_id):
    """
    获取会话消息
    
    优先级：
    1. 从数据库读取（content 字段，JSON 丢失时也能工作）
    2. 从 JSON 文件读取（备用）
    """
    # 1. 尝试从数据库读取
    store = get_memory_store(agent_name)
    try:
        _ = store.client.client  # 确保连接
        
        # 获取所有记忆，按 session_id 过滤
        all_memories = store.client.get_memories(limit=5000)
        session_memories = [m for m in all_memories if m.get('session_id') == session_id]
        
        if session_memories:
            # 从数据库读取成功
            messages = []
            seen_contents = set()  # 去重
            
            for m in session_memories:
                content = m.get('content', '')
                role = m.get('role', 'user')
                timestamp = m.get('timestamp', '')
                
                # 解析 content（可能是 "user: xxx" 或 "assistant: xxx" 格式）
                if content.startswith('user: '):
                    role = 'user'
                    content = content[6:]
                elif content.startswith('assistant: '):
                    role = 'assistant'
                    content = content[11:]
                
                # 去重：基于内容前 100 字符
                content_key = f"{role}:{content[:100]}"
                if content_key in seen_contents:
                    continue
                seen_contents.add(content_key)
                
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    "summary": m.get('summary', ''),
                    "quality": m.get('quality', ''),
                    "importance": m.get('importance', 0.5)
                })
            
            # 按时间倒序排序（最新的在最上面）
            messages.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return jsonify({
                "sessionId": session_id,
                "source": "database",
                "messageCount": len(messages),
                "messages": messages
            })
    except Exception as e:
        print(f"从数据库读取会话失败: {e}")
    
    # 2. 从 JSON 文件读取（备用）
    sessions = session_reader.list_sessions(agent_name)
    session = next((s for s in sessions if s.get("id") == session_id or s["session_id"] == session_id), None)

    if not session:
        return jsonify({"error": "Session not found", "message": "会话不存在且数据库中无记录"}), 404

    messages = session_reader.read_session_messages(session["file_path"])

    # 过滤无效消息
    def should_skip_message(msg):
        role = msg.get("role", "")
        if role not in ["user", "assistant"]:
            return True
        content = msg.get("content", "")
        if not content:
            return True
        if isinstance(content, list):
            has_text = any(isinstance(part, dict) and part.get("type") == "text" for part in content)
            if not has_text:
                return True
        return False

    def extract_text_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if text:
                        texts.append(text)
            return "\n".join(texts)
        return ""

    filtered_messages = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")

        if role not in ["user", "assistant"]:
            continue
        if should_skip_message(m):
            continue

        clean_content = extract_text_content(content)
        if not clean_content or len(clean_content.strip()) < 5:
            continue

        filtered_messages.append({
            "role": role,
            "content": clean_content,
            "timestamp": m.get("timestamp", "")
        })

    # 按时间倒序排序（最新的在最上面）
    filtered_messages.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    return jsonify({
        "sessionId": session_id,
        "source": "json_file",
        "filePath": session["file_path"],
        "messageCount": len(filtered_messages),
        "messages": filtered_messages
    })


# ==================== 智能记忆保存 API ====================

def assess_message_quality(content: str, role: str) -> dict:
    """
    评估消息质量，判断是否值得保存

    返回: {
        "quality": "high/medium/low",
        "score": 0.0-1.0,
        "reason": "原因",
        "should_save": True/False
    }
    """
    if not content or len(content.strip()) < 10:
        return {"quality": "low", "score": 0.1, "reason": "内容过短", "should_save": False}

    content = content.strip()
    length = len(content)

    # 低质量模式
    low_quality_patterns = [
        r'^好的?[，。！]?$',           # 单纯的"好的"
        r'^明白[，。！]?$',            # 单纯的"明白"
        r'^收到[，。！]?$',            # 单纯的"收到"
        r'^可以[，。！]?$',            # 单纯的"可以"
        r'^没问题[，。！]?$',          # 单纯的"没问题"
        r'^谢谢[，。！]?$',            # 单纯的"谢谢"
        r'^嗯[，。！]?$',              # 单纯的"嗯"
        r'^是[，。！]?$',              # 单纯的"是"
        r'^对[，。！]?$',              # 单纯的"对"
        r'^行[，。！]?$',              # 单纯的"行"
        r'^OK[，。！]?$',              # OK
        r'^ok[，。！]?$',              # ok
        r'^好的，我(现在|马上|这就)',  # 简单确认后开始行动
    ]

    import re
    for pattern in low_quality_patterns:
        if re.match(pattern, content, re.IGNORECASE):
            return {"quality": "low", "score": 0.2, "reason": "简单确认，无实质内容", "should_save": False}

    # 检查是否主要是代码/命令输出
    code_blocks = len(re.findall(r'```', content))
    if code_blocks > 3 and len(re.sub(r'```[\s\S]*?```', '', content).strip()) < 50:
        return {"quality": "low", "score": 0.3, "reason": "主要是代码输出", "should_save": False}

    # 高质量指标
    high_quality_indicators = [
        r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}',  # 日期
        r'\d+[%％]',                          # 百分比
        r'\d+[万千百十亿]',                   # 数字+单位
        r'https?://[^\s]+',                   # URL
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+',  # 邮箱
        r'项目|任务|需求|问题|方案|计划',       # 关键词
        r'决定|确认|同意|修改|添加|删除',       # 动作词
        r'因为|所以|但是|如果|那么|首先|然后',  # 逻辑词
    ]

    high_count = sum(1 for p in high_quality_indicators if re.search(p, content))

    # 计算分数
    score = 0.5  # 基础分

    # 长度加分
    if length > 100:
        score += 0.1
    if length > 300:
        score += 0.1
    if length > 500:
        score += 0.1

    # 高质量指标加分
    score += high_count * 0.05

    # user 消息通常更重要
    if role == "user":
        score += 0.1

    # 限制分数范围
    score = min(1.0, max(0.1, score))

    # 判断质量等级
    if score >= 0.7:
        return {"quality": "high", "score": score, "reason": "包含关键信息", "should_save": True}
    elif score >= 0.4:
        return {"quality": "medium", "score": score, "reason": "有一定价值", "should_save": True}
    else:
        return {"quality": "low", "score": score, "reason": "信息量不足", "should_save": False}


@app.route('/agents/<agent_name>/memories/save-message', methods=['POST'])
def save_message_to_memory(agent_name):
    """
    保存单条消息到记忆

    流程：
    1. 评估消息质量
    2. 生成摘要（用于向量检索）
    3. 存储原文 + 摘要
    """
    data = request.json
    content = data.get("content", "")
    role = data.get("role", "user")
    timestamp = data.get("timestamp", "")
    session_id = data.get("session_id", "")

    if not content:
        return jsonify({"success": False, "error": "内容为空"})

    # 1. 评估质量
    quality_info = assess_message_quality(content, role)

    if not quality_info["should_save"]:
        return jsonify({
            "skipped": True,
            "reason": quality_info["reason"],
            "quality": quality_info["quality"]
        })

    # 2. 生成摘要
    try:
        from core.summarizer import get_summarizer
        summarizer = get_summarizer()
        extracted = summarizer.clean_and_extract(content)
        summary = extracted.get("summary", "")
        importance = extracted.get("importance", quality_info["score"])
    except Exception as e:
        # 降级：使用原文前150字
        summary = content[:150] if len(content) > 150 else content
        importance = quality_info["score"]

    if not summary:
        summary = content[:150]

    # 3. 存储到记忆
    store = get_memory_store(agent_name)

    # 构建记忆数据
    memory_data = {
        "content": content,           # 原文（返回给AI）
        "summary": summary,           # 摘要（用于检索）
        "importance": importance,
        "memory_type": "conversation",
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "session_id": session_id,
        "role": role,
        "quality": quality_info["quality"]
    }

    try:
        memory_id = store.store(content, memory_data)
        return jsonify({
            "success": True,
            "memory_id": memory_id,
            "summary": summary,
            "quality": quality_info["quality"],
            "importance": round(importance, 2)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/agents/<agent_name>/sessions/<session_id>/save-to-memory', methods=['POST'])
def save_session_to_memory(agent_name, session_id):
    """
    批量保存会话消息到记忆

    智能过滤低质量消息，只保存有价值的内容
    """
    # 获取会话消息
    sessions = session_reader.list_sessions(agent_name)
    session = next((s for s in sessions if s.get("id") == session_id or s["session_id"] == session_id), None)

    if not session:
        return jsonify({"success": False, "error": "会话不存在"})

    messages = session_reader.read_session_messages(session["file_path"])

    if not messages:
        return jsonify({"success": False, "error": "会话无消息"})

    results = []
    saved_count = 0
    skipped_count = 0

    for idx, msg in enumerate(messages):
        content = msg.get("content", "")
        role = msg.get("role", "user")
        timestamp = msg.get("timestamp", "")

        # 评估质量
        quality_info = assess_message_quality(content, role)

        if not quality_info["should_save"]:
            results.append({
                "idx": idx,
                "skipped": True,
                "reason": quality_info["reason"]
            })
            skipped_count += 1
            continue

        # 生成摘要并保存
        try:
            from core.summarizer import get_summarizer
            summarizer = get_summarizer()
            extracted = summarizer.clean_and_extract(content)
            summary = extracted.get("summary", content[:150])
            importance = extracted.get("importance", quality_info["score"])
        except Exception:
            summary = content[:150]
            importance = quality_info["score"]

        # 确保时间戳格式正确
        if timestamp and not timestamp.endswith('Z'):
            timestamp = timestamp + 'Z' if '+' not in timestamp else timestamp.replace('+00:00', 'Z')

        # 存储
        store = get_memory_store(agent_name)
        memory_data = {
            "content": content,
            "summary": summary,
            "importance": importance,
            "memory_type": "conversation",
            "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "session_id": session_id,
            "role": role,
            "quality": quality_info["quality"]
        }

        try:
            memory_id = store.store(content, memory_data)
            results.append({
                "idx": idx,
                "saved": True,
                "memory_id": memory_id,
                "quality": quality_info["quality"]
            })
            saved_count += 1
        except Exception as e:
            results.append({
                "idx": idx,
                "error": str(e)
            })

    return jsonify({
        "success": True,
        "total": len(messages),
        "saved": saved_count,
        "skipped": skipped_count,
        "results": results
    })




# ==================== Importance 相关 API ====================

@app.route('/agents/<agent_name>/memories/importance')
def get_memories_by_importance(agent_name):
    """获取高 importance 的记忆"""
    min_importance = request.args.get("minImportance", 0.7, type=float)
    limit = request.args.get("limit", 50, type=int)
    
    store = get_memory_store(agent_name)
    
    if store.client.client is None:
        return jsonify({
            "memories": [],
            "warning": "Weaviate未连接"
        })
    
    try:
        memories = store.client.get_memories_by_importance(min_importance, limit)
        return jsonify({
            "memories": memories,
            "minImportance": min_importance,
            "count": len(memories)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/memories/cleanup', methods=['POST'])
def cleanup_memories(agent_name):
    """清理低 importance 的记忆"""
    data = request.json or {}
    threshold = data.get("threshold", 0.3)
    keep_recent = data.get("keepRecent", 100)
    
    store = get_memory_store(agent_name)
    
    if store.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        result = store.client.cleanup_low_importance_memories(threshold, keep_recent)
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/memories/<memory_id>/importance', methods=['PUT'])
def update_memory_importance(agent_name, memory_id):
    """更新记忆的 importance"""
    data = request.json
    importance = data.get("importance")
    
    if importance is None:
        return jsonify({"error": "importance required"}), 400
    
    if not 0 <= importance <= 1:
        return jsonify({"error": "importance must be between 0 and 1"}), 400
    
    store = get_memory_store(agent_name)
    
    if store.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        # Weaviate 需要先获取再更新
        collection = store.client.client.collections.get(store.client.memory_collection)
        # 更新属性
        collection.data.update(
            uuid=memory_id,
            properties={"importance": importance}
        )
        return jsonify({
            "success": True,
            "memoryId": memory_id,
            "importance": importance
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/memories/<memory_id>', methods=['DELETE'])
def delete_memory(agent_name, memory_id):
    """删除单个记忆"""
    store = get_memory_store(agent_name)
    
    if store.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        success = store.client.delete_memory_by_id(memory_id)
        if success:
            return jsonify({"success": True, "message": "记忆已删除"})
        else:
            return jsonify({"error": "删除失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 知识图谱 ====================

@app.route('/agents/<agent_name>/graph')
def knowledge_graph(agent_name):
    """获取知识图谱（支持 NetworkX、NebulaGraph 自动切换）"""
    limit = request.args.get("limit", 100, type=int)
    
    # 优先使用 NetworkX 后端
    if KG_MANAGER_AVAILABLE and NETWORKX_AVAILABLE:
        from core.networkx_kg_client import get_nx_client
        
        nx_client = get_nx_client(agent_name)
        stats = nx_client.get_stats()
        
        # 构建节点和边
        nodes = {}
        edges = []
        
        # 获取所有实体
        entities = nx_client.get_all_entities(limit=limit)
        for entity in entities:
            nodes[entity['id']] = {
                "id": entity['id'],
                "name": entity['name'],
                "type": entity['type'],
                "mentionCount": entity.get('access_count', 1),
                "size": min(50, 10 + (entity.get('access_count', 1) * 3)),
                "color": _get_entity_color(entity['type']),
            }
        
        # 获取所有关系
        relations = nx_client.get_all_relations(limit=limit * 2)
        for rel in relations:
            edges.append({
                "source": rel['source_id'],
                "target": rel['target_id'],
                "type": rel['relation'],
                "label": rel['relation'],
                "weight": rel.get('confidence', 0.5)
            })
        
        return jsonify({
            "nodes": list(nodes.values()),
            "links": edges,
            "stats": stats,
            "backend": "NetworkX"
        })
    
    # 回退到 Weaviate
    kg = get_knowledge_graph(agent_name)
    if kg.client.client is None:
        return jsonify({
            "nodes": [],
            "links": [],
            "warning": "知识图谱服务未连接"
        })

    kg_data = kg.client.get_kg(limit=limit)

    # 构建节点和边
    nodes = {}
    edges = []

    for item in kg_data:
        entity_name = item.get("entity_name")
        entity_type = item.get("entity_type", "unknown")
        relation_type = item.get("relation_type")
        target = item.get("target_entity")

        # 添加节点
        if entity_name and entity_name not in nodes:
            nodes[entity_name] = {
                "id": entity_name,
                "name": entity_name,
                "type": entity_type,
                "mentionCount": item.get("access_count", 1),
                "size": min(50, 10 + (item.get("access_count", 1) * 3)),
                "color": _get_entity_color(entity_type),
            }

        # 添加边
        if entity_name and relation_type and target:
            edges.append({
                "source": entity_name,
                "target": target,
                "type": relation_type,
                "strength": item.get("confidence", 1),
            })

            # 确保目标节点存在
            if target not in nodes:
                nodes[target] = {
                    "id": target,
                    "name": target,
                    "type": "unknown",
                    "mentionCount": 1,
                    "size": 15,
                    "color": _get_entity_color("unknown"),
                }

    return jsonify({
        "nodes": list(nodes.values()),
        "links": edges,
    })


@app.route('/agents/<agent_name>/graph/stats')
def graph_stats(agent_name):
    """知识图谱统计 - 优先使用 NetworkX"""
    # 调试：记录 NETWORKX_AVAILABLE 的值
    print(f"[DEBUG] NETWORKX_AVAILABLE = {NETWORKX_AVAILABLE}")
    
    # 优先使用 NetworkX
    if NETWORKX_AVAILABLE:
        try:
            from core.networkx_kg_client import get_nx_client
            
            nx_client = get_nx_client(agent_name)
            stats = nx_client.get_stats()
            
            print(f"[DEBUG] NetworkX stats: {stats}")
            
            return jsonify({
                "totalEntities": stats.get("totalEntities", 0),
                "totalRelations": stats.get("totalRelations", 0),
                "typeDistribution": stats.get("typeDistribution", {}),
                "mostConnected": stats.get("mostConnected", []),
                "relationTypes": stats.get("relationTypes", {}),
                "backend": "NetworkX"
            })
        except Exception as e:
            print(f"[DEBUG] NetworkX 失败: {e}")
    
    # 回退到 Weaviate
    print("[DEBUG] 回退到 Weaviate")
    kg = get_knowledge_graph(agent_name)
    
    # 检查Weaviate连接状态
    if kg.client.client is None:
        return jsonify({
            "totalEntities": 0,
            "totalRelations": 0,
            "typeDistribution": {},
            "mostConnected": [],
            "relationTypes": [],
            "warning": "Weaviate未连接"
        })

    kg_data = kg.client.get_kg(limit=1000)

    # 类型分布
    type_dist = {}
    for item in kg_data:
        entity_type = item.get("entity_type") or "unknown"
        type_dist[entity_type] = type_dist.get(entity_type, 0) + 1

    # 连接度
    connectivity = {}
    for item in kg_data:
        name = item.get("entity_name")
        target = item.get("target_entity")
        if name:
            connectivity[name] = connectivity.get(name, 0) + 1
        if target:
            connectivity[target] = connectivity.get(target, 0) + 1

    most_connected = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)[:10]

    # 过滤掉None值
    relation_types = [r for r in set([k.get("relation_type") or "" for k in kg_data]) if r]

    return jsonify({
        "totalEntities": len(set([k.get("entity_name") for k in kg_data if k.get("entity_name")])),
        "totalRelations": len(kg_data),
        "typeDistribution": type_dist,
        "mostConnected": most_connected,
        "relationTypes": relation_types,
        "backend": "Weaviate"
    })


@app.route('/agents/<agent_name>/entities')
def list_entities(agent_name):
    """列出实体"""
    entity_type = request.args.get("type")
    kg = get_knowledge_graph(agent_name)
    
    # 检查Weaviate连接状态
    if kg.client.client is None:
        return jsonify([])
    
    kg_data = kg.client.get_kg(limit=500)

    entities = {}
    for item in kg_data:
        name = item.get("entity_name")
        etype = item.get("entity_type", "unknown")

        if entity_type and etype != entity_type:
            continue

        if name and name not in entities:
            entities[name] = {
                "name": name,
                "type": etype,
                "mentionCount": item.get("access_count", 1),
            }
        elif name:
            entities[name]["mentionCount"] += 1

    return jsonify(list(entities.values()))


@app.route('/agents/<agent_name>/relations')
def get_relations(agent_name):
    """获取所有关系"""
    limit = request.args.get("limit", 100, type=int)
    kg = get_knowledge_graph(agent_name)
    
    # 检查连接状态
    if kg.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        # 从图谱中获取所有关系
        relations = kg.client.get_all_relations(limit=limit)
        return jsonify(relations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 知识图谱编辑 ====================

@app.route('/agents/<agent_name>/entities', methods=['POST'])
def add_entity(agent_name):
    """添加实体"""
    data = request.json
    entity_name = data.get("name")
    entity_type = data.get("type", "entity")
    
    if not entity_name:
        return jsonify({"error": "Entity name required"}), 400
    
    kg = get_knowledge_graph(agent_name)
    
    # 检查连接状态
    if kg.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        entity_id = kg.add_entity(entity_name, entity_type, source="user_defined")
        return jsonify({
            "success": True,
            "entity": {
                "id": entity_id,
                "name": entity_name,
                "type": entity_type
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/entities/<entity_name>', methods=['DELETE'])
def delete_entity(agent_name, entity_name):
    """删除实体及其所有关系"""
    kg = get_knowledge_graph(agent_name)
    
    # 检查连接状态
    if kg.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        deleted_count = kg.client.delete_kg_by_entity(entity_name)
        return jsonify({
            "success": True,
            "deletedCount": deleted_count,
            "message": f"已删除实体 '{entity_name}' 及其 {deleted_count} 个关联关系"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/relations', methods=['POST'])
def add_relation(agent_name):
    """添加关系"""
    data = request.json
    source_entity = data.get("source")
    relation_type = data.get("relation")
    target_entity = data.get("target")
    context = data.get("context")
    
    if not source_entity or not relation_type or not target_entity:
        return jsonify({"error": "source, relation, target are required"}), 400
    
    kg = get_knowledge_graph(agent_name)
    
    # 检查连接状态
    if kg.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        relation_id = kg.add_relation(
            source_entity, 
            relation_type, 
            target_entity, 
            context=context,
            source="user_defined"
        )
        return jsonify({
            "success": True,
            "relation": {
                "id": relation_id,
                "source": source_entity,
                "relation": relation_type,
                "target": target_entity
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/relations', methods=['DELETE'])
def delete_relation(agent_name):
    """删除关系"""
    data = request.json
    source_entity = data.get("source")
    relation_type = data.get("relation")
    target_entity = data.get("target")
    
    if not source_entity or not relation_type or not target_entity:
        return jsonify({"error": "source, relation, target are required"}), 400
    
    kg = get_knowledge_graph(agent_name)
    
    # 检查连接状态
    if kg.client.client is None:
        return jsonify({"error": "Weaviate未连接"}), 503
    
    try:
        deleted_count = kg.client.delete_kg_by_relation(source_entity, relation_type, target_entity)
        return jsonify({
            "success": True,
            "deletedCount": deleted_count,
            "message": f"已删除关系 '{source_entity} -> {relation_type} -> {target_entity}'"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 进化知识图谱 API ====================

from core.evolutionary_kg import EvolutionaryKnowledgeGraph
from core.enterprise_kg import EnterpriseKnowledgeGraph

evolutionary_kgs: dict = {}
enterprise_kgs: dict = {}

def get_evolutionary_kg(agent_id: str) -> EvolutionaryKnowledgeGraph:
    """获取进化知识图谱实例"""
    if agent_id not in evolutionary_kgs:
        evolutionary_kgs[agent_id] = EvolutionaryKnowledgeGraph(agent_id)
    return evolutionary_kgs[agent_id]

def get_enterprise_kg(agent_id: str) -> EnterpriseKnowledgeGraph:
    """获取企业级知识图谱实例"""
    if agent_id not in enterprise_kgs:
        enterprise_kgs[agent_id] = EnterpriseKnowledgeGraph(agent_id)
    return enterprise_kgs[agent_id]


@app.route('/agents/<agent_name>/kg/extract', methods=['POST'])
def extract_with_learning(agent_name):
    """智能提取实体关系（带学习）"""
    data = request.json
    text = data.get("text", "")
    context = data.get("context", {})

    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        ekg = get_evolutionary_kg(agent_name)
        result = ekg.extract_with_context(text, context)
        stored = ekg.store_with_learning(result, source=data.get("source"))

        return jsonify({
            "extracted": result,
            "stored": stored
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/learn/entity-type', methods=['POST'])
def learn_entity_type(agent_name):
    """学习新的实体类型"""
    data = request.json
    type_name = data.get("name")
    color = data.get("color", "#999999")
    icon = data.get("icon", "📌")

    if not type_name:
        return jsonify({"error": "Entity type name required"}), 400

    try:
        ekg = get_evolutionary_kg(agent_name)
        result = ekg.learn_entity_type(type_name, color, icon)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/learn/relation-type', methods=['POST'])
def learn_relation_type(agent_name):
    """学习新的关系类型"""
    data = request.json
    relation_name = data.get("name")
    category = data.get("category", "custom")
    opposite = data.get("opposite")
    transitive = data.get("transitive", False)

    if not relation_name:
        return jsonify({"error": "Relation type name required"}), 400

    try:
        ekg = get_evolutionary_kg(agent_name)
        result = ekg.learn_relation_type(relation_name, category, opposite, transitive)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/infer', methods=['POST'])
def infer_relations(agent_name):
    """推理新关系"""
    try:
        ekg = get_evolutionary_kg(agent_name)
        inferred = ekg.infer_new_relations()

        # 可选：自动存储推理结果
        auto_store = request.json.get("autoStore", False)
        if auto_store:
            for inf in inferred:
                ekg._store_relation(
                    inf["source"], inf["relation"], inf["target"],
                    context=inf["reason"], confidence=inf["confidence"]
                )

        return jsonify({
            "inferred": inferred,
            "count": len(inferred)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/contradictions', methods=['GET'])
def detect_contradictions(agent_name):
    """检测矛盾关系"""
    try:
        ekg = get_evolutionary_kg(agent_name)
        contradictions = ekg.detect_contradictions()
        return jsonify({
            "contradictions": contradictions,
            "count": len(contradictions)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/contradictions/resolve', methods=['POST'])
def resolve_contradiction(agent_name):
    """解决矛盾"""
    data = request.json
    contradiction = data.get("contradiction")
    strategy = data.get("strategy", "higher_confidence")

    if not contradiction:
        return jsonify({"error": "Contradiction data required"}), 400

    try:
        ekg = get_evolutionary_kg(agent_name)
        ekg.resolve_contradiction(contradiction, strategy)
        return jsonify({"success": True, "message": "矛盾已解决"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/merge', methods=['POST'])
def merge_entities(agent_name):
    """合并实体（处理同义词）"""
    data = request.json
    entity_a = data.get("entityA")
    entity_b = data.get("entityB")
    canonical = data.get("canonical")

    if not all([entity_a, entity_b, canonical]):
        return jsonify({"error": "entityA, entityB, and canonical required"}), 400

    try:
        ekg = get_evolutionary_kg(agent_name)
        ekg.merge_entities(entity_a, entity_b, canonical)
        return jsonify({
            "success": True,
            "message": f"已将 '{entity_a}' 和 '{entity_b}' 合并为 '{canonical}'"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/feedback', methods=['POST'])
def learn_from_feedback(agent_name):
    """从用户反馈学习"""
    data = request.json
    entity = data.get("entity")
    relation = data.get("relation")
    target = data.get("target")
    correct = data.get("correct", True)

    if not all([entity, relation, target]):
        return jsonify({"error": "entity, relation, and target required"}), 400

    try:
        ekg = get_evolutionary_kg(agent_name)
        ekg.learn_from_feedback(entity, relation, target, "", correct)
        return jsonify({
            "success": True,
            "learned": "correct" if correct else "incorrect"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/search', methods=['GET'])
def kg_search(agent_name):
    """
    知识图谱搜索
    
    Query params:
        q: 查询文本
        mode: 检索模式 (vector/hybrid/bm25)
        limit: 返回数量
    """
    query = request.args.get("q", "")
    mode = request.args.get("mode", "hybrid")
    limit = request.args.get("limit", 5, type=int)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    try:
        kg = get_knowledge_graph(agent_name)
        
        if mode == "vector":
            results = kg.search_similar_entities(query, limit)
        elif mode == "bm25":
            # 使用 Weaviate 原生 BM25
            collection = kg.client.client.collections.get(kg.client.kg_collection)
            response = collection.query.bm25(query=query, limit=limit)
            results = [{"id": str(o.uuid), **o.properties} for o in response.objects]
        else:  # hybrid
            results = kg.search_relations(query, limit)
        
        return jsonify({
            "query": query,
            "mode": mode,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/entity/<entity_name>/neighbors', methods=['GET'])
def kg_entity_neighbors(agent_name, entity_name):
    """
    获取实体的邻居节点 (图遍历)
    
    返回该实体的所有出边、入边和邻居实体
    """
    try:
        kg = get_knowledge_graph(agent_name)
        result = kg.get_entity_neighbors(entity_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/types', methods=['GET'])
def get_kg_types(agent_name):
    """获取所有实体类型和关系类型"""
    try:
        ekg = get_evolutionary_kg(agent_name)
        return jsonify({
            "entityTypes": ekg.all_entity_types,
            "relationTypes": ekg.all_relation_types
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/kg/graph-enhanced', methods=['GET'])
def get_enhanced_graph(agent_name):
    """获取增强版知识图谱数据"""
    try:
        ekg = get_evolutionary_kg(agent_name)
        graph_data = ekg.get_graph_data()
        return jsonify(graph_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 智能查询 ====================

@app.route('/agents/<agent_name>/query', methods=['POST'])
def smart_query(agent_name):
    """智能自然语言查询"""
    data = request.json
    query = data.get("query", "")
    time_hint = data.get("timeHint")

    if not query:
        return jsonify({"error": "Query required"}), 400

    store = get_memory_store(agent_name)

    # 检查Weaviate连接状态
    if store.client.client is None:
        return jsonify({
            "memories": [],
            "timeRange": None,
            "warning": "Weaviate未连接，无法执行语义搜索"
        })

    # 解析时间提示
    time_range = None
    now = datetime.now()

    # 中文时间提示
    if "上周" in query or time_hint == "lastweek":
        time_range = (now - timedelta(weeks=2), now - timedelta(weeks=1))
    elif "昨天" in query or time_hint == "yesterday":
        time_range = (now - timedelta(days=2), now - timedelta(days=1))
    elif "今天" in query or time_hint == "today":
        time_range = (now - timedelta(days=1), now)
    elif "本周" in query or time_hint == "week":
        time_range = (now - timedelta(weeks=1), now)
    elif "本月" in query or time_hint == "month":
        time_range = (now - timedelta(days=30), now)

    # 搜索
    results = store.recall_interactive(query, limit=data.get("limit", 5))

    # 时间过滤
    if time_range:
        filtered = []
        for r in results:
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    if time_range[0] <= ts <= time_range[1]:
                        filtered.append(r)
                except Exception:
                    pass
        results = filtered

    return jsonify({
        "memories": results,
        "timeRange": {
            "start": time_range[0].isoformat() if time_range else None,
            "end": time_range[1].isoformat() if time_range else None,
        },
    })


@app.route('/agents/<agent_name>/rerank', methods=['POST'])
def rerank_query(agent_name):
    """混合检索 + RRF 融合 + 重排序"""
    from openclaw_memory.core.vector_store.adapters.weaviate_adapter import WeaviateAdapter

    data = request.json
    query = data.get("query", "")
    vector_limit = data.get("vector_limit", 20)  # 向量检索数量
    bm25_limit = data.get("bm25_limit", 20)      # BM25 检索数量
    top_k = data.get("top_k", 10)                # 最终返回数量
    time_decay = data.get("time_decay", 0.1)
    importance_weight = data.get("importance_weight", 0.3)

    if not query:
        return jsonify({"error": "Query required"}), 400

    try:
        adapter = WeaviateAdapter(agent_name)
        if not adapter.connect():
            return jsonify({"memories": [], "warning": "Weaviate未连接"})

        results = adapter.search_with_rerank(
            query=query,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit,
            rerank_top_k=top_k,
            time_decay=time_decay,
            importance_weight=importance_weight
        )

        return jsonify({
            "memories": results,
            "count": len(results),
            "params": {
                "vector_limit": vector_limit,
                "bm25_limit": bm25_limit,
                "top_k": top_k,
                "time_decay": time_decay,
                "importance_weight": importance_weight
            }
        })
    except Exception as e:
        return jsonify({"error": str(e), "memories": []}), 500


# ==================== 数据导出/导入 ====================

@app.route('/agents/<agent_name>/export', methods=['GET'])
def export_agent_data(agent_name):
    """
    导出 Agent 的所有数据

    包括：
    - 所有记忆（含原文和摘要）
    - 知识图谱（实体和关系）
    - 配置文件
    """
    try:
        store = get_memory_store(agent_name)
        kg = get_knowledge_graph(agent_name)

        # 确保连接
        _ = store.client.client
        _ = kg.client.client

        # 获取所有记忆
        memories = store.client.get_memories(limit=10000)

        # 获取知识图谱
        kg_data = kg.client.get_kg(limit=10000)

        # 获取配置文件
        config_files = {}
        config_dir = PROJECT_ROOT / "data" / "agents" / agent_name / "config"
        if config_dir.exists():
            for config_file in config_dir.glob("*.md"):
                try:
                    config_files[config_file.name] = config_file.read_text(encoding='utf-8')
                except Exception:
                    pass

        export_data = {
            "version": "1.0",
            "agent_name": agent_name,
            "export_time": datetime.now(timezone.utc).isoformat(),
            "memories": memories,
            "knowledge_graph": kg_data,
            "config_files": config_files,
            "stats": {
                "total_memories": len(memories),
                "total_kg_items": len(kg_data)
            }
        }

        return jsonify(export_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/import', methods=['POST'])
def import_agent_data(agent_name):
    """
    导入 Agent 数据

    支持合并或覆盖模式
    """
    try:
        data = request.json
        mode = data.get("mode", "merge")  # merge 或 overwrite

        store = get_memory_store(agent_name)
        kg = get_knowledge_graph(agent_name)

        # 确保连接
        _ = store.client.client
        _ = kg.client.client

        results = {
            "memories_imported": 0,
            "memories_skipped": 0,
            "kg_imported": 0,
            "kg_skipped": 0,
            "config_imported": 0
        }

        # 导入记忆
        if "memories" in data:
            for memory in data["memories"]:
                try:
                    content = memory.get("content", "")
                    if not content:
                        continue

                    # 检查是否已存在（通过时间戳和内容hash）
                    # 简单起见，直接导入
                    memory_data = {
                        "summary": memory.get("summary"),
                        "importance": memory.get("importance", 0.5),
                        "memory_type": memory.get("memory_type", "conversation"),
                        "timestamp": memory.get("timestamp"),
                        "keywords": memory.get("keywords", []),
                        "source": "import"
                    }

                    memory_id = store.store(content, memory_data)
                    if memory_id:
                        results["memories_imported"] += 1
                    else:
                        results["memories_skipped"] += 1
                except Exception as e:
                    results["memories_skipped"] += 1

        # 导入知识图谱
        if "knowledge_graph" in data:
            for item in data["knowledge_graph"]:
                try:
                    kg.client.insert_kg(item)
                    results["kg_imported"] += 1
                except Exception:
                    results["kg_skipped"] += 1

        # 导入配置文件
        if "config_files" in data:
            config_dir = PROJECT_ROOT / "data" / "agents" / agent_name / "config"
            config_dir.mkdir(parents=True, exist_ok=True)

            for filename, content in data["config_files"].items():
                try:
                    config_file = config_dir / filename
                    config_file.write_text(content, encoding='utf-8')
                    results["config_imported"] += 1
                except Exception:
                    pass

        return jsonify({
            "success": True,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ==================== 静态文件 ====================

@app.route('/')
def index():
    """首页"""
    response = send_from_directory(str(PROJECT_ROOT / "ui"), "index.html")
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/<path:filename>')
def static_files(filename):
    """静态文件"""
    response = send_from_directory(str(PROJECT_ROOT / "ui"), filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ==================== 辅助函数 ====================

def _get_entity_color(entity_type: str) -> str:
    """获取实体类型颜色"""
    colors = {
        "advertiser": "#FF6B6B",
        "product": "#4ECDC4",
        "campaign": "#45B7D1",
        "video": "#96CEB4",
        "metric": "#FFEAA7",
        "platform": "#DDA0DD",
        "person": "#98D8C8",
        "organization": "#F7DC6F",
        "project": "#BB8FCE",
        "date": "#85C1E9",
        "location": "#F8B500",
        "unknown": "#AAAAAA",
    }
    return colors.get(entity_type, colors["unknown"])


# ==================== 企业级知识图谱 API ====================

@app.route('/agents/<agent_name>/ekg/extract', methods=['POST'])
def enterprise_extract(agent_name):
    """企业级知识抽取"""
    data = request.json
    text = data.get("text", "")
    source = data.get("source", "conversation")

    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        ekg = get_enterprise_kg(agent_name)
        result = ekg.extract_and_store(text, source)

        return jsonify({
            "status": "success",
            "entities": result["entities"],
            "relations": result["relations"],
            "events": result["events"],
            "conflicts": result["conflicts"],
            "aliases": result["aliases"],
            "stats": {
                "entities_found": len(result["entities"]),
                "relations_found": len(result["relations"]),
                "events_found": len(result["events"]),
                "conflicts_found": len(result["conflicts"])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/stats', methods=['GET'])
def enterprise_stats(agent_name):
    """企业级知识图谱统计"""
    try:
        ekg = get_enterprise_kg(agent_name)
        stats = ekg.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/quality', methods=['GET'])
def enterprise_quality(agent_name):
    """知识质量评估"""
    threshold = float(request.args.get("threshold", 0.5))

    try:
        ekg = get_enterprise_kg(agent_name)
        low_confidence = ekg.get_low_confidence_items(threshold)

        return jsonify({
            "threshold": threshold,
            "needs_review": {
                "entities_count": len(low_confidence["entities"]),
                "relations_count": len(low_confidence["relations"]),
                "entities": low_confidence["entities"][:10],
                "relations": low_confidence["relations"][:10]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/decay', methods=['POST'])
def apply_confidence_decay(agent_name):
    """应用置信度衰减"""
    try:
        ekg = get_enterprise_kg(agent_name)
        ekg.apply_decay()
        return jsonify({"status": "success", "message": "置信度衰减已应用"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/entity/<entity_name>', methods=['GET'])
def get_entity_info(agent_name, entity_name):
    """获取实体详情"""
    try:
        ekg = get_enterprise_kg(agent_name)
        entity = ekg.get_entity(entity_name)

        if entity:
            return jsonify(entity.to_dict())
        else:
            return jsonify({"error": "Entity not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/relations', methods=['GET'])
def get_entity_relations(agent_name):
    """获取关系列表"""
    entity_name = request.args.get("entity")

    try:
        ekg = get_enterprise_kg(agent_name)
        relations = ekg.get_relations(entity_name)

        return jsonify({
            "relations": [r.to_dict() for r in relations],
            "count": len(relations)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 企业级知识图谱 API (补充) ====================

@app.route('/agents/<agent_name>/ekg/extract-and-store', methods=['POST'])
def ekg_extract_and_store(agent_name):
    """提取并存储"""
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        ekg = get_enterprise_kg(agent_name)
        extracted = ekg.extract(text)
        stored = ekg.store(extracted)
        return jsonify({
            "extracted": extracted,
            "stored": stored
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/search', methods=['GET'])
def ekg_search(agent_name):
    """语义搜索实体"""
    query = request.args.get("q", "")
    limit = request.args.get("limit", 5, type=int)

    if not query:
        return jsonify({"error": "Query required"}), 400

    try:
        ekg = get_enterprise_kg(agent_name)
        results = ekg.search_entities(query, limit)
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/extract-from-memories', methods=['POST'])
def ekg_extract_from_memories(agent_name):
    """从记忆中提取知识（企业级）"""
    data = request.json or {}
    limit = data.get("limit", 20)

    try:
        store = get_memory_store(agent_name)
        memories = store.client.get_memories(limit=limit)

        if not memories:
            return jsonify({"status": "no_memories"})

        ekg = get_enterprise_kg(agent_name)
        total_entities = 0
        total_relations = 0

        for m in memories:
            content = m.get("content", "")
            if not content or len(content) < 20:
                continue

            # 提取
            result = ekg.extract(content)
            # 存储
            stored = ekg.store(result)

            total_entities += len(stored.get("entities", []))
            total_relations += len(stored.get("relations", []))

        return jsonify({
            "status": "success",
            "memoriesProcessed": len(memories),
            "entitiesExtracted": total_entities,
            "relationsExtracted": total_relations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/self-update', methods=['POST'])
def ekg_self_update(agent_name):
    """知识图谱自我更新"""
    data = request.json or {}
    new_memories = data.get("memories", [])
    limit = data.get("limit", 5)  # 限制处理数量，避免超时

    try:
        ekg = get_enterprise_kg(agent_name)

        # 如果没有提供新记忆，从存储中获取最近的
        if not new_memories:
            store = get_memory_store(agent_name)
            memories = store.client.get_memories(limit=limit)
            new_memories = [m.get("content", "") for m in memories if m.get("content")]

        stats = ekg.self_update(new_memories)

        return jsonify({
            "status": "success",
            "stats": stats,
            "processed_memories": len(new_memories)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/learn', methods=['POST'])
def ekg_learn_from_feedback(agent_name):
    """从用户反馈中学习"""
    data = request.json or {}
    entity_name = data.get("entityName")
    correct_type = data.get("correctType")
    correct_relation = data.get("correctRelation")
    is_wrong = data.get("isWrong", False)

    if not entity_name:
        return jsonify({"error": "entityName required"}), 400

    try:
        ekg = get_enterprise_kg(agent_name)
        ekg.learn_from_feedback(
            entity_name=entity_name,
            correct_type=correct_type,
            correct_relation=correct_relation,
            is_wrong=is_wrong
        )
        return jsonify({"status": "learned", "entity": entity_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/ekg/summary', methods=['GET'])
def ekg_update_summary(agent_name):
    """获取知识图谱更新摘要"""
    try:
        ekg = get_enterprise_kg(agent_name)
        summary = ekg.get_update_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 启动 ====================

def main():
    """启动 API 服务器"""
    import argparse

    parser = argparse.ArgumentParser(description="记忆系统 API")
    parser.add_argument("--port", type=int, default=API_PORT, help="API 端口")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--no-scheduler", action="store_true", help="禁用后台调度器")
    parser.add_argument("--no-sync", action="store_true", help="禁用实时同步")

    args = parser.parse_args()

    # 启动后台调度器
    if not args.no_scheduler:
        try:
            scheduler = start_scheduler()
            print(f"✅ 后台调度器已启动")
        except Exception as e:
            print(f"⚠️ 后台调度器启动失败: {e}")

    # 启动实时同步服务
    global sync_service
    if not args.no_sync:
        try:
            sync_service = RealtimeSyncService()
            sync_service.start(background=True)
            print(f"✅ 实时同步服务已启动")
        except Exception as e:
            print(f"⚠️ 实时同步服务启动失败: {e}")

    print("\n" + "="*50)
    print("🚀 记忆系统 API 启动")
    print("="*50)
    print(f"📡 API: http://localhost:{args.port}")
    print(f"📊 UI: http://localhost:{args.port}/index.html")
    print(f"🧠 AI: http://localhost:{args.port}/health/ai")
    print(f"⏰ 调度器: http://localhost:{args.port}/scheduler/status")
    print(f"🔄 实时同步: {'已启用' if not args.no_sync else '已禁用'}")
    print("="*50 + "\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


# ==================== 多模态 API ====================

@app.route('/agents/<agent_name>/multimodal/upload', methods=['POST'])
def upload_multimodal(agent_name):
    """
    上传多模态内容（图片、文档、音频、视频）
    
    Form Data:
        file: 文件
        description: 描述（可选）
        tags: 标签（可选，JSON 数组）
        importance: 重要性（可选，0-1）
    
    Returns:
        {
            "success": true,
            "memory_id": "...",
            "content_type": "image|document|audio|video",
            "description": "..."
        }
    """
    try:
        from core.multimodal import MultimodalProcessor
        
        if 'file' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
        
        # 保存临时文件
        import tempfile
        import werkzeug.utils
        
        filename = werkzeug.utils.secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # 处理多模态内容
        processor = MultimodalProcessor()
        store = get_memory_store(agent_name)
        
        # 获取额外参数
        description = request.form.get('description')
        tags = json.loads(request.form.get('tags', '[]'))
        importance = float(request.form.get('importance', 0.5))
        
        # 存储多模态内容
        memory_id = processor.store_multimodal(
            temp_path, 
            store,
            metadata={
                "description": description,
                "tags": tags,
                "importance": importance
            }
        )
        
        # 检测类型
        category, mime_type = processor.detect_type(temp_path)
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({
            "success": True,
            "memory_id": memory_id,
            "content_type": category,
            "mime_type": mime_type,
            "message": f"已存储 {category} 内容"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/multimodal/image', methods=['POST'])
def upload_image(agent_name):
    """
    上传图片并生成描述
    
    Form Data:
        file: 图片文件
        description: 自定义描述（可选）
        tags: 标签（可选）
    
    Returns:
        {
            "success": true,
            "memory_id": "...",
            "description": "AI 生成的图片描述",
            "thumbnail": "base64 缩略图"
        }
    """
    try:
        from core.multimodal import MultimodalProcessor, MultimodalMemoryStore
        
        if 'file' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400
        
        file = request.files['file']
        
        # 保存临时文件
        import tempfile
        import werkzeug.utils
        
        filename = werkzeug.utils.secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # 处理图片
        processor = MultimodalProcessor()
        store = get_memory_store(agent_name)
        mm_store = MultimodalMemoryStore(store, processor)
        
        description = request.form.get('description')
        tags = json.loads(request.form.get('tags', '[]'))
        importance = float(request.form.get('importance', 0.5))
        
        memory_id = mm_store.remember_image(
            temp_path,
            description=description,
            tags=tags,
            importance=importance
        )
        
        # 获取处理结果
        result = processor.process_image(temp_path)
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({
            "success": True,
            "memory_id": memory_id,
            "description": result["description"],
            "tags": result["tags"],
            "thumbnail": result["thumbnail"][:100] + "..." if result["thumbnail"] else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/multimodal/search', methods=['GET'])
def search_multimodal(agent_name):
    """
    搜索多模态内容
    
    Query Parameters:
        q: 搜索关键词
        type: 内容类型 (image|document|audio|video|all)
        limit: 返回数量
    
    Returns:
        {
            "results": [...]
        }
    """
    try:
        from core.multimodal import MultimodalMemoryStore
        
        query = request.args.get('q', '')
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 5))
        
        store = get_memory_store(agent_name)
        
        if content_type == 'all':
            results = store.search(query, limit=limit)
            # 过滤多模态内容
            results = [r for r in results if r.get('content_type')]
        else:
            mm_store = MultimodalMemoryStore(store)
            results = mm_store.search_by_content_type(content_type, query, limit)
        
        return jsonify({
            "success": True,
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/multimodal/types', methods=['GET'])
def get_multimodal_types(agent_name):
    """
    获取各类型多模态内容的统计
    """
    try:
        store = get_memory_store(agent_name)
        memories = store.get_recent(limit=10000)
        
        # 统计各类型
        type_counts = {
            "image": 0,
            "document": 0,
            "audio": 0,
            "video": 0,
            "text": 0
        }
        
        for m in memories:
            content_type = m.get('content_type', 'text')
            if content_type in type_counts:
                type_counts[content_type] += 1
            else:
                type_counts['text'] += 1
        
        return jsonify({
            "success": True,
            "counts": type_counts,
            "total": len(memories)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 用户画像 API ====================

@app.route('/agents/<agent_name>/profile', methods=['GET'])
def get_agent_profile(agent_name):
    """
    获取用户画像
    
    GET /agents/main/profile?days=30
    """
    if not PROFILE_AVAILABLE:
        return jsonify({"error": "用户画像模块未启用"}), 500
    
    try:
        days = request.args.get('days', 30, type=int)
        
        profile = get_user_profile(agent_name)
        result = profile.build_profile(days=days)
        
        return jsonify({
            "success": True,
            "agent": agent_name,
            "profile": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/profile/entities', methods=['GET'])
def get_profile_entities(agent_name):
    """获取用户实体聚合"""
    if not PROFILE_AVAILABLE:
        return jsonify({"error": "用户画像模块未启用"}), 500
    
    try:
        profile = get_user_profile(agent_name)
        memories = profile._get_all_memories()
        entities = profile._extract_entities(memories)
        
        return jsonify({
            "success": True,
            "entities": entities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/profile/behavior', methods=['GET'])
def get_profile_behavior(agent_name):
    """获取行为分析"""
    if not PROFILE_AVAILABLE:
        return jsonify({"error": "用户画像模块未启用"}), 500
    
    try:
        profile = get_user_profile(agent_name)
        memories = profile._get_all_memories()
        behavior = profile._analyze_behavior(memories)
        
        return jsonify({
            "success": True,
            "behavior": behavior
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/profile/classify', methods=['GET'])
def get_memory_classification(agent_name):
    """获取记忆分类 (Fact vs Experience)"""
    if not PROFILE_AVAILABLE:
        return jsonify({"error": "用户画像模块未启用"}), 500
    
    try:
        profile = get_user_profile(agent_name)
        memories = profile._get_all_memories()
        classified = profile._classify_memories(memories)
        
        return jsonify({
            "success": True,
            "classification": {
                "facts": len(classified["facts"]),
                "experiences": len(classified["experiences"]),
                "mixed": len(classified["mixed"]),
                "unknown": len(classified["unknown"])
            },
            "sample_facts": classified["facts"][:5],
            "sample_experiences": classified["experiences"][:5]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 智能召回 API ====================

@app.route('/agents/<agent_name>/smart_recall', methods=['GET', 'POST'])
def smart_recall_memories(agent_name):
    """
    智能召回 - 根据查询自动决定是否召回、召回多少、是否压缩
    
    GET /agents/main/smart_recall?q=上次讨论的项目
    
    返回：
    {
        "success": true,
        "memories": [...],       // 压缩后的记忆
        "entities": [...],       // 相关实体（可选）
        "decision": {...},       // 召回决策
        "stats": {...}           // 统计信息
    }
    """
    try:
        # 获取查询
        if request.method == 'POST':
            data = request.json or {}
            query = data.get('query', '')
            max_tokens = data.get('max_tokens', 1000)
            force_recall = data.get('force_recall', False)
        else:
            query = request.args.get('q') or request.args.get('query', '')
            max_tokens = request.args.get('max_tokens', 1000, type=int)
            force_recall = request.args.get('force_recall', 'false').lower() == 'true'
        
        if not query:
            return jsonify({"error": "query required"}), 400
        
        # 执行智能召回
        from core.smart_recall import get_smart_recaller
        
        recaller = get_smart_recaller(agent_name)
        result = recaller.smart_recall(
            query=query,
            force_recall=force_recall,
            max_tokens=max_tokens
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/smart_recall/decision', methods=['GET'])
def get_recall_decision(agent_name):
    """
    获取召回决策 - 只返回决策信息，不执行召回
    
    GET /agents/main/smart_recall/decision?q=上次讨论的项目
    
    返回：
    {
        "should_recall": true,
        "level": "full",
        "memory_limit": 3,
        "include_entities": false,
        "compression_ratio": 0.7,
        "reason": "触发类别: ['time', 'query']"
    }
    """
    try:
        query = request.args.get('q') or request.args.get('query', '')
        
        if not query:
            return jsonify({"error": "query required"}), 400
        
        from core.smart_recall import analyze_recall_needs
        
        decision = analyze_recall_needs(query)
        
        return jsonify({
            "success": True,
            "query": query,
            "decision": decision
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents/<agent_name>/smart_recall/context', methods=['GET'])
def get_recall_context(agent_name):
    """
    获取召回上下文 - 返回可直接注入 prompt 的文本
    
    GET /agents/main/smart_recall/context?q=上次讨论的项目
    
    返回：
    {
        "success": true,
        "context": "## 相关记忆\n- ...\n## 相关实体\n- ..."
    }
    """
    try:
        query = request.args.get('q') or request.args.get('query', '')
        
        if not query:
            return jsonify({"error": "query required"}), 400
        
        from core.smart_recall import get_smart_recaller
        
        recaller = get_smart_recaller(agent_name)
        context = recaller.quick_recall(query)
        
        return jsonify({
            "success": True,
            "query": query,
            "context": context
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 知识库 API ====================

@app.route('/api/kb/upload', methods=['POST'])
def kb_upload():
    """
    上传文档到知识库
    
    POST /api/kb/upload
    Content-Type: multipart/form-data
    File: (文件)
    Title: (可选，文档标题)
    """
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "未上传文件"}), 400
        
        file = request.files['file']
        title = request.form.get('title', '')
        
        if file.filename == '':
            return jsonify({"success": False, "error": "文件名无效"}), 400
        
        # 检查文件类型
        allowed_extensions = {'.pdf', '.txt', '.md', '.markdown', '.docx', '.doc'}
        ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        
        if ext not in allowed_extensions:
            return jsonify({
                "success": False, 
                "error": f"不支持的文件类型: {ext}。支持: {', '.join(allowed_extensions)}"
            }), 400
        
        # 保存到临时文件
        import tempfile
        import uuid
        
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        file.save(temp_path)
        
        # 添加到知识库
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        result = kb.add_document(temp_path, title or None)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/kb/documents', methods=['GET'])
def kb_list_documents():
    """列出知识库中的文档"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        documents = kb.list_documents()
        
        return jsonify({
            "success": True,
            "documents": documents,
            "total": len(documents)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/kb/documents/<document_id>', methods=['DELETE'])
def kb_delete_document(document_id):
    """删除知识库文档"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        success = kb.delete_document(document_id)
        
        return jsonify({
            "success": success,
            "document_id": document_id
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/kb/search', methods=['GET', 'POST'])
def kb_search():
    """
    搜索知识库
    
    GET /api/kb/search?q=查询内容&limit=5&mode=hybrid
    POST /api/kb/search (JSON body)
    
    mode: 
      - "vector": 向量相似度搜索
      - "bm25": 关键词搜索
      - "hybrid": 混合搜索 (向量+BM25, 默认)
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            query = data.get('q') or data.get('query', '')
            limit = data.get('limit', 5)
            mode = data.get('mode', 'hybrid')
        else:
            query = request.args.get('q') or request.args.get('query', '')
            limit = int(request.args.get('limit', 5))
            mode = request.args.get('mode', 'hybrid')
        
        if not query:
            return jsonify({"success": False, "error": "查询内容不能为空"}), 400
        
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        results = kb.search(query, limit, mode=mode)
        
        return jsonify({
            "success": True,
            "query": query,
            "mode": mode,
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== Agent 级别知识库 API ====================

@app.route('/agents/<agent_name>/kb/search', methods=['GET', 'POST'])
def agent_kb_search(agent_name):
    """
    搜索指定 Agent 的知识库
    
    GET /agents/main/kb/search?q=查询内容&limit=5&mode=hybrid
    POST /agents/main/kb/search (JSON body)
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            query = data.get('q') or data.get('query', '')
            limit = data.get('limit', 5)
            mode = data.get('mode', 'hybrid')
        else:
            query = request.args.get('q') or request.args.get('query', '')
            limit = int(request.args.get('limit', 5))
            mode = request.args.get('mode', 'hybrid')
        
        if not query:
            return jsonify({"success": False, "error": "查询内容不能为空"}), 400
        
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id=agent_name)
        results = kb.search(query, limit, mode=mode, agent_id=agent_name)
        
        return jsonify({
            "success": True,
            "agent": agent_name,
            "query": query,
            "mode": mode,
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/agents/<agent_name>/kb/upload', methods=['POST'])
def agent_kb_upload(agent_name):
    """上传文档到指定 Agent 的知识库"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有文件"}), 400
        
        file = request.files['file']
        title = request.form.get('title')
        
        if file.filename == '':
            return jsonify({"success": False, "error": "文件名不能为空"}), 400
        
        # 保存到临时文件
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_filename = f"kb_{agent_name}_{int(time.time())}_{file.filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        file.save(temp_path)
        
        # 添加到指定 agent 的知识库
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id=agent_name)
        result = kb.add_document(temp_path, title or None, agent_id=agent_name)
        
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        
        return jsonify({
            "success": result.get("success", False),
            "agent": agent_name,
            "document_id": result.get("document_id"),
            "title": result.get("title"),
            "chunks": result.get("chunks"),
            "error": result.get("error")
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/agents/<agent_name>/kb/documents', methods=['GET'])
def agent_kb_list_documents(agent_name):
    """列出指定 Agent 的知识库文档"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id=agent_name)
        
        # 搜索该 agent 的所有文档
        results = kb.search("*", limit=100, agent_id=agent_name)
        
        # 按 document_id 分组
        docs = {}
        for r in results:
            doc_id = r.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "chunks": len(r.get("chunks", []))
                }
            else:
                docs[doc_id]["chunks"] += len(r.get("chunks", []))
        
        return jsonify({
            "success": True,
            "agent": agent_name,
            "documents": list(docs.values()),
            "total": len(docs)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/agents/<agent_name>/kb/documents/<document_id>', methods=['DELETE'])
def agent_kb_delete_document(agent_name, document_id):
    """删除指定 Agent 的知识库文档"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id=agent_name)
        success = kb.delete_document(document_id)
        
        return jsonify({
            "success": success,
            "agent": agent_name,
            "document_id": document_id
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/agents/<agent_name>/kb/clear', methods=['DELETE'])
def agent_kb_clear(agent_name):
    """清空指定 Agent 的知识库"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id=agent_name)
        
        # 搜索该 agent 的所有文档并删除
        results = kb.search("*", limit=1000, agent_id=agent_name)
        
        doc_ids = list(set([r.get("document_id") for r in results if r.get("document_id")]))
        
        for doc_id in doc_ids:
            kb.delete_document(doc_id)
        
        return jsonify({
            "success": True,
            "agent": agent_name,
            "deleted_count": len(doc_ids)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 共享知识库 API (所有 Agent 可用) ====================

@app.route('/kb/shared/search', methods=['GET', 'POST'])
def shared_kb_search():
    """
    搜索共享知识库
    
    所有 Agent 都可以访问的公共知识库
    
    GET /kb/shared/search?q=查询内容&limit=5&mode=hybrid
    POST /kb/shared/search (JSON body)
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            query = data.get('q') or data.get('query', '')
            limit = data.get('limit', 5)
            mode = data.get('mode', 'hybrid')
        else:
            query = request.args.get('q') or request.args.get('query', '')
            limit = int(request.args.get('limit', 5))
            mode = request.args.get('mode', 'hybrid')
        
        if not query:
            return jsonify({"success": False, "error": "查询内容不能为空"}), 400
        
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id='shared')
        results = kb.search(query, limit, mode=mode, agent_id='shared')
        
        return jsonify({
            "success": True,
            "type": "shared",  # 标记为共享知识库
            "query": query,
            "mode": mode,
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/kb/shared/upload', methods=['POST'])
def shared_kb_upload():
    """
    上传到共享知识库
    
    上传的文档所有 Agent 都可以访问
    """
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有文件"}), 400
        
        file = request.files['file']
        title = request.form.get('title')
        
        if file.filename == '':
            return jsonify({"success": False, "error": "文件名不能为空"}), 400
        
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_filename = f"kb_shared_{int(time.time())}_{file.filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        file.save(temp_path)
        
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id='shared')
        result = kb.add_document(temp_path, title or None, agent_id='shared')
        
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        
        return jsonify({
            "success": result.get("success", False),
            "type": "shared",
            "document_id": result.get("document_id"),
            "title": result.get("title"),
            "chunks": result.get("chunks"),
            "error": result.get("error")
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/kb/shared/documents', methods=['GET'])
def shared_kb_list():
    """列出共享知识库的文档"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id='shared')
        results = kb.search("*", limit=100, agent_id='shared')
        
        docs = {}
        for r in results:
            doc_id = r.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "chunks": len(r.get("chunks", []))
                }
            else:
                docs[doc_id]["chunks"] += len(r.get("chunks", []))
        
        return jsonify({
            "success": True,
            "type": "shared",
            "documents": list(docs.values()),
            "total": len(docs)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/kb/shared/clear', methods=['DELETE'])
def shared_kb_clear():
    """清空共享知识库"""
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base(agent_id='shared')
        results = kb.search("*", limit=1000, agent_id='shared')
        
        doc_ids = list(set([r.get("document_id") for r in results if r.get("document_id")]))
        
        for doc_id in doc_ids:
            kb.delete_document(doc_id)
        
        return jsonify({
            "success": True,
            "type": "shared",
            "deleted_count": len(doc_ids)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    main()

