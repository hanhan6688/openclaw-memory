#!/usr/bin/env python3
"""
Memory System 定时检查任务
每小时自动运行，检查并完善系统
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 日志文件
LOG_FILE = os.path.expanduser("~/.openclaw/memory_system/health_checks.json")


def log_check(result: dict):
    """记录检查结果"""
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    
    logs.append(result)
    logs = logs[-24:]
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


def check_weaviate() -> dict:
    """检查 Weaviate 状态"""
    try:
        import requests
        response = requests.get("http://localhost:8080/v1/.well-known/ready", timeout=5)
        return {"status": "ok" if response.status_code == 200 else "error"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_memory_api() -> dict:
    """检查 Memory System API"""
    try:
        import requests
        response = requests.get("http://localhost:8082/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {"status": data.get("status", "unknown"), "ai": data.get("ai", False)}
        return {"status": "error"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_hybrid_recall() -> dict:
    """检查混合检索"""
    try:
        import requests
        response = requests.get(
            "http://localhost:8082/agents/main/memories/search",
            params={"q": "test", "limit": 1},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return {"status": "ok" if data.get("success") else "error", "mode": data.get("mode")}
        return {"status": "error"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_kg() -> dict:
    """检查知识图谱"""
    try:
        import requests
        response = requests.get("http://localhost:8082/agents/main/graph/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {"status": "ok", "entities": data.get("totalEntities", 0), "relations": data.get("totalRelations", 0)}
        return {"status": "error"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_health_check():
    """运行健康检查"""
    print(f"\n{'='*50}")
    print(f"Memory System 健康检查 - {datetime.now().isoformat()}")
    print(f"{'='*50}\n")
    
    results = {}
    
    print("1. Weaviate:", check_weaviate()["status"])
    results["weaviate"] = check_weaviate()
    
    print("2. Memory API:", check_memory_api()["status"])
    results["api"] = check_memory_api()
    
    print("3. Hybrid Recall:", check_hybrid_recall()["status"])
    results["hybrid_recall"] = check_hybrid_recall()
    
    print("4. Knowledge Graph:", check_kg()["status"])
    results["knowledge_graph"] = check_kg()
    
    log_check(results)
    
    print(f"\n{'='*50}")
    print("检查完成")
    return results


if __name__ == "__main__":
    run_health_check()