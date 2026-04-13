#!/usr/bin/env python3
"""
OpenClaw Memory System - 主入口
================================

AI Agent 长期记忆系统

使用方法:
    python main.py api          # 启动 API 服务
    python main.py sync         # 启动实时同步
    python main.py summarize    # 运行摘要任务
    python main.py check        # 检查系统状态
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查系统依赖...\n")
    
    issues = []
    
    # 检查 Weaviate
    try:
        import weaviate
        print(f"  ✅ weaviate-client: {weaviate.__version__}")
    except ImportError:
        print("  ❌ weaviate-client 未安装")
        issues.append("pip install weaviate-client")
    
    # 检查 Flask
    try:
        import flask
        print(f"  ✅ flask: {flask.__version__}")
    except ImportError:
        print("  ❌ flask 未安装")
        issues.append("pip install flask")
    
    # 检查 Ollama 连接
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            print(f"  ✅ Ollama: {len(models)} 个模型可用")
        else:
            print("  ⚠️ Ollama 连接异常")
    except:
        print("  ⚠️ Ollama 未运行或未安装")
    
    # 检查 Weaviate 连接
    try:
        weaviate_url = f"http://{os.getenv('WEAVIATE_HOST', 'localhost')}:{os.getenv('WEAVIATE_PORT', '8080')}"
        r = requests.get(f"{weaviate_url}/v1/.well-known/ready", timeout=5)
        if r.status_code == 200:
            print(f"  ✅ Weaviate: 运行中")
        else:
            print("  ⚠️ Weaviate 未就绪")
    except:
        print("  ❌ Weaviate 未运行")
        issues.append("请启动 Weaviate: docker run -p 8080:8080 cr.weaviate.io/semitechnologies/weaviate:latest")
    
    if issues:
        print("\n⚠️ 需要修复的问题:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("\n✅ 所有依赖检查通过!\n")
    return True


def start_api_server(port=None):
    """启动 API 服务器"""
    port = port or int(os.getenv("API_PORT", "8082"))

    print(f"\n🚀 启动 API 服务器 (端口: {port})...\n")

    from openclaw_memory.api.server import app, RealtimeSyncService, start_scheduler

    # 配置 CORS
    from flask_cors import CORS
    CORS(app)

    # 启动后台调度器
    try:
        scheduler = start_scheduler()
        print(f"✅ 后台调度器已启动")
    except Exception as e:
        print(f"⚠️ 后台调度器启动失败: {e}")

    # 启动实时同步服务
    sync_service = None
    try:
        sync_service = RealtimeSyncService()
        sync_service.start(background=True)
        print(f"✅ 实时同步服务已启动")
    except Exception as e:
        print(f"⚠️ 实时同步服务启动失败: {e}")

    print("\n" + "="*50)
    print("🚀 记忆系统 API 启动")
    print("="*50)
    print(f"📡 API: http://localhost:{port}")
    print(f"📊 UI: http://localhost:{port}/index.html")
    print(f"🔄 实时同步: 已启用")
    print("="*50 + "\n")

    # 启动服务
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )


def start_sync_service():
    """启动实时同步服务"""
    print("\n🔄 启动实时同步服务...\n")
    
    from openclaw_memory.sync.realtime_sync import RealtimeSyncService
    
    service = RealtimeSyncService()
    
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()


def run_summarizer(agent=None):
    """运行摘要任务"""
    agent = agent or os.getenv("DEFAULT_AGENT", "main")
    
    print(f"\n📝 运行摘要任务 (Agent: {agent})...\n")

    try:
        from scripts.summarizer import Summarizer
    except ImportError:
        print("⚠️ 未找到批量摘要脚本 `scripts/summarizer.py`。")
        print("   当前仓库仅提供 `openclaw_memory.core.summarizer` 中的文本摘要能力，")
        print("   但没有按 Agent 批量执行摘要的 CLI 任务。")
        return False

    summarizer = Summarizer(agent)
    result = summarizer.run_summarization()

    print(f"\n📊 结果: {result}\n")

    summarizer.close()
    return True


def run_learner(action=None):
    """运行学习任务"""
    print(f"\n🧠 运行学习任务 ({action or 'optimize'})...\n")

    try:
        from scripts.learner import run_optimization
    except ImportError:
        print("⚠️ 未找到学习优化脚本 `scripts/learner.py`。")
        print("   当前版本尚未包含 CLI 学习任务入口，请先补充对应脚本后再运行该命令。")
        return False

    run_optimization(action)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Memory System - AI Agent 长期记忆系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py api --port 8082    启动 API 服务
  python main.py sync               启动实时同步
  python main.py summarize          运行摘要任务
  python main.py check              检查系统状态
        """
    )
    
    parser.add_argument(
        "command",
        choices=["api", "sync", "summarize", "learn", "check"],
        help="要执行的命令"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="API 服务端口 (默认: 8082)"
    )
    
    parser.add_argument(
        "--agent", "-a",
        help="指定的 Agent 名称"
    )
    
    parser.add_argument(
        "--action",
        help="学习任务动作 (dedup/optimize)"
    )
    
    args = parser.parse_args()
    
    # 执行命令
    if args.command == "check":
        return 0 if check_dependencies() else 1
    elif args.command == "api":
        if check_dependencies():
            start_api_server(args.port)
            return 0
        return 1
    elif args.command == "sync":
        if check_dependencies():
            start_sync_service()
            return 0
        return 1
    elif args.command == "summarize":
        return 0 if run_summarizer(args.agent) else 1
    elif args.command == "learn":
        return 0 if run_learner(args.action) else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
