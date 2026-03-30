"""
Workspace 文件解析器
从 agent 的 workspace 目录中的 markdown 文件提取实体和关系
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class WorkspaceParser:
    """解析 workspace 目录中的 markdown 文件"""

    # 重要文件及其优先级
    IMPORTANT_FILES = {
        "SOUL.md": {"priority": 10, "type": "identity", "description": "Agent身份定义"},
        "USER.md": {"priority": 9, "type": "user", "description": "用户信息"},
        "IDENTITY.md": {"priority": 8, "type": "identity", "description": "身份信息"},
        "MEMORY.md": {"priority": 7, "type": "memory", "description": "长期记忆"},
        "AGENTS.md": {"priority": 6, "type": "config", "description": "Agent配置"},
        "TOOLS.md": {"priority": 5, "type": "config", "description": "工具配置"},
        "HEARTBEAT.md": {"priority": 4, "type": "task", "description": "定时任务"},
        "BOOTSTRAP.md": {"priority": 3, "type": "config", "description": "启动配置"},
    }

    # 实体类型映射
    ENTITY_TYPE_MAP = {
        # 人物相关
        "产品经理": "人物",
        "后端负责人": "人物",
        "工程师": "人物",
        "开发": "人物",
        "用户": "人物",

        # 项目相关
        "项目": "项目",
        "系统": "系统",
        "平台": "平台",

        # 技术相关
        "API": "技术",
        "数据库": "技术",
        "服务器": "技术",

        # 配置相关
        "密码": "配置",
        "token": "配置",
        "key": "配置",
    }

    def __init__(self, openclaw_root: str = None):
        """
        Args:
            openclaw_root: .openclaw 根目录路径
        """
        self.openclaw_root = openclaw_root or os.path.expanduser("~/.openclaw")

    def get_workspace_path(self, agent_name: str) -> Optional[str]:
        """获取 agent 的 workspace 路径"""
        # 尝试多种可能的路径
        possible_paths = [
            os.path.join(self.openclaw_root, f"workspace-{agent_name}"),
            os.path.join(self.openclaw_root, "workspace"),  # main agent
            os.path.join(self.openclaw_root, "agents", agent_name, "workspace"),
        ]

        # main agent 使用默认 workspace
        if agent_name == "main":
            possible_paths.insert(0, os.path.join(self.openclaw_root, "workspace"))
        elif agent_name == "copy":
            possible_paths.insert(0, os.path.join(self.openclaw_root, "workspace-copy"))

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def list_markdown_files(self, agent_name: str) -> List[Dict]:
        """列出 workspace 中的所有 markdown 文件"""
        workspace_path = self.get_workspace_path(agent_name)
        if not workspace_path:
            return []

        files = []
        for root, dirs, filenames in os.walk(workspace_path):
            # 跳过 node_modules 等目录
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__']]

            for filename in filenames:
                if filename.endswith('.md'):
                    filepath = os.path.join(root, filename)
                    relpath = os.path.relpath(filepath, workspace_path)

                    # 获取文件信息
                    stat = os.stat(filepath)
                    file_info = self.IMPORTANT_FILES.get(filename, {
                        "priority": 1,
                        "type": "document",
                        "description": relpath
                    })

                    files.append({
                        "path": filepath,
                        "relative_path": relpath,
                        "filename": filename,
                        "priority": file_info["priority"],
                        "type": file_info["type"],
                        "description": file_info["description"],
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

        # 按优先级排序
        files.sort(key=lambda x: (-x["priority"], x["filename"]))
        return files

    def read_file(self, filepath: str) -> Optional[str]:
        """读取文件内容"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取文件失败 {filepath}: {e}")
            return None

    def extract_entities_from_content(self, content: str, filename: str) -> Dict:
        """从内容中提取实体和关系"""
        result = {
            "entities": [],
            "relations": [],
            "source": filename
        }

        # 提取标题作为实体
        titles = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for title in titles:
            title = title.strip()
            if len(title) > 2 and len(title) < 50:
                result["entities"].append({
                    "name": title,
                    "type": "概念",
                    "source": "title"
                })

        # 提取键值对（如 Name: xxx）
        kv_pairs = re.findall(r'[-*]\s+\*\*([^*]+)\*\*[：:]\s*(.+?)(?:\n|$)', content)
        for key, value in kv_pairs:
            key = key.strip()
            value = value.strip()
            if value and len(value) < 100:
                entity_type = self.ENTITY_TYPE_MAP.get(key, "属性")
                result["entities"].append({
                    "name": value,
                    "type": entity_type,
                    "source": f"field:{key}"
                })
                result["relations"].append({
                    "source": filename.replace('.md', ''),
                    "relation": "has_" + key.lower().replace(' ', '_'),
                    "target": value
                })

        # 提取列表项
        list_items = re.findall(r'^[-*]\s+(.+?)$', content, re.MULTILINE)
        for item in list_items:
            item = item.strip()
            # 跳过太长或太短的项
            if len(item) > 5 and len(item) < 100 and not item.startswith('['):
                # 尝试识别实体类型
                entity_type = self._infer_entity_type(item)
                result["entities"].append({
                    "name": item[:50],  # 限制长度
                    "type": entity_type,
                    "source": "list_item"
                })

        # 提取特殊模式
        # 密码模式
        passwords = re.findall(r'密码[是为：:]+\s*(\S+)', content)
        for pwd in passwords:
            result["entities"].append({
                "name": "系统密码",
                "type": "配置",
                "source": "password_field"
            })

        # 角色/身份
        roles = re.findall(r'(?:我是|你是|负责)[，,。]?\s*([^，,。\n]{2,20})', content)
        for role in roles:
            role = role.strip()
            if role:
                result["entities"].append({
                    "name": role,
                    "type": "角色",
                    "source": "role_definition"
                })

        return result

    def _infer_entity_type(self, text: str) -> str:
        """推断实体类型"""
        text_lower = text.lower()

        # 技术相关
        if any(kw in text_lower for kw in ['api', 'http', 'server', '数据库', '服务器', 'ssh', 'git']):
            return "技术"

        # 人物相关
        if any(kw in text for kw in ['经理', '工程师', '负责人', '开发', '用户', '我']):
            return "人物"

        # 项目相关
        if any(kw in text for kw in ['项目', '系统', '平台', '应用']):
            return "项目"

        # 配置相关
        if any(kw in text_lower for kw in ['password', 'token', 'key', '密码', '配置']):
            return "配置"

        return "概念"

    def parse_workspace(self, agent_name: str, max_files: int = 10) -> Dict:
        """解析整个 workspace"""
        files = self.list_markdown_files(agent_name)

        if not files:
            return {
                "agent": agent_name,
                "workspace_found": False,
                "files_processed": 0,
                "entities": [],
                "relations": []
            }

        all_entities = []
        all_relations = []

        for file_info in files[:max_files]:
            content = self.read_file(file_info["path"])
            if not content:
                continue

            # 提取实体
            extracted = self.extract_entities_from_content(content, file_info["filename"])

            # 添加文件来源信息
            for entity in extracted["entities"]:
                entity["file"] = file_info["relative_path"]
                entity["file_type"] = file_info["type"]

            for relation in extracted["relations"]:
                relation["file"] = file_info["relative_path"]

            all_entities.extend(extracted["entities"])
            all_relations.extend(extracted["relations"])

        # 去重
        seen_entities = set()
        unique_entities = []
        for e in all_entities:
            key = (e["name"], e["type"])
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(e)

        seen_relations = set()
        unique_relations = []
        for r in all_relations:
            key = (r["source"], r["relation"], r["target"])
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)

        return {
            "agent": agent_name,
            "workspace_found": True,
            "workspace_path": self.get_workspace_path(agent_name),
            "files_processed": min(len(files), max_files),
            "total_files": len(files),
            "entities": unique_entities,
            "relations": unique_relations
        }

    def get_file_summary(self, agent_name: str) -> List[Dict]:
        """获取文件摘要列表"""
        files = self.list_markdown_files(agent_name)

        summary = []
        for f in files:
            content = self.read_file(f["path"])
            if content:
                # 提取第一段作为摘要
                lines = content.strip().split('\n')
                first_content = ""
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('---'):
                        first_content = line[:100]
                        break

                summary.append({
                    "filename": f["filename"],
                    "type": f["type"],
                    "description": f["description"],
                    "size": f["size"],
                    "modified": f["modified"],
                    "preview": first_content
                })

        return summary


if __name__ == "__main__":
    # 测试
    print("🧪 测试 Workspace 解析器...")

    parser = WorkspaceParser()

    for agent in ["main", "copy"]:
        print(f"\n=== {agent} ===")
        result = parser.parse_workspace(agent)

        if result["workspace_found"]:
            print(f"Workspace: {result['workspace_path']}")
            print(f"文件数: {result['files_processed']}/{result['total_files']}")
            print(f"实体数: {len(result['entities'])}")
            print(f"关系数: {len(result['relations'])}")

            print("\n实体示例:")
            for e in result['entities'][:5]:
                print(f"  - {e['name']} ({e['type']})")

            print("\n关系示例:")
            for r in result['relations'][:5]:
                print(f"  - {r['source']} -> {r['relation']} -> {r['target']}")
        else:
            print("Workspace 未找到")