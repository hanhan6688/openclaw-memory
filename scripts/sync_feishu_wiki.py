#!/usr/bin/env python3
"""
飞书 Wiki 同步到本地 KnowledgeBase
====================================

功能：
1. 获取飞书 Wiki 空间和节点
2. 获取节点文档内容
3. 解析并存入本地 KnowledgeBase

依赖：
- pip install requests lark-parser

用法：
    python sync_feishu_wiki.py [--space-id SPACE_ID] [--dry-run]
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw_memory.knowledge_base import KnowledgeBase


class FeishuWikiSync:
    """飞书 Wiki 同步器"""
    
    def __init__(
        self,
        feishu_app_id: str = None,
        feishu_app_secret: str = None,
        knowledge_base: KnowledgeBase = None
    ):
        self.app_id = feishu_app_id or os.getenv("FEISHU_APP_ID")
        self.app_secret = feishu_app_secret or os.getenv("FEISHU_APP_SECRET")
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.access_token = None
    
    def _get_tenant_access_token(self) -> str:
        """获取 tenant_access_token"""
        import requests
        
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        response = requests.post(url, json={
            "app_id": self.app_id,
            "app_secret": self.app_secret
        })
        data = response.json()
        
        if data.get("code") != 0:
            raise Exception(f"获取 token 失败: {data}")
        
        return data["tenant_access_token"]
    
    def _get_wiki_spaces(self) -> List[Dict]:
        """获取 Wiki 空间列表"""
        import requests
        
        if not self.access_token:
            self.access_token = self._get_tenant_access_token()
        
        url = "https://open.feishu.cn/open-apis/wiki/v2/spaces"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data.get("code") != 0:
            raise Exception(f"获取 Wiki 空间失败: {data}")
        
        return data.get("data", {}).get("items", [])
    
    def _get_space_nodes(self, space_id: str) -> List[Dict]:
        """获取空间下的节点列表"""
        import requests
        
        if not self.access_token:
            self.access_token = self._get_tenant_access_token()
        
        url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/{space_id}/nodes"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data.get("code") != 0:
            raise Exception(f"获取节点失败: {data}")
        
        return data.get("data", {}).get("items", [])
    
    def _get_node_content(self, node_token: str) -> str:
        """获取节点文档内容 (docx)"""
        import requests
        
        if not self.access_token:
            self.access_token = self._get_tenant_access_token()
        
        # 获取文档块内容
        url = f"https://open.feishu.cn/open-apis/doc/v1/documents/{node_token}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data.get("code") != 0:
            print(f"⚠️ 获取文档失败: {data}")
            return ""
        
        # 提取文本内容
        doc_data = data.get("data", {})
        blocks = doc_data.get("document", {}).get("blocks", [])
        
        content_parts = []
        for block in blocks:
            block_type = block.get("type")
            
            if block_type == "paragraph":
                text_elems = block.get("paragraph", {}).get("elements", [])
                for elem in text_elems:
                    if "text_run" in elem:
                        content_parts.append(elem["text_run"].get("text", ""))
            
            elif block_type == "heading1":
                text_elems = block.get("heading1", {}).get("elements", [])
                for elem in text_elems:
                    if "text_run" in elem:
                        content_parts.append("## " + elem["text_run"].get("text", ""))
            
            elif block_type == "heading2":
                text_elems = block.get("heading2", {}).get("elements", [])
                for elem in text_elems:
                    if "text_run" in elem:
                        content_parts.append("### " + elem["text_run"].get("text", ""))
            
            elif block_type == "heading3":
                text_elems = block.get("heading3", {}).get("elements", [])
                for elem in text_elems:
                    if "text_run" in elem:
                        content_parts.append("#### " + elem["text_run"].get("text", ""))
            
            elif block_type == "code":
                content_parts.append("```\n" + block.get("code", {}).get("content", "") + "\n```")
            
            elif block_type == "quote":
                text_elems = block.get("quote", {}).get("elements", [])
                for elem in text_elems:
                    if "text_run" in elem:
                        content_parts.append("> " + elem["text_run"].get("text", ""))
        
        return "\n\n".join(content_parts)
    
    def sync_space(
        self,
        space_id: str,
        dry_run: bool = False
    ) -> Dict:
        """同步指定 Wiki 空间"""
        
        print(f"\n📂 开始同步 Wiki 空间: {space_id}")
        
        # 获取节点列表
        nodes = self._get_space_nodes(space_id)
        print(f"   找到 {len(nodes)} 个节点")
        
        results = {
            "total": len(nodes),
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        for node in nodes:
            node_token = node.get("node_token")
            title = node.get("title", "Untitled")
            obj_type = node.get("obj_type", "docx")
            
            print(f"\n   📄 处理: {title}")
            
            # 只处理 docx 类型
            if obj_type != "docx":
                print(f"      ⏭️ 跳过非 docx 类型: {obj_type}")
                results["failed"] += 1
                continue
            
            try:
                # 获取内容
                content = self._get_node_content(node_token)
                
                if not content.strip():
                    print(f"      ⚠️ 内容为空")
                    results["failed"] += 1
                    continue
                
                if dry_run:
                    print(f"      [DRY-RUN] 内容长度: {len(content)} 字符")
                    results["success"] += 1
                else:
                    # 写入临时文件
                    with tempfile.NamedTemporaryFile(
                        mode='w',
                        suffix='.md',
                        delete=False,
                        encoding='utf-8'
                    ) as f:
                        f.write(content)
                        temp_path = f.name
                    
                    try:
                        # 添加到知识库
                        result = self.knowledge_base.add_document(
                            file_path=temp_path,
                            title=title
                        )
                        
                        if result.get("success"):
                            print(f"      ✅ 已存储 {result.get('chunks')} 个块")
                            results["success"] += 1
                        else:
                            print(f"      ❌ 存储失败: {result.get('error')}")
                            results["failed"] += 1
                    finally:
                        os.unlink(temp_path)
                
                results["details"].append({
                    "title": title,
                    "status": "success" if not dry_run else "dry_run"
                })
                
            except Exception as e:
                print(f"      ❌ 错误: {e}")
                results["failed"] += 1
        
        return results
    
    def list_spaces(self):
        """列出所有 Wiki 空间"""
        spaces = self._get_wiki_spaces()
        
        print("\n📚 飞书 Wiki 空间列表:")
        print("-" * 50)
        
        for space in spaces:
            print(f"  ID: {space.get('space_id')}")
            print(f"  名称: {space.get('name')}")
            print(f"  类型: {space.get('space_type')}")
            print()


def main():
    parser = argparse.ArgumentParser(description="飞书 Wiki 同步到本地 KnowledgeBase")
    parser.add_argument("--space-id", "-s", help="指定 Wiki 空间 ID")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有 Wiki 空间")
    parser.add_argument("--dry-run", "-n", action="store_true", help="仅显示不存储")
    parser.add_argument("--app-id", help="飞书 App ID")
    parser.add_argument("--app-secret", help="飞书 App Secret")
    
    args = parser.parse_args()
    
    # 加载配置
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    
    # 创建同步器
    sync = FeishuWikiSync(
        feishu_app_id=args.app_id or os.getenv("FEISHU_APP_ID"),
        feishu_app_secret=args.app_secret or os.getenv("FEISHU_APP_SECRET")
    )
    
    if args.list:
        sync.list_spaces()
        return
    
    if not args.space_id:
        print("错误: 请指定 --space-id 或使用 --list 查看空间")
        print("\n用法:")
        print("  python sync_feishu_wiki.py --list                    # 查看空间列表")
        print("  python sync_feishu_wiki.py -s SPACE_ID               # 同步指定空间")
        print("  python sync_feishu_wiki.py -s SPACE_ID --dry-run     # 预览不存储")
        sys.exit(1)
    
    # 执行同步
    result = sync.sync_space(args.space_id, dry_run=args.dry_run)
    
    print("\n" + "=" * 50)
    print(f"📊 同步完成:")
    print(f"   总计: {result['total']}")
    print(f"   成功: {result['success']}")
    print(f"   失败: {result['failed']}")


if __name__ == "__main__":
    main()