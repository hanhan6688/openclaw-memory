#!/usr/bin/env python3
"""
快速清理精确重复记忆
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw_memory.core.weaviate_client import WeaviateClient
from collections import defaultdict

def cleanup_exact_duplicates(agent_id: str = "main", dry_run: bool = True):
    """清理精确重复记忆"""
    print(f"\n{'='*50}")
    print(f"清理精确重复记忆 - {agent_id}")
    print(f"模式: {'预览' if dry_run else '删除'}")
    print(f"{'='*50}\n")
    
    client = WeaviateClient(agent_id)
    c = client.client
    
    if not c:
        print("❌ 无法连接 Weaviate")
        return
    
    try:
        collection = c.collections.get(client.memory_collection)
        
        # 获取所有记忆
        print("📦 获取所有记忆...")
        all_memories = []
        after = None
        batch = 0
        
        while True:
            if after:
                results = collection.query.fetch_objects(limit=500, after=after)
            else:
                results = collection.query.fetch_objects(limit=500)
            
            if not results.objects:
                break
                
            for obj in results.objects:
                all_memories.append({
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "timestamp": str(obj.properties.get("timestamp", "")),
                })
            
            after = results.objects[-1].uuid
            batch += 1
            print(f"   已获取 {len(all_memories)} 条 (批次 {batch})...")
            
            if len(results.objects) < 500:
                break
        
        print(f"\n📊 总记忆数: {len(all_memories)}")
        
        # 按内容分组
        print("\n🔍 检测精确重复...")
        content_groups = defaultdict(list)
        for m in all_memories:
            content_groups[m["content"]].append(m)
        
        # 找出重复的
        duplicates_to_remove = []
        for content, memories in content_groups.items():
            if len(memories) > 1:
                # 保留最早的一条，删除其他
                sorted_memories = sorted(memories, key=lambda x: x["timestamp"])
                for m in sorted_memories[1:]:
                    duplicates_to_remove.append(m["id"])
        
        print(f"   精确重复: {len(duplicates_to_remove)} 条")
        
        if duplicates_to_remove:
            print(f"\n📋 前 10 条重复内容:")
            shown = 0
            for content, memories in content_groups.items():
                if len(memories) > 1 and shown < 10:
                    print(f"   - [{len(memories)} 次] {content[:60]}...")
                    shown += 1
        
        # 删除重复
        if not dry_run and duplicates_to_remove:
            print(f"\n🗑️ 删除重复记忆...")
            deleted = 0
            for i, mem_id in enumerate(duplicates_to_remove):
                try:
                    collection.data.delete_by_id(mem_id)
                    deleted += 1
                    if (i + 1) % 100 == 0:
                        print(f"   进度: {i + 1}/{len(duplicates_to_remove)}")
                except Exception as e:
                    print(f"   删除失败: {e}")
            print(f"   ✅ 已删除 {deleted} 条重复")
        elif dry_run and duplicates_to_remove:
            print(f"\n💡 运行 --delete 来删除这 {len(duplicates_to_remove)} 条重复")
        
        print(f"\n{'='*50}")
        print(f"📊 清理统计:")
        print(f"   原始记忆: {len(all_memories)}")
        print(f"   唯一内容: {len(content_groups)}")
        print(f"   重复数量: {len(duplicates_to_remove)}")
        if not dry_run:
            print(f"   清理后: {len(all_memories) - len(duplicates_to_remove)}")
        print(f"{'='*50}\n")
        
    finally:
        c.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清理精确重复记忆")
    parser.add_argument("--agent", default="main", help="Agent ID")
    parser.add_argument("--delete", action="store_true", help="执行删除（默认只预览）")
    args = parser.parse_args()
    
    cleanup_exact_duplicates(
        agent_id=args.agent,
        dry_run=not args.delete
    )