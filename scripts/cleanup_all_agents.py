#!/usr/bin/env python3
"""
清理所有 Agent 的重复记忆
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weaviate
from weaviate.connect import ConnectionParams
from collections import defaultdict

def cleanup_agent(client, col_name: str, dry_run: bool = True):
    """清理单个 agent 的重复记忆"""
    try:
        col = client.collections.get(col_name)
        
        # 获取所有记忆
        all_memories = []
        after = None
        
        while True:
            if after:
                results = col.query.fetch_objects(limit=500, after=after)
            else:
                results = col.query.fetch_objects(limit=500)
            
            if not results.objects:
                break
                
            for obj in results.objects:
                all_memories.append({
                    "id": str(obj.uuid),
                    "content": str(obj.properties.get("content", "")),
                    "timestamp": str(obj.properties.get("timestamp", "")),
                })
            
            if len(results.objects) < 500:
                break
            after = results.objects[-1].uuid
        
        if not all_memories:
            return 0, 0
        
        # 按内容分组
        content_groups = defaultdict(list)
        for m in all_memories:
            content_groups[m["content"]].append(m)
        
        # 找出重复
        duplicates = []
        for content, memories in content_groups.items():
            if len(memories) > 1:
                sorted_memories = sorted(memories, key=lambda x: x["timestamp"])
                for m in sorted_memories[1:]:
                    duplicates.append(m["id"])
        
        # 删除重复
        if not dry_run and duplicates:
            for mem_id in duplicates:
                try:
                    col.data.delete_by_id(mem_id)
                except Exception:
                    pass
        
        return len(all_memories), len(duplicates)
    
    except Exception as e:
        print(f"  错误: {e}")
        return 0, 0


def main(dry_run: bool = True):
    print("\n" + "=" * 60)
    print(f"清理所有 Agent 重复记忆 - {'预览' if dry_run else '删除'}模式")
    print("=" * 60 + "\n")
    
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="localhost",
            http_port=8080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
        ),
        skip_init_checks=True
    )
    client.connect()
    
    memory_collections = [
        "AgentMemory_main",
        "AgentMemory_copy", 
        "AgentMemory_default",
        "AgentMemory_hr",
    ]
    
    total_memories = 0
    total_duplicates = 0
    
    for col_name in memory_collections:
        print(f"📦 {col_name}...")
        total, dups = cleanup_agent(client, col_name, dry_run)
        if total > 0:
            print(f"   总数: {total}, 重复: {dups}")
            total_memories += total
            total_duplicates += dups
    
    client.close()
    
    print("\n" + "=" * 60)
    print(f"总计: {total_memories} 条记忆, {total_duplicates} 条重复")
    if dry_run and total_duplicates > 0:
        print(f"\n💡 运行 --delete 来删除这 {total_duplicates} 条重复")
    elif not dry_run:
        print(f"\n✅ 已删除 {total_duplicates} 条重复")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="执行删除")
    args = parser.parse_args()
    
    main(dry_run=not args.delete)