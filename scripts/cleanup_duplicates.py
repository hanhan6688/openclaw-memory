#!/usr/bin/env python3
"""
清理重复记忆脚本
通过向量相似度检测并删除重复的记忆
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw_memory.core.weaviate_client import WeaviateClient
from openclaw_memory.core.embeddings import OllamaEmbedding
from collections import defaultdict
from datetime import datetime

def cleanup_duplicates(agent_id: str = "main", similarity_threshold: float = 0.95, dry_run: bool = True):
    """
    清理重复记忆
    
    Args:
        agent_id: Agent ID
        similarity_threshold: 相似度阈值（0.95 = 95% 相似）
        dry_run: 是否只预览不删除
    """
    print(f"\n{'='*50}")
    print(f"清理重复记忆 - {agent_id}")
    print(f"相似度阈值: {similarity_threshold * 100}%")
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
        
        while True:
            if after:
                results = collection.query.fetch_objects(limit=100, after=after)
            else:
                results = collection.query.fetch_objects(limit=100)
            
            if not results.objects:
                break
                
            for obj in results.objects:
                all_memories.append({
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "summary": obj.properties.get("summary", ""),
                    "timestamp": str(obj.properties.get("timestamp", "")),
                })
            
            after = results.objects[-1].uuid
            print(f"   已获取 {len(all_memories)} 条...")
            
            if len(results.objects) < 100:
                break
        
        print(f"\n📊 总记忆数: {len(all_memories)}")
        
        # 按内容分组（精确匹配）
        print("\n🔍 检测精确重复...")
        content_groups = defaultdict(list)
        for m in all_memories:
            content_groups[m["content"]].append(m)
        
        exact_duplicates = sum(len(v) - 1 for v in content_groups.values() if len(v) > 1)
        print(f"   精确重复: {exact_duplicates} 条")
        
        # 检测相似重复（基于向量）
        print("\n🔍 检测相似重复...")
        embedder = OllamaEmbedding()
        
        # 只检查较长的记忆（短消息容易误判）
        long_memories = [m for m in all_memories if len(m["content"]) > 50]
        print(f"   需要检查的长记忆: {len(long_memories)} 条")
        
        duplicates_to_remove = []
        checked = set()
        
        for i, m1 in enumerate(long_memories[:100]):  # 先检查前 100 条
            if m1["id"] in checked:
                continue
                
            # 生成向量
            try:
                vector = embedder.embed(m1["summary"] or m1["content"][:200])
            except Exception:
                continue
            
            # 搜索相似记忆
            results = collection.query.near_vector(
                near_vector=vector,
                limit=5,
                return_metadata={"distance": True}
            )
            
            for obj in results.objects[1:]:  # 跳过自己
                distance = obj.metadata.distance
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    dup_id = str(obj.uuid)
                    if dup_id not in checked:
                        duplicates_to_remove.append({
                            "id": dup_id,
                            "content": str(obj.properties.get("content", ""))[:80],
                            "similarity": similarity,
                            "original": m1["content"][:80]
                        })
                        checked.add(dup_id)
            
            checked.add(m1["id"])
            
            if i % 20 == 0:
                print(f"   检查进度: {i}/{min(100, len(long_memories))}")
        
        print(f"\n📋 发现 {len(duplicates_to_remove)} 条相似重复:")
        for dup in duplicates_to_remove[:10]:
            print(f"   - [{dup['similarity']*100:.0f}%] {dup['content'][:50]}...")
        
        if len(duplicates_to_remove) > 10:
            print(f"   ... 还有 {len(duplicates_to_remove) - 10} 条")
        
        # 删除重复
        if not dry_run and duplicates_to_remove:
            print(f"\n🗑️ 删除重复记忆...")
            for dup in duplicates_to_remove:
                try:
                    collection.data.delete_by_id(dup["id"])
                except Exception as e:
                    print(f"   删除失败 {dup['id'][:8]}: {e}")
            print(f"   ✅ 已删除 {len(duplicates_to_remove)} 条重复")
        elif dry_run and duplicates_to_remove:
            print(f"\n💡 运行 --delete 来删除这些重复")
        
        # 统计
        print(f"\n{'='*50}")
        print(f"📊 清理统计:")
        print(f"   总记忆: {len(all_memories)}")
        print(f"   精确重复: {exact_duplicates}")
        print(f"   相似重复: {len(duplicates_to_remove)}")
        print(f"   可清理: {exact_duplicates + len(duplicates_to_remove)}")
        print(f"{'='*50}\n")
        
    finally:
        c.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清理重复记忆")
    parser.add_argument("--agent", default="main", help="Agent ID")
    parser.add_argument("--threshold", type=float, default=0.95, help="相似度阈值")
    parser.add_argument("--delete", action="store_true", help="执行删除（默认只预览）")
    args = parser.parse_args()
    
    cleanup_duplicates(
        agent_id=args.agent,
        similarity_threshold=args.threshold,
        dry_run=not args.delete
    )