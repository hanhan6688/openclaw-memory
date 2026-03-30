"""
知识图谱服务 - 使用 weaviate-client 4.x
处理实体关系存储、查询和可视化数据
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weaviate_client import WeaviateClient
from embeddings import OllamaEmbedding, OllamaChat


class KnowledgeGraph:
    """知识图谱服务"""
    
    def __init__(self, agent_id: str, user_id: str = "default"):
        self.client = WeaviateClient(agent_id, user_id)
        self.embedder = OllamaEmbedding()
        self.chat = OllamaChat()
        self.agent_id = agent_id
    
    def add_entity(self, entity_name: str, entity_type: str, 
                   source: str = None, confidence: float = 1.0) -> str:
        """添加实体"""
        # 生成实体嵌入向量
        vector = self.embedder.embed(f"{entity_name} {entity_type}")
        
        entity_id = self.client.insert_kg({
            "entity_name": entity_name,
            "entity_type": entity_type,
            "relation_type": "is_a",
            "target_entity": entity_type,
            "context": f"{entity_name} 是一个 {entity_type}",
            "source": source,
            "confidence": confidence,
            "access_count": 0
        }, vector)
        
        return entity_id
    
    def add_relation(self, source_entity: str, relation_type: str, 
                     target_entity: str, context: str = None,
                     source: str = None, confidence: float = 1.0) -> str:
        """添加关系"""
        # 生成关系嵌入向量
        relation_text = f"{source_entity} {relation_type} {target_entity}"
        vector = self.embedder.embed(relation_text)
        
        relation_id = self.client.insert_kg({
            "entity_name": source_entity,
            "entity_type": "entity",
            "relation_type": relation_type,
            "target_entity": target_entity,
            "context": context or relation_text,
            "source": source,
            "confidence": confidence,
            "access_count": 0
        }, vector)
        
        return relation_id
    
    def extract_and_store(self, text: str, source: str = None) -> Dict:
        """从文本提取实体关系并存储"""
        entities_data = self.chat.extract_entities(text)
        
        stored_entities = []
        stored_relations = []
        
        # 存储实体
        for entity in entities_data.get("entities", []):
            entity_id = self.add_entity(
                entity["name"], 
                entity.get("type", "entity"),
                source=source
            )
            stored_entities.append({"name": entity["name"], "id": entity_id})
        
        # 存储关系
        for relation in entities_data.get("relations", []):
            relation_id = self.add_relation(
                relation["source"],
                relation["relation"],
                relation["target"],
                context=text[:200],
                source=source
            )
            stored_relations.append({
                "source": relation["source"],
                "relation": relation["relation"],
                "target": relation["target"],
                "id": relation_id
            })
        
        return {
            "entities": stored_entities,
            "relations": stored_relations
        }
    
    def find_path(self, entity_a: str, entity_b: str, max_depth: int = 3) -> List[Dict]:
        """查找两个实体之间的路径"""
        # 获取所有知识图谱数据
        all_kg = self.client.get_kg(limit=500)
        
        # 筛选相关关系
        paths = []
        for obj in all_kg:
            source = obj.get("entity_name")
            target = obj.get("target_entity")
            if source == entity_a or target == entity_a or source == entity_b or target == entity_b:
                paths.append({
                    "source": source,
                    "relation": obj.get("relation_type"),
                    "target": target,
                    "context": obj.get("context")
                })
        
        return paths
    
    def get_all_entities(self) -> List[Dict]:
        """获取所有实体"""
        all_kg = self.client.get_kg(limit=500)
        
        entities = {}
        for obj in all_kg:
            name = obj.get("entity_name")
            if name and name not in entities:
                entities[name] = {
                    "name": name,
                    "type": obj.get("entity_type"),
                    "access_count": obj.get("access_count", 0)
                }
        
        return list(entities.values())
    
    def get_all_relations(self) -> List[Dict]:
        """获取所有关系"""
        all_kg = self.client.get_kg(limit=500)
        
        relations = []
        seen = set()
        
        for obj in all_kg:
            key = (obj.get("entity_name"), obj.get("relation_type"), obj.get("target_entity"))
            if key not in seen and obj.get("relation_type") != "is_a":
                seen.add(key)
                relations.append({
                    "source": obj.get("entity_name"),
                    "relation": obj.get("relation_type"),
                    "target": obj.get("target_entity"),
                    "context": obj.get("context"),
                    "confidence": obj.get("confidence")
                })
        
        return relations
    
    def get_graph_data(self) -> Dict:
        """获取知识图谱可视化数据（用于前端可视化）"""
        entities = self.get_all_entities()
        relations = self.get_all_relations()
        
        # 转换为可视化格式
        nodes = [
            {
                "id": e["name"],
                "label": e["name"],
                "group": e.get("type", "entity"),
                "size": 10 + e.get("access_count", 0) * 2
            }
            for e in entities
        ]
        
        links = [
            {
                "source": r["source"],
                "target": r["target"],
                "relation": r["relation"],
                "value": r.get("confidence", 1.0)
            }
            for r in relations
            if r["source"] and r["target"]
        ]
        
        return {
            "nodes": nodes,
            "links": links
        }
    

    def search_similar_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """
        向量检索相似实体
        
        Args:
            query: 查询文本
            limit: 返回数量
        
        Returns:
            相似实体列表
        """
        query_vector = self.embedder.embed(query)
        results = self.client.search_kg_entities(query_vector, limit)
        return results

    def search_relations(self, query: str, limit: int = 10) -> List[Dict]:
        """
        混合检索关系 (向量 + BM25)
        
        Args:
            query: 查询文本
            limit: 返回数量
        """
        try:
            collection = self.client.client.collections.get(self.client.kg_collection)
            
            # 生成查询向量
            query_vector = self.embedder.embed(query)
            
            # 使用 Weaviate 原生混合检索，手动提供向量
            results = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=0.5,
                limit=limit
            )
            
            relations = []
            for obj in results.objects:
                relations.append({
                    "id": str(obj.uuid),
                    "entity_name": obj.properties.get("entity_name"),
                    "entity_type": obj.properties.get("entity_type"),
                    "relation_type": obj.properties.get("relation_type"),
                    "target_entity": obj.properties.get("target_entity"),
                    "context": obj.properties.get("context"),
                    "confidence": obj.properties.get("confidence"),
                    "score": obj.metadata.score if obj.metadata else None
                })
            return relations
            
        except Exception as e:
            print(f"⚠️ 混合检索出错: {e}")
            return self.search_similar_entities(query, limit)

    def get_entity_neighbors(self, entity_name: str) -> Dict:
        """
        获取实体的邻居节点 (图遍历)
        
        Args:
            entity_name: 实体名称
        """
        all_kg = self.client.get_kg(limit=500)
        
        outgoing = []
        incoming = []
        neighbors = set()
        
        for obj in all_kg:
            source = obj.get("entity_name")
            target = obj.get("target_entity")
            
            if source == entity_name:
                outgoing.append(obj)
                neighbors.add(target)
            elif target == entity_name:
                incoming.append(obj)
                neighbors.add(source)
        
        neighbor_info = []
        for neighbor in neighbors:
            for obj in all_kg:
                if obj.get("entity_name") == neighbor:
                    neighbor_info.append({
                        "name": neighbor,
                        "type": obj.get("entity_type")
                    })
                    break
        
        return {
            "entity": entity_name,
            "outgoing": outgoing,
            "incoming": incoming,
            "neighbors": neighbor_info
        }

    def hebbian_learning(self, entity_name: str):
        """Hebbian 学习 - 访问时增强连接"""
        # 在 REST API 版本中，这需要额外实现
        pass
    
    def detect_contradictions(self) -> List[Dict]:
        """检测矛盾关系"""
        relations = self.get_all_relations()
        
        contradictions = []
        opposite_relations = {
            "合作": "竞争",
            "依赖": "独立",
            "喜欢": "讨厌"
        }
        
        for r in relations:
            opposite = opposite_relations.get(r["relation"])
            if opposite:
                for other in relations:
                    if (other["source"] == r["source"] and 
                        other["target"] == r["target"] and 
                        other["relation"] == opposite):
                        contradictions.append({
                            "entity": r["source"],
                            "relation_a": r,
                            "relation_b": other
                        })
        
        return contradictions
    
    def close(self):
        """关闭连接"""
        self.client.close()


if __name__ == "__main__":
    # 测试知识图谱
    print("🧪 测试知识图谱...")
    kg = KnowledgeGraph("test_agent")
    
    # 添加测试数据
    print("\n📝 添加实体和关系...")
    kg.add_entity("抖音", "平台")
    kg.add_entity("广告投放", "业务")
    kg.add_relation("抖音", "支持", "广告投放")
    kg.add_relation("广告投放", "包含", "信息流广告")
    kg.add_relation("广告投放", "包含", "搜索广告")
    
    # 提取测试
    print("\n🔍 测试实体提取...")
    result = kg.extract_and_store("张三正在和李四合作开发抖音广告投放系统")
    print(f"  实体: {result['entities']}")
    print(f"  关系: {result['relations']}")
    
    # 获取图谱数据
    print("\n📊 获取图谱数据...")
    graph_data = kg.get_graph_data()
    print(f"  节点数: {len(graph_data['nodes'])}")
    print(f"  边数: {len(graph_data['links'])}")