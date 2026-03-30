"""
电商知识图谱服务
================

管理电商场景的实体关系网络，支持：
- 用户-商品关系（浏览/购买/收藏/加购）
- 商品-商品关系（相似/搭配/竞品）
- 用户-用户关系（相似用户）
- 智能推荐和关联分析
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weaviate_client import CommerceWeaviateClient
from embeddings import OllamaEmbedding, OllamaChat


class CommerceKnowledgeGraph:
    """电商知识图谱服务"""
    
    # 关系类型定义
    USER_PRODUCT_RELATIONS = [
        "viewed",        # 浏览
        "purchased",     # 购买
        "favorited",     # 收藏
        "cart_added",    # 加购
        "reviewed",     # 评价
        "shared",        # 分享
        "returned"      # 退货
    ]
    
    PRODUCT_PRODUCT_RELATIONS = [
        "similar",       # 相似
        "complementary", # 搭配
        "competitor",    # 竞品
        "alternative",   # 替代品
        "up_sell",       # 向上销售
        "cross_sell"    # 关联销售
    ]
    
    USER_USER_RELATIONS = [
        "similar",       # 相似用户
        "friend",        # 好友
        "follower"       # 关注
    ]
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.client = CommerceWeaviateClient(agent_id, "commerce_kg")
        self.embedder = OllamaEmbedding()
        self.chat = OllamaChat()
        self.CLASS_NAME = f"CommerceKG_{agent_id}"
    
    # ==================== 用户-商品关系 ====================
    
    def add_user_product_relation(self, user_id: str, relation_type: str,
                                   product_id: str, context: Dict = None) -> str:
        """添加用户-商品关系
        
        Args:
            user_id: 用户ID
            relation_type: 关系类型 (viewed/purchased/favorited/cart_added等)
            product_id: 商品ID
            context: 上下文信息（价格、时间、渠道等）
        """
        if relation_type not in self.USER_PRODUCT_RELATIONS:
            raise ValueError(f"无效的关系类型: {relation_type}")
        
        relation_text = f"用户{user_id}{relation_type}{product_id}"
        vector = self.embedder.embed(relation_text)
        
        relation_id = self.client.insert_object(
            class_name=self.CLASS_NAME,
            properties={
                "source_type": "user",
                "source_id": user_id,
                "target_type": "product",
                "target_id": product_id,
                "relation_type": relation_type,
                "context": json.dumps(context or {}, ensure_ascii=False),
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                "weight": self._get_relation_weight(relation_type),
                "agent_id": self.agent_id
            },
            vector=vector
        )
        
        return relation_id
    
    def _get_relation_weight(self, relation_type: str) -> float:
        """获取关系权重"""
        weights = {
            "viewed": 0.1,
            "cart_added": 0.3,
            "favorited": 0.4,
            "purchased": 1.0,
            "reviewed": 0.8,
            "shared": 0.6,
            "returned": -0.5
        }
        return weights.get(relation_type, 0.1)
    
    def get_user_products(self, user_id: str, relation_type: str = None,
                           limit: int = 50) -> List[Dict]:
        """获取用户关联的商品"""
        filters = {
            "path": ["source_id"],
            "operator": "Equal",
            "valueString": user_id
        }
        
        if relation_type:
            filters = {
                "operator": "And",
                "operands": [
                    filters,
                    {"path": ["relation_type"], "operator": "Equal", "valueString": relation_type}
                ]
            }
        
        results = self.client.query_objects(
            class_name=self.CLASS_NAME,
            where=filters,
            limit=limit
        )
        
        return [
            {
                "product_id": r.get("target_id"),
                "relation": r.get("relation_type"),
                "weight": r.get("weight"),
                "timestamp": r.get("timestamp"),
                "context": json.loads(r.get("context", "{}"))
            }
            for r in results
        ]
    
    def get_product_users(self, product_id: str, relation_type: str = None,
                          limit: int = 50) -> List[Dict]:
        """获取商品关联的用户"""
        filters = {
            "path": ["target_id"],
            "operator": "Equal",
            "valueString": product_id
        }
        
        if relation_type:
            filters = {
                "operator": "And",
                "operands": [
                    filters,
                    {"path": ["relation_type"], "operator": "Equal", "valueString": relation_type}
                ]
            }
        
        results = self.client.query_objects(
            class_name=self.CLASS_NAME,
            where=filters,
            limit=limit
        )
        
        return [
            {
                "user_id": r.get("source_id"),
                "relation": r.get("relation_type"),
                "weight": r.get("weight"),
                "timestamp": r.get("timestamp")
            }
            for r in results
        ]
    
    # ==================== 商品-商品关系 ====================
    
    def add_product_relation(self, product_a_id: str, relation_type: str,
                              product_b_id: str, context: Dict = None) -> str:
        """添加商品-商品关系"""
        if relation_type not in self.PRODUCT_PRODUCT_RELATIONS:
            raise ValueError(f"无效的关系类型: {relation_type}")
        
        relation_text = f"商品{product_a_id}{relation_type}{product_b_id}"
        vector = self.embedder.embed(relation_text)
        
        relation_id = self.client.insert_object(
            class_name=self.CLASS_NAME,
            properties={
                "source_type": "product",
                "source_id": product_a_id,
                "target_type": "product",
                "target_id": product_b_id,
                "relation_type": relation_type,
                "context": json.dumps(context or {}, ensure_ascii=False),
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                "weight": context.get("weight", 1.0) if context else 1.0,
                "agent_id": self.agent_id
            },
            vector=vector
        )
        
        return relation_id
    
    def get_related_products(self, product_id: str, relation_type: str = None,
                              limit: int = 20) -> List[Dict]:
        """获取关联商品"""
        filters = {
            "path": ["source_id"],
            "operator": "Equal",
            "valueString": product_id
        }
        
        if relation_type:
            filters = {
                "operator": "And",
                "operands": [
                    filters,
                    {"path": ["relation_type"], "operator": "Equal", "valueString": relation_type}
                ]
            }
        
        results = self.client.query_objects(
            class_name=self.CLASS_NAME,
            where=filters,
            limit=limit
        )
        
        return [
            {
                "product_id": r.get("target_id"),
                "relation": r.get("relation_type"),
                "weight": r.get("weight"),
                "context": json.loads(r.get("context", "{}"))
            }
            for r in results
            if r.get("target_type") == "product"
        ]
    
    # ==================== 用户-用户关系 ====================
    
    def add_user_relation(self, user_a_id: str, relation_type: str,
                           user_b_id: str, context: Dict = None) -> str:
        """添加用户-用户关系"""
        if relation_type not in self.USER_USER_RELATIONS:
            raise ValueError(f"无效的关系类型: {relation_type}")
        
        relation_text = f"用户{user_a_id}{relation_type}{user_b_id}"
        vector = self.embedder.embed(relation_text)
        
        relation_id = self.client.insert_object(
            class_name=self.CLASS_NAME,
            properties={
                "source_type": "user",
                "source_id": user_a_id,
                "target_type": "user",
                "target_id": user_b_id,
                "relation_type": relation_type,
                "context": json.dumps(context or {}, ensure_ascii=False),
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                "weight": context.get("weight", 1.0) if context else 1.0,
                "agent_id": self.agent_id
            },
            vector=vector
        )
        
        return relation_id
    
    def get_similar_users(self, user_id: str, limit: int = 20) -> List[Dict]:
        """获取相似用户"""
        results = self.client.query_objects(
            class_name=self.CLASS_NAME,
            where={
                "operator": "Or",
                "operands": [
                    {"path": ["source_id"], "operator": "Equal", "valueString": user_id},
                    {"path": ["target_id"], "operator": "Equal", "valueString": user_id}
                ]
            },
            limit=limit
        )
        
        similar_users = []
        for r in results:
            if r.get("source_type") == "user" and r.get("target_type") == "user":
                other_user = r.get("target_id") if r.get("source_id") == user_id else r.get("source_id")
                similar_users.append({
                    "user_id": other_user,
                    "relation": r.get("relation_type"),
                    "weight": r.get("weight")
                })
        
        return similar_users
    
    # ==================== 推荐相关 ====================
    
    def get_recommendations_for_user(self, user_id: str, limit: int = 20) -> List[Dict]:
        """为用户推荐商品（基于知识图谱）"""
        # 获取用户已交互的商品
        user_products = self.get_user_products(user_id)
        
        if not user_products:
            return []
        
        # 基于用户行为计算商品权重
        product_weights = {}
        for up in user_products:
            product_id = up["product_id"]
            weight = up["weight"]
            if product_id in product_weights:
                product_weights[product_id] += weight
            else:
                product_weights[product_id] = weight
        
        # 获取这些商品的关联商品
        recommendations = {}
        for product_id, base_weight in product_weights.items():
            related = self.get_related_products(product_id)
            for r in related:
                related_id = r["product_id"]
                relation_weight = r["weight"]
                score = base_weight * relation_weight * self._get_recommendation_multiplier(r["relation"])
                
                if related_id in recommendations:
                    recommendations[related_id] += score
                else:
                    recommendations[related_id] = score
        
        # 排除已购买的商品
        purchased_ids = {up["product_id"] for up in user_products if up["relation"] == "purchased"}
        
        # 排序返回
        sorted_recs = sorted(
            [(pid, score) for pid, score in recommendations.items() if pid not in purchased_ids],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"product_id": pid, "score": score}
            for pid, score in sorted_recs[:limit]
        ]
    
    def _get_recommendation_multiplier(self, relation_type: str) -> float:
        """获取推荐权重系数"""
        multipliers = {
            "similar": 0.8,
            "complementary": 1.2,
            "alternative": 0.6,
            "up_sell": 1.0,
            "cross_sell": 1.1,
            "competitor": 0.3
        }
        return multipliers.get(relation_type, 0.5)
    
    def get_frequently_bought_together(self, product_id: str, 
                                        limit: int = 10) -> List[Dict]:
        """获取经常一起购买的商品"""
        # 找到购买过该商品的用户
        buyers = self.get_product_users(product_id, relation_type="purchased")
        
        # 统计这些用户还购买了什么
        co_purchased = {}
        for buyer in buyers:
            user_products = self.get_user_products(buyer["user_id"], relation_type="purchased")
            for up in user_products:
                if up["product_id"] != product_id:
                    pid = up["product_id"]
                    co_purchased[pid] = co_purchased.get(pid, 0) + 1
        
        # 排序返回
        sorted_products = sorted(co_purchased.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"product_id": pid, "count": count}
            for pid, count in sorted_products[:limit]
        ]
    
    # ==================== 图谱可视化 ====================
    
    def get_graph_data(self, entity_type: str = None, entity_id: str = None,
                        depth: int = 2) -> Dict:
        """获取图谱可视化数据"""
        nodes = []
        links = []
        seen_nodes = set()
        seen_links = set()
        
        # 获取所有关系
        results = self.client.get_objects(self.CLASS_NAME, limit=500)
        
        for obj in results:
            source_type = obj.get("source_type")
            source_id = obj.get("source_id")
            target_type = obj.get("target_type")
            target_id = obj.get("target_id")
            relation = obj.get("relation_type")
            weight = obj.get("weight", 1.0)
            
            # 过滤
            if entity_id:
                if source_id != entity_id and target_id != entity_id:
                    continue
            
            if entity_type:
                if source_type != entity_type and target_type != entity_type:
                    continue
            
            # 添加节点
            if source_id not in seen_nodes:
                nodes.append({
                    "id": source_id,
                    "type": source_type,
                    "group": source_type
                })
                seen_nodes.add(source_id)
            
            if target_id not in seen_nodes:
                nodes.append({
                    "id": target_id,
                    "type": target_type,
                    "group": target_type
                })
                seen_nodes.add(target_id)
            
            # 添加边
            link_key = (source_id, relation, target_id)
            if link_key not in seen_links:
                links.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": relation,
                    "value": weight
                })
                seen_links.add(link_key)
        
        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "node_count": len(nodes),
                "link_count": len(links)
            }
        }
    
    # ==================== 分析功能 ====================
    
    def analyze_user_interests(self, user_id: str) -> Dict:
        """分析用户兴趣"""
        user_products = self.get_user_products(user_id)
        
        interests = {
            "categories": {},
            "price_ranges": [],
            "behaviors": {},
            "top_products": []
        }
        
        for up in user_products:
            relation = up["relation"]
            weight = up["weight"]
            
            # 统计行为
            interests["behaviors"][relation] = interests["behaviors"].get(relation, 0) + 1
            
            # 记录高分商品
            if weight >= 0.5:
                interests["top_products"].append({
                    "product_id": up["product_id"],
                    "relation": relation,
                    "weight": weight
                })
        
        # 排序
        interests["top_products"].sort(key=lambda x: x["weight"], reverse=True)
        interests["top_products"] = interests["top_products"][:10]
        
        return interests
    
    def analyze_product_performance(self, product_id: str) -> Dict:
        """分析商品表现"""
        users = self.get_product_users(product_id)
        related = self.get_related_products(product_id)
        
        performance = {
            "total_interactions": len(users),
            "interactions_by_type": {},
            "unique_users": len(set(u["user_id"] for u in users)),
            "related_products": len(related),
            "relations_by_type": {}
        }
        
        for u in users:
            relation = u["relation"]
            performance["interactions_by_type"][relation] = \
                performance["interactions_by_type"].get(relation, 0) + 1
        
        for r in related:
            relation = r["relation"]
            performance["relations_by_type"][relation] = \
                performance["relations_by_type"].get(relation, 0) + 1
        
        return performance
    
    def close(self):
        """关闭连接"""
        self.client.close()


if __name__ == "__main__":
    # 测试电商知识图谱
    print("🧪 测试电商知识图谱...")
    kg = CommerceKnowledgeGraph("test_agent")
    
    # 添加用户-商品关系
    print("\n📝 添加用户-商品关系...")
    kg.add_user_product_relation("user_001", "viewed", "prod_001", {"channel": "search"})
    kg.add_user_product_relation("user_001", "purchased", "prod_001", {"price": 268})
    kg.add_user_product_relation("user_001", "favorited", "prod_002")
    kg.add_user_product_relation("user_002", "purchased", "prod_001")
    kg.add_user_product_relation("user_002", "purchased", "prod_003")
    
    # 添加商品-商品关系
    print("\n📝 添加商品-商品关系...")
    kg.add_product_relation("prod_001", "similar", "prod_002", {"similarity": 0.85})
    kg.add_product_relation("prod_001", "complementary", "prod_003", {"confidence": 0.9})
    
    # 获取用户商品
    print("\n🔍 获取用户关联商品...")
    products = kg.get_user_products("user_001")
    for p in products:
        print(f"  - {p['product_id']}: {p['relation']} (权重: {p['weight']})")
    
    # 获取推荐
    print("\n🎯 获取推荐...")
    recs = kg.get_recommendations_for_user("user_001")
    for r in recs[:3]:
        print(f"  - {r['product_id']}: 分数 {r['score']:.2f}")
    
    # 分析用户兴趣
    print("\n📊 分析用户兴趣...")
    interests = kg.analyze_user_interests("user_001")
    print(f"  行为统计: {interests['behaviors']}")
    
    kg.close()