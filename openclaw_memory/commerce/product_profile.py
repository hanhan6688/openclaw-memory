"""
商品画像存储服务
================

存储和管理商品画像数据，支持：
- 基础属性（名称/价格/类目/品牌）
- 销售数据（销量/评价/转化率）
- 标签体系（风格/场景/人群）
- 向量嵌入（支持相似商品检索）
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weaviate_client import CommerceWeaviateClient
from embeddings import OllamaEmbedding


class ProductProfileStore:
    """商品画像存储服务"""
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.client = CommerceWeaviateClient(agent_id, "product_profiles")
        self.embedder = OllamaEmbedding()
        self.CLASS_NAME = f"ProductProfile_{agent_id}"
    
    def _default_profile(self, product_id: str) -> Dict:
        """创建默认商品画像"""
        return {
            "product_id": product_id,
            "agent_id": self.agent_id,
            "basic": {
                "name": None,
                "price": 0,
                "original_price": 0,
                "category": [],
                "brand": None,
                "sku": None,
                "description": None,
                "images": [],
                "status": "active"
            },
            "sales": {
                "monthly_sales": 0,
                "total_sales": 0,
                "total_reviews": 0,
                "avg_rating": 0,
                "conversion_rate": 0,
                "return_rate": 0
            },
            "tags": {
                "style": [],
                "scene": [],
                "crowd": [],
                "season": [],
                "custom": []
            },
            "keywords": {
                "positive": [],
                "negative": [],
                "search": []
            },
            "relations": {
                "similar_products": [],
                "complementary": [],
                "competitors": [],
                "alternatives": []
            },
            "inventory": {
                "stock": 0,
                "sku_variants": [],
                "low_stock_threshold": 10
            },
            "pricing": {
                "cost": 0,
                "margin": 0,
                "discount_history": []
            },
            "created_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            "updated_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }
    
    def _serialize_profile(self, profile: Dict) -> Dict:
        """将画像序列化为 Weaviate 存储格式"""
        return {
            "product_id": profile.get("product_id"),
            "agent_id": profile.get("agent_id"),
            "basic": json.dumps(profile.get("basic", {}), ensure_ascii=False),
            "sales": json.dumps(profile.get("sales", {}), ensure_ascii=False),
            "tags": json.dumps(profile.get("tags", {}), ensure_ascii=False),
            "keywords": json.dumps(profile.get("keywords", {}), ensure_ascii=False),
            "relations": json.dumps(profile.get("relations", {}), ensure_ascii=False),
            "inventory": json.dumps(profile.get("inventory", {}), ensure_ascii=False),
            "pricing": json.dumps(profile.get("pricing", {}), ensure_ascii=False),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at")
        }
    
    def _deserialize_profile(self, obj: Dict) -> Dict:
        """将 Weaviate 存储格式反序列化为画像"""
        def parse_json(value, default):
            if value is None:
                return default
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    return default
            return value
        
        return {
            "id": obj.get("_additional", {}).get("id"),
            "product_id": obj.get("product_id"),
            "agent_id": obj.get("agent_id"),
            "basic": parse_json(obj.get("basic"), {}),
            "sales": parse_json(obj.get("sales"), {}),
            "tags": parse_json(obj.get("tags"), {}),
            "keywords": parse_json(obj.get("keywords"), {}),
            "relations": parse_json(obj.get("relations"), {}),
            "inventory": parse_json(obj.get("inventory"), {}),
            "pricing": parse_json(obj.get("pricing"), {}),
            "created_at": obj.get("created_at"),
            "updated_at": obj.get("updated_at")
        }
    
    def get_profile(self, product_id: str) -> Dict:
        """获取商品画像"""
        try:
            results = self.client.query_objects(
                class_name=self.CLASS_NAME,
                where={"path": ["product_id"], "operator": "Equal", "valueString": product_id},
                limit=1
            )
            
            if results:
                return self._deserialize_profile(results[0])
        except Exception as e:
            print(f"获取商品画像失败: {e}")
        
        return self._default_profile(product_id)
    
    def create_profile(self, product_id: str, profile_data: Dict = None) -> str:
        """创建商品画像"""
        profile = self._default_profile(product_id)
        
        if profile_data:
            profile = self._merge_profile(profile, profile_data)
        
        # 生成商品向量
        product_text = self._product_to_text(profile)
        vector = self.embedder.embed(product_text)
        
        # 序列化存储
        serialized = self._serialize_profile(profile)
        
        profile_id = self.client.insert_object(
            class_name=self.CLASS_NAME,
            properties=serialized,
            vector=vector
        )
        
        return profile_id
    
    def update_profile(self, product_id: str, updates: Dict) -> bool:
        """更新商品画像"""
        existing = self.get_profile(product_id)
        
        if not existing.get("id"):
            self.create_profile(product_id, updates)
            return True
        
        # 合并更新
        updated = self._merge_profile(existing, updates)
        updated["updated_at"] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        # 重新生成向量
        product_text = self._product_to_text(updated)
        vector = self.embedder.embed(product_text)
        
        # 序列化存储
        serialized = self._serialize_profile(updated)
        
        self.client.update_object(
            class_name=self.CLASS_NAME,
            object_id=existing["id"],
            properties=serialized,
            vector=vector
        )
        
        return True
    
    def _merge_profile(self, base: Dict, updates: Dict) -> Dict:
        """合并画像数据"""
        result = json.loads(json.dumps(base))  # 深拷贝
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = {**result[key], **value}
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = list(set(result[key] + value))
            else:
                result[key] = value
        
        return result
    
    def _product_to_text(self, profile: Dict) -> str:
        """将商品画像转换为文本（用于生成向量）"""
        parts = []
        
        # 基础信息
        basic = profile.get("basic", {})
        if basic.get("name"):
            parts.append(basic["name"])
        if basic.get("category"):
            parts.append(f"类目:{','.join(basic['category'])}")
        if basic.get("brand"):
            parts.append(f"品牌:{basic['brand']}")
        if basic.get("description"):
            parts.append(basic["description"][:200])
        
        # 标签
        tags = profile.get("tags", {})
        if tags.get("style"):
            parts.append(f"风格:{','.join(tags['style'])}")
        if tags.get("scene"):
            parts.append(f"场景:{','.join(tags['scene'])}")
        if tags.get("crowd"):
            parts.append(f"人群:{','.join(tags['crowd'])}")
        
        # 关键词
        keywords = profile.get("keywords", {})
        if keywords.get("positive"):
            parts.append(f"优点:{','.join(keywords['positive'][:5])}")
        if keywords.get("search"):
            parts.append(f"搜索词:{','.join(keywords['search'][:5])}")
        
        return " ".join(parts)
    
    def batch_import(self, products: List[Dict]) -> Dict:
        """批量导入商品"""
        imported = 0
        updated = 0
        failed = 0
        
        for product in products:
            product_id = product.get("product_id")
            if not product_id:
                failed += 1
                continue
            
            existing = self.get_profile(product_id)
            
            if existing.get("id"):
                self.update_profile(product_id, product)
                updated += 1
            else:
                self.create_profile(product_id, product)
                imported += 1
        
        return {
            "imported": imported,
            "updated": updated,
            "failed": failed,
            "total": len(products)
        }
    
    def find_similar_products(self, product_id: str, limit: int = 10) -> List[Dict]:
        """查找相似商品"""
        existing = self.get_profile(product_id)
        
        if not existing.get("id"):
            return []
        
        product_text = self._product_to_text(existing)
        vector = self.embedder.embed(product_text)
        
        results = self.client.vector_search(
            class_name=self.CLASS_NAME,
            vector=vector,
            limit=limit + 1
        )
        
        similar_products = []
        for obj in results:
            obj_id = obj.get("_additional", {}).get("id")
            if obj_id != existing.get("id"):
                deserialized = self._deserialize_profile(obj)
                similar_products.append({
                    "product_id": deserialized.get("product_id"),
                    "name": deserialized.get("basic", {}).get("name"),
                    "price": deserialized.get("basic", {}).get("price"),
                    "category": deserialized.get("basic", {}).get("category"),
                    "similarity": 1 - obj.get("_additional", {}).get("distance", 0),
                    "tags": deserialized.get("tags", {})
                })
        
        return similar_products[:limit]
    
    def search_by_text(self, query: str, limit: int = 10, 
                       filters: Dict = None) -> List[Dict]:
        """文本搜索商品"""
        vector = self.embedder.embed(query)
        
        results = self.client.vector_search(
            class_name=self.CLASS_NAME,
            vector=vector,
            limit=limit,
            filters=filters
        )
        
        products = []
        for obj in results:
            deserialized = self._deserialize_profile(obj)
            products.append({
                "product_id": deserialized.get("product_id"),
                "name": deserialized.get("basic", {}).get("name"),
                "price": deserialized.get("basic", {}).get("price"),
                "category": deserialized.get("basic", {}).get("category"),
                "sales": deserialized.get("sales", {}),
                "tags": deserialized.get("tags", {}),
                "similarity": 1 - obj.get("_additional", {}).get("distance", 0)
            })
        
        return products
    
    def search_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """按类目搜索商品"""
        results = self.client.query_objects(
            class_name=self.CLASS_NAME,
            limit=limit
        )
        
        # 过滤包含该类目的商品
        filtered = []
        for obj in results:
            deserialized = self._deserialize_profile(obj)
            categories = deserialized.get("basic", {}).get("category", [])
            if category in categories:
                filtered.append(deserialized)
        
        return filtered
    
    def update_sales_data(self, product_id: str, sales_data: Dict) -> bool:
        """更新销售数据"""
        return self.update_profile(product_id, {"sales": sales_data})
    
    def update_inventory(self, product_id: str, stock: int, 
                         sku_variants: List[Dict] = None) -> bool:
        """更新库存"""
        updates = {"inventory": {"stock": stock}}
        if sku_variants:
            updates["inventory"]["sku_variants"] = sku_variants
        return self.update_profile(product_id, updates)
    
    def add_relation(self, product_id: str, relation_type: str, 
                     related_product_id: str) -> bool:
        """添加商品关联"""
        existing = self.get_profile(product_id)
        
        if not existing.get("id"):
            return False
        
        relations = existing.get("relations", {})
        related_list = relations.get(relation_type, [])
        
        if related_product_id not in related_list:
            related_list.append(related_product_id)
            relations[relation_type] = related_list
            return self.update_profile(product_id, {"relations": relations})
        
        return True
    
    def get_all_products(self, limit: int = 100) -> List[Dict]:
        """获取所有商品"""
        results = self.client.get_objects(self.CLASS_NAME, limit=limit)
        return [self._deserialize_profile(obj) for obj in results]
    
    def delete_profile(self, product_id: str) -> bool:
        """删除商品画像"""
        existing = self.get_profile(product_id)
        
        if existing.get("id"):
            self.client.delete_object(self.CLASS_NAME, existing["id"])
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """获取商品统计信息"""
        products = self.get_all_products(limit=1000)
        
        if not products:
            return {"total": 0}
        
        stats = {
            "total": len(products),
            "categories": {},
            "brands": {},
            "price_ranges": {
                "0-50": 0,
                "50-100": 0,
                "100-300": 0,
                "300-500": 0,
                "500+": 0
            },
            "avg_rating": 0,
            "total_sales": 0
        }
        
        total_rating = 0
        rating_count = 0
        
        for p in products:
            # 类目统计
            for cat in p.get("basic", {}).get("category", []):
                stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            
            # 品牌统计
            brand = p.get("basic", {}).get("brand")
            if brand:
                stats["brands"][brand] = stats["brands"].get(brand, 0) + 1
            
            # 价格区间
            price = p.get("basic", {}).get("price", 0)
            if price <= 50:
                stats["price_ranges"]["0-50"] += 1
            elif price <= 100:
                stats["price_ranges"]["50-100"] += 1
            elif price <= 300:
                stats["price_ranges"]["100-300"] += 1
            elif price <= 500:
                stats["price_ranges"]["300-500"] += 1
            else:
                stats["price_ranges"]["500+"] += 1
            
            # 评分
            rating = p.get("sales", {}).get("avg_rating", 0)
            if rating > 0:
                total_rating += rating
                rating_count += 1
            
            # 销量
            stats["total_sales"] += p.get("sales", {}).get("monthly_sales", 0)
        
        if rating_count > 0:
            stats["avg_rating"] = round(total_rating / rating_count, 2)
        
        return stats
    
    def close(self):
        """关闭连接"""
        self.client.close()


if __name__ == "__main__":
    # 测试商品画像
    print("🧪 测试商品画像存储...")
    store = ProductProfileStore("test_agent")
    
    # 创建商品
    print("\n📝 创建商品画像...")
    product_id = store.create_profile("prod_001", {
        "basic": {
            "name": "法式复古连衣裙女夏季",
            "price": 268,
            "original_price": 399,
            "category": ["女装", "连衣裙", "复古风"],
            "brand": "自有品牌"
        },
        "tags": {
            "style": ["法式", "复古", "优雅"],
            "scene": ["约会", "度假", "通勤"],
            "crowd": ["25-35岁女性", "白领"]
        },
        "keywords": {
            "positive": ["显瘦", "质量好", "版型好看"],
            "search": ["连衣裙", "法式", "复古"]
        }
    })
    print(f"✅ 创建成功: {product_id}")
    
    # 获取商品
    print("\n🔍 获取商品画像...")
    profile = store.get_profile("prod_001")
    print(f"  商品名: {profile.get('basic', {}).get('name')}")
    print(f"  价格: {profile.get('basic', {}).get('price')}")
    
    store.close()