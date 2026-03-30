"""
用户画像存储服务
================

存储和管理用户画像数据，支持：
- 基础属性（年龄/性别/地区）
- 消费特征（价格敏感度/消费档次）
- 兴趣标签（类目偏好/品牌偏好）
- 行为特征（活跃时段/购买频次）
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


class UserProfileStore:
    """用户画像存储服务"""
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.client = CommerceWeaviateClient(agent_id, "user_profiles")
        self.embedder = OllamaEmbedding()
        self.CLASS_NAME = f"UserProfile_{agent_id}"
    
    def _default_profile(self, user_id: str) -> Dict:
        """创建默认用户画像"""
        return {
            "user_id": user_id,
            "agent_id": self.agent_id,
            "basic": {
                "age_range": None,
                "gender": None,
                "region": None,
                "member_level": None
            },
            "consumption": {
                "price_sensitivity": "medium",
                "avg_order_value": 0,
                "purchase_frequency": None,
                "preferred_price_range": [0, 10000],
                "total_orders": 0,
                "total_spent": 0
            },
            "interests": {
                "categories": [],
                "brands": [],
                "styles": [],
                "keywords": []
            },
            "behavior": {
                "active_hours": [],
                "browse_to_buy_ratio": 0,
                "favorite_channels": [],
                "last_active": None
            },
            "tags": [],
            "insights": [],
            "created_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            "updated_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }
    
    def _serialize_profile(self, profile: Dict) -> Dict:
        """将画像序列化为 Weaviate 存储格式"""
        def to_str(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            return str(val)

        return {
            "user_id": profile.get("user_id"),
            "agent_id": profile.get("agent_id"),
            "basic": json.dumps(profile.get("basic", {}), ensure_ascii=False, default=str),
            "consumption": json.dumps(profile.get("consumption", {}), ensure_ascii=False, default=str),
            "interests": json.dumps(profile.get("interests", {}), ensure_ascii=False, default=str),
            "behavior": json.dumps(profile.get("behavior", {}), ensure_ascii=False, default=str),
            "tags": json.dumps(profile.get("tags", []), ensure_ascii=False, default=str),
            "insights": json.dumps(profile.get("insights", []), ensure_ascii=False, default=str),
            "created_at": to_str(profile.get("created_at")),
            "updated_at": to_str(profile.get("updated_at"))
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
            "user_id": obj.get("user_id"),
            "agent_id": obj.get("agent_id"),
            "basic": parse_json(obj.get("basic"), {}),
            "consumption": parse_json(obj.get("consumption"), {}),
            "interests": parse_json(obj.get("interests"), {}),
            "behavior": parse_json(obj.get("behavior"), {}),
            "tags": parse_json(obj.get("tags"), []),
            "insights": parse_json(obj.get("insights"), []),
            "created_at": obj.get("created_at"),
            "updated_at": obj.get("updated_at")
        }
    
    def get_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        try:
            results = self.client.query_objects(
                class_name=self.CLASS_NAME,
                where={"path": ["user_id"], "operator": "Equal", "valueString": user_id},
                limit=1
            )
            
            if results:
                return self._deserialize_profile(results[0])
        except Exception as e:
            print(f"获取用户画像失败: {e}")
        
        return self._default_profile(user_id)
    
    def create_profile(self, user_id: str, profile_data: Dict = None) -> str:
        """创建用户画像"""
        profile = self._default_profile(user_id)
        
        if profile_data:
            profile = self._merge_profile(profile, profile_data)
        
        # 生成画像向量（用于相似用户检索）
        profile_text = self._profile_to_text(profile)
        vector = self.embedder.embed(profile_text)
        
        # 序列化存储
        serialized = self._serialize_profile(profile)
        
        profile_id = self.client.insert_object(
            class_name=self.CLASS_NAME,
            properties=serialized,
            vector=vector
        )
        
        return profile_id
    
    def update_profile(self, user_id: str, updates: Dict) -> bool:
        """更新用户画像"""
        existing = self.get_profile(user_id)
        
        if not existing.get("id"):
            # 创建新画像
            self.create_profile(user_id, updates)
            return True
        
        # 合并更新
        updated = self._merge_profile(existing, updates)
        updated["updated_at"] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        # 重新生成向量
        profile_text = self._profile_to_text(updated)
        vector = self.embedder.embed(profile_text)
        
        # 序列化存储
        serialized = self._serialize_profile(updated)
        
        # 更新
        self.client.update_object(
            class_name=self.CLASS_NAME,
            object_id=existing["id"],
            properties=serialized,
            vector=vector
        )
        
        return True
    
    def _merge_profile(self, base: Dict, updates: Dict) -> Dict:
        """合并画像数据"""
        import copy
        result = copy.deepcopy(base)  # 使用deepcopy避免datetime序列化问题
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 深度合并字典
                result[key] = {**result[key], **value}
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # 合并列表并去重
                result[key] = list(set(result[key] + value))
            else:
                result[key] = value
        
        return result
    
    def _profile_to_text(self, profile: Dict) -> str:
        """将画像转换为文本（用于生成向量）"""
        parts = []
        
        # 基础信息
        basic = profile.get("basic", {})
        if basic.get("age_range"):
            parts.append(f"年龄段:{basic['age_range']}")
        if basic.get("gender"):
            parts.append(f"性别:{basic['gender']}")
        if basic.get("region"):
            parts.append(f"地区:{basic['region']}")
        
        # 消费特征
        consumption = profile.get("consumption", {})
        if consumption.get("price_sensitivity"):
            parts.append(f"价格敏感度:{consumption['price_sensitivity']}")
        if consumption.get("avg_order_value"):
            parts.append(f"平均客单价:{consumption['avg_order_value']}")
        
        # 兴趣
        interests = profile.get("interests", {})
        if interests.get("categories"):
            parts.append(f"偏好类目:{','.join(interests['categories'][:5])}")
        if interests.get("brands"):
            parts.append(f"偏好品牌:{','.join(interests['brands'][:5])}")
        if interests.get("styles"):
            parts.append(f"偏好风格:{','.join(interests['styles'][:5])}")
        
        # 标签
        if profile.get("tags"):
            parts.append(f"标签:{','.join(profile['tags'][:10])}")
        
        return " ".join(parts)
    
    def add_behavior(self, user_id: str, behavior_type: str, item_id: str, 
                     item_info: Dict = None) -> bool:
        """记录用户行为（浏览/购买/收藏/加购）"""
        existing = self.get_profile(user_id)
        
        if not existing.get("id"):
            self.create_profile(user_id)
            existing = self.get_profile(user_id)
        
        updates = {}
        
        # 更新行为统计
        if behavior_type == "purchase":
            consumption = existing.get("consumption", {})
            price = item_info.get("price", 0) if item_info else 0
            total_orders = consumption.get("total_orders", 0) + 1
            total_spent = consumption.get("total_spent", 0) + price
            updates["consumption"] = {
                "total_orders": total_orders,
                "total_spent": total_spent,
                "avg_order_value": total_spent / total_orders if total_orders > 0 else 0
            }
        
        # 更新兴趣
        if item_info:
            interests = existing.get("interests", {})
            categories = interests.get("categories", [])
            category = item_info.get("category")
            if category and category not in categories:
                categories.append(category)
                updates["interests"] = {"categories": categories[:20]}
        
        # 更新活跃时间
        updates["behavior"] = {
            "last_active": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }
        
        return self.update_profile(user_id, updates)
    
    def add_tag(self, user_id: str, tag: str) -> bool:
        """添加用户标签"""
        existing = self.get_profile(user_id)
        
        if not existing.get("id"):
            self.create_profile(user_id)
            existing = self.get_profile(user_id)
        
        tags = existing.get("tags", [])
        if tag not in tags:
            tags.append(tag)
            return self.update_profile(user_id, {"tags": tags})
        
        return True
    
    def add_insight(self, user_id: str, insight: str) -> bool:
        """添加用户洞察"""
        existing = self.get_profile(user_id)
        
        if not existing.get("id"):
            self.create_profile(user_id)
            existing = self.get_profile(user_id)
        
        insights = existing.get("insights", [])
        if insight not in insights:
            insights.append(insight)
            # 保留最近50条洞察
            insights = insights[-50:]
            return self.update_profile(user_id, {"insights": insights})
        
        return True
    
    def find_similar_users(self, user_id: str, limit: int = 10) -> List[Dict]:
        """查找相似用户"""
        existing = self.get_profile(user_id)
        
        if not existing.get("id"):
            return []
        
        # 使用向量检索相似用户
        profile_text = self._profile_to_text(existing)
        vector = self.embedder.embed(profile_text)
        
        results = self.client.vector_search(
            class_name=self.CLASS_NAME,
            vector=vector,
            limit=limit + 1  # 包含自己
        )
        
        similar_users = []
        for obj in results:
            obj_id = obj.get("_additional", {}).get("id")
            if obj_id != existing.get("id"):  # 排除自己
                deserialized = self._deserialize_profile(obj)
                similar_users.append({
                    "user_id": deserialized.get("user_id"),
                    "similarity": 1 - obj.get("_additional", {}).get("distance", 0),
                    "tags": deserialized.get("tags", []),
                    "interests": deserialized.get("interests", {})
                })
        
        return similar_users[:limit]
    
    def get_all_profiles(self, limit: int = 100) -> List[Dict]:
        """获取所有用户画像"""
        results = self.client.get_objects(self.CLASS_NAME, limit=limit)
        return [self._deserialize_profile(obj) for obj in results]
    
    def delete_profile(self, user_id: str) -> bool:
        """删除用户画像"""
        existing = self.get_profile(user_id)
        
        if existing.get("id"):
            self.client.delete_object(self.CLASS_NAME, existing["id"])
            return True
        
        return False
    
    def close(self):
        """关闭连接"""
        self.client.close()


if __name__ == "__main__":
    # 测试用户画像
    print("🧪 测试用户画像存储...")
    store = UserProfileStore("test_agent")
    
    # 创建画像
    print("\n📝 创建用户画像...")
    profile_id = store.create_profile("user_001", {
        "basic": {
            "age_range": "25-34",
            "gender": "female",
            "region": "上海"
        },
        "interests": {
            "categories": ["女装", "美妆"],
            "brands": ["ZARA", "优衣库"]
        },
        "tags": ["品质优先", "直播活跃"]
    })
    print(f"✅ 创建成功: {profile_id}")
    
    # 获取画像
    print("\n🔍 获取用户画像...")
    profile = store.get_profile("user_001")
    print(f"  用户ID: {profile.get('user_id')}")
    print(f"  标签: {profile.get('tags')}")
    
    store.close()