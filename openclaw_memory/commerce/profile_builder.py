"""
画像构建器
==========

自动从对话和行为数据中构建用户画像和商品画像：
- 从对话提取用户偏好
- 从行为更新用户画像
- 自动标签生成
- 商品属性提取
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weaviate_client import WeaviateClient
from embeddings import OllamaEmbedding, OllamaChat
from .user_profile import UserProfileStore
from .product_profile import ProductProfileStore
from .commerce_kg import CommerceKnowledgeGraph


class ProfileBuilder:
    """画像构建器"""
    
    # 用户偏好提取提示词
    USER_PREFERENCE_PROMPT = """分析以下对话内容，提取用户的购物偏好和特征。

对话内容：
{conversation}

请以JSON格式返回以下信息（只返回JSON，不要其他内容）：
{{
    "age_range": "年龄段（如：18-24, 25-34, 35-44等）",
    "gender": "性别（male/female/unknown）",
    "region": "地区",
    "price_sensitivity": "价格敏感度（low/medium/high）",
    "preferred_categories": ["偏好类目列表"],
    "preferred_brands": ["偏好品牌列表"],
    "preferred_styles": ["偏好风格列表"],
    "interests": ["兴趣标签"],
    "shopping_habits": ["购物习惯描述"],
    "tags": ["用户标签"]
}}
"""
    
    # 商品属性提取提示词
    PRODUCT_ATTRIBUTE_PROMPT = """分析以下商品信息，提取商品属性和标签。

商品信息：
{product_info}

请以JSON格式返回以下信息（只返回JSON，不要其他内容）：
{{
    "name": "商品名称",
    "category": ["类目层级"],
    "brand": "品牌",
    "price_range": "价格区间（如：100-300）",
    "style_tags": ["风格标签"],
    "scene_tags": ["适用场景"],
    "crowd_tags": ["适用人群"],
    "season_tags": ["适用季节"],
    "keywords": ["关键词"],
    "selling_points": ["卖点"]
}}
"""
    
    # 行为分析提示词
    BEHAVIOR_ANALYSIS_PROMPT = """分析用户的购物行为，生成用户洞察。

用户行为数据：
{behavior_data}

用户画像：
{user_profile}

请以JSON格式返回用户洞察（只返回JSON，不要其他内容）：
{{
    "insights": ["洞察1", "洞察2", ...],
    "recommendations": ["推荐策略1", "推荐策略2", ...],
    "potential_needs": ["潜在需求1", "潜在需求2", ...]
}}
"""
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.user_store = UserProfileStore(agent_id)
        self.product_store = ProductProfileStore(agent_id)
        self.kg = CommerceKnowledgeGraph(agent_id)
        self.chat = OllamaChat()
        self.embedder = OllamaEmbedding()
    
    # ==================== 用户画像构建 ====================
    
    def build_user_profile_from_conversation(self, user_id: str, 
                                               conversation: str) -> Dict:
        """从对话构建用户画像"""
        # 使用LLM提取偏好
        prompt = self.USER_PREFERENCE_PROMPT.format(conversation=conversation)
        
        try:
            result = self.chat.chat(prompt)
            # 解析JSON
            preferences = self._parse_json_response(result)
            
            if not preferences:
                return {"success": False, "error": "无法解析偏好"}
            
            # 构建画像更新数据
            profile_updates = {}
            
            # 基础信息
            if preferences.get("age_range"):
                profile_updates.setdefault("basic", {})["age_range"] = preferences["age_range"]
            if preferences.get("gender"):
                profile_updates.setdefault("basic", {})["gender"] = preferences["gender"]
            if preferences.get("region"):
                profile_updates.setdefault("basic", {})["region"] = preferences["region"]
            
            # 消费特征
            if preferences.get("price_sensitivity"):
                profile_updates.setdefault("consumption", {})["price_sensitivity"] = preferences["price_sensitivity"]
            
            # 兴趣偏好
            if preferences.get("preferred_categories"):
                profile_updates.setdefault("interests", {})["categories"] = preferences["preferred_categories"]
            if preferences.get("preferred_brands"):
                profile_updates.setdefault("interests", {})["brands"] = preferences["preferred_brands"]
            if preferences.get("preferred_styles"):
                profile_updates.setdefault("interests", {})["styles"] = preferences["preferred_styles"]
            
            # 标签
            if preferences.get("tags"):
                profile_updates["tags"] = preferences["tags"]
            
            # 更新画像
            self.user_store.update_profile(user_id, profile_updates)
            
            # 添加洞察
            if preferences.get("shopping_habits"):
                for habit in preferences.get("shopping_habits", []):
                    self.user_store.add_insight(user_id, habit)
            
            return {
                "success": True,
                "preferences": preferences,
                "updates": profile_updates
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def update_user_profile_from_behavior(self, user_id: str, 
                                           behavior_type: str,
                                           product_id: str,
                                           product_info: Dict = None) -> Dict:
        """从行为更新用户画像"""
        # 记录行为到知识图谱
        self.kg.add_user_product_relation(
            user_id, behavior_type, product_id, product_info
        )
        
        # 更新用户画像
        self.user_store.add_behavior(user_id, behavior_type, product_id, product_info)
        
        # 根据行为类型更新标签
        tags_to_add = []
        if behavior_type == "purchased":
            tags_to_add.append("已购用户")
        elif behavior_type == "favorited":
            tags_to_add.append("收藏活跃")
        elif behavior_type == "cart_added":
            tags_to_add.append("加购用户")
        
        for tag in tags_to_add:
            self.user_store.add_tag(user_id, tag)
        
        return {
            "success": True,
            "behavior_type": behavior_type,
            "product_id": product_id
        }
    
    def analyze_user_and_generate_insights(self, user_id: str) -> Dict:
        """分析用户行为并生成洞察"""
        # 获取用户画像
        profile = self.user_store.get_profile(user_id)
        
        # 获取用户行为
        behaviors = self.kg.get_user_products(user_id, limit=100)
        
        # 构建行为数据
        behavior_data = {
            "total_interactions": len(behaviors),
            "by_type": {},
            "recent_products": []
        }
        
        for b in behaviors[:20]:
            relation = b["relation"]
            behavior_data["by_type"][relation] = behavior_data["by_type"].get(relation, 0) + 1
            behavior_data["recent_products"].append({
                "product_id": b["product_id"],
                "relation": relation
            })
        
        # 使用LLM生成洞察
        prompt = self.BEHAVIOR_ANALYSIS_PROMPT.format(
            behavior_data=json.dumps(behavior_data, ensure_ascii=False, indent=2),
            user_profile=json.dumps(profile, ensure_ascii=False, indent=2)
        )
        
        try:
            result = self.chat.chat(prompt)
            insights = self._parse_json_response(result)
            
            if insights:
                # 添加洞察到用户画像
                for insight in insights.get("insights", []):
                    self.user_store.add_insight(user_id, insight)
                
                return {
                    "success": True,
                    "insights": insights
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "无法生成洞察"}
    
    # ==================== 商品画像构建 ====================
    
    def build_product_profile_from_info(self, product_id: str,
                                         product_info: str) -> Dict:
        """从商品信息构建商品画像"""
        # 使用LLM提取属性
        prompt = self.PRODUCT_ATTRIBUTE_PROMPT.format(product_info=product_info)
        
        try:
            result = self.chat.chat(prompt)
            attributes = self._parse_json_response(result)
            
            if not attributes:
                return {"success": False, "error": "无法解析属性"}
            
            # 构建画像数据
            profile_data = {}
            
            # 基础信息
            if attributes.get("name"):
                profile_data.setdefault("basic", {})["name"] = attributes["name"]
            if attributes.get("category"):
                profile_data.setdefault("basic", {})["category"] = attributes["category"]
            if attributes.get("brand"):
                profile_data.setdefault("basic", {})["brand"] = attributes["brand"]
            
            # 标签
            if attributes.get("style_tags"):
                profile_data.setdefault("tags", {})["style"] = attributes["style_tags"]
            if attributes.get("scene_tags"):
                profile_data.setdefault("tags", {})["scene"] = attributes["scene_tags"]
            if attributes.get("crowd_tags"):
                profile_data.setdefault("tags", {})["crowd"] = attributes["crowd_tags"]
            if attributes.get("season_tags"):
                profile_data.setdefault("tags", {})["season"] = attributes["season_tags"]
            
            # 关键词
            if attributes.get("keywords"):
                profile_data.setdefault("keywords", {})["search"] = attributes["keywords"]
            if attributes.get("selling_points"):
                profile_data.setdefault("keywords", {})["positive"] = attributes["selling_points"]
            
            # 创建或更新画像
            existing = self.product_store.get_profile(product_id)
            if existing.get("id"):
                self.product_store.update_profile(product_id, profile_data)
            else:
                self.product_store.create_profile(product_id, profile_data)
            
            return {
                "success": True,
                "attributes": attributes,
                "profile_data": profile_data
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def batch_import_products(self, products: List[Dict]) -> Dict:
        """批量导入商品"""
        return self.product_store.batch_import(products)
    
    def extract_product_from_description(self, description: str) -> Dict:
        """从描述提取商品属性"""
        prompt = self.PRODUCT_ATTRIBUTE_PROMPT.format(product_info=description)
        
        try:
            result = self.chat.chat(prompt)
            return self._parse_json_response(result) or {}
        except Exception:
            return {}
    
    # ==================== 自动标签生成 ====================
    
    def auto_tag_user(self, user_id: str) -> List[str]:
        """自动为用户生成标签"""
        profile = self.user_store.get_profile(user_id)
        behaviors = self.kg.get_user_products(user_id)
        
        tags = []
        
        # 基于消费金额
        total_spent = profile.get("consumption", {}).get("total_spent", 0)
        if total_spent > 10000:
            tags.append("高消费用户")
        elif total_spent > 5000:
            tags.append("中高消费用户")
        elif total_spent > 1000:
            tags.append("中等消费用户")
        
        # 基于购买频次
        total_orders = profile.get("consumption", {}).get("total_orders", 0)
        if total_orders > 20:
            tags.append("高频购买")
        elif total_orders > 10:
            tags.append("活跃用户")
        
        # 基于行为
        behavior_counts = {}
        for b in behaviors:
            relation = b["relation"]
            behavior_counts[relation] = behavior_counts.get(relation, 0) + 1
        
        if behavior_counts.get("favorited", 0) > 10:
            tags.append("收藏达人")
        if behavior_counts.get("cart_added", 0) > behavior_counts.get("purchased", 0) * 2:
            tags.append("加购未购")
        
        # 基于价格敏感度
        price_sensitivity = profile.get("consumption", {}).get("price_sensitivity")
        if price_sensitivity == "high":
            tags.append("价格敏感")
        elif price_sensitivity == "low":
            tags.append("品质优先")
        
        # 添加标签
        for tag in tags:
            self.user_store.add_tag(user_id, tag)
        
        return tags
    
    def auto_tag_product(self, product_id: str, 
                          reviews: List[str] = None) -> List[str]:
        """自动为商品生成标签"""
        profile = self.product_store.get_profile(product_id)
        tags = []
        
        # 基于销量
        monthly_sales = profile.get("sales", {}).get("monthly_sales", 0)
        if monthly_sales > 1000:
            tags.append("爆款")
        elif monthly_sales > 500:
            tags.append("热销")
        
        # 基于评分
        avg_rating = profile.get("sales", {}).get("avg_rating", 0)
        if avg_rating >= 4.8:
            tags.append("好评如潮")
        elif avg_rating >= 4.5:
            tags.append("口碑好")
        
        # 基于价格
        price = profile.get("basic", {}).get("price", 0)
        original_price = profile.get("basic", {}).get("original_price", 0)
        if original_price > 0 and price < original_price * 0.5:
            tags.append("超值")
        
        # 从评论提取标签
        if reviews:
            positive_keywords = []
            negative_keywords = []
            
            for review in reviews[:50]:  # 分析最近50条评论
                # 简单的关键词提取
                if "质量好" in review or "质量不错" in review:
                    positive_keywords.append("质量好")
                if "便宜" in review or "性价比" in review:
                    positive_keywords.append("性价比高")
                if "物流快" in review:
                    positive_keywords.append("物流快")
                if "色差" in review:
                    negative_keywords.append("有色差")
                if "偏大" in review:
                    negative_keywords.append("尺码偏大")
                if "偏小" in review:
                    negative_keywords.append("尺码偏小")
            
            # 更新关键词
            if positive_keywords:
                self.product_store.update_profile(product_id, {
                    "keywords": {"positive": list(set(positive_keywords))}
                })
            if negative_keywords:
                self.product_store.update_profile(product_id, {
                    "keywords": {"negative": list(set(negative_keywords))}
                })
        
        # 添加标签到商品
        if tags:
            self.product_store.update_profile(product_id, {
                "tags": {"custom": tags}
            })
        
        return tags
    
    # ==================== 辅助方法 ====================
    
    def _parse_json_response(self, response: str) -> Dict:
        """解析LLM返回的JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except Exception:
            pass
        
        # 尝试提取JSON块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass
        
        # 尝试找到JSON对象
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        
        return {}
    
    def close(self):
        """关闭连接"""
        self.user_store.close()
        self.product_store.close()
        self.kg.close()


if __name__ == "__main__":
    # 测试画像构建器
    print("🧪 测试画像构建器...")
    builder = ProfileBuilder("test_agent")
    
    # 测试从对话构建用户画像
    print("\n📝 从对话构建用户画像...")
    conversation = """
    用户: 我想找一件夏天穿的连衣裙，不要太贵的，200左右吧
    助手: 好的，我为您推荐几款夏季连衣裙，价格在200元左右
    用户: 我比较喜欢法式风格的，颜色不要太花哨
    助手: 法式风格很优雅，我为您筛选了简约款
    用户: 之前买过你们家的衣服，质量还不错
    """
    
    result = builder.build_user_profile_from_conversation("user_test_001", conversation)
    print(f"  结果: {result}")
    
    # 测试从行为更新画像
    print("\n📊 从行为更新用户画像...")
    result = builder.update_user_profile_from_behavior(
        "user_test_001", "purchased", "prod_test_001",
        {"price": 199, "category": "连衣裙"}
    )
    print(f"  结果: {result}")
    
    # 测试自动标签
    print("\n🏷️ 自动生成用户标签...")
    tags = builder.auto_tag_user("user_test_001")
    print(f"  标签: {tags}")
    
    # 测试商品画像构建
    print("\n📦 构建商品画像...")
    product_info = """
    商品名称：法式复古连衣裙女夏季新款
    价格：268元
    类目：女装/连衣裙/复古风
    描述：法式优雅风格，适合约会、度假、通勤场景，适合25-35岁女性
    """
    
    result = builder.build_product_profile_from_info("prod_test_001", product_info)
    print(f"  结果: {result}")
    
    builder.close()