"""
CRM 数据导入服务
================

支持从各种CRM系统导入用户数据：
- CSV/Excel 文件导入
- JSON 数据导入
- API 对接导入
- 自动映射到用户画像
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import csv
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_profile import UserProfileStore
from commerce_kg import CommerceKnowledgeGraph


class CRMImporter:
    """CRM数据导入器"""
    
    # 常见CRM字段映射
    FIELD_MAPPINGS = {
        # 用户ID
        "user_id": ["user_id", "customer_id", "会员ID", "客户ID", "member_id", "id"],
        
        # 基础信息
        "age_range": ["age_range", "年龄段", "age", "年龄"],
        "gender": ["gender", "性别", "sex"],
        "region": ["region", "地区", "city", "城市", "province", "省份", "address", "地址"],
        "member_level": ["member_level", "会员等级", "level", "等级", "vip_level"],
        
        # 消费信息
        "total_spent": ["total_spent", "总消费", "total_amount", "累计消费", "lifetime_value"],
        "total_orders": ["total_orders", "订单数", "order_count", "购买次数", "total_purchases"],
        "avg_order_value": ["avg_order_value", "客单价", "average_order", "平均消费"],
        
        # 联系方式
        "phone": ["phone", "手机", "mobile", "电话", "tel"],
        "email": ["email", "邮箱", "mail"],
        "name": ["name", "姓名", "customer_name", "用户名", "real_name"],
        
        # 时间
        "register_time": ["register_time", "注册时间", "created_at", "join_date", "注册日期"],
        "last_purchase": ["last_purchase", "最后购买", "last_order", "最近购买"],
    }
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.user_store = UserProfileStore(agent_id)
        self.kg = CommerceKnowledgeGraph(agent_id)
    
    def import_from_csv(self, file_path: str = None, content: str = None, 
                         field_mapping: Dict = None, skip_header: bool = True) -> Dict:
        """从CSV导入用户数据
        
        Args:
            file_path: CSV文件路径
            content: CSV内容字符串（二选一）
            field_mapping: 自定义字段映射
            skip_header: 是否跳过首行
        """
        if content:
            lines = content.strip().split('\n')
            reader = csv.DictReader(lines)
            rows = list(reader)
        elif file_path:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            return {"error": "需要提供 file_path 或 content"}
        
        return self._process_rows(rows, field_mapping)
    
    def import_from_json(self, data: List[Dict] = None, file_path: str = None,
                         field_mapping: Dict = None) -> Dict:
        """从JSON导入用户数据
        
        Args:
            data: JSON数据列表
            file_path: JSON文件路径（二选一）
            field_mapping: 自定义字段映射
        """
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        if not data:
            return {"error": "需要提供 data 或 file_path"}
        
        # 如果是单个对象，转为列表
        if isinstance(data, dict):
            data = [data]
        
        return self._process_rows(data, field_mapping)
    
    def import_orders(self, orders: List[Dict]) -> Dict:
        """导入订单数据，自动更新用户画像和知识图谱
        
        Args:
            orders: 订单列表，每个订单包含:
                - user_id: 用户ID
                - order_id: 订单ID
                - products: 商品列表 [{product_id, price, quantity}]
                - total_amount: 订单金额
                - order_time: 下单时间
        """
        stats = {
            "orders_processed": 0,
            "users_updated": set(),
            "products_linked": 0,
            "errors": []
        }
        
        for order in orders:
            try:
                user_id = order.get("user_id")
                if not user_id:
                    continue
                
                # 更新用户画像
                total_amount = order.get("total_amount", 0)
                products = order.get("products", [])
                
                for product in products:
                    product_id = product.get("product_id")
                    price = product.get("price", 0)
                    
                    if product_id:
                        # 添加购买关系到知识图谱
                        self.kg.add_user_product_relation(
                            user_id, "purchased", product_id,
                            {"price": price, "order_id": order.get("order_id")}
                        )
                        stats["products_linked"] += 1
                
                # 更新用户消费统计
                profile = self.user_store.get_profile(user_id)
                consumption = profile.get("consumption", {})
                
                updates = {
                    "consumption": {
                        "total_orders": consumption.get("total_orders", 0) + 1,
                        "total_spent": consumption.get("total_spent", 0) + total_amount,
                    }
                }
                
                # 更新最后购买时间
                order_time = order.get("order_time")
                if order_time:
                    updates["behavior"] = {"last_active": order_time}
                
                self.user_store.update_profile(user_id, updates)
                
                stats["users_updated"].add(user_id)
                stats["orders_processed"] += 1
                
            except Exception as e:
                stats["errors"].append(str(e))
        
        stats["users_updated"] = len(stats["users_updated"])
        return stats
    
    def import_behaviors(self, behaviors: List[Dict]) -> Dict:
        """导入用户行为数据
        
        Args:
            behaviors: 行为列表，每个行为包含:
                - user_id: 用户ID
                - product_id: 商品ID
                - behavior_type: 行为类型 (viewed/favorited/cart_added等)
                - timestamp: 时间
                - context: 上下文信息
        """
        stats = {
            "behaviors_imported": 0,
            "users_updated": set(),
            "errors": []
        }
        
        for behavior in behaviors:
            try:
                user_id = behavior.get("user_id")
                product_id = behavior.get("product_id")
                behavior_type = behavior.get("behavior_type", "viewed")
                
                if not user_id or not product_id:
                    continue
                
                # 添加到知识图谱
                context = behavior.get("context", {})
                if behavior.get("timestamp"):
                    context["timestamp"] = behavior["timestamp"]
                
                self.kg.add_user_product_relation(
                    user_id, behavior_type, product_id, context
                )
                
                # 更新用户画像
                self.user_store.add_behavior(user_id, behavior_type, product_id, context)
                
                stats["users_updated"].add(user_id)
                stats["behaviors_imported"] += 1
                
            except Exception as e:
                stats["errors"].append(str(e))
        
        stats["users_updated"] = len(stats["users_updated"])
        return stats
    
    def _process_rows(self, rows: List[Dict], custom_mapping: Dict = None) -> Dict:
        """处理数据行"""
        mapping = custom_mapping or self._auto_detect_mapping(rows[0] if rows else {})
        
        stats = {
            "total": len(rows),
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "errors": []
        }
        
        for row in rows:
            try:
                # 提取用户ID
                user_id = self._extract_field(row, mapping, "user_id")
                if not user_id:
                    stats["skipped"] += 1
                    continue
                
                # 构建用户画像数据
                profile_data = self._build_profile_data(row, mapping)
                
                # 检查是否已存在
                existing = self.user_store.get_profile(user_id)
                
                if existing.get("id"):
                    self.user_store.update_profile(user_id, profile_data)
                    stats["updated"] += 1
                else:
                    self.user_store.create_profile(user_id, profile_data)
                    stats["imported"] += 1
                    
            except Exception as e:
                stats["errors"].append(f"行处理错误: {str(e)}")
        
        return stats
    
    def _auto_detect_mapping(self, sample_row: Dict) -> Dict:
        """自动检测字段映射"""
        mapping = {}
        
        for standard_field, possible_names in self.FIELD_MAPPINGS.items():
            for name in possible_names:
                if name in sample_row:
                    mapping[standard_field] = name
                    break
        
        return mapping
    
    def _extract_field(self, row: Dict, mapping: Dict, field_name: str) -> Any:
        """提取字段值"""
        source_field = mapping.get(field_name)
        if source_field and source_field in row:
            return row[source_field]
        
        # 尝试直接匹配
        for possible_name in self.FIELD_MAPPINGS.get(field_name, []):
            if possible_name in row:
                return row[possible_name]
        
        return None
    
    def _build_profile_data(self, row: Dict, mapping: Dict) -> Dict:
        """构建用户画像数据"""
        profile_data = {}
        
        # 基础信息
        basic = {}
        for field in ["age_range", "gender", "region", "member_level"]:
            value = self._extract_field(row, mapping, field)
            if value:
                basic[field] = value
        
        if basic:
            profile_data["basic"] = basic
        
        # 消费信息
        consumption = {}
        for field in ["total_spent", "total_orders", "avg_order_value"]:
            value = self._extract_field(row, mapping, field)
            if value:
                try:
                    consumption[field] = float(value) if field != "total_orders" else int(value)
                except Exception:
                    pass
        
        if consumption:
            profile_data["consumption"] = consumption
        
        # 联系方式（存储到basic）
        for field in ["phone", "email", "name"]:
            value = self._extract_field(row, mapping, field)
            if value:
                profile_data.setdefault("basic", {})[field] = value
        
        return profile_data
    
    def get_import_template(self, format: str = "csv") -> str:
        """获取导入模板"""
        if format == "csv":
            return "user_id,name,phone,email,gender,age_range,region,member_level,total_spent,total_orders\n用户001,张三,13800138000,test@example.com,male,25-34,上海,金卡,5000,10"
        elif format == "json":
            return json.dumps([
                {
                    "user_id": "用户001",
                    "name": "张三",
                    "phone": "13800138000",
                    "email": "test@example.com",
                    "gender": "male",
                    "age_range": "25-34",
                    "region": "上海",
                    "member_level": "金卡",
                    "total_spent": 5000,
                    "total_orders": 10
                }
            ], ensure_ascii=False, indent=2)
        
        return ""


class CRMConnector:
    """CRM系统连接器（预留接口）"""
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.importer = CRMImporter(agent_id)
    
    def connect_salesforce(self, config: Dict) -> bool:
        """连接Salesforce（预留）"""
        # TODO: 实现Salesforce API对接
        pass
    
    def connect_hubspot(self, config: Dict) -> bool:
        """连接HubSpot（预留）"""
        # TODO: 实现HubSpot API对接
        pass
    
    def connect_shopify(self, config: Dict) -> bool:
        """连接Shopify（预留）"""
        # TODO: 实现Shopify API对接
        pass
    
    def connect_custom_api(self, api_url: str, headers: Dict = None, 
                           data_path: str = None) -> Dict:
        """连接自定义API
        
        Args:
            api_url: API地址
            headers: 请求头
            data_path: 数据在响应中的路径（如 data.users）
        """
        import requests
        
        try:
            response = requests.get(api_url, headers=headers or {})
            response.raise_for_status()
            
            data = response.json()
            
            # 提取数据路径
            if data_path:
                for key in data_path.split('.'):
                    data = data.get(key, {})
            
            if isinstance(data, list):
                return self.importer.import_from_json(data)
            else:
                return {"error": "API返回的数据格式不正确"}
                
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # 测试CRM导入
    print("🧪 测试CRM数据导入...")
    importer = CRMImporter("test_agent")
    
    # 测试CSV导入
    print("\n📝 测试CSV导入...")
    csv_content = """user_id,name,phone,gender,age_range,region,total_spent,total_orders
user_001,张三,13800138000,male,25-34,上海,5000,10
user_002,李四,13900139000,female,35-44,北京,8000,15
"""
    
    result = importer.import_from_csv(content=csv_content)
    print(f"  导入结果: {result}")
    
    # 测试订单导入
    print("\n📦 测试订单导入...")
    orders = [
        {
            "user_id": "user_001",
            "order_id": "order_001",
            "products": [{"product_id": "prod_001", "price": 299, "quantity": 1}],
            "total_amount": 299,
            "order_time": "2026-03-14T10:00:00Z"
        }
    ]
    
    result = importer.import_orders(orders)
    print(f"  订单导入结果: {result}")
    
    # 获取导入模板
    print("\n📋 导入模板:")
    print(importer.get_import_template("json"))