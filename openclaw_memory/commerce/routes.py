"""
电商模块 API 路由
================

提供用户画像、商品画像、知识图谱等电商功能的 REST API
"""

from flask import Blueprint, jsonify, request
import json
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openclaw_memory.commerce.user_profile import UserProfileStore
from openclaw_memory.commerce.product_profile import ProductProfileStore
from openclaw_memory.commerce.commerce_kg import CommerceKnowledgeGraph
from openclaw_memory.commerce.profile_builder import ProfileBuilder
from openclaw_memory.commerce.crm_importer import CRMImporter, CRMConnector

# 创建 Blueprint
commerce_bp = Blueprint('commerce', __name__, url_prefix='/commerce')

# 全局实例缓存
_user_stores = {}
_product_stores = {}
_commerce_kgs = {}
_profile_builders = {}
_crm_importers = {}
_crm_connectors = {}


def get_user_store(agent_id: str) -> UserProfileStore:
    """获取用户画像存储实例"""
    if agent_id not in _user_stores:
        _user_stores[agent_id] = UserProfileStore(agent_id)
    return _user_stores[agent_id]


def get_product_store(agent_id: str) -> ProductProfileStore:
    """获取商品画像存储实例"""
    if agent_id not in _product_stores:
        _product_stores[agent_id] = ProductProfileStore(agent_id)
    return _product_stores[agent_id]


def get_commerce_kg(agent_id: str) -> CommerceKnowledgeGraph:
    """获取电商知识图谱实例"""
    if agent_id not in _commerce_kgs:
        _commerce_kgs[agent_id] = CommerceKnowledgeGraph(agent_id)
    return _commerce_kgs[agent_id]


def get_profile_builder(agent_id: str) -> ProfileBuilder:
    """获取画像构建器实例"""
    if agent_id not in _profile_builders:
        _profile_builders[agent_id] = ProfileBuilder(agent_id)
    return _profile_builders[agent_id]


# ==================== 用户画像 API ====================

@commerce_bp.route('/agents/<agent_id>/users/<user_id>', methods=['GET'])
def get_user_profile(agent_id, user_id):
    """获取用户画像"""
    store = get_user_store(agent_id)
    profile = store.get_profile(user_id)
    return jsonify(profile)


@commerce_bp.route('/agents/<agent_id>/users/<user_id>', methods=['POST'])
def create_user_profile(agent_id, user_id):
    """创建用户画像"""
    store = get_user_store(agent_id)
    data = request.json or {}
    
    profile_id = store.create_profile(user_id, data)
    return jsonify({
        "success": True,
        "profile_id": profile_id,
        "user_id": user_id
    })


@commerce_bp.route('/agents/<agent_id>/users/<user_id>', methods=['PUT'])
def update_user_profile(agent_id, user_id):
    """更新用户画像"""
    store = get_user_store(agent_id)
    data = request.json or {}
    
    success = store.update_profile(user_id, data)
    return jsonify({
        "success": success,
        "user_id": user_id
    })


@commerce_bp.route('/agents/<agent_id>/users/<user_id>/behavior', methods=['POST'])
def add_user_behavior(agent_id, user_id):
    """记录用户行为"""
    store = get_user_store(agent_id)
    kg = get_commerce_kg(agent_id)
    data = request.json or {}
    
    behavior_type = data.get("behavior_type")
    product_id = data.get("product_id")
    product_info = data.get("product_info")
    
    if not behavior_type or not product_id:
        return jsonify({"error": "behavior_type and product_id required"}), 400
    
    # 更新用户画像
    store.add_behavior(user_id, behavior_type, product_id, product_info)
    
    # 添加到知识图谱
    kg.add_user_product_relation(user_id, behavior_type, product_id, product_info)
    
    return jsonify({
        "success": True,
        "user_id": user_id,
        "behavior_type": behavior_type,
        "product_id": product_id
    })


@commerce_bp.route('/agents/<agent_id>/users/<user_id>/tags', methods=['POST'])
def add_user_tag(agent_id, user_id):
    """添加用户标签"""
    store = get_user_store(agent_id)
    data = request.json or {}
    tag = data.get("tag")
    
    if not tag:
        return jsonify({"error": "tag required"}), 400
    
    success = store.add_tag(user_id, tag)
    return jsonify({"success": success, "tag": tag})


@commerce_bp.route('/agents/<agent_id>/users/<user_id>/insights', methods=['POST'])
def add_user_insight(agent_id, user_id):
    """添加用户洞察"""
    store = get_user_store(agent_id)
    data = request.json or {}
    insight = data.get("insight")
    
    if not insight:
        return jsonify({"error": "insight required"}), 400
    
    success = store.add_insight(user_id, insight)
    return jsonify({"success": success, "insight": insight})


@commerce_bp.route('/agents/<agent_id>/users/<user_id>/similar', methods=['GET'])
def find_similar_users(agent_id, user_id):
    """查找相似用户"""
    store = get_user_store(agent_id)
    limit = request.args.get("limit", 10, type=int)
    
    similar_users = store.find_similar_users(user_id, limit)
    return jsonify(similar_users)


@commerce_bp.route('/agents/<agent_id>/users', methods=['GET'])
def list_user_profiles(agent_id):
    """列出所有用户画像"""
    store = get_user_store(agent_id)
    limit = request.args.get("limit", 100, type=int)
    
    profiles = store.get_all_profiles(limit)
    return jsonify(profiles)


# ==================== 商品画像 API ====================

@commerce_bp.route('/agents/<agent_id>/products/<product_id>', methods=['GET'])
def get_product_profile(agent_id, product_id):
    """获取商品画像"""
    store = get_product_store(agent_id)
    profile = store.get_profile(product_id)
    return jsonify(profile)


@commerce_bp.route('/agents/<agent_id>/products/<product_id>', methods=['POST'])
def create_product_profile(agent_id, product_id):
    """创建商品画像"""
    store = get_product_store(agent_id)
    data = request.json or {}
    
    profile_id = store.create_profile(product_id, data)
    return jsonify({
        "success": True,
        "profile_id": profile_id,
        "product_id": product_id
    })


@commerce_bp.route('/agents/<agent_id>/products/<product_id>', methods=['PUT'])
def update_product_profile(agent_id, product_id):
    """更新商品画像"""
    store = get_product_store(agent_id)
    data = request.json or {}
    
    success = store.update_profile(product_id, data)
    return jsonify({
        "success": success,
        "product_id": product_id
    })


@commerce_bp.route('/agents/<agent_id>/products/batch', methods=['POST'])
def batch_import_products(agent_id):
    """批量导入商品"""
    store = get_product_store(agent_id)
    data = request.json or {}
    products = data.get("products", [])
    
    if not products:
        return jsonify({"error": "products required"}), 400
    
    result = store.batch_import(products)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/products/<product_id>/similar', methods=['GET'])
def find_similar_products(agent_id, product_id):
    """查找相似商品"""
    store = get_product_store(agent_id)
    limit = request.args.get("limit", 10, type=int)
    
    similar_products = store.find_similar_products(product_id, limit)
    return jsonify(similar_products)


@commerce_bp.route('/agents/<agent_id>/products/search', methods=['POST'])
def search_products(agent_id):
    """搜索商品"""
    store = get_product_store(agent_id)
    data = request.json or {}
    query = data.get("query", "")
    limit = data.get("limit", 10)
    filters = data.get("filters")
    
    if not query:
        return jsonify({"error": "query required"}), 400
    
    results = store.search_by_text(query, limit, filters)
    return jsonify(results)


@commerce_bp.route('/agents/<agent_id>/products/category/<category>', methods=['GET'])
def search_by_category(agent_id, category):
    """按类目搜索商品"""
    store = get_product_store(agent_id)
    limit = request.args.get("limit", 50, type=int)
    
    results = store.search_by_category(category, limit)
    return jsonify(results)


@commerce_bp.route('/agents/<agent_id>/products', methods=['GET'])
def list_product_profiles(agent_id):
    """列出所有商品"""
    store = get_product_store(agent_id)
    limit = request.args.get("limit", 100, type=int)
    
    products = store.get_all_products(limit)
    return jsonify(products)


@commerce_bp.route('/agents/<agent_id>/products/stats', methods=['GET'])
def get_product_stats(agent_id):
    """获取商品统计"""
    store = get_product_store(agent_id)
    stats = store.get_statistics()
    return jsonify(stats)


# ==================== 电商知识图谱 API ====================

@commerce_bp.route('/agents/<agent_id>/kg/user-product', methods=['POST'])
def add_user_product_relation(agent_id):
    """添加用户-商品关系"""
    kg = get_commerce_kg(agent_id)
    data = request.json or {}
    
    user_id = data.get("user_id")
    relation_type = data.get("relation_type")
    product_id = data.get("product_id")
    context = data.get("context")
    
    if not all([user_id, relation_type, product_id]):
        return jsonify({"error": "user_id, relation_type, product_id required"}), 400
    
    relation_id = kg.add_user_product_relation(user_id, relation_type, product_id, context)
    return jsonify({
        "success": True,
        "relation_id": relation_id
    })


@commerce_bp.route('/agents/<agent_id>/kg/product-relation', methods=['POST'])
def add_product_relation(agent_id):
    """添加商品-商品关系"""
    kg = get_commerce_kg(agent_id)
    data = request.json or {}
    
    product_a = data.get("product_a_id")
    relation_type = data.get("relation_type")
    product_b = data.get("product_b_id")
    context = data.get("context")
    
    if not all([product_a, relation_type, product_b]):
        return jsonify({"error": "product_a_id, relation_type, product_b_id required"}), 400
    
    relation_id = kg.add_product_relation(product_a, relation_type, product_b, context)
    return jsonify({
        "success": True,
        "relation_id": relation_id
    })


@commerce_bp.route('/agents/<agent_id>/kg/users/<user_id>/products', methods=['GET'])
def get_user_products(agent_id, user_id):
    """获取用户关联的商品"""
    kg = get_commerce_kg(agent_id)
    relation_type = request.args.get("relation_type")
    limit = request.args.get("limit", 50, type=int)
    
    products = kg.get_user_products(user_id, relation_type, limit)
    return jsonify(products)


@commerce_bp.route('/agents/<agent_id>/kg/products/<product_id>/related', methods=['GET'])
def get_related_products(agent_id, product_id):
    """获取关联商品"""
    kg = get_commerce_kg(agent_id)
    relation_type = request.args.get("relation_type")
    limit = request.args.get("limit", 20, type=int)
    
    products = kg.get_related_products(product_id, relation_type, limit)
    return jsonify(products)


@commerce_bp.route('/agents/<agent_id>/kg/users/<user_id>/recommendations', methods=['GET'])
def get_user_recommendations(agent_id, user_id):
    """获取用户推荐"""
    kg = get_commerce_kg(agent_id)
    limit = request.args.get("limit", 20, type=int)
    
    recommendations = kg.get_recommendations_for_user(user_id, limit)
    return jsonify(recommendations)


@commerce_bp.route('/agents/<agent_id>/kg/products/<product_id>/bought-together', methods=['GET'])
def get_frequently_bought_together(agent_id, product_id):
    """获取经常一起购买的商品"""
    kg = get_commerce_kg(agent_id)
    limit = request.args.get("limit", 10, type=int)
    
    products = kg.get_frequently_bought_together(product_id, limit)
    return jsonify(products)


@commerce_bp.route('/agents/<agent_id>/kg/graph', methods=['GET'])
def get_commerce_graph(agent_id):
    """获取电商知识图谱可视化数据"""
    kg = get_commerce_kg(agent_id)
    entity_type = request.args.get("entity_type")
    entity_id = request.args.get("entity_id")
    depth = request.args.get("depth", 2, type=int)
    
    graph_data = kg.get_graph_data(entity_type, entity_id, depth)
    return jsonify(graph_data)


@commerce_bp.route('/agents/<agent_id>/kg/users/<user_id>/analyze', methods=['GET'])
def analyze_user_interests(agent_id, user_id):
    """分析用户兴趣"""
    kg = get_commerce_kg(agent_id)
    interests = kg.analyze_user_interests(user_id)
    return jsonify(interests)


@commerce_bp.route('/agents/<agent_id>/kg/products/<product_id>/analyze', methods=['GET'])
def analyze_product_performance(agent_id, product_id):
    """分析商品表现"""
    kg = get_commerce_kg(agent_id)
    performance = kg.analyze_product_performance(product_id)
    return jsonify(performance)


# ==================== 画像构建 API ====================

@commerce_bp.route('/agents/<agent_id>/builder/user-from-conversation', methods=['POST'])
def build_user_from_conversation(agent_id):
    """从对话构建用户画像"""
    builder = get_profile_builder(agent_id)
    data = request.json or {}
    
    user_id = data.get("user_id")
    conversation = data.get("conversation")
    
    if not user_id or not conversation:
        return jsonify({"error": "user_id and conversation required"}), 400
    
    result = builder.build_user_profile_from_conversation(user_id, conversation)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/builder/product-from-info', methods=['POST'])
def build_product_from_info(agent_id):
    """从商品信息构建商品画像"""
    builder = get_profile_builder(agent_id)
    data = request.json or {}
    
    product_id = data.get("product_id")
    product_info = data.get("product_info")
    
    if not product_id or not product_info:
        return jsonify({"error": "product_id and product_info required"}), 400
    
    result = builder.build_product_profile_from_info(product_id, product_info)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/builder/users/<user_id>/auto-tag', methods=['POST'])
def auto_tag_user(agent_id, user_id):
    """自动为用户生成标签"""
    builder = get_profile_builder(agent_id)
    tags = builder.auto_tag_user(user_id)
    return jsonify({"tags": tags})


@commerce_bp.route('/agents/<agent_id>/builder/products/<product_id>/auto-tag', methods=['POST'])
def auto_tag_product(agent_id, product_id):
    """自动为商品生成标签"""
    builder = get_profile_builder(agent_id)
    data = request.json or {}
    reviews = data.get("reviews")
    
    tags = builder.auto_tag_product(product_id, reviews)
    return jsonify({"tags": tags})


@commerce_bp.route('/agents/<agent_id>/builder/users/<user_id>/insights', methods=['POST'])
def generate_user_insights(agent_id, user_id):
    """生成用户洞察"""
    builder = get_profile_builder(agent_id)
    result = builder.analyze_user_and_generate_insights(user_id)
    return jsonify(result)


# ==================== 综合推荐 API ====================

@commerce_bp.route('/agents/<agent_id>/recommend/<user_id>', methods=['GET'])
def recommend_for_user(agent_id, user_id):
    """为用户推荐商品（综合推荐）"""
    user_store = get_user_store(agent_id)
    product_store = get_product_store(agent_id)
    kg = get_commerce_kg(agent_id)
    
    limit = request.args.get("limit", 20, type=int)
    
    # 获取用户画像
    profile = user_store.get_profile(user_id)
    
    # 获取基于知识图谱的推荐
    kg_recommendations = kg.get_recommendations_for_user(user_id, limit)
    
    # 补充商品详情
    recommendations = []
    for rec in kg_recommendations:
        product_id = rec["product_id"]
        product_profile = product_store.get_profile(product_id)
        
        recommendations.append({
            "product_id": product_id,
            "score": rec["score"],
            "name": product_profile.get("basic", {}).get("name"),
            "price": product_profile.get("basic", {}).get("price"),
            "category": product_profile.get("basic", {}).get("category"),
            "tags": product_profile.get("tags", {}),
            "sales": product_profile.get("sales", {})
        })
    
    return jsonify({
        "user_id": user_id,
        "user_tags": profile.get("tags", []),
        "recommendations": recommendations
    })


# ==================== CRM 数据导入 API ====================

def get_crm_importer(agent_id: str) -> CRMImporter:
    """获取CRM导入器实例"""
    if agent_id not in _crm_importers:
        _crm_importers[agent_id] = CRMImporter(agent_id)
    return _crm_importers[agent_id]


@commerce_bp.route('/agents/<agent_id>/crm/import/csv', methods=['POST'])
def import_crm_csv(agent_id):
    """从CSV导入CRM用户数据
    
    请求体:
    {
        "content": "CSV内容字符串",
        "field_mapping": {"user_id": "customer_id", ...}  // 可选
    }
    """
    importer = get_crm_importer(agent_id)
    data = request.json or {}
    
    content = data.get("content")
    field_mapping = data.get("field_mapping")
    
    if not content:
        return jsonify({"error": "content required"}), 400
    
    result = importer.import_from_csv(content=content, field_mapping=field_mapping)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/crm/import/json', methods=['POST'])
def import_crm_json(agent_id):
    """从JSON导入CRM用户数据
    
    请求体:
    {
        "users": [{"user_id": "001", "name": "张三", ...}],
        "field_mapping": {}  // 可选
    }
    """
    importer = get_crm_importer(agent_id)
    data = request.json or {}
    
    users = data.get("users") or data.get("data")
    field_mapping = data.get("field_mapping")
    
    if not users:
        return jsonify({"error": "users or data required"}), 400
    
    result = importer.import_from_json(data=users, field_mapping=field_mapping)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/crm/import/orders', methods=['POST'])
def import_crm_orders(agent_id):
    """导入订单数据
    
    请求体:
    {
        "orders": [
            {
                "user_id": "user_001",
                "order_id": "order_001",
                "products": [{"product_id": "p001", "price": 299, "quantity": 1}],
                "total_amount": 299,
                "order_time": "2026-03-14T10:00:00Z"
            }
        ]
    }
    """
    importer = get_crm_importer(agent_id)
    data = request.json or {}
    
    orders = data.get("orders")
    if not orders:
        return jsonify({"error": "orders required"}), 400
    
    result = importer.import_orders(orders)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/crm/import/behaviors', methods=['POST'])
def import_crm_behaviors(agent_id):
    """导入用户行为数据
    
    请求体:
    {
        "behaviors": [
            {
                "user_id": "user_001",
                "product_id": "p001",
                "behavior_type": "viewed",
                "timestamp": "2026-03-14T10:00:00Z",
                "context": {}
            }
        ]
    }
    """
    importer = get_crm_importer(agent_id)
    data = request.json or {}
    
    behaviors = data.get("behaviors")
    if not behaviors:
        return jsonify({"error": "behaviors required"}), 400
    
    result = importer.import_behaviors(behaviors)
    return jsonify(result)


@commerce_bp.route('/agents/<agent_id>/crm/template', methods=['GET'])
def get_crm_template(agent_id):
    """获取CRM导入模板"""
    importer = get_crm_importer(agent_id)
    format = request.args.get("format", "json")
    
    template = importer.get_import_template(format)
    
    if format == "json":
        return jsonify({"template": json.loads(template)})
    else:
        return jsonify({"template": template, "format": "csv"})


@commerce_bp.route('/agents/<agent_id>/crm/connect/api', methods=['POST'])
def connect_crm_api(agent_id):
    """连接自定义CRM API
    
    请求体:
    {
        "api_url": "https://your-crm.com/api/users",
        "headers": {"Authorization": "Bearer xxx"},
        "data_path": "data.users"
    }
    """
    connector = CRMConnector(agent_id)
    data = request.json or {}
    
    api_url = data.get("api_url")
    headers = data.get("headers")
    data_path = data.get("data_path")
    
    if not api_url:
        return jsonify({"error": "api_url required"}), 400
    
    result = connector.connect_custom_api(api_url, headers, data_path)
    return jsonify(result)


def register_commerce_routes(app):
    """注册电商路由到 Flask app"""
    app.register_blueprint(commerce_bp)
    print("✅ 电商模块 API 已注册")


if __name__ == "__main__":
    # 测试路由
    from flask import Flask
    app = Flask(__name__)
    register_commerce_routes(app)
    
    print("\n📋 电商 API 路由:")
    for rule in app.url_map.iter_rules():
        if rule.endpoint.startswith('commerce.'):
            print(f"  {rule.methods} {rule.rule}")