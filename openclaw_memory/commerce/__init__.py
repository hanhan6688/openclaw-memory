"""
电商模块 - OpenClaw Memory System
================================

提供电商场景的用户画像、商品画像、知识图谱等功能
"""

from .user_profile import UserProfileStore
from .product_profile import ProductProfileStore
from .commerce_kg import CommerceKnowledgeGraph
from .profile_builder import ProfileBuilder
from .crm_importer import CRMImporter, CRMConnector
from .routes import register_commerce_routes, commerce_bp

__all__ = [
    'UserProfileStore',
    'ProductProfileStore', 
    'CommerceKnowledgeGraph',
    'ProfileBuilder',
    'CRMImporter',
    'CRMConnector',
    'register_commerce_routes',
    'commerce_bp'
]