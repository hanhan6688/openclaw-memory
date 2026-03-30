"""
用户画像模块

功能：
1. 实体聚合 - 按类型分组用户相关实体
2. 行为分析 - 记忆时间分布、话题分布
3. World Fact / Experience 区分
4. 偏好挖掘
"""

from .user_profile import UserProfile, get_user_profile

__all__ = ["UserProfile", "get_user_profile"]