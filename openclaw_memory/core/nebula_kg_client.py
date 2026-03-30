"""
NebulaGraph 知识图谱客户端

用于实体关系存储和查询，替代 Weaviate 的知识图谱功能。
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig


class NebulaKGClient:
    """
    NebulaGraph 知识图谱客户端
    
    功能:
    - 实体管理 (创建/查询/删除)
    - 关系管理 (创建/查询/删除)
    - 图算法 (最短路径、邻居查询)
    """
    
    # 默认图空间
    DEFAULT_SPACE = "knowledge_graph"
    
    # 实体类型标签
    ENTITY_TYPES = [
        "人物", "组织", "项目", "技术", "产品", 
        "地点", "事件", "概念", "资源", "其他"
    ]
    
    # 关系类型
    RELATION_TYPES = [
        "合作", "任职于", "管理", "开发", "使用",
        "包含", "依赖", "是一种", "位于", "相关",
        "创建", "参与", "拥有", "属于", "连接"
    ]
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9669,
        user: str = "root",
        password: str = "nebula",
        space: str = None
    ):
        """
        初始化 NebulaGraph 客户端
        
        Args:
            host: Graphd 服务地址
            port: Graphd 服务端口
            user: 用户名
            password: 密码
            space: 图空间名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.space = space or self.DEFAULT_SPACE
        
        self.pool: Optional[ConnectionPool] = None
        self.session = None
        self._connected = False
    
    def connect(self) -> bool:
        """连接到 NebulaGraph"""
        if self._connected:
            return True
        
        try:
            config = NebulaConfig()
            config.max_connection_pool_size = 10
            
            self.pool = ConnectionPool()
            ok = self.pool.init([(self.host, self.port)], config)
            
            if not ok:
                print("❌ 连接 NebulaGraph 失败")
                return False
            
            self.session = self.pool.get_session(self.user, self.password)
            self._connected = True
            print(f"✅ 已连接 NebulaGraph: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ 连接 NebulaGraph 失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self.session:
            self.session.release()
        if self.pool:
            self.pool.close()
        self._connected = False
    
    def ensure_space(self, agent_id: str = "main") -> bool:
        """
        确保图空间存在
        
        Args:
            agent_id: Agent ID，用于创建独立的图空间
        """
        if not self._connected:
            if not self.connect():
                return False
        
        space_name = f"{self.DEFAULT_SPACE}_{agent_id}"
        
        try:
            # 创建图空间（如果不存在）
            create_space = f"""
            CREATE SPACE IF NOT EXISTS {space_name} (
                partition_num = 10,
                replica_factor = 1,
                vid_type = FIXED_STRING(256)
            );
            """
            self.session.execute(create_space)
            
            # 等待图空间生效
            import time
            time.sleep(2)
            
            # 使用图空间
            self.session.execute(f"USE {space_name};")
            self.space = space_name
            
            # 创建标签（实体类型）
            self._create_tags()
            
            # 创建边类型（关系类型）
            self._create_edge_types()
            
            print(f"✅ 图空间已就绪: {space_name}")
            return True
            
        except Exception as e:
            print(f"❌ 创建图空间失败: {e}")
            return False
    
    def _create_tags(self):
        """创建实体标签"""
        for entity_type in self.ENTITY_TYPES:
            tag_name = self._sanitize_name(entity_type)
            try:
                create_tag = f"""
                CREATE TAG IF NOT EXISTS `{tag_name}` (
                    name STRING,
                    type STRING,
                    confidence DOUBLE,
                    source STRING,
                    created_at STRING,
                    updated_at STRING,
                    properties STRING
                );
                """
                self.session.execute(create_tag)
            except Exception as e:
                print(f"⚠️ 创建标签 {tag_name} 失败: {e}")
    
    def _create_edge_types(self):
        """创建边类型"""
        for relation_type in self.RELATION_TYPES:
            edge_name = self._sanitize_name(relation_type)
            try:
                create_edge = f"""
                CREATE EDGE IF NOT EXISTS `{edge_name}` (
                    confidence DOUBLE,
                    source STRING,
                    evidence STRING,
                    created_at STRING
                );
                """
                self.session.execute(create_edge)
            except Exception as e:
                print(f"⚠️ 创建边类型 {edge_name} 失败: {e}")
    
    def _sanitize_name(self, name: str) -> str:
        """清理名称，使其符合 NebulaGraph 命名规范"""
        # 替换特殊字符
        name = name.replace(" ", "_")
        name = name.replace("-", "_")
        name = name.replace(".", "_")
        return name
    
    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: Dict = None,
        confidence: float = 0.8,
        source: str = "extraction"
    ) -> bool:
        """
        添加实体
        
        Args:
            entity_id: 实体唯一ID
            name: 实体名称
            entity_type: 实体类型
            properties: 额外属性
            confidence: 置信度
            source: 来源
        """
        if not self._connected:
            if not self.connect():
                return False
        
        tag_name = self._sanitize_name(entity_type)
        if tag_name not in [self._sanitize_name(t) for t in self.ENTITY_TYPES]:
            tag_name = "其他"
        
        props = properties or {}
        props_json = json.dumps(props, ensure_ascii=False).replace("'", '"')
        now = datetime.now(timezone.utc).isoformat()
        
        try:
            # 使用 UPSERT 插入或更新
            query = f"""
            INSERT VERTEX `{tag_name}` (
                name, type, confidence, source, created_at, updated_at, properties
            ) VALUES "{entity_id}": (
                "{name}", "{entity_type}", {confidence}, "{source}", 
                "{now}", "{now}", "{props_json}"
            );
            """
            result = self.session.execute(query)
            return result.is_succeeded()
        except Exception as e:
            print(f"❌ 添加实体失败: {e}")
            return False
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 0.7,
        evidence: str = "",
        source: str = "extraction"
    ) -> bool:
        """
        添加关系
        
        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            confidence: 置信度
            evidence: 证据文本
            source: 来源
        """
        if not self._connected:
            if not self.connect():
                return False
        
        edge_name = self._sanitize_name(relation_type)
        if edge_name not in [self._sanitize_name(r) for r in self.RELATION_TYPES]:
            edge_name = "相关"
        
        now = datetime.now(timezone.utc).isoformat()
        
        try:
            query = f"""
            INSERT EDGE `{edge_name}` (
                confidence, source, evidence, created_at
            ) VALUES "{source_id}"->"{target_id}": (
                {confidence}, "{source}", "{evidence[:200]}", "{now}"
            );
            """
            result = self.session.execute(query)
            return result.is_succeeded()
        except Exception as e:
            print(f"❌ 添加关系失败: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """获取实体"""
        if not self._connected:
            if not self.connect():
                return None
        
        try:
            query = f'FETCH PROP ON * "{entity_id}" YIELD vertex as v;'
            result = self.session.execute(query)
            
            if result.is_succeeded() and result.row_size() > 0:
                row = result.row_values(0)[0]
                return {
                    "id": entity_id,
                    "properties": row.as_node().properties()
                }
            return None
        except Exception as e:
            print(f"❌ 获取实体失败: {e}")
            return None
    
    def search_entities(
        self,
        query: str,
        entity_type: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        搜索实体
        
        Args:
            query: 搜索关键词
            entity_type: 实体类型过滤
            limit: 返回数量限制
        """
        if not self._connected:
            if not self.connect():
                return []
        
        results = []
        
        try:
            if entity_type:
                tag_name = self._sanitize_name(entity_type)
                nsql = f"""
                MATCH (e:`{tag_name}`)
                WHERE e.name CONTAINS "{query}"
                RETURN e.name, e.type, e.confidence, e.source
                LIMIT {limit};
                """
            else:
                nsql = f"""
                MATCH (e)
                WHERE e.name CONTAINS "{query}"
                RETURN e.name, e.type, e.confidence, e.source
                LIMIT {limit};
                """
            
            result = self.session.execute(nsql)
            
            if result.is_succeeded():
                for i in range(result.row_size()):
                    row = result.row_values(i)
                    results.append({
                        "name": row[0].as_string(),
                        "type": row[1].as_string() if len(row) > 1 else "",
                        "confidence": row[2].as_double() if len(row) > 2 else 0.5,
                        "source": row[3].as_string() if len(row) > 3 else ""
                    })
            
        except Exception as e:
            print(f"❌ 搜索实体失败: {e}")
        
        return results
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        获取实体的邻居节点
        
        Args:
            entity_id: 实体ID
            relation_type: 关系类型过滤
            limit: 返回数量限制
        """
        if not self._connected:
            if not self.connect():
                return []
        
        results = []
        
        try:
            if relation_type:
                edge_name = self._sanitize_name(relation_type)
                nsql = f"""
                MATCH (e1)-[r:`{edge_name}`]->(e2)
                WHERE id(e1) == "{entity_id}"
                RETURN e2.name, e2.type, type(r), r.confidence
                LIMIT {limit};
                """
            else:
                nsql = f"""
                MATCH (e1)-[r]->(e2)
                WHERE id(e1) == "{entity_id}"
                RETURN e2.name, e2.type, type(r), r.confidence
                LIMIT {limit};
                """
            
            result = self.session.execute(nsql)
            
            if result.is_succeeded():
                for i in range(result.row_size()):
                    row = result.row_values(i)
                    results.append({
                        "target_name": row[0].as_string(),
                        "target_type": row[1].as_string(),
                        "relation": row[2].as_string(),
                        "confidence": row[3].as_double() if len(row) > 3 else 0.5
                    })
            
        except Exception as e:
            print(f"❌ 获取邻居失败: {e}")
        
        return results
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        查找两个实体之间的路径
        
        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            max_depth: 最大搜索深度
        """
        if not self._connected:
            if not self.connect():
                return []
        
        try:
            nsql = f"""
            FIND SHORTEST PATH FROM "{source_id}" TO "{target_id}"
            OVER * UPTO {max_depth} STEPS
            YIELD path as p;
            """
            
            result = self.session.execute(nsql)
            
            paths = []
            if result.is_succeeded():
                for i in range(result.row_size()):
                    row = result.row_values(i)
                    path = row[0].as_path()
                    nodes = [n for n in path.nodes()]
                    paths.append(nodes)
            
            return paths
            
        except Exception as e:
            print(f"❌ 查找路径失败: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """获取图统计信息"""
        if not self._connected:
            if not self.connect():
                return {"error": "未连接"}
        
        stats = {
            "space": self.space,
            "entities": 0,
            "relations": 0
        }
        
        try:
            # 统计实体数量
            result = self.session.execute("MATCH (n) RETURN count(n);")
            if result.is_succeeded() and result.row_size() > 0:
                stats["entities"] = result.row_values(0)[0].as_int()
            
            # 统计关系数量
            result = self.session.execute("MATCH ()-[r]->() RETURN count(r);")
            if result.is_succeeded() and result.row_size() > 0:
                stats["relations"] = result.row_values(0)[0].as_int()
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def clear(self) -> bool:
        """清空图空间"""
        if not self._connected:
            if not self.connect():
                return False
        
        try:
            # 删除所有数据
            self.session.execute("CLEAR SPACE;")
            return True
        except Exception as e:
            print(f"❌ 清空图空间失败: {e}")
            return False


# ============================================================================
# 工厂函数
# ============================================================================

_nebula_clients: Dict[str, NebulaKGClient] = {}


def get_nebula_client(agent_id: str = "main") -> NebulaKGClient:
    """获取或创建 NebulaGraph 客户端"""
    if agent_id not in _nebula_clients:
        client = NebulaKGClient()
        if client.connect():
            client.ensure_space(agent_id)
            _nebula_clients[agent_id] = client
        else:
            # 返回未连接的客户端
            _nebula_clients[agent_id] = client
    return _nebula_clients[agent_id]