"""
NetworkX 知识图谱客户端

轻量级内存图数据库，不需要 Docker，立即可用。
支持持久化到 JSON 文件。
"""

import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path


class NetworkXKGClient:
    """
    NetworkX 知识图谱客户端
    
    特点：
    - 纯 Python，无需 Docker
    - 内存图 + 文件持久化
    - 支持图算法（最短路径、社区检测等）
    """
    
    # 实体类型
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
    
    def __init__(self, agent_id: str = "main", data_dir: str = None):
        """
        初始化 NetworkX 知识图谱客户端
        
        Args:
            agent_id: Agent ID
            data_dir: 数据存储目录
        """
        self.agent_id = agent_id
        self.data_dir = data_dir or os.path.expanduser("~/.openclaw/memory_system/kg_data")
        self.data_file = os.path.join(self.data_dir, f"{agent_id}_kg.json")
        
        # 创建有向图
        self.graph = nx.DiGraph()
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载已有数据
        self._load()
        
        self._connected = True
    
    def _load(self):
        """从文件加载图谱"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载节点
                for node in data.get('nodes', []):
                    self.graph.add_node(
                        node['id'],
                        **node.get('attrs', {})
                    )
                
                # 加载边
                for edge in data.get('edges', []):
                    self.graph.add_edge(
                        edge['source'],
                        edge['target'],
                        **edge.get('attrs', {})
                    )
                
                print(f"📂 加载知识图谱: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
            except Exception as e:
                print(f"⚠️ 加载知识图谱失败: {e}")
    
    def _save(self):
        """保存图谱到文件"""
        try:
            data = {
                'nodes': [
                    {'id': n, 'attrs': dict(self.graph.nodes[n])}
                    for n in self.graph.nodes()
                ],
                'edges': [
                    {'source': u, 'target': v, 'attrs': dict(self.graph.edges[u, v])}
                    for u, v in self.graph.edges()
                ],
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存知识图谱失败: {e}")
    
    def connect(self) -> bool:
        """连接（始终成功）"""
        self._connected = True
        return True
    
    def close(self):
        """关闭连接（保存数据）"""
        self._save()
        self._connected = False
    
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
            entity_id: 实体ID
            name: 实体名称
            entity_type: 实体类型
            properties: 额外属性
            confidence: 置信度
            source: 来源
        """
        if not self._connected:
            return False
        
        # 检查是否已存在
        if self.graph.has_node(entity_id):
            # 更新属性
            self.graph.nodes[entity_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
            self.graph.nodes[entity_id]['access_count'] = self.graph.nodes[entity_id].get('access_count', 0) + 1
            self._save()
            return True
        
        # 添加节点
        self.graph.add_node(
            entity_id,
            name=name,
            type=entity_type,
            confidence=confidence,
            source=source,
            properties=properties or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            access_count=1
        )
        
        self._save()
        return True
    
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
            evidence: 证据
            source: 来源
        """
        if not self._connected:
            return False
        
        # 确保两个节点存在
        if not self.graph.has_node(source_id):
            self.add_entity(source_id, source_id, "未知")
        if not self.graph.has_node(target_id):
            self.add_entity(target_id, target_id, "未知")
        
        # 添加边
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation_type,
            confidence=confidence,
            evidence=evidence,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        self._save()
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """获取实体"""
        if not self._connected or not self.graph.has_node(entity_id):
            return None
        
        attrs = dict(self.graph.nodes[entity_id])
        attrs['id'] = entity_id
        return attrs
    
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
            return []
        
        results = []
        query_lower = query.lower()
        
        for node_id in self.graph.nodes():
            attrs = self.graph.nodes[node_id]
            name = attrs.get('name', str(node_id)).lower()
            
            # 名称匹配
            if query_lower in name:
                # 类型过滤
                if entity_type and attrs.get('type') != entity_type:
                    continue
                
                results.append({
                    'id': node_id,
                    'name': attrs.get('name', str(node_id)),
                    'type': attrs.get('type', '未知'),
                    'confidence': attrs.get('confidence', 0.5),
                    'source': attrs.get('source', ''),
                    'access_count': attrs.get('access_count', 0)
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        获取邻居节点
        
        Args:
            entity_id: 实体ID
            relation_type: 关系类型过滤
            limit: 返回数量限制
        """
        if not self._connected or not self.graph.has_node(entity_id):
            return []
        
        results = []
        
        # 出边（entity_id -> others）
        for _, target in self.graph.out_edges(entity_id):
            edge_data = self.graph.edges[entity_id, target]
            
            if relation_type and edge_data.get('relation') != relation_type:
                continue
            
            target_data = self.graph.nodes[target]
            results.append({
                'target_id': target,
                'target_name': target_data.get('name', str(target)),
                'target_type': target_data.get('type', '未知'),
                'relation': edge_data.get('relation', '相关'),
                'confidence': edge_data.get('confidence', 0.5),
                'evidence': edge_data.get('evidence', ''),
                'direction': 'out'
            })
        
        # 入边（others -> entity_id）
        for source, _ in self.graph.in_edges(entity_id):
            edge_data = self.graph.edges[source, entity_id]
            
            if relation_type and edge_data.get('relation') != relation_type:
                continue
            
            source_data = self.graph.nodes[source]
            results.append({
                'target_id': source,
                'target_name': source_data.get('name', str(source)),
                'target_type': source_data.get('type', '未知'),
                'relation': edge_data.get('relation', '相关'),
                'confidence': edge_data.get('confidence', 0.5),
                'evidence': edge_data.get('evidence', ''),
                'direction': 'in'
            })
        
        return results[:limit]
    
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
            return []
        
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return []
        
        try:
            # 使用 BFS 找最短路径
            path = nx.shortest_path(
                self.graph,
                source_id,
                target_id,
                cutoff=max_depth
            )
            return [path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_stats(self) -> Dict:
        """获取图统计信息"""
        if not self._connected:
            return {"connected": False}
        
        # 统计实体类型分布
        type_dist = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get('type', '未知')
            type_dist[node_type] = type_dist.get(node_type, 0) + 1
        
        # 统计关系类型分布
        relation_dist = {}
        for _, _, edge_data in self.graph.edges(data=True):
            relation = edge_data.get('relation', '相关')
            relation_dist[relation] = relation_dist.get(relation, 0) + 1
        
        # 度中心性最高的节点
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            most_connected = sorted(
                degree_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            most_connected = [
                {'id': n, 'name': self.graph.nodes[n].get('name', str(n)), 'connections': int(self.graph.degree(n))}
                for n, _ in most_connected
            ]
        else:
            most_connected = []
        
        return {
            "connected": True,
            "totalEntities": self.graph.number_of_nodes(),
            "totalRelations": self.graph.number_of_edges(),
            "typeDistribution": type_dist,
            "relationTypes": relation_dist,
            "mostConnected": most_connected
        }
    
    def get_all_entities(self, limit: int = 100) -> List[Dict]:
        """获取所有实体"""
        results = []
        for i, node_id in enumerate(self.graph.nodes()):
            if i >= limit:
                break
            attrs = self.graph.nodes[node_id]
            results.append({
                'id': node_id,
                'name': attrs.get('name', str(node_id)),
                'type': attrs.get('type', '未知'),
                'confidence': attrs.get('confidence', 0.5),
                'access_count': attrs.get('access_count', 0)
            })
        return results
    
    def get_all_relations(self, limit: int = 100) -> List[Dict]:
        """获取所有关系"""
        results = []
        for i, (source, target) in enumerate(self.graph.edges()):
            if i >= limit:
                break
            edge_data = self.graph.edges[source, target]
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            results.append({
                'source_id': source,
                'source_name': source_data.get('name', str(source)),
                'source_type': source_data.get('type', '未知'),
                'target_id': target,
                'target_name': target_data.get('name', str(target)),
                'target_type': target_data.get('type', '未知'),
                'relation': edge_data.get('relation', '相关'),
                'confidence': edge_data.get('confidence', 0.5)
            })
        return results
    
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        if not self._connected or not self.graph.has_node(entity_id):
            return False
        
        self.graph.remove_node(entity_id)
        self._save()
        return True
    
    def delete_relation(self, source_id: str, target_id: str) -> bool:
        """删除关系"""
        if not self._connected or not self.graph.has_edge(source_id, target_id):
            return False
        
        self.graph.remove_edge(source_id, target_id)
        self._save()
        return True
    
    def clear(self) -> bool:
        """清空图谱"""
        self.graph.clear()
        self._save()
        return True


# ============================================================================
# 工厂函数
# ============================================================================

_nx_clients: Dict[str, NetworkXKGClient] = {}


def get_nx_client(agent_id: str = "main") -> NetworkXKGClient:
    """获取或创建 NetworkX 客户端"""
    if agent_id not in _nx_clients:
        _nx_clients[agent_id] = NetworkXKGClient(agent_id)
    return _nx_clients[agent_id]