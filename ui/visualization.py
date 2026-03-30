"""
知识图谱可视化 UI
基于 D3.js 的交互式知识图谱可视化
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KG_VIS_PORT

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识图谱可视化</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .sidebar {
            width: 300px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
        }
        .main {
            flex: 1;
            position: relative;
        }
        h1 { font-size: 1.5rem; margin-bottom: 20px; color: #e94560; }
        h2 { font-size: 1.1rem; margin: 20px 0 10px; color: #0f3460; background: #eee; padding: 8px; border-radius: 4px; }
        
        .control-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #333;
            border-radius: 4px;
            background: #1a1a2e;
            color: #eee;
        }
        button {
            width: 100%;
            padding: 10px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            margin-top: 10px;
        }
        button:hover { background: #ff6b6b; }
        
        .stats {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        .stat-value { font-weight: bold; color: #e94560; }
        
        #graph { width: 100%; height: 100%; }
        .node { cursor: pointer; }
        .node circle { stroke: #fff; stroke-width: 2px; }
        .node text { font-size: 12px; fill: #eee; }
        .link { stroke: #999; stroke-opacity: 0.6; }
        .link-label { font-size: 10px; fill: #aaa; }
        
        .tooltip {
            position: absolute;
            background: #16213e;
            border: 1px solid #e94560;
            padding: 10px;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .legend {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(22, 33, 62, 0.9);
            padding: 15px;
            border-radius: 8px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .query-result {
            background: #0f3460;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            max-height: 200px;
            overflow-y: auto;
        }
        .query-item {
            padding: 8px;
            border-bottom: 1px solid #333;
            cursor: pointer;
        }
        .query-item:hover { background: #16213e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>🧠 知识图谱</h1>
            
            <div class="control-group">
                <label>Agent ID</label>
                <input type="text" id="agentId" value="default" />
            </div>
            
            <div class="control-group">
                <label>用户 ID</label>
                <input type="text" id="userId" value="default" />
            </div>
            
            <button onclick="loadGraph()">加载图谱</button>
            
            <div class="stats">
                <h2>📊 统计信息</h2>
                <div class="stat-item">
                    <span>节点数量</span>
                    <span class="stat-value" id="nodeCount">0</span>
                </div>
                <div class="stat-item">
                    <span>关系数量</span>
                    <span class="stat-value" id="linkCount">0</span>
                </div>
                <div class="stat-item">
                    <span>记忆数量</span>
                    <span class="stat-value" id="memoryCount">0</span>
                </div>
            </div>
            
            <h2>🔍 智能查询</h2>
            <div class="control-group">
                <label>查询内容（支持时间表达式）</label>
                <input type="text" id="queryInput" placeholder="如：上周说的广告商" />
            </div>
            <button onclick="queryMemory()">查询</button>
            <div class="query-result" id="queryResult"></div>
            
            <h2>🔗 路径查询</h2>
            <div class="control-group">
                <label>起点实体</label>
                <input type="text" id="entityA" placeholder="实体 A" />
            </div>
            <div class="control-group">
                <label>终点实体</label>
                <input type="text" id="entityB" placeholder="实体 B" />
            </div>
            <button onclick="findPath()">查找路径</button>
        </div>
        
        <div class="main">
            <svg id="graph"></svg>
            <div class="tooltip" id="tooltip"></div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #e94560;"></div>
                    <span>平台/公司</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ecdc4;"></div>
                    <span>人物</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffe66d;"></div>
                    <span>项目/产品</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #95e1d3;"></div>
                    <span>概念</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const API_URL = window.location.origin;
        
        // 颜色映射
        const colorMap = {
            '平台': '#e94560',
            '公司': '#e94560',
            '人物': '#4ecdc4',
            'Person': '#4ecdc4',
            '项目': '#ffe66d',
            '产品': '#ffe66d',
            'Project': '#ffe66d',
            '概念': '#95e1d3',
            'Concept': '#95e1d3',
            'entity': '#aaa',
            'default': '#aaa'
        };
        
        async function loadGraph() {
            const agentId = document.getElementById('agentId').value;
            const userId = document.getElementById('userId').value;
            
            try {
                // 获取知识图谱
                const kgResponse = await fetch(`${API_URL}/api/kg?agent_id=${agentId}&user_id=${userId}`);
                const kgData = await kgResponse.json();
                
                // 获取统计
                const statsResponse = await fetch(`${API_URL}/api/stats?agent_id=${agentId}&user_id=${userId}`);
                const statsData = await statsResponse.json();
                
                // 更新统计
                document.getElementById('nodeCount').textContent = kgData.graph.nodes.length;
                document.getElementById('linkCount').textContent = kgData.graph.links.length;
                document.getElementById('memoryCount').textContent = statsData.memory_count || 0;
                
                // 渲染图谱
                renderGraph(kgData.graph);
            } catch (error) {
                console.error('加载失败:', error);
                alert('加载失败: ' + error.message);
            }
        }
        
        function renderGraph(data) {
            const svg = d3.select('#graph');
            const width = svg.node().parentElement.clientWidth;
            const height = svg.node().parentElement.clientHeight;
            
            svg.selectAll('*').remove();
            
            const g = svg.append('g');
            
            // 缩放
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => g.attr('transform', event.transform));
            
            svg.call(zoom);
            
            // 力导向图
            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(40));
            
            // 绘制连线
            const link = g.append('g')
                .selectAll('line')
                .data(data.links)
                .enter().append('line')
                .attr('class', 'link')
                .attr('stroke-width', d => Math.sqrt(d.value || 1) * 2);
            
            // 绘制连线标签
            const linkLabel = g.append('g')
                .selectAll('text')
                .data(data.links)
                .enter().append('text')
                .attr('class', 'link-label')
                .attr('text-anchor', 'middle')
                .text(d => d.relation || '');
            
            // 绘制节点
            const node = g.append('g')
                .selectAll('g')
                .data(data.nodes)
                .enter().append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            
            node.append('circle')
                .attr('r', d => d.size || 15)
                .attr('fill', d => colorMap[d.group] || colorMap['default']);
            
            node.append('text')
                .attr('dy', -20)
                .attr('text-anchor', 'middle')
                .text(d => d.label);
            
            // 节点悬停
            node.on('mouseover', function(event, d) {
                const tooltip = document.getElementById('tooltip');
                tooltip.innerHTML = `<strong>${d.label}</strong><br>类型: ${d.group}<br>访问次数: ${d.size || 0}`;
                tooltip.style.opacity = 1;
                tooltip.style.left = event.pageX + 10 + 'px';
                tooltip.style.top = event.pageY - 10 + 'px';
            });
            
            node.on('mouseout', function() {
                document.getElementById('tooltip').style.opacity = 0;
            });
            
            // 更新位置
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                linkLabel
                    .attr('x', d => (d.source.x + d.target.x) / 2)
                    .attr('y', d => (d.source.y + d.target.y) / 2);
                
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
        
        async function queryMemory() {
            const agentId = document.getElementById('agentId').value;
            const userId = document.getElementById('userId').value;
            const query = document.getElementById('queryInput').value;
            
            if (!query) return;
            
            try {
                const response = await fetch(`${API_URL}/api/recall/interactive?agent_id=${agentId}&user_id=${userId}&query=${encodeURIComponent(query)}`);
                const data = await response.json();
                
                const resultDiv = document.getElementById('queryResult');
                resultDiv.innerHTML = '';
                
                data.results.forEach(r => {
                    const item = document.createElement('div');
                    item.className = 'query-item';
                    item.innerHTML = `
                        <div style="font-size: 0.8rem; color: #aaa;">${r.timestamp || ''}</div>
                        <div>${r.summary || r.content || ''}</div>
                    `;
                    resultDiv.appendChild(item);
                });
                
                if (data.results.length === 0) {
                    resultDiv.innerHTML = '<div style="padding: 10px; color: #aaa;">无匹配结果</div>';
                }
            } catch (error) {
                console.error('查询失败:', error);
            }
        }
        
        async function findPath() {
            const agentId = document.getElementById('agentId').value;
            const userId = document.getElementById('userId').value;
            const entityA = document.getElementById('entityA').value;
            const entityB = document.getElementById('entityB').value;
            
            if (!entityA || !entityB) return;
            
            try {
                const response = await fetch(`${API_URL}/api/kg/path?agent_id=${agentId}&user_id=${userId}&from=${encodeURIComponent(entityA)}&to=${encodeURIComponent(entityB)}`);
                const data = await response.json();
                
                alert(`找到 ${data.path.length} 条路径关系`);
            } catch (error) {
                console.error('路径查询失败:', error);
            }
        }
        
        // 初始加载
        loadGraph();
    </script>
</body>
</html>
"""


def run_visualization():
    """运行可视化服务"""
    from flask import Flask, render_template_string, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    # API 代理
    @app.route("/api/<path:path>", methods=["GET", "POST"])
    def api_proxy(path):
        import requests
        api_url = f"http://localhost:8082/{path}"
        
        if request.method == "GET":
            response = requests.get(api_url, params=request.args)
        else:
            response = requests.post(api_url, json=request.json)
        
        return jsonify(response.json())
    
    print(f"🎨 启动知识图谱可视化: http://localhost:{KG_VIS_PORT}")
    app.run(host="0.0.0.0", port=KG_VIS_PORT, debug=False)


if __name__ == "__main__":
    run_visualization()