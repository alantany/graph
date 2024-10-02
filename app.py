import streamlit as st
import neo4j
from neo4j import GraphDatabase
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import asyncio
import concurrent.futures
import time
import threading
import traceback

# Neo4j连接配置
AURA_URI = "neo4j+s://b76a61f2.databases.neo4j.io:7687"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"

LOCAL_URI = "bolt://localhost:7687"
LOCAL_USERNAME = "test"
LOCAL_PASSWORD = "Mikeno01"

def get_available_databases(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            result = session.run("SHOW DATABASES")
            return [record["name"] for record in result]
    finally:
        driver.close()

def get_driver(use_aura, database_name="neo4j"):
    uri = AURA_URI if use_aura else LOCAL_URI
    username = AURA_USERNAME if use_aura else LOCAL_USERNAME
    password = AURA_PASSWORD if use_aura else LOCAL_PASSWORD
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # 验证数据库是否存在
    with driver.session() as session:
        result = session.run("SHOW DATABASES")
        databases = [record["name"] for record in result]
        if database_name not in databases:
            st.error(f"数据库 '{database_name}' 不存在。可用的数据库有: {', '.join(databases)}")
            driver.close()
            st.stop()
    
    return driver

def test_connection(driver):
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] == 1:
                return True, "成功连接到Neo4j数据库"
            else:
                return False, "连接测试失败"
    except Exception as e:
        return False, f"连接Neo4j数据库时发生错误: {str(e)}"

def main():
    try:
        st.title("图数据分析平台")
        
        use_aura = st.sidebar.checkbox("使用Neo4j Aura", value=False)
        
        # 获取可用的数据库列表
        uri = AURA_URI if use_aura else LOCAL_URI
        username = AURA_USERNAME if use_aura else LOCAL_USERNAME
        password = AURA_PASSWORD if use_aura else LOCAL_PASSWORD
        available_databases = get_available_databases(uri, username, password)
        
        # 使用下拉菜单选择数据库
        database_name = st.sidebar.selectbox("选择数据库", available_databases, index=0)
        
        driver = get_driver(use_aura, database_name)
        
        st.info(f"Neo4j驱动版本: {neo4j.__version__}")
        
        st.write("连接详情:")
        st.write(f"URI: {uri}")
        st.write(f"Username: {username}")
        st.write(f"选择的数据库: {database_name}")
        
        success, message = test_connection(driver)
        if success:
            st.success(f"成功连接到Neo4j数据库: {message}")
        else:
            st.error(f"无法连接到Neo4j数据库: {message}")
            st.write("请检查以下几点：")
            st.write("1. 确保Neo4j实例处于活动状态")
            st.write("2. 检查网络连接，确保没有防火墙阻止")
            st.write("3. 验证连接字符串、用户名和密码是否正确")
            st.stop()
        
        menu = ["数据导入", "图数据可视化", "图算法分析", "查询界面"]
        choice = st.sidebar.selectbox("选择功能", menu)
        
        if choice == "数据导入":
            data_import(driver)
        elif choice == "图数据可视化":
            graph_visualization(driver)
        elif choice == "图算法分析":
            graph_analysis(driver)
        elif choice == "查询界面":
            query_interface(driver)
    except Exception as e:
        st.error(f"程序运行时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())

def import_file(driver, file, stop_flag, clear_existing=False):
    try:
        st.write(f"开始导入文件: {file.name}")
        st.write(f"文件大小: {file.size} bytes")
        
        # 直接打印文件内容
        st.write("文件内容预览（前10行）:")
        content = file.read().decode('utf-8')
        lines = content.split('\n')[:10]
        for line in lines:
            st.write(line)
        
        # 重置文件指针
        file.seek(0)
        
        file_type = file.name.split('.')[-1].lower()
        st.write(f"文件类型: {file_type}")
        
        # 读取文件的前几行来检查文件格式
        st.write("尝试读取文件前5行...")
        sample_lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')][:5]
        sample = [line.split() for line in sample_lines]
        st.write("文件前5行预览:")
        st.write(sample)
        
        # 获取文总数（用于进度计算）
        st.write("计算文件总行数...")
        total_lines = len([line for line in content.split('\n') if line.strip() and not line.startswith('#')])
        st.write(f"文件总行数: {total_lines}")
        
        if len(sample[0]) == 2:  # 如果是边数据（FromNodeId ToNodeId）
            st.write("检测到边数据格式")
            chunk_size = 500000  # 可以根据内存情况调整这个值
            total_rows = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.write("准备开始数据库操作...")
            with driver.session() as session:
                if clear_existing:
                    clear_database(driver)
                
                for i in range(0, total_lines, chunk_size):
                    if stop_flag.is_set():
                        st.write("导入过程被用户中断")
                        return False, "导入过程被用户中断"
                    
                    chunk = [line.split() for line in content.split('\n')[i:i+chunk_size] if line.strip() and not line.startswith('#')]
                    st.write(f"正在处理 {len(chunk)} 行数据...")
                    try:
                        result = session.run("""
                        UNWIND $rows AS row
                        CREATE (from:Node {id: row[0]})
                        CREATE (to:Node {id: row[1]})
                        CREATE (from)-[:CONNECTS]->(to)
                        RETURN count(*) as operations
                        """, rows=chunk)
                        
                        summary = result.consume()
                        st.write("Neo4j 操作摘要:")
                        st.write(f"- 创建的节点数: {summary.counters.nodes_created}")
                        st.write(f"- 创建的关系数: {summary.counters.relationships_created}")
                        st.write(f"- 设置的属性数: {summary.counters.properties_set}")
                    except Exception as db_error:
                        st.error(f"数据库操作错误: {str(db_error)}")
                        return False, f"数据库操作错误: {str(db_error)}"
                    
                    total_rows += len(chunk)
                    progress = total_rows / total_lines
                    progress_bar.progress(progress)
                    status_text.text(f"已处理 {total_rows}/{total_lines} 行数据 ({progress:.2%})")
        else:
            st.write(f"无法处理文件格式: {file.name}，列数: {len(sample[0])}")
            return False, f"无法处理文件格式: {file.name}，列数: {len(sample[0])}"
        
        st.write(f"{file.name} 导入成功！共处理 {total_rows} 行数据。")
        return True, f"{file.name} 导入成功！共处理 {total_rows} 数据。"
    except Exception as e:
        st.error(f"导入过程中发生异常: {str(e)}")
        return False, f"导入 {file.name} 时发生错误: {str(e)}"

def data_import(driver):
    st.header("数据导入")
    
    import_type = st.radio("选择导入方式", ["单个文件", "多个文件"])
    clear_existing = st.checkbox("在导入前清空数据库")
    
    if import_type == "单个文件":
        uploaded_file = st.file_uploader("选择要导入的文件", type=["txt", "csv"])
        if uploaded_file is not None:
            st.write(f"文件已上传: {uploaded_file.name}")
            col1, col2 = st.columns(2)
            start_button = col1.button("开始导入")
            stop_button = col2.button("停止导入")
            
            if start_button:
                with st.expander("导入日志", expanded=True):
                    st.write("开始导入过程...")
                    stop_flag = threading.Event()
                    
                    status_area = st.empty()
                    
                    def update_status():
                        while not stop_flag.is_set():
                            status_area.text("导入进行中...")
                            time.sleep(1)
                    
                    status_thread = threading.Thread(target=update_status)
                    status_thread.start()
                    
                    try:
                        success, message = import_file(driver, uploaded_file, stop_flag, clear_existing)
                    except Exception as e:
                        success = False
                        message = f"导入过程中发生错误: {str(e)}"
                    
                    stop_flag.set()  # 停止状态更新线程
                    status_thread.join()
                    
                    st.write("导入过程已结束")
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    else:
        uploaded_files = st.file_uploader("选择要导入的文件", type=["txt", "csv"], accept_multiple_files=True)
        if uploaded_files:
            st.write(f"选择了 {len(uploaded_files)} 个文件")
            col1, col2 = st.columns(2)
            start_button = col1.button("开始导入")
            stop_button = col2.button("停止导入")
            
            if start_button:
                stop_flag = threading.Event()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_files = len(uploaded_files)
                for i, file in enumerate(uploaded_files):
                    if stop_flag.is_set():
                        st.warning("导入过程被用户中断")
                        break
                    
                    status_text.text(f"正在处理文件 {i+1}/{total_files}: {file.name}")
                    success, message = import_file(driver, file, stop_flag)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                        if message == "导入过程被用户中断":
                            break
                    
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    
                    if stop_button:
                        stop_flag.set()
                        st.warning("正在停止导入过程...")
                        break
                
                if not stop_flag.is_set():
                    st.success("所有文件导入完成！")

def graph_visualization(driver):
    st.header("图数据可视化")
    
    visualization_options = [
        "节点度分布",
        "最高度中心性点",
        "最短路径可视化",
        "连通组件分析",
        "网络拓扑概览"
    ]
    
    visualization_choice = st.selectbox("选择可视化类型", visualization_options)
    
    if visualization_choice == "节点度分布":
        visualize_degree_distribution(driver)
    elif visualization_choice == "最高度中心性节点":
        visualize_high_degree_centrality(driver)
    elif visualization_choice == "最短路径可视化":
        visualize_shortest_path(driver)
    elif visualization_choice == "连通组件分析":
        visualize_connected_components(driver)
    elif visualization_choice == "网络拓扑概览":
        visualize_network_topology(driver)

def visualize_degree_distribution(driver):
    st.subheader("节点度分布")
    with driver.session() as session:
        result = session.run("""
        MATCH (n:Node)
        WITH n, size([(n)-[]-() | 1]) AS degree
        RETURN degree, count(*) AS count
        ORDER BY degree
        """)
        df = pd.DataFrame([dict(record) for record in result])
    
    if df.empty:
        st.warning("没有找到数据。请确保图数据已正确加载。")
    else:
        fig = px.bar(df, x="degree", y="count", log_y=True, title="节点度分布")
        fig.update_layout(xaxis_title="度", yaxis_title="节点数量（对数刻度）")
        st.plotly_chart(fig)

def visualize_high_degree_centrality(driver):
    st.subheader("最高度中心性节点")
    limit = st.slider("显示前N个节点", 5, 50, 20)
    with driver.session() as session:
        result = session.run(f"""
        MATCH (n:Node)
        WITH n, size([(n)-[]-() | 1]) AS degree
        ORDER BY degree DESC
        LIMIT {limit}
        RETURN n.id AS node_id, degree
        """)
        df = pd.DataFrame([dict(record) for record in result])
    
    if df.empty:
        st.warning("没有找到数据。请确保图数据已正确加载。")
    else:
        fig = px.bar(df, x="node_id", y="degree", title=f"前{limit}个最高度中心性节点")
        fig.update_layout(xaxis_title="节点ID", yaxis_title="度")
        st.plotly_chart(fig)

def visualize_shortest_path(driver):
    st.subheader("最短路径可视化")
    existing_nodes = get_existing_nodes(driver, 100)
    source = st.selectbox("选择起始节点", existing_nodes)
    target = st.selectbox("选择目标节点", existing_nodes)
    
    if source != target:
        with driver.session() as session:
            result = session.run("""
            MATCH (start:Node {id: $source}), (end:Node {id: $target}),
                  p = shortestPath((start)-[:CONNECTS*]-(end))
            RETURN [node in nodes(p) | node.id] AS path
            """, source=source, target=target)
            path = result.single()
            
            if path:
                path = path['path']
                st.write(f"从节点 {source} 到节点 {target} 的最短路径:")
                st.write(" -> ".join(path))
                
                # 创建路径的可视化
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                G = nx.Graph()
                G.add_edges_from(edges)
                pos = nx.spring_layout(G)
                
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', marker=dict(size=10), text=list(G.nodes()), textposition="top center")

                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(showlegend=False, hovermode='closest',
                                                 margin=dict(b=20,l=5,r=5,t=40),
                                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                
                st.plotly_chart(fig)
            else:
                st.write("未找到路径")
    else:
        st.write("请选择不同的起始和目标节点")

def visualize_connected_components(driver):
    st.subheader("连通组件分析")
    with driver.session() as session:
        result = session.run("""
        CALL {
          MATCH (n:Node)
          WHERE NOT EXISTS((n)--())
          RETURN n.id AS nodeId, -1 AS componentId
          UNION
          MATCH (n:Node)
          WHERE EXISTS((n)--())
          CALL apoc.path.subgraphNodes(n, {
            relationshipFilter: 'CONNECTS'
          }) YIELD node
          WITH collect(node) AS component, n
          RETURN n.id AS nodeId, id(component[0]) AS componentId
        }
        WITH componentId, collect(nodeId) AS nodeIds, count(*) AS size
        RETURN componentId, size
        ORDER BY size DESC
        LIMIT 10
        """)
        df = pd.DataFrame([dict(record) for record in result])
    
    if not df.empty:
        st.write("前10个最大连通组件:")
        st.write(df)
        
        fig = px.bar(df, x="componentId", y="size", title="前10个最大连通组件")
        fig.update_layout(xaxis_title="组件ID", yaxis_title="节点数量")
        st.plotly_chart(fig)
    else:
        st.write("未找到连通组件")

def visualize_network_topology(driver):
    st.subheader("网络拓扑概览")
    sample_size = st.slider("选择样本大小", 100, 1000, 500)
    
    with driver.session() as session:
        result = session.run(f"""
        MATCH (n1)-[r]->(n2)
        WITH n1, n2, r
        ORDER BY rand()
        LIMIT {sample_size}
        RETURN n1.id AS source, n2.id AS target
        """)
        df = pd.DataFrame([dict(record) for record in result])
    
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    pos = nx.spring_layout(G)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, colorbar=dict(thickness=15, title='节点连接数', xanchor='left', titleside='right'))
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'节点ID: {node}<br># 连接: {len(adjacencies)}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=f'网络拓扑概览 (样本大小: {sample_size})', titlefont_size=16, showlegend=False, hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    st.plotly_chart(fig)

def graph_analysis(driver):
    st.header("图算法分析")
    
    existing_nodes = get_existing_nodes(driver)
    st.write("数据库中的一些节点示例：", existing_nodes)
    
    analysis_options = [
        "基本图统计",
        "度分布分析",
        "最短路径分析",
        "节点连接查询",
        "中心性分析",
        "社区检测",
        "网络拓扑可视化"
    ]
    
    analysis_choice = st.selectbox("选择分析类型", analysis_options)
    
    if analysis_choice == "基本图统计":
        show_basic_stats(driver)
    elif analysis_choice == "度分布分析":
        degree_distribution_analysis(driver)
    elif analysis_choice == "最短路径分析":
        shortest_path_analysis(driver, existing_nodes)
    elif analysis_choice == "节点连接查询":
        node_connections(driver, existing_nodes)
    elif analysis_choice == "中心性分析":
        centrality_analysis(driver)
    elif analysis_choice == "社区检测":
        community_detection(driver)
    elif analysis_choice == "网络拓扑可视化":
        network_topology_visualization(driver)

def show_basic_stats(driver):
    st.subheader("基本图统计")
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        
    st.write(f"节点数量：{node_count}")
    st.write(f"边数量：{edge_count}")
    st.write(f"平均度：{2 * edge_count / node_count:.2f}")

def degree_distribution_analysis(driver):
    st.subheader("度分布分析")
    with driver.session() as session:
        result = session.run("""
        MATCH (n)
        WITH n
        MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        RETURN degree, count(*) AS count
        ORDER BY degree
        """)
        df = pd.DataFrame([dict(record) for record in result])
    
    fig = px.bar(df, x="degree", y="count", log_y=True, title="节点度分布")
    fig.update_layout(xaxis_title="度", yaxis_title="节点数量（对数刻度）")
    st.plotly_chart(fig)

def shortest_path_analysis(driver, existing_nodes):
    st.subheader("最短路径分析")
    source = st.selectbox("选择起始节点ID", existing_nodes)
    target = st.selectbox("选择目标节点ID", existing_nodes)
    
    if source == target:
        st.warning("起始节点和目标节点相同，请选择不同的节点。")
        return
    
    with driver.session() as session:
        if is_apoc_available(driver):
            result = session.run("""
            MATCH (source:Node {id: $source}), (target:Node {id: $target})
            CALL apoc.algo.dijkstra(source, target, 'CONNECTS', 'weight')
            YIELD path, weight
            RETURN [node in nodes(path) | node.id] AS path, weight
            """, source=source, target=target)
        else:
            st.warning("APOC库不可用。使用基本Cypher查询查找最短路径。")
            result = session.run("""
            MATCH (source:Node {id: $source}), (target:Node {id: $target}),
                  p = shortestPath((source)-[:CONNECTS*]-(target))
            RETURN [node in nodes(p) | node.id] AS path, length(p) AS weight
            """, source=source, target=target)
        
        path = result.single()
        if path:
            st.write(f"从节点 {source} 到节点 {target} 的最短路径:")
            st.write(f"路径长度: {path['weight']}")
            st.write(f"路径节点: {path['path']}")
        else:
            st.write("未找到路径")

def node_connections(driver, existing_nodes):
    st.subheader("节点连接查询")
    node_id = st.selectbox("选择节点ID", existing_nodes)
    
    with driver.session() as session:
        result = session.run("""
        MATCH (n:Node {id: $node_id})-[r:CONNECTS]-(neighbor)
        RETURN neighbor.id AS neighbor_id, type(r) AS relationship_type, count(*) AS connection_count
        ORDER BY connection_count DESC
        LIMIT 10
        """, node_id=node_id)
        
        df = pd.DataFrame([dict(record) for record in result])
        if not df.empty:
            st.write(f"节点 {node_id} 的前10个连接:")
            st.write(df)
        else:
            st.write(f"节点 {node_id} 没有连接。")

def centrality_analysis(driver):
    st.subheader("中心性分析")
    centrality_type = st.selectbox("选择中心性类型", ["度中心性", "介数中心性", "接近中心性"])
    
    if is_gds_available(driver):
        # 使用GDS库的中心性算法
        if centrality_type == "度中心性":
            query = """
            CALL gds.degree.stream('myGraph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS node, score
            ORDER BY score DESC
            LIMIT 10
            """
        elif centrality_type == "介数中心性":
            query = """
            CALL gds.betweenness.stream('myGraph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS node, score
            ORDER BY score DESC
            LIMIT 10
            """
        else:  # 接近中心性
            query = """
            CALL gds.closeness.stream('myGraph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS node, score
            ORDER BY score DESC
            LIMIT 10
            """
    else:
        # 使用基本的Cypher查询来计算中心性
        st.warning("GDS库不可用。使用基本Cypher查���计算中心性���")
        if centrality_type == "度中心性":
            query = """
            MATCH (n:Node)
            RETURN n.id AS node, size([(n)-[]-() | 1]) AS score
            ORDER BY score DESC
            LIMIT 10
            """
        elif centrality_type == "介数中心性":
            st.error("无法计算介数中心性，需要GDS库支持。")
            return
        else:  # 接近中心性
            st.error("无法计算接近中心性，需要GDS库支持。")
            return
    
    with driver.session() as session:
        result = session.run(query)
        df = pd.DataFrame([dict(record) for record in result])
    
    if df.empty:
        st.warning("没有找到数据。请确保图数据已正确加载。")
    else:
        st.write(f"Top 10 {centrality_type}节点:")
        st.write(df)
        
        fig = px.bar(df, x="node", y="score", title=f"Top 10 {centrality_type}节点")
        st.plotly_chart(fig)

def community_detection(driver):
    st.subheader("社区检测")
    algorithm = st.selectbox("选择社区检测算法", ["基本连通分量", "Louvain (需要GDS库)", "Label Propagation (需要GDS库)"])
    
    if algorithm == "基本连通分量":
        query = """
        CALL {
          MATCH (n:Node)
          WHERE NOT EXISTS((n)--())
          RETURN n.id AS nodeId, -1 AS communityId
          UNION
          MATCH (n:Node)
          WHERE EXISTS((n)--())
          CALL apoc.path.subgraphNodes(n, {
            relationshipFilter: 'CONNECTS'
          }) YIELD node
          WITH collect(node) AS community, n
          RETURN n.id AS nodeId, id(community[0]) AS communityId
        }
        WITH communityId, collect(nodeId) AS nodeIds, count(*) AS size
        RETURN communityId, size
        ORDER BY size DESC
        LIMIT 10
        """
    elif algorithm == "Louvain (需要GDS库)":
        if is_gds_available(driver):
            query = """
            CALL gds.louvain.stream('myGraph')
            YIELD nodeId, communityId
            RETURN communityId, count(*) AS size
            ORDER BY size DESC
            LIMIT 10
            """
        else:
            st.error("GDS库不可用。无法执行Louvain算法。")
            return
    else:  # Label Propagation
        if is_gds_available(driver):
            query = """
            CALL gds.labelPropagation.stream('myGraph')
            YIELD nodeId, communityId
            RETURN communityId, count(*) AS size
            ORDER BY size DESC
            LIMIT 10
            """
        else:
            st.error("GDS库不可用。无法执行Label Propagation算法。")
            return
    
    with driver.session() as session:
        result = session.run(query)
        df = pd.DataFrame([dict(record) for record in result])
    
    if df.empty:
        st.warning("没有找到社区。请确保图数据已正确加载。")
    else:
        st.write(f"Top 10 最大社区:")
        st.write(df)
        
        fig = px.bar(df, x="communityId", y="size", title="Top 10 最大社区")
        st.plotly_chart(fig)

def network_topology_visualization(driver):
    st.subheader("网络拓扑可视化")
    sample_size = st.slider("选择样本大小", 100, 1000, 500)
    
    with driver.session() as session:
        result = session.run(f"""
        MATCH (n1)-[r]->(n2)
        WITH n1, n2, r
        ORDER BY rand()
        LIMIT {sample_size}
        RETURN n1.id AS source, n2.id AS target
        """)
        df = pd.DataFrame([dict(record) for record in result])
    
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='节点连接数',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'节点ID: {node}<br># 连接: {len(adjacencies)}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'网络拓扑可视化 (样本大小: {sample_size})',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig)

def query_interface(driver):
    st.header("自定义查询")
    
    query = st.text_area("输入Cypher查询")
    if st.button("执行查询"):
        try:
            with driver.session() as session:
                result = session.run(query)
                df = pd.DataFrame([dict(record) for record in result])
                if not df.empty:
                    st.write(df)
                else:
                    st.info("查询执行成功，但没有返回结果。")
        except Exception as e:
            st.error(f"执行查询时发生错误: {str(e)}")

def get_existing_nodes(driver, limit=10):
    with driver.session() as session:
        result = session.run("""
        MATCH (n:Node)
        RETURN n.id AS id
        ORDER BY n.id
        LIMIT $limit
        """, limit=limit)
        return [record["id"] for record in result]

def is_apoc_available(driver):
    try:
        with driver.session() as session:
            result = session.run("CALL apoc.help('dijkstra')")
            return True
    except Exception:
        return False

def is_gds_available(driver):
    try:
        with driver.session() as session:
            result = session.run("CALL gds.list()")
            return True
    except Exception:
        return False

if __name__ == "__main__":
    main()