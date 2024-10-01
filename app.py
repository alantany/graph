import streamlit as st
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import socket
import logging
from neo4j.exceptions import ServiceUnavailable, AuthError
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j连接配置
URI = "neo4j+s://b76a61f2.databases.neo4j.io:7687"
USERNAME = "neo4j"
PASSWORD = "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"

# 创建Neo4j驱动
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def check_network():
    try:
        host = URI.split("://")[1].split(":")[0]
        port = 7687
        sock = socket.create_connection((host, port), timeout=10)
        sock.close()
        return True, "网络连接正常"
    except Exception as e:
        return False, f"网络连接失败: {str(e)}"

def test_connection():
    try:
        logger.info(f"Attempting to connect to {URI}")
        with driver.session(database="neo4j") as session:
            logger.info("Session created, running test query")
            result = session.run("RETURN 'Hello, Neo4j!' AS message")
            record = result.single()
            if record:
                logger.info(f"Query successful: {record['message']}")
                return True, f"连接成功: {record['message']}"
            else:
                logger.warning("Query returned no results")
                return False, "连接成功但没有返回预期的消息"
    except AuthError as e:
        logger.error(f"Authentication failed: {str(e)}")
        return False, f"认证失败: {str(e)}"
    except ServiceUnavailable as e:
        logger.error(f"Service unavailable: {str(e)}")
        return False, f"服务不可用: {str(e)}"
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}", exc_info=True)
        return False, f"连接失败: {str(e)}"

def retry_connection(max_retries=3, delay=2):
    for attempt in range(max_retries):
        success, message = test_connection()
        if success:
            return success, message
        logger.warning(f"Connection attempt {attempt + 1} failed. Retrying in {delay} seconds...")
        time.sleep(delay)
    return False, "连接失败，已达到最大重试次数"

def main():
    st.title("图数据分析平台")

    # 检查网络连接
    net_success, net_message = check_network()
    if not net_success:
        st.error(f"网络诊断: {net_message}")
    else:
        st.success(f"网络诊断: {net_message}")

    # 测试Neo4j连接
    success, message = retry_connection()
    if success:
        st.success(f"成功连接到Neo4j Aura数据库: {message}")
    else:
        st.error(f"无法连接到Neo4j Aura数据库: {message}")
        st.write("连接详情:")
        st.write(f"URI: {URI}")
        st.write(f"Username: {USERNAME}")
        st.write("请检查以下几点：")
        st.write("1. 确保Neo4j Aura实例处于活动状态")
        st.write("2. 检查网络连接，确保没有防火墙阻止")
        st.write("3. 验证连接字符串、用户名和密码是否正确")
        st.write("4. 检查Neo4j Aura控制台中的连接信息是否匹配")
        st.write("5. 尝试在Neo4j Browser中直接连接")
        st.stop()

    # 侧边栏菜单
    menu = ["数据导入", "图数据可视化", "图算法分析", "查询界面"]
    choice = st.sidebar.selectbox("选择功能", menu)

    if choice == "数据导入":
        data_import()
    elif choice == "图数据可视化":
        graph_visualization()
    elif choice == "图算法分析":
        graph_analysis()
    elif choice == "查询界面":
        query_interface()

def data_import():
    st.header("数据导入")
    
    upload_file = st.file_uploader("上传CSV文件", type="csv")
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write("CSV文件预览：")
        st.write(df.head())
        
        st.write("CSV文件包含以下列：")
        st.write(df.columns.tolist())
        
        name_column = st.selectbox("选择用作name的列", df.columns)
        
        if st.button("导入到Neo4j"):
            try:
                with driver.session(database="neo4j") as session:
                    def create_person(tx, name):
                        tx.run("MERGE (p:Person {name: $name})", name=name)
                    
                    for _, row in df.iterrows():
                        session.execute_write(create_person, str(row[name_column]))
                st.success("数据已成功导入Neo4j!")
            except Exception as e:
                st.error(f"导入数据时发生错误: {str(e)}")
                st.write("请确保选择了正确的列，并且Neo4j连接配置正确。")

def graph_visualization():
    st.header("图数据可视化")
    
    try:
        with driver.session(database="neo4j") as session:
            result = session.run("MATCH (p:Person) RETURN p.name")
            nodes = [record["p.name"] for record in result]
            
            result = session.run("MATCH (p1:Person)-[r:KNOWS]->(p2:Person) RETURN p1.name, p2.name")
            edges = [(record["p1.name"], record["p2.name"]) for record in result]
        
        if not nodes:
            st.warning("没有找到节点数据。请确保数据库中有Person节点。")
            return

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"图数据可视化时发生错误: {str(e)}")

def graph_analysis():
    st.header("图算法分析")
    
    algorithm = st.selectbox("选择算法", ["页面排名", "社区检测", "最短路径"])
    
    if algorithm == "页面排名":
        try:
            with driver.session(database="neo4j") as session:
                session.run("""
                CALL gds.graph.project('myGraph', 'Person', 'KNOWS')
                YIELD graphName;
                """)
                
                result = session.run("""
                CALL gds.pageRank.stream('myGraph')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS name, score
                ORDER BY score DESC
                LIMIT 10
                """)
                df = pd.DataFrame([dict(record) for record in result])
                st.write(df)
        except Exception as e:
            st.error(f"执行页面排名算法时发生错误: {str(e)}")
    
    elif algorithm == "社区检测":
        st.info("社区检测算法尚未实现")
    
    elif algorithm == "最短路径":
        st.info("最短路径算法尚未实现")

def query_interface():
    st.header("查询界面")
    
    query = st.text_area("输入Cypher查询")
    if st.button("执行查询"):
        try:
            with driver.session(database="neo4j") as session:
                result = session.run(query)
                df = pd.DataFrame([dict(record) for record in result])
                st.write(df)
        except Exception as e:
            st.error(f"执行查询时发生错误: {str(e)}")

if __name__ == "__main__":
    main()