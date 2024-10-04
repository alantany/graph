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
from pyvis.network import Network
import streamlit.components.v1 as components
from generate_financial_data import generate_data, write_to_csv
import matplotlib.pyplot as plt

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

def show_business_scenario():
    st.header("金融风控中的图数据库应用")
    st.write("""
    在复杂的金融环境中，传统的风控方法往往难以应对日益复杂的欺诈和风险行为。
    图数据库技术为我们提供了一种新的视角，能够快速识别复杂的关系网络，发现潜在的风险模式。
    
    本演示将展示图数据库如何帮助我们：
    1. 快速识别高风险用户
    2. 揭示潜在的欺诈团伙
    3. 发现异常的行为模式
    
    通过这些分析，我们可以更有效地预防金融风险，保护客户和机构的利益。
    """)

def main():
    try:
        st.title("金融风控图数据分析平台")
        
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
            st.write("3. 验证连接字符、用户和密码是否正确")
            st.stop()
        
        st.sidebar.title("金融风控演示")
        menu = ["数据管理", "业务场景介绍", "数据概览", "风险分析"]
        choice = st.sidebar.selectbox("选择功能", menu)
        
        if choice == "数据管理":
            data_management(driver)
        elif choice == "业务场景介绍":
            show_business_scenario()
        elif choice == "数据概览":
            show_data_overview(driver)
        elif choice == "风险分析":
            risk_analysis(driver)
    except Exception as e:
        st.error(f"程序运行时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())

def data_management(driver):
    st.header("数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空数据库"):
            clear_database(driver)
            st.success("数据库已清空")
            show_database_stats(driver)
    
    with col2:
        if st.button("导入数据"):
            import_data(driver)
            st.success("数据导入完成")
            show_database_stats(driver)

def clear_database(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def import_data(driver):
    risk_dir = 'risk'
    files = {
        "用户数据": "users.csv",
        "银行账户数据": "bank_accounts.csv",
        "商户数据": "merchants.csv",
        "设备数据": "devices.csv",
        "IP地址数据": "ip_addresses.csv",
        "交易数据": "transactions.csv"
    }
    
    for file_desc, file_name in files.items():
        file_path = os.path.join(risk_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read()
                import_csv_data(driver, file_name, csv_data)
            st.success(f"{file_desc} 导入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。请确保已生成数据文件。")
        except Exception as e:
            st.error(f"导入 {file_desc} 时发生错误: {str(e)}")

def import_file(driver, file, stop_flag, clear_existing=False):
    try:
        st.write(f"开始导入文件: {file.name}")
        st.write(f"文件大小: {file.size} bytes")
        
        # 直接打印文件内
        st.write("文件内容预览（前10行）:")
        content = file.read().decode('utf-8')
        lines = content.split('\n')[:10]
        for line in lines:
            st.write(line)
        
        # 重置文件指针
        file.seek(0)
        
        file_type = file.name.split('.')[-1].lower()
        st.write(f"文类型: {file_type}")
        
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
        
        if len(sample[0]) == 2:  # 果是边数据（FromNodeId ToNodeId）
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
                        st.write(f"- 创的: {summary.counters.relationships_created}")
                        st.write(f"- 设置的属性数: {summary.counters.properties_set}")
                    except Exception as db_error:
                        st.error(f"数据库操作错误: {str(db_error)}")
                        return False, f"数据库操作错误: {str(db_error)}"
                    
                    total_rows += len(chunk)
                    progress = total_rows / total_lines
                    progress_bar.progress(progress)
                    status_text.text(f"已处理 {total_rows}/{total_lines} 行数据 ({progress:.2%})")
        else:
            st.write(f"无法理文件格式: {file.name}，列数: {len(sample[0])}")
            return False, f"无法处理文件格式: {file.name}，列数: {len(sample[0])}"
        
        st.write(f"{file.name} 导入成功！共处理 {total_rows} 行数据。")
        return True, f"{file.name} 导入成功！共处理 {total_rows} 数据。"
    except Exception as e:
        st.error(f"导入过程中发生异常: {str(e)}")
        return False, f"导入 {file.name} 时发生错误: {str(e)}"

def import_generated_data(driver):
    st.subheader("数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空数据库"):
            clear_database(driver)
            st.success("数据库已清空")
            show_database_stats(driver)
    
    with col2:
        if st.button("导入数据"):
            import_data(driver, should_create_indexes=True)
            st.success("数据导入完成")
            show_database_stats(driver)

def import_csv_data(driver, file_name, csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    
    with driver.session() as session:
        if file_name == "users.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (u:User {id: row.id})
            SET u.name = row.name, u.risk_score = toFloat(row.risk_score)
            """, rows=df.to_dict('records'))
        elif file_name == "bank_accounts.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:User {id: row.user_id})
            MERGE (a:BankAccount {id: row.id})
            SET a.balance = toFloat(row.balance)
            MERGE (u)-[:OWNS_ACCOUNT]->(a)
            """, rows=df.to_dict('records'))
        elif file_name == "merchants.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (m:Merchant {id: row.id})
            SET m.name = row.name, m.category = row.category
            """, rows=df.to_dict('records'))
        elif file_name == "devices.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Device {id: row.id})
            SET d.type = row.type
            """, rows=df.to_dict('records'))
        elif file_name == "ip_addresses.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (ip:IPAddress {id: row.id})
            SET ip.address = row.address
            """, rows=df.to_dict('records'))
        elif file_name == "transactions.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:User {id: row.user_id})
            MATCH (m:Merchant {id: row.merchant_id})
            MATCH (d:Device {id: row.device_id})
            MATCH (ip:IPAddress {id: row.ip_id})
            MERGE (t:Transaction {id: row.id})
            SET t.amount = toFloat(row.amount), t.timestamp = row.timestamp, t.status = row.status
            MERGE (u)-[:MADE_TRANSACTION]->(t)
            MERGE (t)-[:INVOLVES_MERCHANT]->(m)
            MERGE (t)-[:USES_DEVICE]->(d)
            MERGE (t)-[:USES_IP]->(ip)
            """, rows=df.to_dict('records'))

    st.success(f"{file_name} 导入成功！")

def show_database_stats(driver):
    st.subheader("数据库统计")
    queries = {
        "节点总数": "MATCH (n) RETURN count(n) as count",
        "关系总数": "MATCH ()-[r]->() RETURN count(r) as count",
        "用户数": "MATCH (u:User) RETURN count(u) as count",
        "银行账户数": "MATCH (a:BankAccount) RETURN count(a) as count",
        "商户数": "MATCH (m:Merchant) RETURN count(m) as count",
        "设备数": "MATCH (d:Device) RETURN count(d) as count",
        "IP址数": "MATCH (ip:IPAddress) RETURN count(ip) as count",
        "交易数": "MATCH (t:Transaction) RETURN count(t) as count"
    }
    
    results = {}
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query).single()
            results[label] = result["count"] if result else 0
    
    for label, count in results.items():
        st.write(f"{label}: {count}")

def risk_analysis(driver):
    st.header("风险分析案例展示")
    
    st.subheader("1. 高风险用户识别")
    query = """
    MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)
    WHERE t.status = 'Flagged'
    WITH u, count(t) as flagged_count, sum(toFloat(t.amount)) as total_amount
    WHERE flagged_count > 3
    RETURN u.id AS user_id, u.name AS user_name, flagged_count, total_amount
    ORDER BY flagged_count DESC, total_amount DESC
    LIMIT 10
    """
    results = run_query(driver, query)
    
    if not results.empty:
        fig = go.Figure(data=[go.Scatter(
            x=results['flagged_count'],
            y=results['total_amount'],
            mode='markers',
            marker=dict(
                size=results['flagged_count'] * 5,
                color=results['total_amount'],
                colorscale='Viridis',
                showscale=True
            ),
            text=results['user_name'],
            hovertemplate=
            "<b>%{text}</b><br>" +
            "可疑交易次数: %{x}<br>" +
            "交易总额: %{y:.2f}<br>" +
            "<extra></extra>"
        )])
        fig.update_layout(
            title="高风险用户分布",
            xaxis_title="可疑交易次数",
            yaxis_title="交易总额",
            showlegend=False
        )
        st.plotly_chart(fig)
        
        st.write("在这个高风险用户分布图中，我们可以观察到：")
        st.write(f"1. 共有 {len(results)} 个高风险用户被识别出来。")
        max_flagged_user = results.loc[results['flagged_count'].idxmax()]
        st.write(f"2. 用户 {max_flagged_user['user_name']} (ID: {max_flagged_user['user_id']}) 进行了最多的可疑交易，共 {max_flagged_user['flagged_count']} 次，总金额达到 {max_flagged_user['total_amount']:.2f}。")
        max_amount_user = results.loc[results['total_amount'].idxmax()]
        st.write(f"3. 用户 {max_amount_user['user_name']} (ID: {max_amount_user['user_id']}) 的可疑交易总额最高，达到 {max_amount_user['total_amount']:.2f}共进行了 {max_amount_user['flagged_count']} 次可疑交易。")
        st.write(f"4. 平均每个高风险用户进行了 {results['flagged_count'].mean():.2f} 次可疑交易，平均可疑交易总额为 {results['total_amount'].mean():.2f}。")
        st.write("5. 从图中可以看出，大多数高风险用户集中在图的右上角，表示他们既有较多的可疑交易，交易总额也较高。这些用户应该是我们重点关注的对象。")
    else:
        st.warning("未发现高风险用户。")

    st.subheader("2. 关联网络分析")
    query = """
    MATCH (u1:User)-[:MADE_TRANSACTION]->(t1:Transaction)-[:INVOLVES_MERCHANT]->(m:Merchant)<-[:INVOLVES_MERCHANT]-(t2:Transaction)<-[:MADE_TRANSACTION]-(u2:User)
    WHERE u1 <> u2 AND t1.status = 'Flagged' AND t2.status = 'Flagged'
    WITH u1, u2, m, count(DISTINCT t1) + count(DISTINCT t2) AS shared_flagged_transactions
    WHERE shared_flagged_transactions > 2
    RETURN u1.id AS user1, u2.id AS user2, m.id AS merchant, shared_flagged_transactions
    ORDER BY shared_flagged_transactions DESC
    LIMIT 15
    """
    results = run_query(driver, query)
    
    if not results.empty:
        G = nx.Graph()
        for _, row in results.iterrows():
            G.add_edge(row['user1'], row['user2'], weight=row['shared_flagged_transactions'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        plt.title("高风险用户关联网络")
        st.pyplot(fig)
        
        st.write("在这个高风险用户关联网络图中，我们可以观察到：")
        st.write(f"1. 共有 {len(G.nodes)} 个高风险用户形成了相互关联的网络。")
        st.write(f"2. 网络中共有 {len(G.edges)} 个关联。")
        
        central_node = max(G.degree, key=lambda x: x[1])[0]
        st.write(f"3. 用户 {central_node} 似乎是这个网络的中心，与其他 {G.degree[central_node]} 个用户有直接关联。这个用户可能是一个关键的风险节点。")
        
        max_weight = max(nx.get_edge_attributes(G, 'weight').values())
        max_edge = max(G.edges(data=True), key=lambda x: x[2]['weight'])
        st.write(f"4. 最强的关联是 {max_edge[0]} 和 {max_edge[1]} 之间，共享了 {max_edge[2]['weight']} 次可疑交易。这两个用户之间的交易行为值��进一步调查。")
        
        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 1]
        if isolated_nodes:
            st.write(f"5. 用户 {', '.join(isolated_nodes)} 与网络的联系相对较弱，只与一个其他用户有关联。这些可能是新加入的成员或者边缘参与者。")
        
        st.write("6. 这种网络结构可能表明存在一个有组织的欺诈团伙，中心用户可能是主要协调者。边缘用户可能是被招募的新成员或者是无意中卷入的普通用户。")
        st.write("7. 边上的数字表示共享的可疑交易数量，数字越大表示两个用户之间的关联越强，这种强关联可能意味着更频繁的协作或者更密切的关系。")
    else:
        st.warning("未发现显著的高风险用户关联网络。")

    st.subheader("3. 异常行为模式")
    query = """
    MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)-[:USES_DEVICE]->(d:Device)
    WITH u, d, count(t) as transaction_count, sum(toFloat(t.amount)) as total_amount
    WHERE transaction_count > 5 OR total_amount > 10000
    RETURN u.id AS user_id, d.id AS device_id, transaction_count, total_amount
    ORDER BY transaction_count DESC, total_amount DESC
    LIMIT 50
    """
    results = run_query(driver, query)
    if not results.empty:
        # 添加风险评估
        results['avg_transaction_amount'] = results['total_amount'] / results['transaction_count']
        results['risk_score'] = results.apply(lambda row: calculate_risk_score(row), axis=1)
        
        # 根据风险评分排序
        results = results.sort_values('risk_score', ascending=False)
        
        # 可视化
        fig = go.Figure(data=[go.Scatter(
            x=results['transaction_count'],
            y=results['total_amount'],
            mode='markers',
            marker=dict(
                size=results['avg_transaction_amount'] / 100,  # 使用平均交易金额作为气泡大小
                sizemode='area',
                sizeref=2.*max(results['avg_transaction_amount'])/(40.**2),
                sizemin=4,
                color=results['risk_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="风险评分")
            ),
            text=results.apply(lambda row: f"用户: {row['user_id']}<br>设备: {row['device_id']}<br>风险评分: {row['risk_score']:.2f}<br>平均交易金额: {row['avg_transaction_amount']:.2f}", axis=1),
            hoverinfo='text'
        )])
        fig.update_layout(
            title="用户-设备交易行为分布与风险评估",
            xaxis_title="交易次数",
            yaxis_title="交易总额",
            showlegend=False
        )
        st.plotly_chart(fig)
        
        st.write("""
        图表解读：
        1. 每个气泡代表一个用户-设备组合。
        2. X轴表示交易次数，Y轴表示交易总额。
        3. 气泡大小表示平均交易金额，越大表示单次交易金额越高。
        4. 颜色深浅表示风险评分，颜色越深表示风险越高。
        
        风险评估说明：
        1. 风险评分范围从0到100，分数越高表示风险越大。
        2. 评分考虑了交易次数、总金额和平均交易金额。
        3. 交易次数异常多、总金额异常高或平均交易金额异常高的用户-设备组合会得到更高的风险评分。
        
        基于上述数据，我们可以得出以下结论：
        """)
        
        # 添加具体的风险评估结论
        high_risk_users = results[results['risk_score'] > 80]
        if not high_risk_users.empty:
            st.write(f"1. 发现 {len(high_risk_users)} 个高风险用户-设备组合，风险评分超过80分。这些用户的交易行为需要立即审查。")
            st.write(f"2. 用户 {high_risk_users.iloc[0]['user_id']} 在设备 {high_risk_users.iloc[0]['device_id']} 上的行为最为可疑，风险评分为 {high_risk_users.iloc[0]['risk_score']:.2f}。")
        
        frequent_users = results[results['transaction_count'] > results['transaction_count'].mean() + 2*results['transaction_count'].std()]
        if not frequent_users.empty:
            st.write(f"3. 有 {len(frequent_users)} 个用户-设备组合的交易频率异常高，可能表示自动化欺诈行为。")
        
        high_amount_users = results[results['total_amount'] > results['total_amount'].mean() + 2*results['total_amount'].std()]
        if not high_amount_users.empty:
            st.write(f"4. 有 {len(high_amount_users)} 个用户-设备组合的交易总额异常高，需要进行大额交易审查。")
        
        st.write("5. 建议对风险评分前10%的用户-设备组合进行深入调查，包括交易记录审核、用户身份验证和设备安全检查。")
        st.write("6. 图表右上角和颜色较深的气泡代表最高风险的用户-设备组合，应优先调查。")
    else:
        st.warning("未发现明显的异常行为模式。这可能是因为数据集中没有足够的异常行为，或者阈值设置过高。")

def calculate_risk_score(row):
    # 这是一个简单的风险评分算法，您可以根据实际需求进行调整
    transaction_score = min(row['transaction_count'] / 20 * 100, 100)  # 20次或以上交易视为最高风险
    amount_score = min(row['total_amount'] / 50000 * 100, 100)  # 50000或以上总额视为最高风险
    avg_amount_score = min(row['avg_transaction_amount'] / 5000 * 100, 100)  # 平均5000或以上视为最高风险
    
    # 综合评分，可以调整权重
    return (transaction_score * 0.3 + amount_score * 0.4 + avg_amount_score * 0.3)

def run_query(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return pd.DataFrame([dict(record) for record in result])

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

def check_data_import(driver):
    st.subheader("数据导入检查")
    queries = {
        "用户数": "MATCH (u:User) RETURN count(u) as count",
        "设备数": "MATCH (d:Device) RETURN count(d) as count",
        "IP地址数": "MATCH (ip:IPAddress) RETURN count(ip) as count",
        "交易数": "MATCH (t:Transaction) RETURN count(t) as count"
    }
    
    results = {}
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query).single()
            results[label] = result["count"] if result else 0
    
    for label, count in results.items():
        st.write(f"{label}: {count}")

def show_sample_transaction(driver):
    st.subheader("示例交易")
    query = """
    MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)-[:INVOLVES_MERCHANT]->(m:Merchant)
    RETURN u.id AS user_id, t.id AS transaction_id, 
           t.device_id AS device_id, t.ip_id AS ip_id,
           m.id AS merchant_id, t.amount AS amount, t.status AS status
    LIMIT 1
    """
    result = run_query(driver, query)
    if not result.empty:
        st.write(result)
    else:
        st.warning("未找到任何交易数据")

def check_relationships(driver):
    st.subheader("关系检查")
    queries = {
        "用户-交易关系": "MATCH ()-[r:MADE_TRANSACTION]->() RETURN count(r) as count",
        "交易-商户关系": "MATCH ()-[r:INVOLVES_MERCHANT]->() RETURN count(r) as count",
        "交易-设备关系": "MATCH ()-[r:USES_DEVICE]->() RETURN count(r) as count",
        "交易-IP关系": "MATCH ()-[r:USES_IP]->() RETURN count(r) as count"
    }
    
    results = {}
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query).single()
            results[label] = result["count"] if result else 0
    
    for label, count in results.items():
        st.write(f"{label}: {count}")

def create_indexes(driver):
    with driver.session() as session:
        # 检查 Neo4j 版本
        result = session.run("CALL dbms.components() YIELD name, versions, edition UNWIND versions AS version RETURN version")
        version = result.single()['version']
        major_version = int(version.split('.')[0])

        if major_version >= 4:
            # Neo4j 4.x 及以上版本
            session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Merchant) ON (m.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Device) ON (d.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (ip:IPAddress) ON (ip.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Transaction) ON (t.id)")
        else:
            # Neo4j 3.x 版本
            session.run("CREATE INDEX ON :User(id)")
            session.run("CREATE INDEX ON :Merchant(id)")
            session.run("CREATE INDEX ON :Device(id)")
            session.run("CREATE INDEX ON :IPAddress(id)")
            session.run("CREATE INDEX ON :Transaction(id)")

def show_data_overview(driver):
    st.header("数据概览")
    with driver.session() as session:
        user_count = session.run("MATCH (u:User) RETURN count(u) AS count").single()["count"]
        transaction_count = session.run("MATCH (t:Transaction) RETURN count(t) AS count").single()["count"]
        flagged_count = session.run("MATCH (t:Transaction {status: 'Flagged'}) RETURN count(t) AS count").single()["count"]
    
    st.write(f"总用户数：{user_count}")
    st.write(f"总交易数：{transaction_count}")
    st.write(f"可疑交易数：{flagged_count}")
    st.write(f"可疑交易占比：{flagged_count/transaction_count:.2%}")

if __name__ == "__main__":
    main()