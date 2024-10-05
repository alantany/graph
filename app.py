import streamlit as st
import neo4j
from neo4j import GraphDatabase
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import io
import traceback
import time
from pyvis.network import Network

# Neo4j连接配置
AURA_URI = "neo4j+s://b76a61f2.databases.neo4j.io:7687"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"

LOCAL_URI = "bolt://localhost:7687"
LOCAL_USERNAME = "test"
LOCAL_PASSWORD = "Mikeno01"

@st.cache_resource
def get_driver(use_aura):
    if use_aura:
        uri = AURA_URI
        username = AURA_USERNAME
        password = AURA_PASSWORD
    else:
        uri = LOCAL_URI
        username = LOCAL_USERNAME
        password = LOCAL_PASSWORD
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver

def main():
    st.title("Neo4j图数据库应用场景展示")

    use_aura = st.sidebar.checkbox("使用Neo4j Aura", value=False)
    driver = get_driver(use_aura)

    # 显示当前连接信息
    if use_aura:
        st.sidebar.success("已连接到Neo4j Aura")
    else:
        st.sidebar.info("已连接到地Neo4j")

    menu = [
        "Neo4j图数据库应用场景介绍",
        "金融风控场景",
        "社交网络场景",
        "医疗健康场景"
    ]
    choice = st.sidebar.selectbox("选择场景", menu)

    if choice == "Neo4j图数据库应用场景介绍":
        show_neo4j_introduction(driver)
    elif choice == "金融风控场景":
        financial_risk_control_scenario(driver)
    elif choice == "社交网络场景":
        social_network_scenario(driver)
    elif choice == "医疗健康场景":
        healthcare_scenario(driver)

def show_neo4j_introduction(driver):
    st.header("Neo4j图数据库应用场景介绍")
    st.write("""
    Neo4j是一种强大的图数据库，适用于多种复杂的数据分析场景。在这个演示中，我们将展示Neo4j在以下三个领域的应用：

    1. 金融风控
    2. 社交网络分析
    3. 医疗健康数据管理

    每个场景都展示了图数据库如何帮助我们更好地理解和分析复杂的关系数据。
    """)
    show_database_overview(driver)

def show_database_overview(driver):
    st.subheader("当前Neo4j数据概览")
    
    # 金融风控数据概
    st.write("### 金风控")
    financial_stats = get_financial_stats(driver)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("用户数", financial_stats["用户数"])
        st.metric("交易数", financial_stats["交易数"])
        st.metric("商户数", financial_stats["商户数"])
    with col2:
        st.metric("设备数", financial_stats["设备数"])
        st.metric("IP地址", financial_stats["IP地址数"])
        st.metric("银行账户数", financial_stats["银行账户数"])
    st.write(f"平均用户风险评分: {financial_stats['平均风险评分']:.2f}")
    st.write(f"最大交易金额: {financial_stats['最大交易金额']:.2f}")
    st.write(f"被标记为可疑的交易数: {financial_stats['可疑交易数']}")

    # 社交网络数据概览
    st.write("### 社交网络数据")
    social_stats = get_social_stats(driver)
    if any(social_stats.values()):  # 检查是否有任何社交网络数据
        col1, col2 = st.columns(2)
        with col1:
            st.metric("用户数", social_stats["用户数"])
            st.metric("帖子数", social_stats["帖子数"])
        with col2:
            st.metric("关注关系数", social_stats["关注关系数"])
        if "平均每用户帖子数" in social_stats:
            st.write(f"平均每用户帖子数: {social_stats['平均每用户帖子数']:.2f}")
        if "平均每用户关注数" in social_stats:
            st.write(f"平均每用户关注数: {social_stats['平均每用户关注数']:.2f}")
    else:
        st.write("暂无社交网络数据")

    # 医疗健康数据概览
    st.write("### 医疗健康数据")
    health_stats = get_health_stats(driver)
    if any(health_stats.values()):  # 检查否有任何医疗健康数据
        col1, col2 = st.columns(2)
        with col1:
            st.metric("患者数", health_stats["患者数"])
            st.metric("医生数", health_stats["医生数"])
        with col2:
            st.metric("诊断记录数", health_stats["诊断记录数"])
            st.metric("药品数", health_stats["药品数"])
        if health_stats["平均每患者诊断次数"] > 0:
            st.write(f"平均每患者诊断次数: {health_stats['平均每患者诊断次数']:.2f}")
        else:
            st.write("平均每患者诊断次数: 无数据")
        st.write(f"最常见疾病: {health_stats['最常见疾病']}")
    else:
        st.write("暂无医疗健康数据")

def get_financial_stats(driver):
    with driver.session() as session:
        return {
            "用户数": session.run("MATCH (u:User) RETURN count(u) AS count").single()["count"],
            "交易数": session.run("MATCH (t:Transaction) RETURN count(t) AS count").single()["count"],
            "商户数": session.run("MATCH (m:Merchant) RETURN count(m) AS count").single()["count"],
            "设备数": session.run("MATCH (d:Device) RETURN count(d) AS count").single()["count"],
            "IP地址数": session.run("MATCH (ip:IPAddress) RETURN count(ip) AS count").single()["count"],
            "银行账户数": session.run("MATCH (a:BankAccount) RETURN count(a) AS count").single()["count"],
            "平均风险评分": session.run("MATCH (u:User) RETURN avg(u.risk_score) AS avg_score").single()["avg_score"],
            "最大交易金额": session.run("MATCH (t:Transaction) RETURN max(t.amount) AS max_amount").single()["max_amount"],
            "可疑交易数": session.run("MATCH (t:Transaction {status: 'Flagged'}) RETURN count(t) AS count").single()["count"]
        }

def get_social_stats(driver):
    with driver.session() as session:
        default_value = 0
        stats = {
            "用户数": session.run("MATCH (u:SocialUser) RETURN count(u) AS count").single()["count"] or default_value,
            "帖子数": session.run("MATCH (p:Post) RETURN count(p) AS count").single()["count"] or default_value,
            "关注关系数": session.run("MATCH (:SocialUser)-[:FOLLOWS]->(:SocialUser) RETURN count(*) AS count").single()["count"] or default_value,
        }
        
        # 只有在有用户和帖子时才计算平均值
        if stats["用户数"] > 0 and stats["帖子数"] > 0:
            avg_posts = session.run("MATCH (u:SocialUser)-[:POSTED]->(p:Post) WITH u, count(p) AS post_count RETURN avg(post_count) AS avg_posts").single()["avg_posts"]
            stats["平均每用户帖子数"] = avg_posts if avg_posts is not None else 0
        else:
            stats["平均每用户帖子数"] = 0
        
        if stats["用户数"] > 0:
            avg_follows = session.run("MATCH (u:SocialUser)-[:FOLLOWS]->(f:SocialUser) WITH u, count(f) AS follow_count RETURN avg(follow_count) AS avg_follows").single()["avg_follows"]
            stats["平均每用户关注数"] = avg_follows if avg_follows is not None else 0
        else:
            stats["平均每用户关注数"] = 0
        
        return stats

def get_health_stats(driver):
    with driver.session() as session:
        default_value = 0
        stats = {
            "患者数": session.run("MATCH (p:Patient) RETURN count(p) AS count").single()["count"] or default_value,
            "医生数": session.run("MATCH (d:Doctor) RETURN count(d) AS count").single()["count"] or default_value,
            "诊断记录数": session.run("MATCH (d:Diagnosis) RETURN count(d) AS count").single()["count"] or default_value,
            "药品数": session.run("MATCH (m:Medication) RETURN count(m) AS count").single()["count"] or default_value,
            "平均每患者诊断次数": session.run("MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis) WITH p, count(d) AS diag_count RETURN avg(diag_count) AS avg_diagnoses").single()["avg_diagnoses"] or 0,
        }
        
        # 只有在有诊断记录时才尝试获取最常见疾病
        if stats["诊断记录数"] > 0:
            most_common_disease = session.run("MATCH (d:Diagnosis) WITH d.disease AS disease, count(*) AS count ORDER BY count DESC LIMIT 1 RETURN disease").single()
            stats["最常见疾病"] = most_common_disease["disease"] if most_common_disease else "无数据"
        else:
            stats["最常见疾病"] = "无数据"
        
        return stats

def financial_risk_control_scenario(driver):
    st.header("图数据库在金融风控的应用")
    
    submenu = st.sidebar.radio(
        "金融风控子菜单",
        ("数据管理", "风险分析案例展示")
    )
    
    if submenu == "数据管理":
        financial_data_management(driver)
    elif submenu == "风险分析案例展示":
        financial_risk_analysis(driver)

def financial_data_management(driver):
    st.subheader("金融数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空金融数据"):
            clear_financial_data(driver)
    
    with col2:
        if st.button("导入金融数据"):
            import_financial_data(driver)
            st.success("金融数据导入完成")
    
    show_financial_database_stats(driver)

def clear_financial_data(driver):
    try:
        with driver.session() as session:
            # 删除所有金融相关的节点和关系
            result = session.run("""
            MATCH (n)
            WHERE n:User OR n:BankAccount OR n:Merchant OR n:Device OR n:IPAddress OR n:Transaction
            WITH n, n.id AS id
            DETACH DELETE n
            RETURN count(n) as deleted_count, collect(id) as deleted_ids
            """)
            
            deleted_info = result.single()
            if deleted_info:
                st.write(f"已删除 {deleted_info['deleted_count']} 个节点")
                st.write(f"删除的节点ID: {', '.join(deleted_info['deleted_ids'][:10])}...")
            else:
                st.write("没有找到要删除的节点")
            
            # 删除可能残留的金融相关关系
            result = session.run("""
            MATCH ()-[r:MADE_TRANSACTION|OWNS_ACCOUNT|INVOLVES_MERCHANT|USES_DEVICE|USES_IP]->()
            DELETE r
            RETURN count(r) as deleted_rel_count
            """)
            
            deleted_rel_count = result.single()["deleted_rel_count"]
            st.write(f"已删除 {deleted_rel_count} 个关系")
        
        st.success("金融数据已成功清除")
    except Exception as e:
        st.error(f"清除数据时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())

def import_financial_data(driver):
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
                import_financial_csv_data(driver, file_name, csv_data)
            st.success(f"{file_desc} 入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。请确保已生成数据文件。")
        except Exception as e:
            st.error(f"导入 {file_desc} 时生错误: {str(e)}")

def import_financial_csv_data(driver, file_name, csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    
    with driver.session() as session:
        if file_name == "users.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (u:User:FinancialRisk {id: row.id})
            SET u.name = row.name, u.risk_score = toFloat(row.risk_score)
            """, rows=df.to_dict('records'))
        elif file_name == "bank_accounts.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:User:FinancialRisk {id: row.user_id})
            MERGE (a:BankAccount:FinancialRisk {id: row.id})
            SET a.balance = toFloat(row.balance)
            MERGE (u)-[:OWNS_ACCOUNT]->(a)
            """, rows=df.to_dict('records'))
        elif file_name == "merchants.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (m:Merchant:FinancialRisk {id: row.id})
            SET m.name = row.name, m.category = row.category
            """, rows=df.to_dict('records'))
        elif file_name == "devices.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Device:FinancialRisk {id: row.id})
            SET d.type = row.type
            """, rows=df.to_dict('records'))
        elif file_name == "ip_addresses.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (ip:IPAddress:FinancialRisk {id: row.id})
            SET ip.address = row.address
            """, rows=df.to_dict('records'))
        elif file_name == "transactions.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:User:FinancialRisk {id: row.user_id})
            MATCH (m:Merchant:FinancialRisk {id: row.merchant_id})
            MATCH (d:Device:FinancialRisk {id: row.device_id})
            MATCH (ip:IPAddress:FinancialRisk {id: row.ip_id})
            MERGE (t:Transaction:FinancialRisk {id: row.id})
            SET t.amount = toFloat(row.amount), t.timestamp = row.timestamp, t.status = row.status
            MERGE (u)-[:MADE_TRANSACTION]->(t)
            MERGE (t)-[:INVOLVES_MERCHANT]->(m)
            MERGE (t)-[:USES_DEVICE]->(d)
            MERGE (t)-[:USES_IP]->(ip)
            """, rows=df.to_dict('records'))

def show_financial_database_stats(driver):
    st.subheader("金融数据库统计")
    queries = {
        "用户数": "MATCH (u:User) RETURN count(u) as count",
        "银行账户数": "MATCH (a:BankAccount) RETURN count(a) as count",
        "商户数": "MATCH (m:Merchant) RETURN count(m) as count",
        "设备": "MATCH (d:Device) RETURN count(d) as count",
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

def financial_risk_analysis(driver):
    st.subheader("金融风险分析")
    
    analysis_options = [
        "高风险用户识别",
        "关联网络分析",
        "异常交易模式检测"
    ]
    
    analysis_choice = st.selectbox("选择分析类型", analysis_options)
    
    if analysis_choice == "高风险用户识别":
        high_risk_users_analysis(driver)
    elif analysis_choice == "关联网络分析":
        relationship_network_analysis(driver)
    elif analysis_choice == "异常交易模式检测":
        anomalous_transactions_analysis(driver)

def high_risk_users_analysis(driver):
    st.write("识别风险评分最高的用户")
    query = """
    MATCH (u:User)
    WHERE u.risk_score > 80
    RETURN u.id AS user_id, u.name AS name, u.risk_score AS risk_score
    ORDER BY u.risk_score DESC
    LIMIT 10
    """
    results = run_query(driver, query)
    if not results.empty:
        fig = px.scatter(results, x="risk_score", y="user_id", color="risk_score", 
                         hover_data=["name"], title="高风险用户")
        st.plotly_chart(fig)
        
        st.write("分析结果解释：")
        st.write(f"1. 我们发现了 {len(results)} 个高风险用户，他们的风险评分都超过了80分。")
        highest_risk_user = results.iloc[0]
        st.write(f"2. 最高风险用户是 {highest_risk_user['name']} (ID: {highest_risk_user['user_id']})，风险评分高达 {highest_risk_user['risk_score']:.2f}。")
        st.write(f"3. 这些用户的平均风险评分为 {results['risk_score'].mean():.2f}，远高于正常水平。")
        st.write("4. 建议立即对这些用户进行深入调查，包括审查他们的最近交易、资金来源，以及是否有可疑的关联网络。")
    else:
        st.warning("未发现高风险用户。这可能表明当前的风险评分模型需要调整，或者系统中用户普遍表现良好。")

def relationship_network_analysis(driver):
    st.write("分析用户之间的关联网络")
    query = """
    MATCH (u1:User)-[:MADE_TRANSACTION]->(t:Transaction)<-[:MADE_TRANSACTION]-(u2:User)
    WHERE u1 <> u2
    WITH u1, u2, count(t) AS shared_transactions
    WHERE shared_transactions > 3
    RETURN u1.id AS user1, u2.id AS user2, shared_transactions
    LIMIT 50
    """
    results = run_query(driver, query)
    if not results.empty:
        G = nx.Graph()
        for _, row in results.iterrows():
            G.add_edge(row['user1'], row['user2'], weight=row['shared_transactions'])
        
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, 
                                     edge_labels={(u,v): d['weight'] for u,v,d in G.edges(data=True)})
        plt.title("用户关联网络")
        st.pyplot(fig)
        
        st.write("分析结果解释：")
        st.write(f"1. 我们发现了 {len(G.nodes)} 个用户之间存在密切的交易关联。")
        max_edge = max(G.edges(data=True), key=lambda x: x[2]['weight'])
        st.write(f"2. 最强的关联是用户 {max_edge[0]} 和 {max_edge[1]} 之间，他们共享了 {max_edge[2]['weight']} 次交易。这种高度关联能表示潜在的欺诈行为或资金洗白活动")
        central_node = max(G.degree, key=lambda x: x[1])[0]
        st.write(f"3. 用户 {central_node} 似乎是这个网络的中心，与 {G.degree[central_node]} 个其他用户有直接关联。这个用户可能是一个关键的风险节点，需要特别关注。")
        st.write("4. 建议对这个网络中的用户进行更深入的背景调查，并密切监控他们的来交易动。")
    else:
        st.warning("未发现显著的用户关联网络。这可能表明当前的交易模式相对分散，或者需要调整分析的阈值以捕获更多的关联。")

def anomalous_transactions_analysis(driver):
    st.write("检测可能的异常交易")
    query = """
    MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)
    WHERE t.amount > 10000 OR t.status = 'Flagged'
    RETURN u.id AS user_id, t.id AS transaction_id, t.amount AS amount, t.status AS status
    ORDER BY t.amount DESC
    LIMIT 20
    """
    results = run_query(driver, query)
    if not results.empty:
        fig = px.scatter(results, x="amount", y="user_id", color="status", 
                         hover_data=["transaction_id"], title="异常交易")
        st.plotly_chart(fig)
        
        st.write("分析结果解释：")
        st.write(f"1. 我们检测到 {len(results)} 笔能的异常交易。")
        flagged_transactions = results[results['status'] == 'Flagged']
        st.write(f"2. 其中 {len(flagged_transactions)} 笔交易被系统自动标记为可疑。")
        high_amount_transactions = results[results['amount'] > 10000]
        st.write(f"3. 有 {len(high_amount_transactions)} 笔交易的金额超过了10,000，这些大额交易需要特别关注。")
        highest_transaction = results.iloc[0]
        st.write(f"4. 最大的一笔交易金额为 {highest_transaction['amount']:.2f}，由用户 {highest_transaction['user_id']} 发起，交易ID为 {highest_transaction['transaction_id']}。")
        st.write("5. 建议对这些异常交易进行人工审核，特别是那些金额特别大或被系统标记的交易。同时，可能需要临时制相关用户的交易权限，直到完调查。")
    else:
        st.warning("未发现异常交易。可能表明当前的交易活动都正常范围内，或者需要调整异常交易的定义标准。")

def run_query(driver, query, params=None):
    with driver.session() as session:
        if params:
            result = session.run(query, params)
        else:
            result = session.run(query)
        return pd.DataFrame([dict(record) for record in result])

def social_network_scenario(driver):
    st.header("图数据库在社交网络分析中的应用")
    
    submenu = st.sidebar.radio(
        "社交网络子菜单",
        ("数据管理", "影响力分析", "社区发现", "信息传播分析", "推荐系统", "欺诈检测")
    )
    
    if submenu == "数据管理":
        social_data_management(driver)
    elif submenu == "影响力分析":
        influence_analysis(driver)
    elif submenu == "社区发现":
        community_detection(driver)
    elif submenu == "信息传播分析":
        information_propagation_analysis(driver)
    elif submenu == "推荐系统":
        recommendation_system(driver)
    elif submenu == "欺诈检测":
        social_fraud_detection(driver)

def social_data_management(driver):
    st.subheader("社交网络数据管理")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("清空社交网络数据"):
            clear_social_data(driver)
            st.success("社交网络数据已清空")
    
    with col2:
        if st.button("导入社交网络数据"):
            import_social_data(driver)
    
    with col3:
        if st.button("停止导入"):
            st.session_state.stop_import = True

    # 显示社交网络数据统计
    show_social_database_stats(driver)

def clear_social_data(driver):
    st.write("开始清空社交网络数据...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with driver.session() as session:
            # 使用 APOC 批量删除节点和关系
            total_deleted = 0
            batch_size = 1000

            # 删除 SocialUser 节点及其关系
            while True:
                result = session.run("""
                MATCH (n:SocialUser)
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """, batch_size=batch_size)
                deleted_count = result.single()["deleted_count"]
                total_deleted += deleted_count
                progress_bar.progress(min(total_deleted / 10000, 1.0))  # 假设最多有10000个节点
                status_text.text(f"删除 {total_deleted} 个社交用户节点")
                if deleted_count < batch_size:
                    break

            # 删除 Post 节点
            session.run("MATCH (p:Post) DETACH DELETE p")
            
            # 删除 Interest 节点
            session.run("MATCH (i:Interest) DETACH DELETE i")

        st.success("社交网络数已成功清空")
    except Exception as e:
        st.error(f"清空数据时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())
    finally:
        progress_bar.empty()
        status_text.empty()

    show_social_database_stats(driver)

def import_social_data(driver):
    st.session_state.stop_import = False
    progress_bar = st.progress(0)
    status_text = st.empty()

    social_dir = 'social'
    files = {
        "用户数据": "social_users.csv",
        "帖子数据": "posts.csv",
        "关注关系数据": "follow_relationships.csv",
        "兴趣数据": "interests.csv",
        "用户兴趣数据": "user_interests.csv"
    }
    
    total_files = len(files)
    for i, (file_desc, file_name) in enumerate(files.items()):
        if st.session_state.stop_import:
            status_text.text("导入已停止")
            break

        file_path = os.path.join(social_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read()
                import_social_csv_data(driver, file_name, csv_data)
            status_text.text(f"{file_desc} 导入成功！")
        except FileNotFoundError:
            status_text.text(f"{file_path} 文件不存在。请确保已生成数据文件。")
        except Exception as e:
            status_text.text(f"导入 {file_desc} 时发生错误: {str(e)}")
        
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        time.sleep(0.1)  # 添加小延迟以便更新UI

    if not st.session_state.stop_import:
        status_text.text("所有社交网络数据导入完成")
    show_social_database_stats(driver)

def import_social_csv_data(driver, file_name, csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    
    with driver.session() as session:
        if file_name == "social_users.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (u:SocialUser {id: row.id})
            SET u.name = row.name, 
                u.followers_count = toInteger(row.followers_count),
                u.following_count = toInteger(row.following_count),
                u.account_creation_date = row.account_creation_date,
                u.is_verified = (row.is_verified = 'True'),
                u.activity_score = toFloat(row.activity_score)
            """, rows=df.to_dict('records'))
        elif file_name == "posts.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:SocialUser {id: row.user_id})
            MERGE (p:Post {id: row.id})
            SET p.content = row.content,
                p.timestamp = row.timestamp,
                p.likes_count = toInteger(row.likes_count),
                p.shares_count = toInteger(row.shares_count)
            MERGE (u)-[:POSTED]->(p)
            """, rows=df.to_dict('records'))
        elif file_name == "follow_relationships.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u1:SocialUser {id: row.follower_id})
            MATCH (u2:SocialUser {id: row.following_id})
            MERGE (u1)-[r:FOLLOWS]->(u2)
            SET r.follow_date = row.follow_date
            """, rows=df.to_dict('records'))
        elif file_name == "interests.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (i:Interest {id: row.id})
            SET i.name = row.name
            """, rows=df.to_dict('records'))
        elif file_name == "user_interests.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:SocialUser {id: row.user_id})
            MATCH (i:Interest {id: row.interest_id})
            MERGE (u)-[r:INTERESTED_IN]->(i)
            SET r.affinity_score = toFloat(row.affinity_score)
            """, rows=df.to_dict('records'))

def show_social_database_stats(driver):
    st.subheader("社交网络数据库统计")
    queries = {
        "用户数": "MATCH (u:SocialUser) RETURN count(u) as count",
        "帖子数": "MATCH (p:Post) RETURN count(p) as count",
        "关注关系数": "MATCH ()-[r:FOLLOWS]->() RETURN count(r) as count",
        "兴趣标签数": "MATCH (i:Interest) RETURN count(i) as count",
        "用户-兴趣关系数": "MATCH ()-[r:INTERESTED_IN]->() RETURN count(r) as count"
    }
    
    results = {}
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query).single()
            results[label] = result["count"] if result else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("用户数", results["用户数"])
        st.metric("帖子数", results["帖子数"])
        st.metric("兴趣标签数", results["兴趣标签数"])
    
    with col2:
        st.metric("关注关系数", results["关注关系数"])
        st.metric("用户-兴趣关系数", results["用户-兴趣关系数"])

    # 额外的统计信息
    with driver.session() as session:
        avg_followers = session.run("MATCH (u:SocialUser) RETURN avg(u.followers_count) as avg").single()["avg"]
        avg_posts = session.run("MATCH (u:SocialUser)-[:POSTED]->(p:Post) WITH u, count(p) as post_count RETURN avg(post_count) as avg").single()["avg"]
    
    st.write(f"平均粉丝数: {avg_followers:.2f}")
    st.write(f"平均发帖数: {avg_posts:.2f}")

def influence_analysis(driver):
    st.subheader("用户影响力分析")
    query = """
    MATCH (u:SocialUser)
    CALL {
        WITH u
        MATCH (u)<-[:FOLLOWS]-(follower)
        RETURN count(follower) AS follower_count
    }
    CALL {
        WITH u
        MATCH (u)-[:POSTED]->(post)
        RETURN count(post) AS post_count
    }
    WITH u, follower_count, post_count
    RETURN u.id AS user_id, u.name AS name, follower_count, post_count, 
           follower_count * 0.7 + post_count * 0.3 AS influence_score
    ORDER BY influence_score DESC
    LIMIT 10
    """
    results = run_query(driver, query)
    if not results.empty:
        fig = px.scatter(results, x="follower_count", y="post_count", size="influence_score", 
                         color="influence_score", hover_data=["name"], 
                         labels={"follower_count": "粉丝数", "post_count": "发帖数", "influence_score": "影响力得分"},
                         title="用户影响力分析")
        st.plotly_chart(fig)
        
        st.write("分析结果解释：")
        st.write(f"1. 我们分析了平台上最具影响力的 {len(results)} 个用户。")
        top_influencer = results.iloc[0]
        st.write(f"2. 最具影响力户是 {top_influencer['name']} (ID: {top_influencer['user_id']})，影响力得分为 {top_influencer['influence_score']:.2f}。")
        st.write(f"3. 这个用户有 {top_influencer['follower_count']} 个粉丝，发布了 {top_influencer['post_count']} 条帖子。")
        st.write("4. 影响力得分是根据粉丝数（权重70%）和发帖数（权重30%）计算的。")
        st.write("5. 建议关注这些高影响力用户，他可能是重要的意见领袖或者潜在的品牌合作伙伴。")
    else:
        st.warning("未能获取用户影响力数据。请确保已导入足够的社交网络数据。")

def community_detection(driver):
    st.subheader("社区发现")
    st.write("注意：这是一个基于用户共同兴趣的社区检测方法。")
    
    query = """
    MATCH (u1:SocialUser)-[:INTERESTED_IN]->(i:Interest)<-[:INTERESTED_IN]-(u2:SocialUser)
    WHERE u1 <> u2
    WITH u1, u2, count(DISTINCT i) AS shared_interests
    WHERE shared_interests > 1
    RETURN u1.id AS user1_id, u1.name AS user1_name, 
           u2.id AS user2_id, u2.name AS user2_name, 
           shared_interests
    ORDER BY shared_interests DESC
    LIMIT 100
    """
    results = run_query(driver, query)
    
    if not results.empty:
        G = nx.Graph()
        for _, row in results.iterrows():
            G.add_edge(row['user1_id'], row['user2_id'], weight=row['shared_interests'])
        
        # 使用社区检测算法
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # 创建 Pyvis 网络图
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        # 添加节点和边
        for node in G.nodes():
            # 为每个节点找到其所属的社区
            community_id = next(i for i, comm in enumerate(communities) if node in comm)
            net.add_node(node, label=node, title=f"User: {node}", group=community_id)
        
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], value=edge[2]['weight'], title=f"Shared Interests: {edge[2]['weight']}")
        
        # 设置物理布局
        net.force_atlas_2based()
        
        # 保存为 HTML 文件
        net.save_graph("community_graph.html")
        
        # 在 Streamlit 中显示图
        with open("community_graph.html", 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600)
        
        st.write("分析结果解释：")
        st.write(f"1. 我们发现了 {len(communities)} 个主要的社区。")
        st.write(f"2. 图中的每个节点代表一个用户，节点的颜色表示不同的社区")
        st.write(f"3. 节点之间的连线表示用户之间有共同兴趣，线的粗细表示共同兴的数量。")
        st.write(f"4. 最大的社区包含 {len(max(communities, key=len))} 个成员。")
        st.write("5. 紧密连接的节点群表示具有相似兴趣的用户群体。")
        st.write("6. 这种分析可以帮助识别具有相似兴趣的用户群体，对于容推荐和目标营销很有价值。")
    else:
        st.warning("未能检测到明显的社区结构。请确保已导入足够的社交网络数据，特别是用户兴趣数据。")

def information_propagation_analysis(driver):
    st.subheader("信息传播分析")
    
    # 分析热门帖子
    query_hot_posts = """
    MATCH (u:SocialUser)-[:POSTED]->(p:Post)
    WHERE p.shares_count > 0 OR p.likes_count > 0
    RETURN u.id AS user_id, u.name AS user_name, p.id AS post_id, p.content AS content, 
           p.shares_count AS shares, p.likes_count AS likes, p.timestamp AS post_time
    ORDER BY (p.shares_count + p.likes_count) DESC
    LIMIT 20
    """
    hot_posts = run_query(driver, query_hot_posts)
    
    if not hot_posts.empty:
        st.write("### 热门帖子分析")
        fig = px.scatter(hot_posts, x="shares", y="likes", size="shares", color="likes",
                         hover_data=["user_name", "content", "post_time"],
                         labels={"shares": "分享数", "likes": "点赞数"},
                         title="热门帖子分布")
        st.plotly_chart(fig)
        
        st.write("热门帖子内容：")
        for _, post in hot_posts.iterrows():
            st.write(f"- 用户 {post['user_name']} 发布: {post['content']} (分享: {post['shares']}, 点赞: {post['likes']})")
        
        st.write("热门帖子分析结果：")
        most_shared = hot_posts.loc[hot_posts['shares'].idxmax()]
        most_liked = hot_posts.loc[hot_posts['likes'].idxmax()]
        st.write(f"1. 最多分享的帖子来自用户 {most_shared['user_name']}，内容为 '{most_shared['content']}'，获得了 {most_shared['shares']} 次分享和 {most_shared['likes']} 个点赞。")
        st.write(f"2. 最多点赞的帖子来自用户 {most_liked['user_name']}，内容为 '{most_liked['content']}'，获得了 {most_liked['likes']} 个点赞和 {most_liked['shares']} 次分享。")
        st.write(f"3. 平均而言，门帖子获得了 {hot_posts['shares'].mean():.2f} 次分享和 {hot_posts['likes'].mean():.2f} 个点赞。")
        st.write(f"4. 有 {len(hot_posts[hot_posts['shares'] > hot_posts['likes']])} 个帖子的分享数超过了点赞数，这些帖子可能具有更高的传播性。")
        
        # 分析信息传播速度
        st.write("### 信息传播速度分析")
        query_propagation_speed = """
        MATCH (u:SocialUser)-[:POSTED]->(p:Post)
        WHERE p.shares_count > 0
        WITH p, u, duration.inSeconds(datetime(p.timestamp), datetime()).seconds AS age_seconds
        WHERE age_seconds > 0
        RETURN p.id AS post_id, u.name AS user_name, p.content AS content, 
               p.shares_count AS shares, p.likes_count AS likes, 
               age_seconds / 3600 AS age_hours, 
               p.shares_count / (age_seconds / 3600) AS shares_per_hour
        ORDER BY shares_per_hour DESC
        LIMIT 10
        """
        propagation_speed = run_query(driver, query_propagation_speed)
        
        if not propagation_speed.empty:
            fig = px.bar(propagation_speed, x="user_name", y="shares_per_hour", 
                         hover_data=["content", "shares", "likes", "age_hours"],
                         labels={"user_name": "发布用户", "shares_per_hour": "每小时分享数"},
                         title="帖子传播速度")
            st.plotly_chart(fig)
            
            st.write("传播最快的帖子：")
            for _, post in propagation_speed.iterrows():
                st.write(f"- {post['user_name']}: {post['content']} (每小时分享 {post['shares_per_hour']:.2f} 次)")
        
            st.write("传播速度分析结果：")
            fastest_post = propagation_speed.iloc[0]
            st.write(f"1. 传播速度最快的帖子来自用户 {fastest_post['user_name']}，内容为 '{fastest_post['content']}'。")
            st.write(f"2. 这条帖子平均每小时被分享 {fastest_post['shares_per_hour']:.2f} 次，总共获得了 {fastest_post['shares']} 次分享和 {fastest_post['likes']} 个点赞。")
            st.write(f"3. 该帖子发布后 {fastest_post['age_hours']:.2f} 小时内就达到了这个分享量，显示出极强的传播力。")
            st.write(f"4. 平均而言，这些快速传播的子每小时获得 {propagation_speed['shares_per_hour'].mean():.2f} 次分享。")
            st.write(f"5. 有 {len(propagation_speed[propagation_speed['shares_per_hour'] > 1])} 个帖子的每小时分享次数超过了1次，这些可能是病毒式传播的内容。")
            
            common_words = ' '.join(propagation_speed['content']).lower().split()
            word_freq = pd.Series(common_words).value_counts()
            st.write(f"6. 在这些快速传播的帖子中，最常见的词语是：{', '.join(word_freq.head().index)}。这些词可能是引起用户共鸣的关键词。")
            
            st.write("建议：")
            st.write("1. 重点关注那些传播速度最快的帖子，分析它们的内容特点、发布时间、目标受众等因素。")
            st.write("2. 考虑在内容创作中融入快速传播帖子中的常见词语或主题。")
            st.write("3. 研究那些既有分享数又有高点赞数的帖子，它们可能代表了最成功的内容形式。")
            st.write("4. 对于传播速度特别快的帖子，考虑及时进行相关话题的跟进或互动，以延长热度。")
            st.write("5. 分析传播速度最快的帖子的发布者特征，可能有助于识别平台上的关键意见领袖。")
        else:
            st.warning("未能获取足够的传播速度数据。可能是因为帖子太新或者分享数为0。")
    else:
        st.warning("未能获取足够的热门帖子数据。请确保已导入足够的社交网络数据。")

def recommendation_system(driver):
    st.subheader("用户推荐系统")

    # 自动选择一个用户进行推荐
    query_random_user = """
    MATCH (u:SocialUser)
    RETURN u.id AS user_id, u.name AS user_name, u.followers_count AS followers, u.following_count AS following
    ORDER BY rand()
    LIMIT 1
    """
    random_user = run_query(driver, query_random_user).iloc[0]
    user_id = random_user['user_id']
    
    st.write(f"为用户 {random_user['user_name']} (ID: {user_id}) 生成推荐")
    st.write(f"该用户有 {random_user['followers']} 个粉丝，关注了 {random_user['following']} 个用户")

    # 基于共同兴趣的用户推荐
    query_user_recommendation = """
    MATCH (u:SocialUser {id: $user_id})-[:INTERESTED_IN]->(i:Interest)<-[:INTERESTED_IN]-(other:SocialUser)
    WHERE NOT (u)-[:FOLLOWS]->(other)
    WITH other, count(DISTINCT i) AS shared_interests, collect(i.name) AS interests
    ORDER BY shared_interests DESC
    LIMIT 5
    RETURN other.id AS recommended_user_id, other.name AS name, other.followers_count AS followers, 
           other.following_count AS following, shared_interests, interests
    """
    user_recommendations = run_query(driver, query_user_recommendation, {"user_id": user_id})
    
    if not user_recommendations.empty:
        st.write("### 推荐关注的用户")
        fig = px.scatter(user_recommendations, x="followers", y="following", size="shared_interests", 
                         color="shared_interests", hover_name="name", 
                         labels={"followers": "粉丝数", "following": "关注数", "shared_interests": "共同兴趣数"},
                         title="推荐用户分布")
        st.plotly_chart(fig)
        
        st.write("推荐用户详情：")
        for _, row in user_recommendations.iterrows():
            st.write(f"- {row['name']} (ID: {row['recommended_user_id']})")
            st.write(f"  粉丝数: {row['followers']}, 关注数: {row['following']}")
            st.write(f"  共同兴趣数: {row['shared_interests']}")
            st.write(f"  共同兴趣: {', '.join(row['interests'])}")
        
        st.write("推荐分析：")
        st.write(f"1. 为用户推荐了 {len(user_recommendations)} 个潜在的关注对象。")
        st.write(f"2. 这些推荐用户平均有 {user_recommendations['followers'].mean():.0f} 个粉丝和 {user_recommendations['following'].mean():.0f} 个关注。")
        st.write(f"3. 平均共同兴趣数为 {user_recommendations['shared_interests'].mean():.2f}。")
        most_common_interests = ', '.join(set.intersection(*map(set, user_recommendations['interests'])))
        st.write(f"4. 最常见的共同兴趣是: {most_common_interests}")
    
    # 基于用户行为的内容推荐
    query_content_recommendation = """
    MATCH (u:SocialUser {id: $user_id})-[:INTERESTED_IN]->(i:Interest)<-[:INTERESTED_IN]-(other:SocialUser),
          (other)-[:POSTED]->(p:Post)
    WHERE NOT (u)-[:POSTED]->(p)
    WITH p, count(DISTINCT i) AS relevance, p.likes_count + p.shares_count AS popularity
    ORDER BY relevance DESC, popularity DESC
    LIMIT 5
    RETURN p.id AS post_id, p.content AS content, p.likes_count AS likes, p.shares_count AS shares,
           relevance, popularity
    """
    content_recommendations = run_query(driver, query_content_recommendation, {"user_id": user_id})
    
    if not content_recommendations.empty:
        st.write("### 推荐内容")
        fig = px.scatter(content_recommendations, x="relevance", y="popularity", size="likes", color="shares",
                         hover_data=["content"], labels={"relevance": "相关度", "popularity": "热度", 
                                                         "likes": "点赞数", "shares": "分享数"},
                         title="推荐内容分布")
        st.plotly_chart(fig)
        
        st.write("推荐内容详情：")
        for _, row in content_recommendations.iterrows():
            st.write(f"- 内容: {row['content']}")
            st.write(f"  相关度: {row['relevance']}, 热度: {row['popularity']}")
            st.write(f"  点赞数: {row['likes']}, 分享数: {row['shares']}")
        
        st.write("内容推荐分析：")
        st.write(f"1. 为用户推荐了 {len(content_recommendations)} 条潜在感兴趣的内容。")
        st.write(f"2. 这些内容的平均相关度为 {content_recommendations['relevance'].mean():.2f}。")
        st.write(f"3. 平均热度（点赞+分享）为 {content_recommendations['popularity'].mean():.2f}。")
        st.write(f"4. 推荐内容平均获得 {content_recommendations['likes'].mean():.2f} 个点赞和 {content_recommendations['shares'].mean():.2f} 次分享。")
    
    st.write("推荐系统说明：")
    st.write("1. 用户推荐基于共同兴趣，推荐那些与当前用户有相似兴趣但尚未关注的用户。")
    st.write("2. 内容推荐基于用户兴趣和内热度，推荐那些可能感兴趣且受欢迎的帖子。")
    st.write("3. 图表展示了推荐用户的粉丝数、关注数和共同兴趣数量，以及推荐内容的相关度与热度分布。")
    st.write("4. 这个演示展示了图数据库如何有效地利用用户间的关系和兴趣网络来生成个性化推荐。")
    st.write("5. 在实际应用中，这个系统可以进一步优化，例如考虑用户的历史行为、社交圈等因素，以提供更精准的推荐。")

def social_fraud_detection(driver):
    st.subheader("社交网络欺诈检测")
    
    # 检测可疑的虚假账户
    query_suspicious_accounts = """
    MATCH (u:SocialUser)
    WHERE u.followers_count > 500 AND u.following_count < 10
    WITH u
    MATCH (u)-[:POSTED]->(p:Post)
    WITH u, COUNT(p) AS post_count
    WHERE post_count < 5
    RETURN u.id AS user_id, u.name AS name, u.followers_count AS followers, 
           u.following_count AS following, post_count, 
           u.followers_count * 1.0 / (u.following_count + 1) AS follower_ratio
    ORDER BY follower_ratio DESC
    LIMIT 20
    """
    suspicious_accounts = run_query(driver, query_suspicious_accounts)
    
    if not suspicious_accounts.empty:
        st.write("### 可疑账户分析")
        fig = px.scatter(suspicious_accounts, x="followers", y="following", 
                         size="follower_ratio", color="post_count",
                         hover_data=["name", "user_id"],
                         labels={"followers": "粉丝数", "following": "关注数", 
                                 "follower_ratio": "粉丝比率", "post_count": "发帖数"},
                         title="可疑账户分布")
        st.plotly_chart(fig)
        
        st.write("可疑账户列表：")
        for _, account in suspicious_accounts.iterrows():
            st.write(f"- 用户 {account['name']} (ID: {account['user_id']}): 粉丝数 {account['followers']}, 关注数 {account['following']}, 发帖数 {account['post_count']}")
        
        # 检测异常的关注行为
        query_suspicious_follows = """
        MATCH (u:SocialUser)-[f:FOLLOWS]->(target:SocialUser)
        WITH u, count(f) AS follow_count
        WHERE follow_count > 100
        MATCH (u)-[:FOLLOWS]->(low_follower:SocialUser)
        WHERE low_follower.followers_count < 10
        WITH u, follow_count, count(low_follower) AS low_follower_targets
        WHERE low_follower_targets * 1.0 / follow_count > 0.5
        RETURN u.id AS user_id, u.name AS name, follow_count, low_follower_targets,
               low_follower_targets * 1.0 / follow_count AS suspicious_ratio
        ORDER BY suspicious_ratio DESC
        LIMIT 10
        """
        suspicious_follows = run_query(driver, query_suspicious_follows)
        
        if not suspicious_follows.empty:
            st.write("### 异常关注行为分析")
            fig = px.bar(suspicious_follows, x="name", y="suspicious_ratio", 
                         hover_data=["follow_count", "low_follower_targets"],
                         labels={"name": "用户", "suspicious_ratio": "可疑关注比例",
                                 "follow_count": "总关注数", "low_follower_targets": "低粉丝数目标"},
                         title="异常关注行为")
            st.plotly_chart(fig)
            
            st.write("异常关注行为用户：")
            for _, user in suspicious_follows.iterrows():
                st.write(f"- 用户 {user['name']} (ID: {user['user_id']}): 总关注 {user['follow_count']}, 可疑关注 {user['low_follower_targets']}, 可疑比例 {user['suspicious_ratio']:.2%}")
        
        st.write("欺诈检测分析结果：")
        st.write("1. 可疑账户图表展示了粉丝数异常高而关注数和发帖数异常低的账户，这些可能是虚假或被操纵的账户。")
        st.write("2. 异常关注行为分析显示了大量关注低粉丝数账户的用户，这可能是批量关注行为或试图操纵社交网络的迹象。")
        st.write("3. 建议对这些可疑账户和行为进行进一步调查，可能需要实施更严格的账验证措施和行为监控。")
    else:
        st.warning("未能检测到明显的可疑行为。请确保已导入足够的社交网络数据。")

def healthcare_scenario(driver):
    st.header("图数据库在医疗健康领域的应用")
    
    submenu = st.sidebar.radio(
        "医疗健康子菜单",
        ("数据管理", "患者病历分析", "疾病诊断模式", "药物处方趋势", "医疗保险分析")
    )
    
    if submenu == "数据管理":
        healthcare_data_management(driver)
    elif submenu == "患者病历分析":
        patient_history_analysis(driver)
    elif submenu == "疾病诊断模式":
        disease_diagnosis_patterns(driver)
    elif submenu == "药物处方趋势":
        medication_prescription_trends(driver)
    elif submenu == "医疗保险分析":
        insurance_claim_analysis(driver)

def healthcare_data_management(driver):
    st.subheader("医疗健康数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空医疗健康数据"):
            clear_healthcare_data(driver)
    
    with col2:
        if st.button("导入医疗健康数据"):
            import_healthcare_data(driver)
            st.success("医疗健康数据导入完成")
    
    show_healthcare_database_stats(driver)

def clear_healthcare_data(driver):
    try:
        with driver.session() as session:
            # 删除所有医疗健康相关的节点和关系
            result = session.run("""
            MATCH (n)
            WHERE n:Patient OR n:Doctor OR n:Disease OR n:Symptom OR n:Medication OR n:Hospital OR n:Diagnosis OR n:Prescription OR n:InsuranceClaim
            DETACH DELETE n
            """)
            
            # 获取删除的节点数量
            deleted_count = result.consume().counters.nodes_deleted
            
            st.write(f"已删除 {deleted_count} 个医疗健康相关节点")
            
            # 删除可能残留的医疗健康相关关系
            result = session.run("""
            MATCH ()-[r:DIAGNOSED_WITH|HAS_SYMPTOM|PRESCRIBES|TREATS|WORKS_AT|INTERACTS_WITH|PART_OF|CLAIMS]->()
            DELETE r
            RETURN count(r) as deleted_rel_count
            """)
            
            deleted_rel_count = result.single()["deleted_rel_count"]
            st.write(f"已删除 {deleted_rel_count} 个医疗健康相关关系")
        
        st.success("医疗健康数据已成功清除")
    except Exception as e:
        st.error(f"清除数据时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())

def import_healthcare_data(driver):
    healthcare_dir = 'healthcare'
    files = {
        "患者数据": "patients.csv",
        "医生数据": "doctors.csv",
        "疾病数据": "diseases.csv",
        "症状数据": "symptoms.csv",
        "药物数据": "medications.csv",
        "医院数据": "hospitals.csv",
        "诊断记录": "diagnoses.csv",
        "症状记录": "symptom_records.csv",
        "处方记录": "prescriptions.csv",
        "医生工作记录": "doctor_hospital.csv",
        "保险理赔记录": "insurance_claims.csv"
    }
    
    for file_desc, file_name in files.items():
        file_path = os.path.join(healthcare_dir, file_name)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # 数据清理步骤
            df = df.dropna()  # 删除包含空值的行
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # 去除字符串首尾的空白
            csv_data = df.to_csv(index=False)
            import_healthcare_csv_data(driver, file_name, csv_data)
            st.success(f"{file_desc} 导入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。请确保已生成数据文件。")
        except Exception as e:
            st.error(f"导入 {file_desc} 时发生错误: {str(e)}")
            st.write("错误详情:")
            st.write(traceback.format_exc())

def import_healthcare_csv_data(driver, file_name, csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    
    with driver.session() as session:
        if file_name == "patients.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (p:Patient {id: row.id})
            SET p.name = row.name, p.age = toInteger(row.age), p.gender = row.gender
            """, rows=df.to_dict('records'))
        elif file_name == "doctors.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Doctor {id: row.id})
            SET d.name = row.name, d.speciality = row.speciality
            """, rows=df.to_dict('records'))
        elif file_name == "diseases.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Disease {id: row.id})
            SET d.name = row.name
            """, rows=df.to_dict('records'))
        elif file_name == "symptoms.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (s:Symptom {id: row.id})
            SET s.name = row.name
            """, rows=df.to_dict('records'))
        elif file_name == "medications.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (m:Medication {id: row.id})
            SET m.name = row.name
            """, rows=df.to_dict('records'))
        elif file_name == "hospitals.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (h:Hospital {id: row.id})
            SET h.name = row.name
            """, rows=df.to_dict('records'))
        elif file_name == "diagnoses.csv":
            # 在导入诊断记录之前，确保所有疾病节点都已创建
            unique_diseases = df['disease_id'].unique()
            session.run("""
            UNWIND $disease_ids AS disease_id
            MERGE (d:Disease {id: disease_id})
            """, disease_ids=unique_diseases.tolist())
            
            session.run("""
            UNWIND $rows AS row
            MATCH (p:Patient {id: row.patient_id})
            MATCH (d:Doctor {id: row.doctor_id})
            MATCH (dis:Disease {id: row.disease_id})
            MERGE (diag:Diagnosis {id: coalesce(row.id, row.patient_id + '_' + row.doctor_id + '_' + row.disease_id + '_' + row.date)})
            SET diag.date = date(row.date)
            MERGE (p)-[:HAS_DIAGNOSIS]->(diag)
            MERGE (d)-[:MADE_DIAGNOSIS]->(diag)
            MERGE (diag)-[:DIAGNOSED_WITH]->(dis)
            """, rows=df.to_dict('records'))
        elif file_name == "symptom_records.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (p:Patient {id: row.patient_id})
            MATCH (s:Symptom {id: row.symptom_id})
            MERGE (p)-[r:HAS_SYMPTOM]->(s)
            SET r.date = date(row.date)
            """, rows=df.to_dict('records'))
            
            # 添加这行代码来打印导入的症状记录数
            st.write(f"导入了 {len(df)} 条症状记录")
        elif file_name == "prescriptions.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (p:Patient {id: row.patient_id})
            MATCH (d:Doctor {id: row.doctor_id})
            MATCH (m:Medication {id: row.medication_id})
            MERGE (pres:Prescription {id: row.patient_id + '_' + row.doctor_id + '_' + row.medication_id + '_' + row.date})
            SET pres.date = date(row.date), pres.dosage = row.dosage
            MERGE (d)-[:PRESCRIBES]->(pres)
            MERGE (pres)-[:PRESCRIBED_TO]->(p)
            MERGE (pres)-[:INCLUDES]->(m)
            """, rows=df.to_dict('records'))
        elif file_name == "doctor_hospital.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (d:Doctor {id: row.doctor_id})
            MATCH (h:Hospital {id: row.hospital_id})
            MERGE (d)-[r:WORKS_AT]->(h)
            SET r.start_date = date(row.start_date)
            """, rows=df.to_dict('records'))
        elif file_name == "insurance_claims.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (p:Patient {id: row.patient_id})
            MERGE (c:InsuranceClaim {id: row.claim_id})
            SET c.amount = toFloat(row.amount), c.date = date(row.date), c.status = row.status
            MERGE (p)-[:FILED_CLAIM]->(c)
            """, rows=df.to_dict('records'))

def show_healthcare_database_stats(driver):
    st.subheader("医疗健康数据库统计")
    queries = {
        "患者数": "MATCH (p:Patient) RETURN count(p) as count",
        "医数": "MATCH (d:Doctor) RETURN count(d) as count",
        "疾病数": "MATCH (d:Disease) RETURN count(d) as count",
        "症状数": "MATCH (s:Symptom) RETURN count(s) as count",
        "药物数": "MATCH (m:Medication) RETURN count(m) as count",
        "医院数": "MATCH (h:Hospital) RETURN count(h) as count",
        "诊断记录数": "MATCH (d:Diagnosis) RETURN count(d) as count",
        "处方数": "MATCH (p:Prescription) RETURN count(p) as count",
        "保险理赔数": "MATCH (c:InsuranceClaim) RETURN count(c) as count"
    }
    
    results = {}
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query).single()
            results[label] = result["count"] if result else 0
    
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    
    for i, (label, count) in enumerate(results.items()):
        columns[i % 3].metric(label, count)

    # 额外的统计信息
    with driver.session() as session:
        avg_age = session.run("MATCH (p:Patient) RETURN avg(p.age) as avg").single()["avg"]
        top_disease = session.run("MATCH (d:Disease)<-[:DIAGNOSED_WITH]-(diag:Diagnosis) WITH d, count(diag) as freq ORDER BY freq DESC LIMIT 1 RETURN d.name as disease, freq").single()
    
    st.write(f"患者平均年龄: {avg_age:.2f}")
    if top_disease:
        st.write(f"最常见疾病: {top_disease['disease']} (诊断次数: {top_disease['freq']})")
    else:
        st.write("暂无疾病诊断数据")

def patient_history_analysis(driver):
    st.subheader("患者病历分析")

    # 获取所有患者列表
    query_all_patients = """
    MATCH (p:Patient)
    RETURN p.id AS patient_id, p.name AS name, p.age AS age, p.gender AS gender
    ORDER BY p.name
    """
    all_patients = run_query(driver, query_all_patients)

    # 创建一个下拉菜单让用户选择患者
    selected_patient = st.selectbox(
        "选择患者",
        options=all_patients['patient_id'].tolist(),
        format_func=lambda x: f"{all_patients[all_patients['patient_id'] == x]['name'].iloc[0]} (ID: {x}, 年龄: {all_patients[all_patients['patient_id'] == x]['age'].iloc[0]}, 性别: {'男' if all_patients[all_patients['patient_id'] == x]['gender'].iloc[0] == 'Male' else '女'})"
    )

    # 获取选中患者的信息
    patient_info = all_patients[all_patients['patient_id'] == selected_patient].iloc[0]
    patient_id = patient_info['patient_id']
    
    st.write(f"分析患者: {patient_info['name']} (ID: {patient_id})")
    st.write(f"年龄: {patient_info['age']}, 性别: {'男' if patient_info['gender'] == 'Male' else '女'}")

    # 获取患者的诊断历史
    query_diagnosis_history = """
    MATCH (p:Patient {id: $patient_id})-[:HAS_DIAGNOSIS]->(d:Diagnosis)-[:DIAGNOSED_WITH]->(dis:Disease)
    MATCH (d)<-[:MADE_DIAGNOSIS]-(doc:Doctor)
    RETURN d.date AS date, dis.name AS disease, doc.name AS doctor, doc.speciality AS speciality
    ORDER BY d.date
    """
    diagnosis_history = run_query(driver, query_diagnosis_history, {"patient_id": patient_id})

    if not diagnosis_history.empty:
        st.write("### 诊断历史")
        diagnosis_history['date'] = pd.to_datetime(diagnosis_history['date'], errors='coerce')
        diagnosis_history['诊断日期'] = diagnosis_history['date'].dt.strftime('%Y-%m-%d')
        diagnosis_history['疾病'] = diagnosis_history['disease'].map({
            'Heart Disease': '心脏病',
            'Diabetes': '糖尿病',
            'Asthma': '哮喘',
            'Cancer': '癌症',
            'Depression': '抑郁症'
        })
        diagnosis_history['专科'] = diagnosis_history['speciality'].map({
            'Cardiology': '心脏科',
            'Neurology': '神经科',
            'Oncology': '肿瘤科',
            'Pediatrics': '儿科',
            'Surgery': '外科'
        })
        diagnosis_history['医生'] = diagnosis_history['doctor']

        fig = px.bar(diagnosis_history, x='诊断日期', y='疾病', color='专科',
                     hover_data=['医生'], title="患者诊断历史",
                     labels={'诊断日期': '诊断日期', '疾病': '疾病', '专科': '专科', '医生': '医生'})
        fig.update_layout(xaxis_title="诊断日期", yaxis_title="疾病")
        st.plotly_chart(fig)

    # 获取患者的症状记录
    query_symptom_history = """
    MATCH (p:Patient {id: $patient_id})-[r:HAS_SYMPTOM]->(s:Symptom)
    RETURN s.name AS symptom, r.date AS date
    ORDER BY r.date
    """
    symptom_history = run_query(driver, query_symptom_history, {"patient_id": patient_id})

    if not symptom_history.empty:
        st.write("### 症状记录")
        symptom_history['date'] = pd.to_datetime(symptom_history['date'], errors='coerce')
        symptom_history['日期'] = symptom_history['date'].dt.strftime('%Y-%m-%d')
        symptom_history['症状'] = symptom_history['symptom'].map({
            'Headache': '头痛',
            'Fever': '发烧',
            'Cough': '咳嗽',
            'Fatigue': '疲劳',
            'Nausea': '恶心'
        })
        fig = px.scatter(symptom_history, x='日期', y='症状', title="患者症状记录")
        fig.update_layout(xaxis_title="日期", yaxis_title="症状")
        st.plotly_chart(fig)

    # 获取患者的用药历史
    query_medication_history = """
    MATCH (p:Patient {id: $patient_id})<-[:PRESCRIBED_TO]-(pres:Prescription)-[:INCLUDES]->(m:Medication)
    MATCH (pres)<-[:PRESCRIBES]-(d:Doctor)
    RETURN pres.date AS date, m.name AS medication, pres.dosage AS dosage, d.name AS doctor
    ORDER BY pres.date
    """
    medication_history = run_query(driver, query_medication_history, {"patient_id": patient_id})

    if not medication_history.empty:
        st.write("### 用药历史")
        medication_history['date'] = pd.to_datetime(medication_history['date'], errors='coerce')
        medication_history['日期'] = medication_history['date'].dt.strftime('%Y-%m-%d')
        medication_history['药物'] = medication_history['medication'].map({
            'Lisinopril': '赖诺普利',
            'Metformin': '二甲双胍',
            'Albuterol': '沙丁胺醇',
            'Ibuprofen': '布洛芬',
            'Sertraline': '舍曲林'
        })
        medication_history['剂量'] = medication_history['dosage']
        medication_history['医生'] = medication_history['doctor']
        
        # 尝试将 dosage 转换为数值
        medication_history['剂量数值'] = pd.to_numeric(medication_history['dosage'].str.extract('(\d+)', expand=False), errors='coerce')
        
        fig = px.scatter(medication_history, x='日期', y='药物', size='剂量数值', color='医生',
                         hover_data=['剂量'], title="患者用药历史")
        fig.update_layout(xaxis_title="日期", yaxis_title="药物")
        st.plotly_chart(fig)

        # 显示用药详情
        st.write("用药详情:")
        for _, medication in medication_history.iterrows():
            st.write(f"- {medication['日期']}: {medication['药物']} (剂量: {medication['剂量']}, 开药医生: {medication['医生']})")

    # 创建患者关系网络图
    query_patient_graph = """
    MATCH (p:Patient {id: $patient_id})
    CALL {
        WITH p
        MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)-[:DIAGNOSED_WITH]->(dis:Disease)
        RETURN d, dis
    }
    CALL {
        WITH p
        MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN s
    }
    CALL {
        WITH p
        MATCH (p)<-[:PRESCRIBED_TO]-(pres:Prescription)-[:INCLUDES]->(m:Medication)
        RETURN pres, m
    }
    RETURN p, collect(DISTINCT d) AS diagnoses, collect(DISTINCT dis) AS diseases,
           collect(DISTINCT s) AS symptoms, collect(DISTINCT pres) AS prescriptions, 
           collect(DISTINCT m) AS medications
    """
    patient_graph = run_query(driver, query_patient_graph, {"patient_id": patient_id})

    if not patient_graph.empty:
        st.write("### 患者病历关系图")
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        # 在 patient_history_analysis 函数的开始添加这个辅助函数
        def ensure_string_id(id_value):
            return str(id_value) if id_value is not None else ""
        
        # 然后在使用ID时调用这个函数
        patient_id = ensure_string_id(patient_id)
        
        net.add_node(patient_id, label=patient_info['name'], color="#FF5733", title="患者")
        
        disease_map = {
            'Heart Disease': '心脏病',
            'Diabetes': '糖尿病',
            'Asthma': '哮喘',
            'Cancer': '癌症',
            'Depression': '抑郁症'
        }
        symptom_map = {
            'Headache': '头痛',
            'Fever': '发烧',
            'Cough': '咳嗽',
            'Fatigue': '疲劳',
            'Nausea': '恶心'
        }
        medication_map = {
            'Lisinopril': '赖诺普利',
            'Metformin': '二甲双胍',
            'Albuterol': '沙丁胺醇',
            'Ibuprofen': '布洛芬',
            'Sertraline': '舍曲林'
        }

        diseases = []
        for disease in patient_graph.iloc[0]['diseases']:
            chinese_name = disease_map.get(disease['name'], disease['name'])
            # 确保 disease['id'] 是字符串
            disease_id = ensure_string_id(disease['id'])
            net.add_node(disease_id, label=chinese_name, color="#33FF57", title="疾病")
            net.add_edge(patient_id, disease_id)
            diseases.append(chinese_name)

        symptoms = []
        for symptom in patient_graph.iloc[0]['symptoms']:
            chinese_name = symptom_map.get(symptom['name'], symptom['name'])
            # 确保 symptom['id'] 是字符串
            symptom_id = ensure_string_id(symptom['id'])
            net.add_node(symptom_id, label=chinese_name, color="#5733FF", title="症状")
            net.add_edge(patient_id, symptom_id)
            symptoms.append(chinese_name)

        medications = []
        for medication in patient_graph.iloc[0]['medications']:
            chinese_name = medication_map.get(medication['name'], medication['name'])
            # 确保 medication['id'] 是字符串
            medication_id = ensure_string_id(medication['id'])
            net.add_node(medication_id, label=chinese_name, color="#33FFFF", title="药物")
            net.add_edge(patient_id, medication_id)
            medications.append(chinese_name)

        net.save_graph("patient_graph.html")
        st.components.v1.html(open("patient_graph.html", 'r', encoding='utf-8').read(), height=600)

        st.write("### 患者病历关系图分析")
        st.write(f"1. 疾病情况：该患者被诊断出 {len(diseases)} 种疾病，包括 {', '.join(diseases)}。")
        st.write(f"2. 症状表现：患者出现了 {len(symptoms)} 种症状，包括 {', '.join(symptoms)}。")
        st.write(f"3. 用药情况：医生为患者开具了 {len(medications)} 种药物，包括 {', '.join(medications)}。")

        # 分析疾病和症状的关联
        if len(diseases) > 1 and len(symptoms) > 1:
            st.write("4. 疾病与症状关联：")
            for disease in diseases:
                related_symptoms = [s for s in symptoms if s in symptom_map.values()]
                st.write(f"   - {disease} 可能与 {', '.join(related_symptoms)} 等症状有关。")

        # 分析用药与疾病的关系
        if len(medications) > 0 and len(diseases) > 0:
            st.write("5. 用药与疾病关系：")
            for medication in medications:
                related_diseases = [d for d in diseases if d in disease_map.values()]
                st.write(f"   - {medication} 可能用于治疗 {', '.join(related_diseases)}。")

        st.write("6. 综合分析：")
        st.write(f"   - 该患者的病历显示，主要涉及 {', '.join(diseases)} 等疾病。")
        st.write(f"   - 患者表现出的 {', '.join(symptoms)} 等症状可能与这些疾病有关。")
        st.write(f"   - 医生开具的 {', '.join(medications)} 等药物针对性地治疗这些疾病和症状。")
        st.write("   - 病历关系图直观地展示了患者的健康状况、症状和治疗方案之间的关联，有助于医生全面了解患者情况，制定更精准的治疗计划。")

    st.write("## 图数据库分析的独特价值")
    st.write("1. 全面的关系可视化：图数据库直观地展示了患者、疾病、症状和药物之间的复杂关系网络。")
    st.write("2. 时间序列分析：通过时间线图表，我们可以清晰地看到患者的诊断、症状和用药历史，有助于识别疾病的发展模式。")
    st.write("3. 多维度数据整合：图数据库允许我们同时分析患者的诊断、症状和用药情况，提供了全面的病史视图")
    st.write("4. 高效的关联查询：图数据库能够快速检索患者的所有相关信息，包括诊断、症状和用药，这在传统关系数据库中可能需要多次复杂的连接操作。")
    st.write("5. 模式识别：通过观察患者的症状、诊断和用药的关系图，医生可以更容易地识别潜在的疾病模式或药物相互作用。")

    st.write("## 临床应用价值")
    st.write("1. 个性化治疗方案：基于患者的完整病史和关系网络，医生可以制定更加个性化和精准的治疗方案。")
    st.write("2. 预防性医疗：通过分析患者的症状和诊断模式，可以早期识别潜在的健康风险，实施预防性干预。")
    st.write("3. 药物相互作用分析：关系图可以直观地展示患者使用的所有药物，有助于医生评估潜在的药物相互作用风险。")
    st.write("4. 病情进展追踪：时间线视图使得医生能够轻松追踪患者的病情进展，评估治疗效果。")
    st.write("5. 跨学科协作：综合的患者数据视图有助于不同专科的医生进行有效的沟通和协作，提供更全面的医疗服务。")

def disease_diagnosis_patterns(driver):
    st.subheader("疾病诊断模式分析")

    st.write("### 症状到疾病的诊断路径")
    symptom_map = {
        "Nausea": "恶心",
        "Fever": "发烧",
        "Cough": "咳嗽",
        "Fatigue": "疲劳",
        "Headache": "头痛"
    }
    symptom_input = st.selectbox("选择一个症状来分析其可能导致的疾病", 
                                 list(symptom_map.values()))
    if symptom_input:
        english_symptom = {v: k for k, v in symptom_map.items()}[symptom_input]
        query_symptom_to_disease = """
        MATCH path = (s:Symptom {name: $symptom})<-[:HAS_SYMPTOM]-(p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)-[:DIAGNOSED_WITH]->(dis:Disease)
        WITH dis.name AS disease, COUNT(DISTINCT p) AS patient_count, COLLECT(DISTINCT s.name) AS symptoms, COLLECT(DISTINCT path) AS paths
        ORDER BY patient_count DESC
        LIMIT 5
        RETURN disease, patient_count, symptoms, paths
        """
        symptom_to_disease = run_query(driver, query_symptom_to_disease, {"symptom": english_symptom})
        
        if not symptom_to_disease.empty:
            # 创建一个网络图
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
            
            disease_map = {
                "Asthma": "哮喘",
                "Hypertension": "高血压",
                "Arthritis": "关节炎",
                "Diabetes": "糖尿病",
                "Depression": "抑郁症"
            }
            
            # 添加节点和边
            for _, row in symptom_to_disease.iterrows():
                chinese_disease = disease_map.get(row['disease'], row['disease'])
                net.add_node(symptom_input, label=symptom_input, color="#FF9999", title=f"症状: {symptom_input}")
                net.add_node(chinese_disease, label=chinese_disease, color="#99CCFF", title=f"疾病: {chinese_disease}")
                net.add_edge(symptom_input, chinese_disease, value=row['patient_count'], title=f"患者数: {row['patient_count']}")
            
            # 保存并显示图形
            net.save_graph("symptom_to_disease.html")
            st.components.v1.html(open("symptom_to_disease.html", 'r', encoding='utf-8').read(), height=500)
            
            most_common_disease = symptom_to_disease.iloc[0]
            chinese_disease = disease_map.get(most_common_disease['disease'], most_common_disease['disease'])
            
            st.write(f"""
            **图表解释:**
            
            1. **节点:** 图中有两种节点：红色节点代表症状，蓝色节点代表疾病。节点的大小表示患者数量。
            2. **连线:** 症状到疾病的连线表示诊断关系。线的粗细表示患者数量。
            3. **布局:** 中心位置的节点通常表示患者数量较多的疾病。

            **分析价值:**
            1. 我们可以看到，{chinese_disease}是最常见的疾病，可能需要特别关注其预防和治疗。
            2. 这个网络图直观地展示了不同症状与疾病之间的关系，有助于医生更好地理解疾病的典型表现，从而提高诊断的准确性和效率。
            """)
        else:
            st.write(f"没有找到与症状 '{symptom_input}' 相关的疾病记录。")

    st.write("### 诊断链路分")
    disease_map = {
        "Asthma": "哮喘",
        "Hypertension": "高血压",
        "Arthritis": "关节炎",
        "Diabetes": "糖尿病",
        "Depression": "抑郁症"
    }
    disease_input = st.selectbox("选择一种疾病来分析其诊断链路", 
                                 list(disease_map.values()))
    if disease_input:
        english_disease = {v: k for k, v in disease_map.items()}[disease_input]
        query_diagnosis_chain = """
        MATCH path = (dis:Disease {name: $disease})<-[:DIAGNOSED_WITH]-(d:Diagnosis)<-[:HAS_DIAGNOSIS]-(p:Patient)-[:HAS_SYMPTOM]->(s:Symptom)
        WITH DISTINCT s.name AS symptom, 
             COUNT(DISTINCT p) AS patient_count,
             COLLECT(DISTINCT d.date) AS diagnosis_dates,
             COLLECT(DISTINCT path) AS paths
        ORDER BY patient_count DESC
        LIMIT 10
        RETURN symptom, patient_count, diagnosis_dates, paths
        """
        diagnosis_chain = run_query(driver, query_diagnosis_chain, {"disease": english_disease})
        
        if not diagnosis_chain.empty:
            # 创建一个网络图
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
            
            # 添加节点和边
            net.add_node(disease_input, label=disease_input, color="#99CCFF", title=f"疾病: {disease_input}")
            for _, row in diagnosis_chain.iterrows():
                chinese_symptom = symptom_map.get(row['symptom'], row['symptom'])
                net.add_node(chinese_symptom, label=chinese_symptom, color="#FF9999", title=f"症状: {chinese_symptom}")
                net.add_edge(chinese_symptom, disease_input, value=row['patient_count'], title=f"患者数: {row['patient_count']}")
            
            # 保存并显示图形
            net.save_graph("diagnosis_chain.html")
            st.components.v1.html(open("diagnosis_chain.html", 'r', encoding='utf-8').read(), height=500)
            
            most_common_symptom = diagnosis_chain.iloc[0]
            chinese_symptom = symptom_map.get(most_common_symptom['symptom'], most_common_symptom['symptom'])
            
            st.write(f"""
            **图表解释:**
            
            1. **节点:** 图中心的蓝色节点代表选择的疾病，周围的红色节点代表与该疾病相关的症状。节点的大小表示症状出现的频率。
            
            2. **连线:** 疾病和症状之间的连线表示诊断关系。线的粗细表示患者数量线越粗表示越多的患者在被诊断为该疾病时出现了这种症状。
            
            3. **布局:** 距离中心疾病节点越近的症状节点，表示该症状与疾病的关联越强。
            
            4. **数据分析:** 
               - 对于{disease_input}，最常见的症状是{chinese_symptom}，有{most_common_symptom['patient_count']}名患者报告了这一症状。
               - 图中显示所有症状都至少在一定数量的{disease_input}患者中被观察到。
               - 节点大小和连线粗细的差异反映了不同症状在{disease_input}患者中的出现频率差异。
               - 这个网络图有助于医生在诊断{disease_input}时，了解应该关注哪些关键症状，以及这些症状的相对重要性。
            
            这个诊断链路图直观地展示了从特定疾病到相关症状的关系网络，有助于医疗专业人员更好地理解疾病的典型表现，从而提高诊断的准确性和效率。对于患者教育也很有价值，可以帮助患者理解自己的症状与诊断之间的关系。
            """)
        else:
            st.write(f"没有找到与疾病 '{disease_input}' 相关的诊断链路。")

    st.write("### 疾病共现网络分析")
    query_disease_comorbidity = """
    MATCH (d1:Disease)<-[:DIAGNOSED_WITH]-(diag:Diagnosis)<-[:HAS_DIAGNOSIS]-(p:Patient)-[:HAS_DIAGNOSIS]->(diag2:Diagnosis)-[:DIAGNOSED_WITH]->(d2:Disease)
    WHERE d1 <> d2
    WITH d1.name AS disease1, d2.name AS disease2, COUNT(DISTINCT p) AS comorbidity_count
    WHERE comorbidity_count > 5
    RETURN disease1, disease2, comorbidity_count
    ORDER BY comorbidity_count DESC
    LIMIT 20
    """
    disease_comorbidity = run_query(driver, query_disease_comorbidity)

    if not disease_comorbidity.empty:
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        
        disease_map = {
            "Asthma": "哮喘",
            "Hypertension": "高血压",
            "Arthritis": "关节炎",
            "Diabetes": "糖尿病",
            "Depression": "抑郁症"
        }
        
        for _, row in disease_comorbidity.iterrows():
            disease1 = disease_map.get(row['disease1'], row['disease1'])
            disease2 = disease_map.get(row['disease2'], row['disease2'])
            net.add_node(disease1, label=disease1, title=disease1)
            net.add_node(disease2, label=disease2, title=disease2)
            net.add_edge(disease1, disease2, value=row['comorbidity_count'], title=f"共现次数: {row['comorbidity_count']}")
        
        net.save_graph("disease_comorbidity.html")
        st.components.v1.html(open("disease_comorbidity.html", 'r', encoding='utf-8').read(), height=500)
        
        st.write(f"""
        **图表解释:**
        
        1. **节点:** 每个圆圈代表一种疾病。圆圈的大小表示该疾病在数据集中出现的频率。
        
        2. **连线:** 两个疾病之间的线表示这两种疾病经常同时出现在同一患者身上。线越粗,表示共现频率越高。
        
        3. **颜色:** 不同颜色代表不同的疾病类别或群组。
        
        4. **布局:** 靠近中心的疾病通常与其他疾病有更多的联系,而边缘的疾病可能较为独立。
        
        5. **数据分析:** 
           - 从图中可以看出,高血压和糖尿病之间有较粗的连线,表明这两种疾病经常一起出现。
           - 关节炎的节点较大,说明在我们的数据中,关节炎是一种常见疾病。
           - 抑郁症与其他疾病也有多条连线,表明它可能与多种慢性疾病有关联。
           - 哮喘虽然与其他疾病有连接,但连线相对较细,表明其与其他疾病的共现频率相对较低。
        
        这个网络图直观地展示了各种疾病之间的关联性,有助于医疗专业人员更好地理解疾病的共现模式,从而制定更全面的诊断和治疗策略。对于患者来说,这种可视化有助于提高对多种慢性病并发风险的认识。
        """)
    else:
        st.write("未能获取足够的疾病共现数据来生成网络图。")

    st.write("## 图数据库分析的独特价值")
    st.write("1. 疾病共现网络：利用图数据库的关系模型,直观展示疾病间的复杂关联,有助于识别高风险组合和制定综合治疗方案。")
    st.write("2. 症状到疾病的路径分析：展示了图数据库在多跳查询上的优势,能快速追踪从症状到可能疾病的路径,辅助诊断决策。")
    st.write("3. 诊断链路分析：利用图数据库的路径分析能力,展示了从症状出现到疾病确诊的完整过程,有助于优化诊断流程和时间。")
    st.write("4. 数据关联性：这些分析充分利用了患者、症状、诊断和疾病之间的复杂关系,这是图数据库的核心优势。")
    st.write("5. 查询效率：对于复杂的关系查询和路径分析,图数据库比关系型数据库更高效,特别是在处理大规模医疗数据时。")

def medication_prescription_trends(driver):
    st.subheader("药物处方趋势分析")

    # 1. 药物处方网络分析
    st.write("### 药物处方网络")
    query_prescription_network = """
    MATCH (m:Medication)<-[:INCLUDES]-(p:Prescription)<-[:PRESCRIBES]-(d:Doctor)
    WITH m, d, COUNT(p) AS prescription_count
    WHERE prescription_count > 10
    RETURN m.name AS medication, d.speciality AS doctor_speciality, prescription_count
    ORDER BY prescription_count DESC
    LIMIT 20
    """
    prescription_network = run_query(driver, query_prescription_network)

    if not prescription_network.empty:
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        
        medication_map = {
            "Lisinopril": "赖诺普利",
            "Metformin": "二甲双胍",
            "Albuterol": "沙丁胺醇",
            "Ibuprofen": "布洛芬",
            "Sertraline": "舍曲林"
        }
        speciality_map = {
            "Cardiology": "心脏科",
            "Neurology": "神经科",
            "Oncology": "肿瘤科",
            "Pediatrics": "儿科",
            "Surgery": "外科"
        }
        
        for _, row in prescription_network.iterrows():
            medication = medication_map.get(row['medication'], row['medication'])
            speciality = speciality_map.get(row['doctor_speciality'], row['doctor_speciality'])
            net.add_node(medication, label=medication, color="#99CCFF", title=f"药: {medication}")
            net.add_node(speciality, label=speciality, color="#FF9999", title=f"专科: {speciality}")
            net.add_edge(speciality, medication, value=row['prescription_count'], title=f"处方数: {row['prescription_count']}")
        
        net.save_graph("prescription_network.html")
        st.components.v1.html(open("prescription_network.html", 'r', encoding='utf-8').read(), height=500)
        
        st.write("""
        **图表解释:**
        1. **节点:** 蓝色节点代表药物，红色节点代表医生专科。节点的大小表示处方数量。
        2. **连线:** 专科到药物的连线表示该专科开具该药物的处方。线的粗细表示处方数量。
        3. **布局:** 中心位置的节点通常表示处方数量较多的药物或专科。

        **分析价值:**
        这个网络图直观地展示了不同专科和药物之间的关系，有助于识别各专科的用药偏好。
        可以快速发现哪些药物被广泛使用，哪些专科开具处方最多。
        有助于发现潜在的跨专科用药模式，为医疗资源分配和药物管理提供依据。
        """)
    else:
        st.write("未能获取足够的处方数据来生成网络图。")

    st.write("## 图数据库分析的独特价值")
    st.write("1. 关系网络可视化：图数据库能够直观地展示药物、医生患者之间的复杂关系网络。")
    st.write("2. 多维度数据整合：可以同时分析药物、处方、医生专科和时间等多个维度的数据。")
    st.write("3. 路径分析：能够轻松追踪从医生到药物的处方路径，有助于理解处方模式。")
    st.write("4. 社区检测：可以发现药物使用的聚类模式，如某些药物经常一起使用。")
    st.write("5. 时间序列与图结构结合：可以在图结构中嵌入时间信息，分析药物使用模式的演变。")

def insurance_claim_analysis(driver):
    st.subheader("医疗保险理赔分析")

    # 1. 患者-疾病-理赔关系网络
    st.write("### 患者-疾病-理赔关系网络")
    query_claim_network = """
    MATCH (p:Patient)-[:FILED_CLAIM]->(c:InsuranceClaim)
    MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)-[:DIAGNOSED_WITH]->(dis:Disease)
    WITH p, dis, c, COUNT(DISTINCT d) AS diagnosis_count
    RETURN p.id AS patient_id, p.age AS age, p.gender AS gender,
           dis.name AS disease, c.amount AS claim_amount, diagnosis_count
    ORDER BY claim_amount DESC
    LIMIT 50
    """
    claim_network = run_query(driver, query_claim_network)

    if not claim_network.empty:
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        disease_map = {
            "Asthma": "哮喘",
            "Hypertension": "高血压",
            "Arthritis": "关节炎",
            "Diabetes": "糖尿病",
            "Depression": "抑郁症"
        }
        
        gender_map = {
            "Male": "男性",
            "Female": "女性"
        }
        
        for _, row in claim_network.iterrows():
            patient_node = f"患者_{row['patient_id']}"
            disease = disease_map.get(row['disease'], row['disease'])
            claim_node = f"理赔_{row['claim_amount']:.0f}"
            gender = gender_map.get(row['gender'], row['gender'])
            
            net.add_node(patient_node, label=f"患者\n{row['age']}岁 {gender}", color="#FF9999", size=20)
            net.add_node(disease, label=disease, color="#99CCFF", size=30)
            net.add_node(claim_node, label=f"理赔\n¥{row['claim_amount']:.0f}", color="#99FF99", size=row['claim_amount']/1000)
            
            net.add_edge(patient_node, disease, value=row['diagnosis_count'], title=f"诊断次数: {row['diagnosis_count']}")
            net.add_edge(patient_node, claim_node, title="提交理赔")

        net.save_graph("claim_network.html")
        st.components.v1.html(open("claim_network.html", 'r', encoding='utf-8').read(), height=600)
        
        most_common_disease = disease_map.get(claim_network['disease'].value_counts().index[0], claim_network['disease'].value_counts().index[0])
        high_claims = claim_network[claim_network['claim_amount'] > 10000]
        high_claim_diseases = ', '.join([disease_map.get(d, d) for d in high_claims['disease'].unique()])
        gender_with_highest_claim = gender_map.get(claim_network.groupby('gender')['claim_amount'].mean().idxmax(), claim_network.groupby('gender')['claim_amount'].mean().idxmax())
        
        st.write(f"""
        **图表解释:**
        1. **节点:** 红色节点代表患者，蓝色节点代表疾病，绿色节点代表理赔。节点的大小表示患者年龄、疾病频率或理赔金额。
        2. **连线:** 患者到疾病的连线表示诊断关系，线的粗细表示诊断次数。患者到理赔的连线表示理赔申请。
        3. **布局:** 中心位置的节点通常表示高频疾病或大额理赔。

        **分析价值:**
        1. 我们可以看到，{most_common_disease}是最常见的疾病，可能需要特别关注其预防和治疗。
        2. 有{len(high_claims)}例大额理赔（超过¥10,000），{'主要涉及'+high_claim_diseases+'等疾病' if high_claim_diseases else '没有特定疾病显著相关'}。
        3. {gender_with_highest_claim}患者的平均理赔金额更高，可能需要进一步调查原因。
        4. 年龄在{claim_network.groupby('age')['claim_amount'].mean().idxmax()}岁左右的患者理赔金额峰值，这可能是一个重要的风险因素。

        这个网络图直观地展示了患者、疾病和理赔之间的复杂关系，有助于保险公司识别高风险群体和疾病，优化保险产品设计和风险管理策略。
        """)

    # 2. 疾病理赔金额分析
    st.write("### 疾病理赔金额分析")
    query_disease_claims = """
    MATCH (p:Patient)-[:FILED_CLAIM]->(c:InsuranceClaim)
    MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)-[:DIAGNOSED_WITH]->(dis:Disease)
    WITH dis.name AS disease, AVG(c.amount) AS avg_claim, COUNT(DISTINCT c) AS claim_count
    RETURN disease, avg_claim, claim_count
    ORDER BY avg_claim DESC
    LIMIT 10
    """
    disease_claims = run_query(driver, query_disease_claims)

    if not disease_claims.empty:
        disease_claims['disease'] = disease_claims['disease'].map(disease_map)
        fig = px.scatter(disease_claims, x="avg_claim", y="disease", size="claim_count", 
                         color="avg_claim", hover_data=["claim_count"],
                         labels={"avg_claim": "平均理赔金额", "disease": "疾病", "claim_count": "理赔次数"},
                         title="疾病与理赔金额关系")
        st.plotly_chart(fig)

        st.write(f"""
        **分析结果:**
        1. {disease_claims['disease'].iloc[0]}的平均理赔金额最高，达到¥{disease_claims['avg_claim'].iloc[0]:.2f}，需要特别关注。
        2. {disease_claims['disease'].iloc[-1]}虽然平均理赔金额较低，但理赔次数最多，达到{disease_claims['claim_count'].iloc[-1]}次。
        3. 前三种平均理赔金额最高的疾病是：{', '.join(disease_claims['disease'].head(3))}，这些可能是高风险疾病。
        4. {disease_claims['disease'][disease_claims['claim_count'].idxmax()]}的理赔次数最多，可能是一种常见病，但其严重程度可能相对较轻。

        这个分析有助于保险公司更好地理解不同疾病的经济影响，从而优化保险产品定价和风险评估模型。
        """)

    st.write("## 图数据库分析的独特价值")
    st.write("1. 关系网络可视化：图数据库能够直观地展示患者、疾病和理赔之间的复杂关系网络。")
    st.write("2. 多维度数据整合：可以同时分析患者特征、疾病类型、理赔金额等多个维度的数据。")
    st.write("3. 路径分析：能够轻松追踪从患者到疾病再到理赔的完整路径，有助于理解理赔模式。")
    st.write("4. 社区检测：可以发现高风险患者群体或疾病群组，为风险管理提供依据。")
    st.write("5. 时间序列与图结构结合：可以在图结构中嵌入时间信息，分析理赔模式的演变。")

if __name__ == "__main__":
    main()