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
    
    # 金融风控数据概览
    st.write("### 金融风控数据")
    financial_stats = get_financial_stats(driver)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("用户数", financial_stats["用户数"])
        st.metric("交易数", financial_stats["交易数"])
        st.metric("商户数", financial_stats["商户数"])
    with col2:
        st.metric("设备数", financial_stats["设备数"])
        st.metric("IP地址数", financial_stats["IP地址数"])
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
            st.metric("评论数", social_stats["评论数"])
            st.metric("关注关系数", social_stats["关注关系数"])
        if social_stats["平均每用户帖子数"] is not None:
            st.write(f"平均用户帖子数: {social_stats['平均每用户帖子数']:.2f}")
        if social_stats["平均每用户关注数"] is not None:
            st.write(f"平均每用户关注数: {social_stats['平均每用户关注数']:.2f}")
    else:
        st.write("暂无社交网络数据")

    # 医疗健康数据概览
    st.write("### 医疗健康数据")
    health_stats = get_health_stats(driver)
    if any(health_stats.values()):  # 检查是否有任何医疗健康数据
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
        return {
            "用户数": session.run("MATCH (u:SocialUser) RETURN count(u) AS count").single()["count"] or default_value,
            "帖子数": session.run("MATCH (p:Post) RETURN count(p) AS count").single()["count"] or default_value,
            "评论数": session.run("MATCH (c:Comment) RETURN count(c) AS count").single()["count"] or default_value,
            "关注关系数": session.run("MATCH (:SocialUser)-[:FOLLOWS]->(:SocialUser) RETURN count(*) AS count").single()["count"] or default_value,
            "平均每用户帖子数": session.run("MATCH (u:SocialUser)-[:POSTED]->(p:Post) WITH u, count(p) AS post_count RETURN avg(post_count) AS avg_posts").single()["avg_posts"],
            "平均每用户关注数": session.run("MATCH (u:SocialUser)-[:FOLLOWS]->(f:SocialUser) WITH u, count(f) AS follow_count RETURN avg(follow_count) AS avg_follows").single()["avg_follows"]
        }

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
            st.success("金融数据已清空")
            show_financial_database_stats(driver)
    
    with col2:
        if st.button("导入金融数据"):
            import_financial_data(driver)
            st.success("金融数据导入完成")
            show_financial_database_stats(driver)

def clear_financial_data(driver):
    with driver.session() as session:
        session.run("""
        MATCH (n:FinancialRisk) 
        DETACH DELETE n
        """)

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
            st.success(f"{file_desc} 导入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。请确保已生成数据文件。")
        except Exception as e:
            st.error(f"导入 {file_desc} 时发生错误: {str(e)}")

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
        st.warning("未发现高风险用户。这可能表明当前的风险评分模型需要调整，或者系统中的用户普遍表现良好。")

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
        st.write(f"2. 最强的关联是用户 {max_edge[0]} 和 {max_edge[1]} 之间，他们共享了 {max_edge[2]['weight']} 次交易。这种高度关联可能表示潜在的欺诈行为或资金洗白活动。")
        central_node = max(G.degree, key=lambda x: x[1])[0]
        st.write(f"3. 用户 {central_node} 似乎是这个网络的中心，与 {G.degree[central_node]} 个其他用户有直接关联。这个用户可能是一个关键的风险节点，需要特别关注。")
        st.write("4. 建议对这个网络中的用户进行更深入的背景调查，并密切监控他们的未来交易活动。")
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
        st.write(f"1. 我们检测到 {len(results)} 笔可能的异常交易。")
        flagged_transactions = results[results['status'] == 'Flagged']
        st.write(f"2. 其中 {len(flagged_transactions)} 笔交易被系统自动标记为可疑。")
        high_amount_transactions = results[results['amount'] > 10000]
        st.write(f"3. 有 {len(high_amount_transactions)} 笔交易的金额超过了10,000，这些大额交易需要特别关注。")
        highest_transaction = results.iloc[0]
        st.write(f"4. 最大的一笔交易金额为 {highest_transaction['amount']:.2f}，由用户 {highest_transaction['user_id']} 发起，交易ID为 {highest_transaction['transaction_id']}。")
        st.write("5. 建议对这些异常交易进行人工审核，特别是那些金额特别大或被系统标记的交易。同时，可能需要临时限制相关用户的交易权限，直到完调查。")
    else:
        st.warning("未发现异常交易。这可能表明当前的交易活动都在正常范围内，或者需要调整异常交易的定义标准。")

def run_query(driver, query):
    with driver.session() as session:
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
        fraud_detection(driver)

def social_data_management(driver):
    st.subheader("社交网络数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空社交网络数据"):
            clear_social_data(driver)
            st.success("社交网络数���已清空")
            show_social_database_stats(driver)
    
    with col2:
        if st.button("导入社交网络数据"):
            import_social_data(driver)
            st.success("社交网络数据导入完成")
            show_social_database_stats(driver)

def clear_social_data(driver):
    with driver.session() as session:
        session.run("""
        MATCH (n) 
        WHERE n:SocialUser OR n:Post OR n:Interest
        DETACH DELETE n
        """)
        session.run("MATCH ()-[r:FOLLOWS|POSTED|INTERESTED_IN]->() DELETE r")

def import_social_data(driver):
    social_dir = 'social'
    files = {
        "用户数据": "social_users.csv",
        "帖子数据": "posts.csv",
        "关注关系数据": "follow_relationships.csv",
        "兴趣数据": "interests.csv",
        "用户兴趣数据": "user_interests.csv"
    }
    
    for file_desc, file_name in files.items():
        file_path = os.path.join(social_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read()
                import_social_csv_data(driver, file_name, csv_data)
            st.success(f"{file_desc} 导入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。请确保已生成数据文件。")
        except Exception as e:
            st.error(f"导入 {file_desc} 时发生错误: {str(e)}")

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
    
    for label, count in results.items():
        st.write(f"{label}: {count}")

def influence_analysis(driver):
    st.subheader("用户影响力分析")
    query = """
    MATCH (u:SocialUser)
    WITH u, size((u)<-[:FOLLOWS]-()) as follower_count, size((u)-[:POSTED]->()) as post_count
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
        st.write(f"2. 最具影响力的用户是 {top_influencer['name']} (ID: {top_influencer['user_id']})，影响力得分为 {top_influencer['influence_score']:.2f}。")
        st.write(f"3. 这个用户有 {top_influencer['follower_count']} 个粉丝，发布了 {top_influencer['post_count']} 条帖子。")
        st.write("4. 影响力得分是根据粉丝数（权重70%）和发帖数（权重30%）计算的。")
        st.write("5. 建议关注这些高影响力用户，他们可能是重要的意见领袖或者潜在的品牌合作伙伴。")
    else:
        st.warning("未能获取用户影响力数据。请确保已导入足够的社交网络数据。")

def healthcare_scenario(driver):
    st.header("图数据库在医疗健康领域的应用")
    st.write("医疗健康数据分析功能正在开发中...")

if __name__ == "__main__":
    main()