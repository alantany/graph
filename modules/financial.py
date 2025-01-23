import streamlit as st
from .utils import run_query
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import traceback

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
            st.success(f"{file_desc}导入成功！")
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
    else:
        st.warning("未发现高风险用户。")

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
    else:
        st.warning("未发现显著的用户关联网络。")

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
    else:
        st.warning("未发现异常交易。") 