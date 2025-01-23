import streamlit as st
from .utils import run_query
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import pandas as pd
import os
import io
import traceback

def social_network_scenario(driver):
    st.header("图数据库在社交网络分析中的应用")
    
    submenu = st.sidebar.radio(
        "社交网络子菜单",
        ("数据管理", "影响力分析", "社区发现", "信息传播分析")
    )
    
    if submenu == "数据管理":
        social_data_management(driver)
    elif submenu == "影响力分析":
        influence_analysis(driver)
    elif submenu == "社区发现":
        community_detection(driver)
    elif submenu == "信息传播分析":
        information_propagation_analysis(driver)

def social_data_management(driver):
    st.subheader("社交网络数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空社交网络数据"):
            clear_social_data(driver)
    
    with col2:
        if st.button("导入社交网络数据"):
            import_social_data(driver)
    
    show_social_database_stats(driver)

def clear_social_data(driver):
    try:
        with driver.session() as session:
            # 删除所有社交网络相关的节点和关系
            result = session.run("""
            MATCH (n)
            WHERE n:SocialUser OR n:Post OR n:Interest
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """)
            
            deleted_count = result.single()["deleted_count"]
            st.write(f"已删除 {deleted_count} 个节点")
        
        st.success("社交网络数据已成功清除")
    except Exception as e:
        st.error(f"清除数据时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())

def import_social_data(driver):
    social_dir = 'social'
    files = {
        "用户数据": "social_users.csv",
        "帖子数据": "posts.csv",
        "关注关系": "follow_relationships.csv",
        "兴趣数据": "interests.csv",
        "用户兴趣": "user_interests.csv"
    }
    
    for file_desc, file_name in files.items():
        file_path = os.path.join(social_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read()
                import_social_csv_data(driver, file_name, csv_data)
            st.success(f"{file_desc}导入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。")
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
                u.following_count = toInteger(row.following_count)
            """, rows=df.to_dict('records'))
        elif file_name == "posts.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u:SocialUser {id: row.user_id})
            MERGE (p:Post {id: row.id})
            SET p.content = row.content,
                p.timestamp = row.timestamp,
                p.likes = toInteger(row.likes)
            MERGE (u)-[:POSTED]->(p)
            """, rows=df.to_dict('records'))
        elif file_name == "follow_relationships.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (u1:SocialUser {id: row.follower_id})
            MATCH (u2:SocialUser {id: row.following_id})
            MERGE (u1)-[:FOLLOWS]->(u2)
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
            MERGE (u)-[:INTERESTED_IN]->(i)
            """, rows=df.to_dict('records'))

def show_social_database_stats(driver):
    st.subheader("社交网络数据统计")
    queries = {
        "用户数": "MATCH (u:SocialUser) RETURN count(u) as count",
        "帖子数": "MATCH (p:Post) RETURN count(p) as count",
        "关注关系数": "MATCH ()-[r:FOLLOWS]->() RETURN count(r) as count",
        "兴趣标签数": "MATCH (i:Interest) RETURN count(i) as count"
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
    RETURN u.id AS user_id, u.name AS name,
           u.followers_count AS followers,
           u.following_count AS following
    ORDER BY u.followers_count DESC
    LIMIT 10
    """
    results = run_query(driver, query)
    
    if not results.empty:
        fig = px.scatter(results, x="following", y="followers",
                         hover_data=["name"], 
                         title="用户影响力分布")
        st.plotly_chart(fig)
    else:
        st.warning("未找到用户数据")

def community_detection(driver):
    st.subheader("社区发现")
    query = """
    MATCH (u1:SocialUser)-[:FOLLOWS]->(u2:SocialUser)
    RETURN u1.id AS source, u2.id AS target
    LIMIT 1000
    """
    results = run_query(driver, query)
    
    if not results.empty:
        G = nx.from_pandas_edgelist(results, 'source', 'target', create_using=nx.DiGraph())
        communities = nx.community.greedy_modularity_communities(G.to_undirected())
        
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        
        for i, community in enumerate(communities):
            for node in community:
                net.add_node(node, label=f"User {node}", group=i)
        
        for _, row in results.iterrows():
            net.add_edge(row['source'], row['target'])
        
        net.save_graph("community_graph.html")
        with open("community_graph.html", 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=500)
    else:
        st.warning("未找到足够的数据进行社区分析")

def information_propagation_analysis(driver):
    st.subheader("信息传播分析")
    query = """
    MATCH (u:SocialUser)-[:POSTED]->(p:Post)
    RETURN p.timestamp AS time, p.likes AS likes,
           u.followers_count AS followers
    ORDER BY p.timestamp DESC
    LIMIT 100
    """
    results = run_query(driver, query)
    
    if not results.empty:
        results['time'] = pd.to_datetime(results['time'])
        fig = px.scatter(results, x="time", y="likes",
                         size="followers",
                         title="帖子传播效果分析")
        st.plotly_chart(fig)
    else:
        st.warning("未找到帖子数据") 