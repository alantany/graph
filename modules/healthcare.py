import streamlit as st
from .utils import run_query
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import pandas as pd
import os
import io
import traceback

def healthcare_scenario(driver):
    st.header("图数据库在医疗健康领域的应用")
    
    submenu = st.sidebar.radio(
        "医疗健康子菜单",
        ("数据管理", "疾病关联分析", "药物相互作用", "患者路径分析")
    )
    
    if submenu == "数据管理":
        healthcare_data_management(driver)
    elif submenu == "疾病关联分析":
        disease_correlation_analysis(driver)
    elif submenu == "药物相互作用":
        drug_interaction_analysis(driver)
    elif submenu == "患者路径分析":
        patient_pathway_analysis(driver)

def healthcare_data_management(driver):
    st.subheader("医疗健康数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("清空医疗健康数据"):
            clear_healthcare_data(driver)
    
    with col2:
        if st.button("导入医疗健康数据"):
            import_healthcare_data(driver)
    
    show_healthcare_database_stats(driver)

def clear_healthcare_data(driver):
    try:
        with driver.session() as session:
            # 删除所有医疗健康相关的节点和关系
            result = session.run("""
            MATCH (n)
            WHERE n:Patient OR n:Disease OR n:Drug OR n:Treatment
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """)
            
            deleted_count = result.single()["deleted_count"]
            st.write(f"已删除 {deleted_count} 个节点")
        
        st.success("医疗健康数据已成功清除")
    except Exception as e:
        st.error(f"清除数据时发生错误: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())

def import_healthcare_data(driver):
    healthcare_dir = 'healthcare'
    files = {
        "患者数据": "patients.csv",
        "疾病数据": "diseases.csv",
        "药物数据": "medications.csv",
        "诊断记录": "diagnoses.csv",
        "用药记录": "prescriptions.csv"
    }
    
    for file_desc, file_name in files.items():
        file_path = os.path.join(healthcare_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read()
                import_healthcare_csv_data(driver, file_name, csv_data)
            st.success(f"{file_desc}导入成功！")
        except FileNotFoundError:
            st.error(f"{file_path} 文件不存在。")
        except Exception as e:
            st.error(f"导入 {file_desc} 时发生错误: {str(e)}")

def import_healthcare_csv_data(driver, file_name, csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    
    with driver.session() as session:
        if file_name == "patients.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (p:Patient {id: row.id})
            SET p.name = row.name,
                p.age = toInteger(row.age),
                p.gender = row.gender
            """, rows=df.to_dict('records'))
        elif file_name == "diseases.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Disease {id: row.id})
            SET d.name = row.name,
                d.category = row.category
            """, rows=df.to_dict('records'))
        elif file_name == "medications.csv":
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Drug {id: row.id})
            SET d.name = row.name,
                d.category = row.category
            """, rows=df.to_dict('records'))
        elif file_name == "diagnoses.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (p:Patient {id: row.patient_id})
            MATCH (d:Disease {id: row.disease_id})
            MERGE (p)-[r:DIAGNOSED_WITH]->(d)
            SET r.date = row.date
            """, rows=df.to_dict('records'))
        elif file_name == "prescriptions.csv":
            session.run("""
            UNWIND $rows AS row
            MATCH (p:Patient {id: row.patient_id})
            MATCH (d:Drug {id: row.medication_id})
            MERGE (p)-[r:PRESCRIBED]->(d)
            SET r.date = row.date,
                r.dosage = row.dosage
            """, rows=df.to_dict('records'))

def show_healthcare_database_stats(driver):
    st.subheader("医疗健康数据统计")
    queries = {
        "患者数": "MATCH (p:Patient) RETURN count(p) as count",
        "疾病数": "MATCH (d:Disease) RETURN count(d) as count",
        "药物数": "MATCH (d:Drug) RETURN count(d) as count",
        "诊断记录数": "MATCH ()-[r:DIAGNOSED_WITH]->() RETURN count(r) as count",
        "用药记录数": "MATCH ()-[r:PRESCRIBED]->() RETURN count(r) as count"
    }
    
    results = {}
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query).single()
            results[label] = result["count"] if result else 0
    
    for label, count in results.items():
        st.write(f"{label}: {count}")

def disease_correlation_analysis(driver):
    st.subheader("疾病关联分析")
    query = """
    MATCH (d1:Disease)<-[:DIAGNOSED_WITH]-(p:Patient)-[:DIAGNOSED_WITH]->(d2:Disease)
    WHERE d1.id < d2.id
    RETURN d1.name AS disease1, d2.name AS disease2,
           count(p) AS co_occurrence
    ORDER BY co_occurrence DESC
    LIMIT 10
    """
    results = run_query(driver, query)
    
    if not results.empty:
        fig = px.bar(results, x="disease1", y="co_occurrence",
                     color="disease2",
                     title="疾病共现分析")
        st.plotly_chart(fig)
    else:
        st.warning("未找到疾病关联数据")

def drug_interaction_analysis(driver):
    st.subheader("药物相互作用分析")
    query = """
    MATCH (d1:Drug)<-[:PRESCRIBED]-(p:Patient)-[:PRESCRIBED]->(d2:Drug)
    WHERE d1.id < d2.id
    RETURN d1.name AS drug1, d2.name AS drug2,
           count(p) AS co_prescription,
           d1.category AS category1,
           d2.category AS category2
    ORDER BY co_prescription DESC
    LIMIT 15
    """
    results = run_query(driver, query)
    
    if not results.empty:
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        
        # 添加节点
        drugs = set(results['drug1'].unique()) | set(results['drug2'].unique())
        for drug in drugs:
            net.add_node(drug, label=drug)
        
        # 添加边
        for _, row in results.iterrows():
            net.add_edge(row['drug1'], row['drug2'], 
                        value=row['co_prescription'],
                        title=f"共同开具次数: {row['co_prescription']}")
        
        net.save_graph("drug_interaction_graph.html")
        with open("drug_interaction_graph.html", 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=500)
    else:
        st.warning("未找到药物相互作用数据")

def patient_pathway_analysis(driver):
    st.subheader("患者诊疗路径分析")
    query = """
    MATCH path = (p:Patient)-[:DIAGNOSED_WITH|PRESCRIBED*2..4]-()
    WITH p, path, relationships(path) as rels
    RETURN p.id AS patient_id,
           [r in rels | type(r)] AS path_types,
           [r in rels | CASE
                         WHEN type(r) = 'DIAGNOSED_WITH' THEN endNode(r).name
                         WHEN type(r) = 'PRESCRIBED' THEN endNode(r).name
                       END] AS path_nodes,
           [r in rels | r.date] AS dates
    ORDER BY patient_id
    LIMIT 100
    """
    results = run_query(driver, query)
    
    if not results.empty:
        # 将路径数据转换为时间序列格式
        pathway_data = []
        for _, row in results.iterrows():
            for i in range(len(row['path_types'])):
                pathway_data.append({
                    'patient_id': row['patient_id'],
                    'event_type': row['path_types'][i],
                    'event_node': row['path_nodes'][i],
                    'start_date': row['dates'][i],
                    'end_date': pd.to_datetime(row['dates'][i]) + pd.Timedelta(days=1)  # 结束时间设为开始时间后一天
                })
        
        pathway_df = pd.DataFrame(pathway_data)
        pathway_df['start_date'] = pd.to_datetime(pathway_df['start_date'])
        pathway_df['end_date'] = pd.to_datetime(pathway_df['end_date'])
        
        fig = px.timeline(pathway_df, 
                         x_start="start_date", 
                         x_end="end_date",
                         y="patient_id",
                         color="event_type",
                         hover_data=["event_node"],
                         title="患者诊疗路径时间线")
        
        # 调整图表布局
        fig.update_layout(
            xaxis_title="日期",
            yaxis_title="患者ID",
            height=600
        )
        
        st.plotly_chart(fig)
    else:
        st.warning("未找到患者路径数据") 