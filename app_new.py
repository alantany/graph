import streamlit as st
from modules.db_config import get_driver
from modules.financial import financial_risk_control_scenario
# 后续会添加其他模块的导入

def main():
    st.title("Neo4j图数据库应用场景展示")

    use_aura = st.sidebar.checkbox("使用Neo4j Aura", value=True)
    driver = get_driver(use_aura)

    # 显示当前连接信息
    if use_aura:
        st.sidebar.success("已连接到Neo4j Aura")
    else:
        st.sidebar.info("已连接到本地Neo4j")

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
        st.write("社交网络场景 - 开发中...")
    elif choice == "医疗健康场景":
        st.write("医疗健康场景 - 开发中...")

def show_neo4j_introduction(driver):
    st.header("Neo4j图数据库应用场景介绍")
    st.write("""
    Neo4j是一种强大的图数据库，适用于多种复杂的数据分析场景。在这个演示中，我们将展示Neo4j在以下三个领域的应用：

    1. 金融风控
    2. 社交网络分析
    3. 医疗健康数据管理

    每个场景都展示了图数据库如何帮助我们更好地理解和分析复杂的关系数据。
    """)

if __name__ == "__main__":
    main() 