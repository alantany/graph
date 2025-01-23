import streamlit as st
from modules.db_config import get_cached_driver, close_driver
from modules.financial import financial_risk_control_scenario
from modules.social import social_network_scenario
from modules.healthcare import healthcare_scenario

def main():
    st.set_page_config(
        page_title="Neo4j图数据库应用场景展示",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("Neo4j图数据库应用场景展示")
    
    # 选择数据库类型
    use_aura = st.sidebar.checkbox("使用Aura云数据库", value=True)
    
    try:
        # 获取数据库连接
        driver = get_cached_driver(use_aura)
        
        # 场景选择
        scenario = st.sidebar.radio(
            "选择应用场景",
            ("Neo4j图数据库应用场景介绍", "金融风控场景", "社交网络场景", "医疗健康场景")
        )
        
        # 根据选择显示不同场景
        if scenario == "Neo4j图数据库应用场景介绍":
            show_neo4j_introduction(driver)
        elif scenario == "金融风控场景":
            financial_risk_control_scenario(driver)
        elif scenario == "社交网络场景":
            social_network_scenario(driver)
        elif scenario == "医疗健康场景":
            healthcare_scenario(driver)
            
    except Exception as e:
        st.error("应用程序运行出错，请检查数据库连接或联系管理员。")
        st.error(str(e))

def show_neo4j_introduction(driver):
    st.header("Neo4j图数据库应用场景介绍")
    
    st.markdown("""
    ### 什么是图数据库？
    
    图数据库是一种存储和查询图形结构数据的数据库。在图数据库中，数据以节点（Node）和关系（Relationship）的形式存储，
    这种结构非常适合表达复杂的关联关系。
    
    ### Neo4j的优势
    
    1. **高性能的关系查询**
       - 相比传统关系型数据库，图数据库在处理复杂关系查询时具有显著的性能优势
       - 支持实时的深度遍历查询
    
    2. **灵活的数据模型**
       - 无需预先定义严格的模式
       - 可以动态添加新的节点类型和关系类型
    
    3. **直观的数据表示**
       - 数据模型与现实世界的概念模型高度一致
       - 便于理解和维护
    
    ### 应用场景展示
    
    本演示系统包含三个典型的图数据库应用场景：
    
    1. **金融风控场景**
       - 欺诈检测
       - 风险评估
       - 关联分析
    
    2. **社交网络场景**
       - 社区发现
       - 影响力分析
       - 推荐系统
    
    3. **医疗健康场景**
       - 疾病关联分析
       - 药物相互作用
       - 患者路径分析
    """)
    
    st.info("请在左侧边栏选择具体场景进行探索")

if __name__ == "__main__":
    main()