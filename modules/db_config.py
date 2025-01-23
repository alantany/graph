from neo4j import GraphDatabase
import streamlit as st
import traceback
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量（如果文件存在）
load_dotenv()

# 数据库连接配置
def get_env_var(key, default=None):
    """
    获取环境变量，优先从Streamlit Secrets获取，
    如果不存在则从环境变量获取
    """
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key, default)

# 从环境变量或Streamlit Secrets获取配置
AURA_URI = get_env_var("AURA_URI")
AURA_AUTH = (get_env_var("AURA_USER"), get_env_var("AURA_PASSWORD"))

LOCAL_URI = get_env_var("LOCAL_URI")
LOCAL_AUTH = (get_env_var("LOCAL_USER"), get_env_var("LOCAL_PASSWORD"))

def get_driver(use_aura=True):
    """
    获取Neo4j数据库驱动实例
    """
    uri = AURA_URI if use_aura else LOCAL_URI
    auth = AURA_AUTH if use_aura else LOCAL_AUTH
    
    try:
        # 创建驱动实例但不使用with语句
        driver = GraphDatabase.driver(uri, auth=auth)
        
        # 验证连接
        driver.verify_connectivity()
        print("连接验证成功!")
        
        # 测试查询
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            print("查询成功!")
            print("测试查询结果:", result.single()["test"])
        
        st.success("数据库连接成功！")
        return driver
            
    except Exception as e:
        print("连接错误:")
        print(str(e))
        st.error("数据库连接失败")
        st.error(str(e))
        raise e

def get_cached_driver(use_aura=True):
    """
    获取缓存的Neo4j驱动实例
    """
    if "neo4j_driver" not in st.session_state:
        st.session_state.neo4j_driver = get_driver(use_aura)
    return st.session_state.neo4j_driver

def close_driver(driver):
    """
    关闭Neo4j驱动连接
    """
    if driver:
        try:
            driver.close()
        except Exception as e:
            st.error(f"关闭数据库连接时发生错误: {str(e)}")
            print("关闭连接错误:", str(e)) 