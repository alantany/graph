import pandas as pd
import streamlit as st
import traceback

def run_query(driver, query, params=None):
    """
    执行Neo4j查询并返回pandas DataFrame
    
    参数:
        driver: Neo4j驱动实例
        query: Cypher查询语句
        params: 查询参数字典（可选）
    
    返回:
        pandas DataFrame包含查询结果
    """
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            records = result.data()
            if not records:
                return pd.DataFrame()
            return pd.DataFrame(records)
    except Exception as e:
        st.error(f"查询执行失败: {str(e)}")
        st.write("错误详情:")
        st.write(traceback.format_exc())
        return pd.DataFrame()

def format_number(number):
    """
    格式化数字，添加千位分隔符
    
    参数:
        number: 要格式化的数字
    
    返回:
        格式化后的字符串
    """
    return "{:,}".format(number)

def safe_list_get(lst, idx, default=None):
    """
    安全地获取列表元素，如果索引越界则返回默认值
    
    参数:
        lst: 列表
        idx: 索引
        default: 默认值
    
    返回:
        列表元素或默认值
    """
    try:
        return lst[idx]
    except (IndexError, TypeError):
        return default

def validate_date_format(date_str):
    """
    验证日期字符串格式是否正确
    
    参数:
        date_str: 日期字符串
    
    返回:
        bool: 日期格式是否有效
    """
    try:
        pd.to_datetime(date_str)
        return True
    except (ValueError, TypeError):
        return False

def create_date_filter():
    """
    创建日期范围选择器
    
    返回:
        tuple: (start_date, end_date)
    """
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期")
    with col2:
        end_date = st.date_input("结束日期")
    return start_date, end_date

def show_error_message(error, show_traceback=True):
    """
    显示错误信息
    
    参数:
        error: 错误对象或错误消息
        show_traceback: 是否显示详细的错误追踪信息
    """
    st.error(str(error))
    if show_traceback:
        st.write("错误详情:")
        st.write(traceback.format_exc())

def show_success_message(message):
    """
    显示成功消息
    
    参数:
        message: 成功消息
    """
    st.success(message)

def show_info_message(message):
    """
    显示信息消息
    
    参数:
        message: 信息消息
    """
    st.info(message)

def show_warning_message(message):
    """
    显示警告消息
    
    参数:
        message: 警告消息
    """
    st.warning(message) 