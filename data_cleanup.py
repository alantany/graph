from neo4j import GraphDatabase
import pandas as pd

# Neo4j远程连接配置
#AURA_URI = "neo4j+s://b76a61f2.databases.neo4j.io:7687"
#AURA_USERNAME = "neo4j"
#AURA_PASSWORD = "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"
# 本地连接配置
LOCAL_URI = "bolt://localhost:7687"
LOCAL_USERNAME = "test"
LOCAL_PASSWORD = "Mikeno01"

#def connect_to_neo4j():
#    return GraphDatabase.driver(AURA_URI, auth=(AURA_USERNAME, AURA_PASSWORD))
def connect_to_neo4j():
    return GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USERNAME, LOCAL_PASSWORD))

def close_driver(driver):
    driver.close()

def delete_all_data(session):
    session.run("MATCH (n) DETACH DELETE n")
    print("所有数据已删除")

def delete_specific_nodes(session, node_type):
    result = session.run(f"MATCH (n:{node_type}) DETACH DELETE n")
    print(f"已删除 {result.consume().counters.nodes_deleted} 个 {node_type} 节点")

def show_node_counts(session):
    result = session.run("""
    MATCH (n)
    WITH labels(n) AS labels, count(n) AS count
    UNWIND labels AS label
    RETURN label, sum(count) AS count
    """)
    stats = pd.DataFrame([dict(record) for record in result])
    print("节点统计:")
    print(stats)

def main():
    driver = connect_to_neo4j()
    with driver.session() as session:
        while True:
            print("\n选择操作:")
            print("1. 删除所有数据")
            print("2. 删除特定类型的节点")
            print("3. 显示节点统计")
            print("4. 退出")
            
            choice = input("请输入选项 (1-4): ")
            
            if choice == '1':
                confirm = input("确定要删除所有数据吗？(y/n): ")
                if confirm.lower() == 'y':
                    delete_all_data(session)
            elif choice == '2':
                node_type = input("输入要删除的节点类型: ")
                delete_specific_nodes(session, node_type)
            elif choice == '3':
                show_node_counts(session)
            elif choice == '4':
                break
            else:
                print("无效选项，请重新选择")
    
    close_driver(driver)

if __name__ == "__main__":
    main()