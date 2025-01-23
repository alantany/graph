from neo4j import GraphDatabase
import time

# Aura连接配置
AUTH = ("neo4j", "XL37Q-0UhF1YA2diY3f9Ah3dLxHmWlyoN6rexDu9sdA")
DB_ID = "85c689ad"

# 不同的URI格式
URIS = [
    f"neo4j+s://{DB_ID}.databases.neo4j.io",           # 标准格式
    f"bolt+s://{DB_ID}.databases.neo4j.io:7687",       # bolt格式带端口
    f"bolt://{DB_ID}.databases.neo4j.io:7687",         # 普通bolt格式
    f"neo4j://{DB_ID}.databases.neo4j.io:7687",        # neo4j格式带端口
    f"bolt+s://{DB_ID}.databases.neo4j.io",            # bolt+s格式不带端口
    f"neo4j+ssc://{DB_ID}.databases.neo4j.io",         # neo4j+ssc格式
]

def test_connection(uri):
    print(f"\n尝试连接: {uri}")
    try:
        with GraphDatabase.driver(uri, auth=AUTH) as driver:
            # 验证连接
            driver.verify_connectivity()
            print("连接验证成功!")
            
            # 测试查询
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                print("查询成功!")
                print("测试查询结果:", result.single()["test"])
            return True
    except Exception as e:
        print("连接错误:")
        print(str(e))
        return False

def test_all_connections():
    successful_uris = []
    
    for uri in URIS:
        if test_connection(uri):
            successful_uris.append(uri)
        time.sleep(1)  # 等待1秒再尝试下一个连接
    
    print("\n测试结果汇总:")
    if successful_uris:
        print("成功的连接URI:")
        for uri in successful_uris:
            print(f"- {uri}")
    else:
        print("所有连接方式都失败了")

if __name__ == "__main__":
    test_all_connections() 