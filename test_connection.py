from neo4j import GraphDatabase

# Aura连接配置
URI = "neo4j+s://85c689ad.databases.neo4j.io"
AUTH = ("neo4j", "XL37Q-0UhF1YA2diY3f9Ah3dLxHmWlyoN6rexDu9sdA")

def test_connection():
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            # 验证连接
            driver.verify_connectivity()
            print("连接验证成功!")
            
            # 测试查询
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                print("查询成功!")
                print("测试查询结果:", result.single()["test"])
    except Exception as e:
        print("连接错误:")
        print(str(e))

if __name__ == "__main__":
    test_connection() 