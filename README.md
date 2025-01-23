# Neo4j图数据分析应用

这是一个基于Neo4j图数据库的数据分析应用，使用Streamlit构建Web界面，支持金融风控、社交网络和医疗健康等多个场景的数据分析。

## 环境配置

### 本地开发环境

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 配置数据库连接：
- 复制 `.env.example` 文件并重命名为 `.env`
- 在 `.env` 文件中填入你的数据库连接信息：
```
# Neo4j Aura数据库连接配置
AURA_DB_ID=your_database_id
AURA_URI=neo4j+ssc://${AURA_DB_ID}.databases.neo4j.io
AURA_USER=neo4j
AURA_PASSWORD=your_password

# 本地Neo4j数据库连接配置
LOCAL_URI=bolt://localhost:7687
LOCAL_USER=neo4j
LOCAL_PASSWORD=your_password
```

3. 运行应用：
```bash
streamlit run app.py
```

### Streamlit Cloud部署

1. 将代码推送到GitHub仓库

2. 在Streamlit Cloud中配置环境变量：
- 登录 [Streamlit Cloud](https://share.streamlit.io)
- 选择你的应用
- 点击右上角的三个点，选择 "Settings"
- 在 "Secrets" 部分，添加以下配置：
```yaml
AURA_URI: "neo4j+ssc://your_database_id.databases.neo4j.io"
AURA_USER: "neo4j"
AURA_PASSWORD: "your_password"
LOCAL_URI: "bolt://localhost:7687"
LOCAL_USER: "neo4j"
LOCAL_PASSWORD: "your_password"
```

注意：
- Streamlit Cloud的Secrets配置使用YAML格式
- 不要将实际的数据库密码提交到Git仓库
- 确保在本地开发时使用 `.env` 文件，在Streamlit Cloud中使用Secrets配置

## 常见问题

### 数据库连接问题

1. SSL证书验证失败
- 确保使用正确的连接协议（neo4j+s或neo4j+ssc）
- 检查数据库实例是否在线
- 验证连接URI格式是否正确

2. 连接超时
- 检查网络连接
- 确认数据库实例是否处于活动状态
- 验证防火墙设置

3. 认证失败
- 检查用户名和密码是否正确
- 确认数据库实例是否已启动
- 验证连接URI是否包含正确的数据库ID

## 项目结构

```
.
├── app.py                 # 主应用入口
├── modules/              # 模块目录
│   ├── __init__.py      # 模块初始化文件
│   ├── db_config.py     # 数据库配置
│   ├── financial.py     # 金融风控模块
│   ├── healthcare.py    # 医疗健康模块
│   └── social.py        # 社交网络模块
├── data/                # 数据目录
├── .env.example        # 环境变量示例
├── requirements.txt    # 依赖包列表
└── README.md          # 项目说明文档
```

## 数据导入

应用支持从CSV文件导入数据到Neo4j数据库。每个分析场景（金融、医疗、社交）都有其对应的数据导入功能。导入前请确保：

1. 数据文件格式正确
2. 文件编码为UTF-8
3. 数据库连接正常
4. 有足够的数据库写入权限

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License
