import os # 确保导入 os 模块

# 输出文件名
OUTPUT_FILE = "active_defi_tokens.csv"

# Santiment API 密钥
SANTIMENT_API_KEY = "YOUR SANTIMENT API KEY"

# GraphQL API 端点
SANTIMENT_GRAPHQL_URL = "https://api.santiment.net/graphql"

# 项目基本路径
# 获取 config.py 文件所在的目录的绝对路径
CONFIG_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# # PROJECT_BASE_DIR 应该是 CONFIG_FILE_DIR 的上两级目录 (即 "DeFi" 文件夹)
# PROJECT_BASE_DIR = os.path.abspath(os.path.join(CONFIG_FILE_DIR, "..", ".."))
PROJECT_BASE_DIR = CONFIG_FILE_DIR

OUTPUT_BASE_DIR = os.path.join(PROJECT_BASE_DIR,'multifactor_strategy_outputs')

# (可选) 确保输出目录存在，如果不存在则创建
if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR)
    print(f"Created directory: {OUTPUT_BASE_DIR}")
