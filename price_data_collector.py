import os
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import logging
from config import SANTIMENT_API_KEY, SANTIMENT_GRAPHQL_URL, PROJECT_BASE_DIR

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("price_data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PriceDataCollector")

class PriceDataCollector:
    def __init__(self):
        self.api_key = SANTIMENT_API_KEY
        self.graphql_url = SANTIMENT_GRAPHQL_URL
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Apikey {self.api_key}'
        }
        self.output_dir = os.path.join(PROJECT_BASE_DIR, 'price_data')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created directory: {self.output_dir}")
    
    def load_common_tokens(self):
        """加载在所有数据集中都存在的token列表"""
        common_tokens_path = os.path.join(PROJECT_BASE_DIR, 'cleaned_data', 'common_tokens.txt')
        tokens = []
        in_token_list = False
        
        with open(common_tokens_path, 'r') as f:
            for line in f:
                if "在所有数据集中都存在的Token列表:" in line:
                    in_token_list = True
                    continue
                if in_token_list and line.strip() and line.startswith('-'):
                    token = line.strip().replace('- ', '')
                    tokens.append(token)
        
        logger.info(f"Loaded {len(tokens)} common tokens")
        return tokens
    
    def fetch_price_data(self, slug, from_date, to_date):
        """获取指定token的价格数据"""
        query = '''
        {
          getMetric(metric: "price_usd") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        ''' % (slug, from_date, to_date)
        
        try:
            response = requests.post(
                self.graphql_url,
                json={'query': query},
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'errors' in data:
                    logger.error(f"Error fetching data for {slug}: {data['errors']}")
                    return None
                
                timeseries = data.get('data', {}).get('getMetric', {}).get('timeseriesData', [])
                if not timeseries:
                    logger.warning(f"No price data found for {slug}")
                    return None
                
                df = pd.DataFrame(timeseries)
                df['slug'] = slug
                return df
            else:
                logger.error(f"Request failed with status code {response.status_code} for {slug}")
                return None
        except Exception as e:
            logger.error(f"Exception when fetching data for {slug}: {str(e)}")
            return None
    
    def collect_price_data(self, from_date, to_date):
        """收集所有token的价格数据"""
        tokens = self.load_common_tokens()
        all_data = []
        
        for i, token in enumerate(tokens):
            logger.info(f"Fetching price data for {token} ({i+1}/{len(tokens)})")
            df = self.fetch_price_data(token, from_date, to_date)
            
            if df is not None:
                all_data.append(df)
            
            # 避免API速率限制，每10个请求暂停5秒
            if (i + 1) % 10 == 0 and i < len(tokens) - 1:
                logger.info("Pausing to avoid API rate limits...")
                time.sleep(5)
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data)
            
            # 将数据透视为宽格式，每个token一列
            pivot_df = combined_df.pivot(index='datetime', columns='slug', values='value')
            
            # 重置索引，使datetime成为一个列
            pivot_df.reset_index(inplace=True)
            
            # 确保列名为token slug，不添加后缀
            pivot_df.columns.name = None
            
            # 保存到CSV
            output_path = os.path.join(self.output_dir, 'token_prices.csv')
            pivot_df.to_csv(output_path, index=False)
            logger.info(f"Saved price data to {output_path}")
            
            return pivot_df
        else:
            logger.warning("No data collected for any token")
            return None

if __name__ == "__main__":
    collector = PriceDataCollector()
    
    # 设置日期范围：从2023年6月23日到2025年6月21日
    from_date = "2023-06-23T00:00:00Z"
    to_date = "2025-06-21T23:59:59Z"
    
    logger.info(f"Starting price data collection from {from_date} to {to_date}")
    price_data = collector.collect_price_data(from_date, to_date)
    
    if price_data is not None:
        logger.info(f"Successfully collected price data for {len(price_data.columns) - 1} tokens")
        logger.info(f"Data shape: {price_data.shape}")
    else:
        logger.error("Failed to collect price data")