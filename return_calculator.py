import os
import pandas as pd
import numpy as np
import logging
import re
from config import PROJECT_BASE_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("return_calculation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ReturnCalculator")

class ReturnCalculator:
    def __init__(self):
        self.price_data_path = os.path.join(PROJECT_BASE_DIR, 'filtered_common_tokens_data', 'price_data', 'token_prices.csv')
        self.output_dir = os.path.join(PROJECT_BASE_DIR, 'return_data')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created directory: {self.output_dir}")
    
    def load_price_data(self):
        """Load price data"""
        if not os.path.exists(self.price_data_path):
            logger.error(f"Price data file not found: {self.price_data_path}")
            return None
        
        try:
            df = pd.read_csv(self.price_data_path)
            # Convert datetime column to datetime format
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"Loaded price data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}")
            return None
    
    def calculate_returns(self, windows=[1, 3, 7, 14, 21, 30, 60], return_types=['simple', 'log']):
        """Calculate returns for different time windows
        
        Args:
            windows (list): List of time windows in days
            return_types (list): List of return types, options are 'simple' (simple returns), 'log' (log returns)
        """
        price_data = self.load_price_data()
        if price_data is None:
            return None
        
        # Set datetime as index
        price_data.set_index('datetime', inplace=True)
        price_data.sort_index(inplace=True)
        
        # Get all token columns
        token_columns = [col for col in price_data.columns if col != 'datetime']
        
        all_returns = {}
        
        # Calculate returns for each time window
        for window in windows:
            logger.info(f"Calculating {window}-day returns")
            
            # Create a new DataFrame to store returns
            returns_df = pd.DataFrame(index=price_data.index)
            
            for col in token_columns:
                token_name = col
                
                # Calculate simple returns
                if 'simple' in return_types:
                    # Calculate 1-day lagged returns
                    if window == 1:
                        # 1-day return = (today's price / yesterday's price) - 1
                        returns_df[f"{token_name}_return_{window}d"] = price_data[col].pct_change(1)
                    else:
                        # n-day return = (today's price / price n days ago) - 1
                        returns_df[f"{token_name}_return_{window}d"] = price_data[col].pct_change(window)
                
                # Calculate log returns
                if 'log' in return_types:
                    # Log return = ln(today's price / price n days ago) = ln(today's price) - ln(price n days ago)
                    if window == 1:
                        returns_df[f"{token_name}_log_return_{window}d"] = np.log(price_data[col]).diff(1)
                    else:
                        returns_df[f"{token_name}_log_return_{window}d"] = np.log(price_data[col]).diff(window)
            
            # Remove first window rows (no return data)
            returns_df = returns_df.iloc[window:]
            
            all_returns[window] = returns_df
            
            # Save return data
            output_path = os.path.join(self.output_dir, f"returns_{window}d.csv")
            returns_df.reset_index().to_csv(output_path, index=False)
            logger.info(f"Saved {window}-day returns to {output_path}")
        
        return all_returns
    

    
    def calculate_market_return(self, windows=[1, 3, 7, 14, 21, 30, 60]):
        """Calculate market returns
        
        Calculation logic:
        1. Load return data for all tokens
        2. For each trading day, calculate the average return of all tokens as equal-weighted market return
        3. Save calculation results as CSV files
        
        Args:
            windows (list): List of time windows in days
        
        Returns:
            market_returns_dict: Dictionary containing market returns for different time windows
        """
        logger.info("Starting market return calculation")
        
        # Create market return save directory
        market_return_dir = os.path.join(PROJECT_BASE_DIR, 'return_data', 'market_returns')
        if not os.path.exists(market_return_dir):
            os.makedirs(market_return_dir, exist_ok=True)
            logger.info(f"Created market return directory: {market_return_dir}")
        
        market_returns_dict = {}
        
        # Calculate market returns for each time window
        for window in windows:
            logger.info(f"Calculating {window}-day market returns")
            
            # Load return data
            return_file_path = os.path.join(self.output_dir, f"returns_{window}d.csv")
            if not os.path.exists(return_file_path):
                logger.error(f"Return data file not found: {return_file_path}")
                continue
            
            try:
                # Read return data
                return_data = pd.read_csv(return_file_path)
                return_data['date'] = pd.to_datetime(return_data['datetime'])
                
                # Extract return columns for all tokens
                return_cols = [col for col in return_data.columns if col.endswith(f"_log_return_{window}d")]
                
                if not return_cols:
                    logger.warning(f"No return columns found for {window}-day window")
                    continue
                
                # Calculate average return for each date (equal-weighted market return)
                market_returns = pd.DataFrame()
                market_returns['date'] = return_data['date']
                
                # Calculate average return of all tokens
                market_returns['market_return'] = return_data[return_cols].mean(axis=1)
                
                # Save market returns
                save_path = os.path.join(market_return_dir, f"market_log_return_{window}d.csv")
                market_returns.to_csv(save_path, index=False)
                logger.info(f"Saved {window}-day market returns to {save_path}")
                
                market_returns_dict[window] = market_returns
                
            except Exception as e:
                logger.error(f"Error calculating {window}-day market returns: {str(e)}")
        
        return market_returns_dict
    
    def run(self, windows=[1, 3, 7, 14, 21, 30, 60]):
        """Run return calculation
        
        Args:
            windows (list): List of time windows in days
        """
        logger.info("Starting return calculation")
        
        # Calculate simple returns and log returns
        return_types = ['simple', 'log']
        
        all_returns = self.calculate_returns(windows=windows, return_types=return_types)
        
        if all_returns:
            logger.info("Return calculation completed")
            
            # Calculate market returns
            market_returns = self.calculate_market_return(windows=windows)
            if market_returns:
                logger.info("Market return calculation completed")
            else:
                logger.error("Market return calculation failed")
            
            return True
        else:
            logger.error("Return calculation failed")
            return False

if __name__ == "__main__":
    calculator = ReturnCalculator()
    calculator.run()