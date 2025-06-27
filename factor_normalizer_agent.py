import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List, Optional

class FactorNormalizerAgent:
    """
    Enhanced factor normalization agent class for standardizing different types of factor data and batch processing
    
    Supported normalization methods:
    - 'zscore': Cross-sectional z-score normalization
    - 'log+zscore': log(x+1) followed by z-score
    - 'rank': Cross-sectional percentile ranking (rank(pct=True))
    - 'log+rank': log(x+1) followed by cross-sectional rank
    
    Features:
    1. Single factor normalization
    2. Batch factor normalization
    3. Automatic saving of normalization results
    4. Support for different analysis type normalization method configurations
    """
    
    def __init__(self, factor_dict: Optional[Dict[str, pd.DataFrame]] = None, method_dict: Optional[Dict[str, str]] = None):
        """
        Initialize factor normalization agent
        
        Parameters:
        - factor_dict: Dictionary, key is factor name, value is DataFrame (index is date, columns are tokens)
        - method_dict: Dictionary, key is factor name, value is processing method ('zscore', 'log+zscore', 'rank', 'log+rank')
        """
        self.factor_dict = factor_dict or {}
        self.method_dict = method_dict or {}
        self.data_dir = "/Users/queenwen/Desktop/QI_Paper/filtered_common_tokens_data"
        self.output_base_dir = "/Users/queenwen/Desktop/QI_Paper/normalized_factors"
        
        # Validate inputs
        if factor_dict and method_dict:
            self._validate_inputs()
    
    def _validate_inputs(self):
        """
        Validate the validity of input data
        """
        # Check if all methods in method_dict are supported
        valid_methods = ['zscore', 'log+zscore', 'rank', 'log+rank']
        for factor_name, method in self.method_dict.items():
            if method not in valid_methods:
                raise ValueError(f"Unsupported normalization method: {method}, factor: {factor_name}")
            
            # Check if factor_name exists in factor_dict
            if factor_name not in self.factor_dict:
                raise ValueError(f"Factor {factor_name} does not exist in factor_dict")
    
    def _apply_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional z-score normalization to DataFrame
        
        Parameters:
        - df: Input DataFrame
        
        Returns:
        - Normalized DataFrame
        """
        # Apply cross-sectional normalization to each row (each date)
        return df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log(x+1) transformation to DataFrame
        
        Parameters:
        - df: Input DataFrame
        
        Returns:
        - Transformed DataFrame
        """
        # Apply log(x+1) transformation to all values
        return np.log1p(df)
    
    def _apply_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional percentile ranking to DataFrame
        
        Parameters:
        - df: Input DataFrame
        
        Returns:
        - Ranked DataFrame
        """
        # Apply cross-sectional percentile ranking to each row (each date)
        return df.rank(axis=1, pct=True)
    
    def normalize_factor(self, factor_name: str) -> Tuple[pd.DataFrame, bool]:
        """
        Normalize specified factor
        
        Parameters:
        - factor_name: Factor name
        
        Returns:
        - Tuple[pd.DataFrame, bool]: Normalized DataFrame and success flag
        """
        # Check if factor exists
        if factor_name not in self.factor_dict:
            print(f"Error: Factor {factor_name} not in factor_dict")
            return pd.DataFrame(), False
        
        # Check if normalization method exists
        if factor_name not in self.method_dict:
            print(f"Error: Factor {factor_name} has no specified normalization method")
            return pd.DataFrame(), False
        
        # Get original data and normalization method
        df = self.factor_dict[factor_name].copy()
        method = self.method_dict[factor_name]
        
        # Apply corresponding normalization method
        try:
            if method == 'zscore':
                normalized_df = self._apply_zscore(df)
            elif method == 'log+zscore':
                log_df = self._apply_log_transform(df)
                normalized_df = self._apply_zscore(log_df)
            elif method == 'rank':
                normalized_df = self._apply_rank(df)
            elif method == 'log+rank':
                log_df = self._apply_log_transform(df)
                normalized_df = self._apply_rank(log_df)
            else:
                print(f"Error: Unsupported normalization method {method}")
                return pd.DataFrame(), False
            
            return normalized_df, True
        except Exception as e:
            print(f"Error normalizing factor {factor_name}: {str(e)}")
            return pd.DataFrame(), False
    
    def load_all_factors(self):
        """
        Load all factor data
        
        Returns:
        - factor_dict: Factor dictionary
        - group1_factors: First group factor list
        - group2_factors: Second group factor list
        """
        # Initialize factor dictionary
        factor_dict = {}
        
        # First group factors
        group1_factors = [
            # Development activity data
            ("dev_activity/dev_activity_1d.csv", "dev_activity_1d"),
            ("dev_activity/dev_activity_contributors_count.csv", "dev_activity_contributors_count"),
            ("dev_activity/github_activity_1d.csv", "github_activity_1d"),
            # Social data
            ("sentiment/social_volume_total.csv", "social_volume_total"),
            # Exchange inflow/outflow data
            ("supply_demand/exchange_inflow_usd.csv", "exchange_inflow_usd"),
            ("supply_demand/exchange_outflow_usd.csv", "exchange_outflow_usd"),
            # Whale effect data
            # ("whale_effect/absolute_amount_in_top_holders.csv", "absolute_amount_in_top_holders"),
            ("whale_effect/whale_transaction_count_100k_usd_to_inf.csv", "whale_transaction_count_100k_usd_to_inf"),
            ("whale_effect/whale_transaction_volume_100k_usd_to_inf.csv", "whale_transaction_volume_100k_usd_to_inf")
        ]
        
        # Second group factors
        group2_factors = [
            # Sentiment data
            ("sentiment/sentiment_weighted_total_1d.csv", "sentiment_weighted_total_1d"),
            # Whale effect data
            ("whale_effect/amount_in_top_holders.csv", "amount_in_top_holders"),
            # Exchange balance data
            ("supply_demand/exchange_balance.csv", "exchange_balance")
        ]
        
        # Load all factors
        all_factors = group1_factors + group2_factors
        
        for file_path, factor_name in all_factors:
            full_path = os.path.join(self.data_dir, file_path)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path)
                df.set_index('datetime', inplace=True)
                
                # Reshape DataFrame to make tokens as columns
                tokens = [col.replace(f"_{factor_name}", "") for col in df.columns]
                df.columns = tokens
                factor_dict[factor_name] = df
                print(f"Loaded factor: {factor_name}, shape: {df.shape}")
            else:
                print(f"Warning: File does not exist {full_path}")
        
        self.factor_dict = factor_dict
        return factor_dict, group1_factors, group2_factors
    
    def create_output_directories(self):
        """
        Create output directories
        
        Returns:
        - List of output directories
        """
        output_dirs = [
            os.path.join(self.output_base_dir, "IC_analysis"),
            os.path.join(self.output_base_dir, "Quantile_analysis"),
            os.path.join(self.output_base_dir, "CAPM_regression")
        ]
        
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        return output_dirs
    
    def save_normalized_factor(self, normalized_df, factor_name, output_dir):
        """
        Save normalized factor data
        
        Parameters:
        - normalized_df: Normalized DataFrame
        - factor_name: Factor name
        - output_dir: Output directory
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output file path
        output_path = os.path.join(output_dir, f"{factor_name}.csv")
        
        # Save data
        normalized_df.to_csv(output_path)
        print(f"Saved normalized factor to: {output_path}")
    
    def batch_normalize_factors(self):
        """
        normalize factors for IC analysis, quantile analysis and CAPM regression
        """
        # load all factors
        factor_dict, group1_factors, group2_factors = self.load_all_factors()
        
        # create output directories
        ic_analysis_dir, quantile_analysis_dir, capm_regression_dir = self.create_output_directories()
        
        # extract token names
        group1_names = [name for _, name in group1_factors]
        group2_names = [name for _, name in group2_factors]
        
        # setup normalization methods for IC analysis
        ic_method_dict = {}
        for factor_name in group1_names:
            if factor_name in factor_dict:
                ic_method_dict[factor_name] = 'log+zscore'  # group1: log(x+1) -> z-score
        
        for factor_name in group2_names:
            if factor_name in factor_dict:
                ic_method_dict[factor_name] = 'zscore'  # group 2: z-score
        
        # set up normalization methods for quantile analysis
        quantile_method_dict = {}
        for factor_name in group1_names:
            if factor_name in factor_dict:
                quantile_method_dict[factor_name] = 'log+rank'  # group 1: log(x+1) -> rank
        
        for factor_name in group2_names:
            if factor_name in factor_dict:
                quantile_method_dict[factor_name] = 'rank'  # group2: rank
        
        # set up normalization methods for CAPM regression
        capm_method_dict = {}
        for factor_name in group1_names:
            if factor_name in factor_dict:
                capm_method_dict[factor_name] = 'log+zscore'  # group 1: log(x+1) -> z-score
        
        for factor_name in group2_names:
            if factor_name in factor_dict:
                capm_method_dict[factor_name] = 'zscore'  # Group 2: z-score
        
        # 1. Process factors for IC analysis
        print("\nProcessing factors for IC analysis...")
        self.method_dict = ic_method_dict
        
        for factor_name in factor_dict.keys():
            if factor_name in ic_method_dict:
                print(f"\nNormalizing factor: {factor_name}, method: {ic_method_dict[factor_name]}")
                normalized_df, success = self.normalize_factor(factor_name)
                
                if success:
                    print(f"Normalization successful!")
                    
                    # Special handling: convert exchange_outflow_usd to negative
                    if factor_name == 'exchange_outflow_usd':
                        normalized_df = -normalized_df
                        print(f"Converted {factor_name} to negative values")
                    
                    self.save_normalized_factor(normalized_df, factor_name, ic_analysis_dir)
                    
                    # Validate normalization results
                    if ic_method_dict[factor_name] in ['zscore', 'log+zscore']:
                        row_means = normalized_df.mean(axis=1)
                        row_stds = normalized_df.std(axis=1)
                        print(f"Cross-sectional mean range: [{row_means.min():.6f}, {row_means.max():.6f}]")
                        print(f"Cross-sectional std range: [{row_stds.min():.6f}, {row_stds.max():.6f}]")
                else:
                    print(f"Normalization failed!")
        
        # 2. Process factors for quantile analysis
        print("\nProcessing factors for quantile analysis...")
        self.method_dict = quantile_method_dict
        
        for factor_name in factor_dict.keys():
            if factor_name in quantile_method_dict:
                print(f"\nNormalizing factor: {factor_name}, method: {quantile_method_dict[factor_name]}")
                normalized_df, success = self.normalize_factor(factor_name)
                
                if success:
                    print(f"Normalization successful!")
                    
                    # Special handling: convert exchange_outflow_usd to negative
                    if factor_name == 'exchange_outflow_usd':
                        normalized_df = -normalized_df
                        print(f"Converted {factor_name} to negative values")
                    
                    self.save_normalized_factor(normalized_df, factor_name, quantile_analysis_dir)
                    
                    # Validate normalization results
                    if quantile_method_dict[factor_name] in ['rank', 'log+rank']:
                        print(f"Minimum value: {normalized_df.min().min():.6f}")
                        print(f"Maximum value: {normalized_df.max().max():.6f}")
                else:
                    print(f"Normalization failed!")
        
        # 3. Process factors for CAPM regression
        print("\nProcessing factors for CAPM regression...")
        self.method_dict = capm_method_dict
        
        for factor_name in factor_dict.keys():
            if factor_name in capm_method_dict:
                print(f"\nNormalizing factor: {factor_name}, method: {capm_method_dict[factor_name]}")
                normalized_df, success = self.normalize_factor(factor_name)
                
                if success:
                    print(f"Normalization successful!")
                    
                    # Special handling: convert exchange_outflow_usd to negative
                    if factor_name == 'exchange_outflow_usd':
                        normalized_df = -normalized_df
                        print(f"Converted {factor_name} to negative values")
                    
                    self.save_normalized_factor(normalized_df, factor_name, capm_regression_dir)
                    
                    # Validate normalization results
                    if capm_method_dict[factor_name] in ['zscore', 'log+zscore']:
                        row_means = normalized_df.mean(axis=1)
                        row_stds = normalized_df.std(axis=1)
                        print(f"Cross-sectional mean range: [{row_means.min():.6f}, {row_means.max():.6f}]")
                        print(f"Cross-sectional std range: [{row_stds.min():.6f}, {row_stds.max():.6f}]")
                else:
                    print(f"Normalization failed!")
    

def run_batch_process():
    """
    Run batch processing functionality
    """
    # Create FactorNormalizerAgent instance
    agent = FactorNormalizerAgent()
    
    # Batch normalize factors
    agent.batch_normalize_factors()


if __name__ == "__main__":
    import sys
    
    # Set pandas display options
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    
    # Run batch processing mode directly
    print("Running batch processing mode...")
    run_batch_process()