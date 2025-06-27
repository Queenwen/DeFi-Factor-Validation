#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Correlation Analysis
Generate IC correlation matrix and quantile return comparison charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set font for plotting
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class FactorCorrelationAnalyzer:
    def __init__(self, normalized_factors_dir, return_data_dir, output_dir):
        self.normalized_factors_dir = normalized_factors_dir
        self.return_data_dir = return_data_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Factor name mapping
        self.factor_name_mapping = {
            'amount_in_top_holders': 'Large Holder Amount',
            'dev_activity_1d': 'Development Activity',
            'dev_activity_contributors_count': 'Developer Count',
            'exchange_balance': 'Exchange Balance',
            'exchange_inflow_usd': 'Exchange Inflow',
            'exchange_outflow_usd': 'Exchange Outflow',
            'github_activity_1d': 'GitHub Activity',
            'sentiment_weighted_total_1d': 'Weighted Sentiment',
            'social_volume_total': 'Social Volume',
            'whale_transaction_count_100k_usd_to_inf': 'Whale Transaction Count',
            'whale_transaction_volume_100k_usd_to_inf': 'Whale Transaction Volume'
        }
    
    def load_factor_data(self, analysis_type='IC_analysis'):
        """Load factor data"""
        factor_dir = os.path.join(self.normalized_factors_dir, analysis_type)
        factor_data = {}
        
        for file in os.listdir(factor_dir):
            if file.endswith('.csv'):
                factor_name = file.replace('.csv', '')
                file_path = os.path.join(factor_dir, file)
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    factor_data[factor_name] = df
                    print(f"Loaded factor data: {factor_name}, shape: {df.shape}")
                except Exception as e:
                    print(f"Failed to load factor data {file}: {e}")
        
        return factor_data
    
    def load_return_data(self, period='7d'):
        """Load return data"""
        return_file = os.path.join(self.return_data_dir, f'returns_{period}.csv')
        try:
            df = pd.read_csv(return_file, index_col=0, parse_dates=True)
            print(f"Loaded return data: {period}, shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Failed to load return data {return_file}: {e}")
            return None
    
    def calculate_ic_matrix(self, factor_data, return_data, method='spearman'):
        """Calculate IC correlation matrix"""
        ic_results = {}
        
        for factor_name, factor_df in factor_data.items():
            ic_results[factor_name] = {}
            
            # Align data
            common_dates = factor_df.index.intersection(return_data.index)
            common_tokens = set()
            
            # Find common tokens - factor data columns are token symbols, return data columns contain _log_return suffix
            for factor_col in factor_df.columns:
                # Search for corresponding columns in return data
                for return_col in return_data.columns:
                    if factor_col in return_col and '_log_return' in return_col:
                        common_tokens.add((factor_col, return_col))
                        break
            
            if len(common_dates) == 0 or len(common_tokens) == 0:
                print(f"Warning: {factor_name} has no common dates or tokens")
                continue
            
            # Calculate IC for each token
            token_ics = []
            for factor_col, return_col in common_tokens:
                factor_values = factor_df.loc[common_dates, factor_col].dropna()
                return_values = return_data.loc[common_dates, return_col].dropna()
                
                # Align data
                common_idx = factor_values.index.intersection(return_values.index)
                if len(common_idx) < 10:  # Need at least 10 observations
                    continue
                
                factor_aligned = factor_values.loc[common_idx]
                return_aligned = return_values.loc[common_idx]
                
                # Calculate correlation
                if method == 'spearman':
                    corr, p_value = spearmanr(factor_aligned, return_aligned)
                else:
                    corr, p_value = pearsonr(factor_aligned, return_aligned)
                
                if not np.isnan(corr):
                    token_ics.append(corr)
            
            # Calculate average IC
            if token_ics:
                ic_results[factor_name]['mean_ic'] = np.mean(token_ics)
                ic_results[factor_name]['std_ic'] = np.std(token_ics)
                ic_results[factor_name]['ir'] = np.mean(token_ics) / np.std(token_ics) if np.std(token_ics) > 0 else 0
                ic_results[factor_name]['count'] = len(token_ics)
            else:
                ic_results[factor_name]['mean_ic'] = 0
                ic_results[factor_name]['std_ic'] = 0
                ic_results[factor_name]['ir'] = 0
                ic_results[factor_name]['count'] = 0
        
        return ic_results
    
    def calculate_ic_series_correlation(self, factor_data, return_data, method='spearman'):
        """Generate correlation matrix between factor IC series"""
        ic_series = {}
        
        for factor_name, factor_df in factor_data.items():
            daily_ic_list = []
            
            for date in factor_df.index.intersection(return_data.index):
                factor_vals = factor_df.loc[date]
                returns = return_data.loc[date]
                
                token_pairs = [(f, r) for f in factor_vals.index for r in returns.index if f in r]
                
                ic_vals = []
                for factor_col, return_col in token_pairs:
                    if factor_col not in factor_vals or return_col not in returns:
                        continue
                    f_val = factor_vals[factor_col]
                    r_val = returns[return_col]
                    if np.isnan(f_val) or np.isnan(r_val):
                        continue
                    ic_vals.append((f_val, r_val))
                
                if len(ic_vals) >= 5:
                    f_arr, r_arr = zip(*ic_vals)
                    if method == 'spearman':
                        ic, _ = spearmanr(f_arr, r_arr)
                    else:
                        ic, _ = pearsonr(f_arr, r_arr)
                    if not np.isnan(ic):
                        daily_ic_list.append(ic)
            
            ic_series[factor_name] = pd.Series(daily_ic_list)
        
        # Construct IC series DataFrame
        ic_df = pd.DataFrame(ic_series)
        corr_matrix = ic_df.corr(method=method)
        return corr_matrix
    
    def create_ic_correlation_matrix(self, factor_data, return_periods=['1d', '3d', '7d', '14d', '21d', '30d','60d']):
        """Create IC correlation matrix"""
        ic_matrix = pd.DataFrame()
        
        for period in return_periods:
            return_data = self.load_return_data(period)
            if return_data is None:
                continue
            
            ic_results = self.calculate_ic_matrix(factor_data, return_data)
            
            # Extract average IC values
            period_ics = {}
            for factor_name, results in ic_results.items():
                period_ics[factor_name] = results['mean_ic']
            
            ic_matrix[period] = pd.Series(period_ics)
        
        return ic_matrix
    

    
    def plot_ic_series_correlation_heatmap(self, ic_series_corr_matrix, title="Factor IC Series Correlation Matrix"):
        """Plot IC series correlation heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Use English factor names
        ic_corr_en = ic_series_corr_matrix.copy()
        ic_corr_en.index = [self.factor_name_mapping.get(idx, idx) for idx in ic_corr_en.index]
        ic_corr_en.columns = [self.factor_name_mapping.get(col, col) for col in ic_corr_en.columns]
        
        # Create heatmap
        sns.heatmap(ic_corr_en, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.3f',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Factor', fontsize=12)
        plt.ylabel('Factor', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'ic_series_correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def calculate_quantile_returns(self, factor_data, return_data, quantiles=5):
        """Calculate quantile returns"""
        quantile_results = {}
        
        for factor_name, factor_df in factor_data.items():
            # Align data
            common_dates = factor_df.index.intersection(return_data.index)
            
            # Find common token pairs
            common_token_pairs = []
            for factor_col in factor_df.columns:
                for return_col in return_data.columns:
                    if factor_col in return_col and '_log_return' in return_col:
                        common_token_pairs.append((factor_col, return_col))
                        break
            
            if len(common_dates) == 0 or len(common_token_pairs) == 0:
                continue
            
            # Collect all observations
            factor_values = []
            return_values = []
            
            for date in common_dates:
                for factor_col, return_col in common_token_pairs:
                    if (factor_col in factor_df.columns and return_col in return_data.columns and
                        date in factor_df.index and date in return_data.index):
                        
                        factor_val = factor_df.loc[date, factor_col]
                        return_val = return_data.loc[date, return_col]
                        
                        if not (np.isnan(factor_val) or np.isnan(return_val)):
                            factor_values.append(factor_val)
                            return_values.append(return_val)
            
            if len(factor_values) < quantiles * 10:  # At least 10 observations per quantile
                continue
            
            # Calculate quantiles
            factor_array = np.array(factor_values)
            return_array = np.array(return_values)
            
            # Sort by factor values
            sorted_indices = np.argsort(factor_array)
            sorted_returns = return_array[sorted_indices]
            
            # Group and calculate average returns
            group_size = len(sorted_returns) // quantiles
            quantile_returns = []
            
            for i in range(quantiles):
                start_idx = i * group_size
                if i == quantiles - 1:  # Last group contains all remaining data
                    end_idx = len(sorted_returns)
                else:
                    end_idx = (i + 1) * group_size
                
                group_returns = sorted_returns[start_idx:end_idx]
                quantile_returns.append(np.mean(group_returns))
            
            quantile_results[factor_name] = quantile_returns
        
        return quantile_results
    
    def plot_quantile_returns(self, quantile_results, return_period='7d', quantiles=5):
        """Plot quantile returns comparison chart"""
        if not quantile_results:
            print("No quantile return data to plot")
            return None
        
        # Calculate chart layout
        n_factors = len(quantile_results)
        cols = 3
        rows = (n_factors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        factor_names = list(quantile_results.keys())
        
        for i, factor_name in enumerate(factor_names):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            returns = quantile_results[factor_name]
            x_labels = [f'Q{j+1}' for j in range(quantiles)]
            
            # Plot bar chart
            bars = ax.bar(x_labels, returns, 
                         color=['red' if r < 0 else 'green' for r in returns],
                         alpha=0.7)
            
            # Add value labels
            for bar, ret in zip(bars, returns):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{ret:.3f}',
                       ha='center', va='bottom' if height >= 0 else 'top')
            
            # Set title and labels
            en_name = self.factor_name_mapping.get(factor_name, factor_name)
            ax.set_title(f'{en_name} - {return_period} Return Quantiles', fontweight='bold')
            ax.set_xlabel('Quantile')
            ax.set_ylabel('Average Return')
            ax.grid(True, alpha=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_factors, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Factor Quantile Returns Comparison ({return_period})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'quantile_returns_{return_period}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_correlation_summary(self, ic_matrix, quantile_results):
        """Generate correlation analysis summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("Factor Correlation Analysis Summary")
        summary.append("=" * 60)
        summary.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Number of Factors: {len(ic_matrix)}")
        summary.append("")
        
        # IC analysis summary
        summary.append("1. IC Correlation Analysis:")
        summary.append("-" * 30)
        
        for period in ic_matrix.columns:
            summary.append(f"\n{period} Return Period:")
            period_data = ic_matrix[period].dropna().sort_values(ascending=False)
            
            summary.append(f"  Strongest Positive Correlation Factors:")
            for i, (factor, ic) in enumerate(period_data.head(3).items()):
                en_name = self.factor_name_mapping.get(factor, factor)
                summary.append(f"    {i+1}. {en_name}: {ic:.4f}")
            
            summary.append(f"  Strongest Negative Correlation Factors:")
            for i, (factor, ic) in enumerate(period_data.tail(3).items()):
                en_name = self.factor_name_mapping.get(factor, factor)
                summary.append(f"    {i+1}. {en_name}: {ic:.4f}")
        
        # Quantile analysis summary
        summary.append("\n\n2. Quantile Return Analysis:")
        summary.append("-" * 30)
        
        for factor_name, returns in quantile_results.items():
            en_name = self.factor_name_mapping.get(factor_name, factor_name)
            spread = returns[-1] - returns[0]  # Highest quantile - Lowest quantile
            summary.append(f"\n{en_name}:")
            summary.append(f"  Quantile Return Spread: {spread:.4f}")
            summary.append(f"  Q1 Return: {returns[0]:.4f}")
            summary.append(f"  Q5 Return: {returns[-1]:.4f}")
        
        summary_text = "\n".join(summary)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'correlation_analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(summary_text)
        return summary_path
    
    def run_analysis(self, return_periods=['1d','3d', '7d', '14d', '21d', '30d', '60d']):
        """Run complete correlation analysis"""
        print("Starting factor correlation analysis...")
        
        # Load factor data
        factor_data = self.load_factor_data('IC_analysis')
        if not factor_data:
            print("Error: Unable to load factor data")
            return
        
        # Create IC correlation matrix
        print("\nCalculating IC correlation matrix...")
        ic_matrix = self.create_ic_correlation_matrix(factor_data, return_periods)
        
        # Calculate IC series correlation matrix
        print("\nCalculating IC series correlation matrix...")
        return_data_7d = self.load_return_data('7d')  # Use 7-day return data
        if return_data_7d is not None:
            ic_series_corr_matrix = self.calculate_ic_series_correlation(factor_data, return_data_7d)
            print("\nPlotting IC series correlation heatmap...")
            ic_series_heatmap_path = self.plot_ic_series_correlation_heatmap(ic_series_corr_matrix)
        
        # Calculate and plot quantile returns
        print("\nCalculating quantile returns...")
        for period in return_periods:
            return_data = self.load_return_data(period)
            if return_data is None:
                continue
            
            quantile_results = self.calculate_quantile_returns(factor_data, return_data)
            if quantile_results:
                print(f"Plotting {period} quantile return chart...")
                self.plot_quantile_returns(quantile_results, period)
        
        # Generate summary report
        print("\nGenerating analysis summary...")
        # Use 7d data to generate summary
        return_data_7d = self.load_return_data('7d')
        if return_data_7d is not None:
            quantile_results_7d = self.calculate_quantile_returns(factor_data, return_data_7d)
            summary_path = self.generate_correlation_summary(ic_matrix, quantile_results_7d)
        
        print(f"\nAnalysis complete! Results saved in: {self.output_dir}")
        return ic_matrix

def main():
    # Set paths
    base_dir = "/Users/queenwen/Desktop/QI_Paper"
    normalized_factors_dir = os.path.join(base_dir, "normalized_factors")
    return_data_dir = os.path.join(base_dir, "return_data")
    output_dir = os.path.join(base_dir, "factor_correlation_analysis")
    
    # Create analyzer
    analyzer = FactorCorrelationAnalyzer(normalized_factors_dir, return_data_dir, output_dir)
    
    # Run analysis
    ic_matrix = analyzer.run_analysis()
    
if __name__ == "__main__":
    main()