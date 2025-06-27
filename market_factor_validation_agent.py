import os
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, ttest_1samp
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

class MarketFactorValidationAgent:
    def __init__(self, output_dir='validation_results', ic_window=60, ic_windows=[20, 40, 60, 120], quantiles=5, add_timestamp=True, return_periods=['1d', '3d', '7d', '14d', '21d', '30d', '60d']):
        self.output_dir = output_dir
        self.ic_window = ic_window
        self.ic_windows = ic_windows  
        self.quantiles = quantiles
        self.add_timestamp = add_timestamp
        self.return_periods = return_periods
        
        # add timestamp to output_dir
        if add_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            self.output_dir = f"{output_dir}_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # create subfolders for IC_analysis, Quantile_analysis, CAPM_analysis
        self.ic_dir = os.path.join(self.output_dir, 'IC_analysis')
        self.quantile_dir = os.path.join(self.output_dir, 'Quantile_analysis')
        self.capm_dir = os.path.join(self.output_dir, 'CAPM_analysis')
        
        os.makedirs(self.ic_dir, exist_ok=True)
        os.makedirs(self.quantile_dir, exist_ok=True)
        os.makedirs(self.capm_dir, exist_ok=True)
        
        # initialize results lists
        self.all_ic_results = []
        self.all_quantile_results = []
        self.all_portfolio_ts = []
        self.all_portfolio_stats = []
        self.all_pooled_stats = []

    def calculate_ic(self, data_df, factor_col, return_col, factor_name=None, return_period=None, window=None, use_multiple_windows=True):
        if window is None:
            window = self.ic_window
        # calculate spearman IC, lag 1 day
        data_df = data_df.sort_values(['date', 'asset'])
        
        # Calculate lagged returns (1 period ahead)
        data_df['next_return'] = data_df.groupby('asset')[return_col].shift(-1)
        
        # Remove missing values
        data_df = data_df.dropna(subset=[factor_col, 'next_return'])
        
        if len(data_df) == 0:
            print(f"Warning: Data is empty, cannot calculate IC")
            return pd.DataFrame()
        
        print(f"Calculating IC: {len(data_df)} records, {data_df['asset'].nunique()} assets, {data_df['date'].nunique()} trading days")
        
        # If multiple rolling windows are enabled, calculate IC for multiple windows
        if use_multiple_windows:
            ic_results = []
            
            for window in self.ic_windows:
                print(f"  Calculating IC for rolling window {window}...")
                
                # Calculate daily IC by date
                daily_ic = []
                dates = sorted(data_df['date'].unique())
                
                for date in dates:
                    date_data = data_df[data_df['date'] == date]
                    if len(date_data) >= 5:  # Need at least 5 observations
                        ic, p_value = spearmanr(date_data[factor_col], date_data['next_return'])
                        if not np.isnan(ic):
                            daily_ic.append({
                                'date': date,
                                'ic': ic,
                                'p_value': p_value,
                                'n_assets': len(date_data)
                            })
                
                if len(daily_ic) == 0:
                    print(f"    Warning: No valid IC values for window {window}")
                    continue
                
                daily_ic_df = pd.DataFrame(daily_ic)
                daily_ic_df = daily_ic_df.sort_values('date')
                
                # Calculate rolling IC
                daily_ic_df['rolling_ic'] = daily_ic_df['ic'].rolling(window=window, min_periods=max(1, window//2)).mean()
                daily_ic_df['rolling_ic_std'] = daily_ic_df['ic'].rolling(window=window, min_periods=max(1, window//2)).std()
                
                # Calculate t-statistic and significance
                daily_ic_df['t_stat'] = daily_ic_df['rolling_ic'] / (daily_ic_df['rolling_ic_std'] / np.sqrt(window))
                daily_ic_df['significant'] = np.abs(daily_ic_df['t_stat']) > 1.96
                
                # Add metadata
                daily_ic_df['factor_name'] = factor_name
                daily_ic_df['return_period'] = return_period
                daily_ic_df['window'] = window
                
                ic_results.append(daily_ic_df)
            
            if ic_results:
                final_ic_df = pd.concat(ic_results, ignore_index=True)
                
                # Save results
                period_suffix = f'_{return_period}' if return_period else ''
                filename = f'ic_{factor_name}{period_suffix}.csv' if factor_name else f'ic_{factor_col}{period_suffix}.csv'
                final_ic_df.to_csv(os.path.join(self.ic_dir, filename), index=False)
                
                return final_ic_df
            else:
                print(f"Warning: No valid IC values for all windows")
                return pd.DataFrame()
        else:
            # Original single window calculation for backward compatibility
            data_df['factor_lag1'] = data_df.groupby('asset')[factor_col].shift(1)
            data_df = data_df.dropna(subset=['factor_lag1', return_col])

            ic_results = []
            dates = data_df['date'].unique()
            for date in dates:
                sub_df = data_df[data_df['date'] == date]
                if len(sub_df) < 5:
                    continue
                spearman_ic, _ = spearmanr(sub_df['factor_lag1'], sub_df[return_col])
                ic_results.append({'date': date, 'spearman_ic': spearman_ic})
            ic_df = pd.DataFrame(ic_results)
            
            # calculate spearman IC for multiple windows
            for w in self.ic_windows:
                ic_df[f'rolling_spearman_ic_{w}d'] = ic_df['spearman_ic'].rolling(w).mean()
            
            # use token name and return period as filename
            period_suffix = f'_{return_period}' if return_period else ''
            filename = f'ic_{factor_name}{period_suffix}.csv' if factor_name else f'ic_{factor_col}{period_suffix}.csv'
            ic_df.to_csv(os.path.join(self.ic_dir, filename), index=False)
            return ic_df

    def quantile_return_analysis(self, data_df, factor_col, return_col, factor_name=None, return_period=None, quantiles=None):
        if quantiles is None:
            quantiles = self.quantiles
        # quantile return analysis
        data_df = data_df.sort_values(['date', 'asset'])
        data_df['factor_lag1'] = data_df.groupby('asset')[factor_col].shift(1)
        data_df = data_df.dropna(subset=['factor_lag1', return_col])

        quantile_returns = []
        dates = data_df['date'].unique()
        for date in dates:
            sub_df = data_df[data_df['date'] == date]
            if len(sub_df) < 5:
                continue
            
            # use more steady rank+qcut method
            sub_df['factor_rank'] = sub_df['factor_lag1'].rank(method='first')
            try:
                sub_df['quantile'] = pd.qcut(sub_df['factor_rank'], quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                # if still fail, use cut method
                sub_df['quantile'] = pd.cut(sub_df['factor_rank'], quantiles, labels=False) + 1
            
            mean_returns = sub_df.groupby('quantile')[return_col].mean()
            mean_returns_dict = mean_returns.to_dict()
            mean_returns_dict['date'] = date
            quantile_returns.append(mean_returns_dict)
            
        quantile_df = pd.DataFrame(quantile_returns)
        if len(quantile_df) > 0:
            quantile_df = quantile_df.sort_values('date')
            # use token name and return period as filename
            period_suffix = f'_{return_period}' if return_period else ''
            filename = f'quantile_returns_{factor_name}{period_suffix}.csv' if factor_name else f'quantile_returns_{factor_col}{period_suffix}.csv'
            quantile_df.to_csv(os.path.join(self.quantile_dir, filename), index=False)
        return quantile_df

    def portfolio_capm_analysis(self, data_df, factor_col, return_col, market_return_col, factor_name=None, return_period=None, top_pct=0.2, bottom_pct=0.2):
        """
        Group by factors and build portfolio, do CAPM regression
        Return_t = α + β·R^market_t + γ·Factor_t + ε_t
        Control market risk, look at factor's explaining power (γ)
        """
        data_df = data_df.sort_values(['date', 'asset'])
        data_df['factor_lag1'] = data_df.groupby('asset')[factor_col].shift(1)
        data_df = data_df.dropna(subset=['factor_lag1', return_col, market_return_col])
        
        portfolio_returns = []
        dates = data_df['date'].unique()
        
        for date in dates:
            sub_df = data_df[data_df['date'] == date]
            if len(sub_df) < 10:  # Need at least 10 assets for analysis
                continue
                
            # sort by factor value
            sub_df = sub_df.sort_values('factor_lag1')
            n_assets = len(sub_df)
            
            # create long and short portfolios
            top_n = max(1, int(n_assets * top_pct))
            bottom_n = max(1, int(n_assets * bottom_pct))
            
            long_portfolio = sub_df.tail(top_n)  # top 20% by factor value
            short_portfolio = sub_df.head(bottom_n)  # bottom 20% by factor value
            
            # calculate long and short portfolio returns(equal weight)
            long_return = long_portfolio[return_col].mean()
            short_return = short_portfolio[return_col].mean()
            long_short_return = long_return - short_return  # long short portfolio return
            
            # get market return, assume all assets have same market return
            market_ret = sub_df[market_return_col].iloc[0]
            
            # get factor value (use difference between long and short portfolio factor values)
            factor_value = long_portfolio['factor_lag1'].mean() - short_portfolio['factor_lag1'].mean()
            
            portfolio_returns.append({
                'date': date,
                'long_return': long_return,
                'short_return': short_return,
                'long_short_return': long_short_return,
                'market_return': market_ret,
                'factor_value': factor_value
            })
        
        portfolio_df = pd.DataFrame(portfolio_returns)
        
        if len(portfolio_df) > 0:
            # do CAPM regression on long short portfolio
            X = add_constant(portfolio_df[['market_return', 'factor_value']])
            y = portfolio_df['long_short_return']
            
            try:
                model = OLS(y, X).fit()
                alpha = model.params.iloc[0] if len(model.params) > 0 else 0
                beta = model.params.iloc[1] if len(model.params) > 1 else 0
                gamma = model.params.iloc[2] if len(model.params) > 2 else 0
                
                alpha_pvalue = model.pvalues.iloc[0] if len(model.pvalues) > 0 else np.nan
                beta_pvalue = model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan
                gamma_pvalue = model.pvalues.iloc[2] if len(model.pvalues) > 2 else np.nan
                
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj
                
                # save results
                portfolio_stats = {
                    'alpha': alpha,
                    'alpha_pvalue': alpha_pvalue,
                    'beta': beta,
                    'beta_pvalue': beta_pvalue,
                    'gamma': gamma,
                    'gamma_pvalue': gamma_pvalue,
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'alpha_significant': alpha_pvalue < 0.05 if not np.isnan(alpha_pvalue) else False,
                    'gamma_significant': gamma_pvalue < 0.05 if not np.isnan(gamma_pvalue) else False,
                    'observations': len(portfolio_df)
                }
                
                # save results to csv
                period_suffix = f'_{return_period}' if return_period else ''
                filename = f'portfolio_capm_{factor_name}{period_suffix}.csv' if factor_name else f'portfolio_capm_{factor_col}{period_suffix}.csv'
                
                # save time series data
                portfolio_df.to_csv(os.path.join(self.capm_dir, f'portfolio_timeseries_{filename}'), index=False)
                
                # save portfolio stats
                stats_df = pd.DataFrame([portfolio_stats])
                stats_df.to_csv(os.path.join(self.capm_dir, f'portfolio_stats_{filename}'), index=False)
                
                return portfolio_df, portfolio_stats
            except Exception as e:
                print(f"Portfolio CAPM regression failed: {e}")
                return portfolio_df, None
        
        return None, None
    
    def pooled_capm_regression(self, data_df, factor_col, return_col, market_return_col, factor_name=None, return_period=None):
        """
        pooled CAPM regression - single factor model
        Return_t = α + β·R^market_t + γ·Factor_t + ε_t
        control market risk, look at factor's explaining power (γ)
        """
        data_df = data_df.sort_values(['date', 'asset'])
        data_df['factor_lag1'] = data_df.groupby('asset')[factor_col].shift(1)
        data_df = data_df.dropna(subset=['factor_lag1', return_col, market_return_col])
        
        if len(data_df) < 50:  # need at least 50 observations
            print("Not enough data for pooled CAPM regression.")
            return None
        
        try:
            # single factor model: control market risk and look at factor's explaining power
            X = add_constant(data_df[[market_return_col, 'factor_lag1']])
            y = data_df[return_col]
            model = OLS(y, X).fit()
            
            results = {
                'alpha': model.params.iloc[0],
                'beta': model.params.iloc[1],
                'gamma': model.params.iloc[2],
                'alpha_pvalue': model.pvalues.iloc[0],
                'beta_pvalue': model.pvalues.iloc[1],
                'gamma_pvalue': model.pvalues.iloc[2],
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'observations': len(data_df),
                'alpha_significant': model.pvalues.iloc[0] < 0.05,
                'gamma_significant': model.pvalues.iloc[2] < 0.05
            }
            
            # save results to csv
            period_suffix = f'_{return_period}' if return_period else ''
            filename = f'pooled_capm_{factor_name}{period_suffix}.csv' if factor_name else f'pooled_capm_{factor_col}{period_suffix}.csv'
            
            results_df = pd.DataFrame([results])
            results_df.to_csv(os.path.join(self.capm_dir, filename), index=False)
            
            return results
        except Exception as e:
            print(f"Pooled CAPM regression failed: {e}")
            return None
    
    def run_validation(self, data_ic, data_quantile, data_capm, factor_col, return_col, market_return_col, factor_name=None, return_period=None, use_multiple_windows=True):
        # calculate IC, support multiple rolling windows
        ic_df = self.calculate_ic(data_ic, factor_col, return_col, factor_name, return_period, use_multiple_windows=use_multiple_windows)
        quantile_df = self.quantile_return_analysis(data_quantile, factor_col, return_col, factor_name=factor_name, return_period=return_period, quantiles=self.quantiles)
        
        portfolio_ts, portfolio_stats = self.portfolio_capm_analysis(data_capm, factor_col, return_col, market_return_col, factor_name=factor_name, return_period=return_period)
        pooled_stats = self.pooled_capm_regression(data_capm, factor_col, return_col, market_return_col, factor_name=factor_name, return_period=return_period)
        
        # return an empty capm_df to keep the interface compatible
        capm_df = pd.DataFrame()
        
        # if use_multiple_windows is True, add the results of different windows to all_ic_results
        if use_multiple_windows:
            # create a summary record for each window
            for window in self.ic_windows:
                window_col = f'rolling_spearman_ic_{window}d'
                if window_col in ic_df.columns:
                    mean_ic = ic_df[window_col].mean()
                    std_ic = ic_df[window_col].std()
                    t_stat, p_value = ttest_1samp(ic_df[window_col].dropna(), 0)
                    
                    # add to the summary list
                    self.all_ic_results.append({
                        'factor_name': factor_name,
                        'return_period': return_period,
                        'window': window,
                        'mean_rolling_ic': mean_ic,
                        'std_rolling_ic': std_ic,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
        
        # add quantile analysis results to summary list
        if len(quantile_df) > 0:
            # calculate the average difference in returns between the highest and lowest quantiles
            if self.quantiles is not None and self.quantiles > 1:
                highest_q = self.quantiles
                lowest_q = 1
                if highest_q in quantile_df.columns and lowest_q in quantile_df.columns:
                    quantile_df['high_minus_low'] = quantile_df[highest_q] - quantile_df[lowest_q]
                    mean_diff = quantile_df['high_minus_low'].mean()
                    t_stat, p_value = ttest_1samp(quantile_df['high_minus_low'].dropna(), 0)
                    
                    self.all_quantile_results.append({
                        'factor_name': factor_name,
                        'return_period': return_period,
                        'mean_high_minus_low': mean_diff,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
        
        # add portfolio analysis results to summary list
        if portfolio_stats is not None:
            portfolio_stats['factor_name'] = factor_name
            portfolio_stats['return_period'] = return_period
            self.all_portfolio_stats.append(portfolio_stats)
            
            if portfolio_ts is not None and len(portfolio_ts) > 0:
                portfolio_ts['factor_name'] = factor_name
                portfolio_ts['return_period'] = return_period
                self.all_portfolio_ts.append(portfolio_ts)
        
        # add pooled regression results to summary list
        if pooled_stats is not None:
            pooled_stats['factor_name'] = factor_name
            pooled_stats['return_period'] = return_period
            self.all_pooled_stats.append(pooled_stats)
        
        return ic_df, quantile_df, capm_df
        
    def save_summary_results(self):
        # save IC summary results
        if self.all_ic_results:
            ic_summary = pd.DataFrame(self.all_ic_results)
            ic_summary.to_csv(os.path.join(self.output_dir, 'ic_summary.csv'), index=False)
            print(f"✅ IC summary results have been saved to : {os.path.join(self.output_dir, 'ic_summary.csv')}")
            
            # If there are multiple rolling window results, generate visualization
            if 'window' in ic_summary.columns and len(ic_summary['window'].unique()) > 1:
                self.visualize_multiple_windows_ic(ic_summary)
        
        # Save quantile results summary
        if self.all_quantile_results:
            quantile_summary = pd.DataFrame(self.all_quantile_results)
            quantile_summary.to_csv(os.path.join(self.output_dir, 'quantile_summary.csv'), index=False)
            print(f"✅ Quantile summary results have been saved to: {os.path.join(self.output_dir, 'quantile_summary.csv')}")
        
        # Save Portfolio statistics results summary
        if self.all_portfolio_stats:
            portfolio_stats_summary = pd.DataFrame(self.all_portfolio_stats)
            portfolio_stats_summary.to_csv(os.path.join(self.output_dir, 'portfolio_stats_summary.csv'), index=False)
            print(f"✅ Portfolio statistics summary results have been saved to: {os.path.join(self.output_dir, 'portfolio_stats_summary.csv')}")
        
        # Save Portfolio time series results summary
        if self.all_portfolio_ts:
            portfolio_ts_summary = pd.concat(self.all_portfolio_ts, ignore_index=True)
            portfolio_ts_summary.to_csv(os.path.join(self.output_dir, 'portfolio_ts_summary.csv'), index=False)
            print(f"✅ Portfolio time series summary results have been saved to: {os.path.join(self.output_dir, 'portfolio_ts_summary.csv')}")
        
        # Save Pooled regression results summary
        if self.all_pooled_stats:
            pooled_stats_summary = pd.DataFrame(self.all_pooled_stats)
            pooled_stats_summary.to_csv(os.path.join(self.output_dir, 'pooled_regression_summary.csv'), index=False)
            print(f"✅ Pooled regression summary results have been saved to: {os.path.join(self.output_dir, 'pooled_regression_summary.csv')}")
            
    def visualize_multiple_windows_ic(self, ic_summary):
        """Create visualization for IC results of multiple rolling windows"""
        # 1. Create bar chart visualization
        plt.figure(figsize=(12, 8))
        
        # Group by factor and return period to plot IC for different windows
        factors = ic_summary['factor_name'].unique()
        periods = ic_summary['return_period'].unique()
        
        for i, factor in enumerate(factors):
            for j, period in enumerate(periods):
                subset = ic_summary[(ic_summary['factor_name'] == factor) & 
                                   (ic_summary['return_period'] == period)]
                
                if len(subset) > 0:
                    plt.subplot(len(factors), len(periods), i*len(periods) + j + 1)
                    
                    # Plot IC trend with window changes
                    windows = subset['window'].values
                    ics = subset['mean_rolling_ic'].values
                    significant = subset['significant'].values
                    
                    # Use different colors to mark significance
                    colors = ['green' if sig else 'red' for sig in significant]
                    plt.bar(windows, ics, color=colors, alpha=0.7)
                    
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title(f'{factor} - {period}')
                    plt.xlabel('Window Size')
                    plt.ylabel('Mean IC')
                    
                    # Add value labels
                    for x, y, c in zip(windows, ics, colors):
                        plt.text(x, y + (0.01 if y >= 0 else -0.01), 
                                f'{y:.3f}', ha='center', va='bottom' if y >= 0 else 'top',
                                color=c, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multiple_windows_ic.png'))
        plt.close()
        print(f"✅ Multiple windows IC bar chart has been saved to: {os.path.join(self.output_dir, 'multiple_windows_ic.png')}")
        
        # 2. Create heatmap visualization
        self.create_window_ic_heatmap(ic_summary)
        
    def create_window_ic_heatmap(self, ic_summary):
        """Create IC heatmap for different rolling windows"""
        # Create a heatmap for each return period
        periods = ic_summary['return_period'].unique()
        
        for period in periods:
            period_data = ic_summary[ic_summary['return_period'] == period]
            
            # Create pivot table: factor x window size
            pivot_data = period_data.pivot_table(
                index='factor_name',
                columns='window',
                values='mean_rolling_ic'
            )
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
            plt.title(f'Average IC under Different Rolling Windows - {period} Return Period')
            plt.xlabel('Rolling Window Size')
            plt.ylabel('Factor')
            
            # Save figure
            save_path = os.path.join(self.output_dir, f'window_ic_heatmap_{period}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"✅ Window IC heatmap for {period} return period has been saved to: {save_path}")
            
            # Create significance heatmap
            sig_pivot = period_data.pivot_table(
                index='factor_name',
                columns='window',
                values='significant',
                aggfunc=lambda x: 1 if any(x) else 0  # If any value is True, then 1, otherwise 0
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(sig_pivot, annot=True, cmap='Greens', vmin=0, vmax=1, fmt='.0f')
            plt.title(f'IC Significance under Different Rolling Windows - {period} Return Period')
            plt.xlabel('Rolling Window Size')
            plt.ylabel('Factor')
            
            # Save figure
            save_path = os.path.join(self.output_dir, f'window_ic_significance_{period}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"✅ Window IC significance heatmap for {period} return period has been saved to: {save_path}")

def load_and_prepare_data(file_path, value_name='factor', token_whitelist=None, winsorize_pct=None):
    """Load and prepare data, convert from wide format to long format"""
    df = pd.read_csv(file_path)
    print(f"Loading factor data: {file_path}, columns: {len(df.columns)}")
    
    # Process date column
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1)
    else:
        # If the first column is date but column name is not datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.rename(columns={df.columns[0]: 'date'})
    
    # Print date range
    print(f"Factor data date range: {df['date'].min()} to {df['date'].max()}, total {len(df)} rows")
    
    # Print asset column examples
    asset_columns = [col for col in df.columns if col != 'date']
    print(f"Found {len(asset_columns)} asset columns, examples: {asset_columns[:5]}")
    
    # Convert from wide format to long format
    df_long = df.melt(id_vars=['date'], var_name='asset', value_name=value_name)
    df_long = df_long.dropna(subset=[value_name])
    print(f"After converting to long format: {len(df_long)} records, {df_long['asset'].nunique()} unique assets")
    
    # Print some asset name examples
    if len(df_long) > 0:
        print(f"Asset name examples: {sorted(df_long['asset'].unique())[:5]}")
    else:
        print("Warning: Data is empty after conversion")
    
    # If token_whitelist is None, load token list from common_tokens.txt
    if token_whitelist is None:
        try:
            common_tokens_path = '/Users/queenwen/Desktop/QI_Paper/cleaned_data/common_tokens.txt'
            with open(common_tokens_path, 'r') as f:
                lines = f.readlines()
                # Skip the first few lines of description text, only extract token names
                token_list = []
                for line in lines:
                    if line.strip().startswith('- '):
                        token_list.append(line.strip()[2:])  # Remove '- ' prefix
            print(f"Loaded {len(token_list)} tokens from {common_tokens_path}")
            token_whitelist = token_list
        except Exception as e:
            print(f"Failed to load common_tokens.txt: {e}")
    
    # Create a mapping dictionary to map asset names in factor data to token names
    asset_to_token = {}
    for asset in df_long['asset'].unique():
        # For factor data, we need to handle possible suffixes like _dev_activity_1d
        for token in token_whitelist:
            if asset.startswith(token + '_') or asset == token:
                asset_to_token[asset] = token
                break
    
    # Apply mapping
    if asset_to_token:  # Only apply mapping when the mapping dictionary is not empty
        df_long['original_asset'] = df_long['asset']  # Save original asset names
        df_long['asset'] = df_long['asset'].map(asset_to_token)
        
        # Print mapping results
        print(f"Asset name mapping examples: {dict(list(asset_to_token.items())[:5])}")
        print(f"Mapped asset examples: {df_long['asset'].iloc[:5].tolist() if len(df_long) > 0 else 'No data'}")
        print(f"Number of unique assets after mapping: {df_long['asset'].nunique()}")
        
        # Remove rows where mapping failed (asset is NaN)
        df_long = df_long.dropna(subset=['asset'])
        print(f"After removing failed mapping rows: {len(df_long)} records, {df_long['asset'].nunique()} unique assets")
    
    # Token filtering
    if token_whitelist is not None:
        df_long = df_long[df_long['asset'].isin(token_whitelist)]
        print(f"After filtering, retained {df_long['asset'].nunique()} tokens")
    
    # Outlier handling (Winsorization)
    if winsorize_pct is not None:
        from scipy.stats import mstats
        df_long[value_name] = mstats.winsorize(df_long[value_name], limits=[winsorize_pct/100, winsorize_pct/100])
        print(f"Applied {winsorize_pct}% Winsorization to {value_name}")
    
    return df_long

def load_and_prepare_return_data(file_path, value_name='return', token_whitelist=None, winsorize_pct=None):
    """Load and prepare return data, handle column names with suffixes"""
    df = pd.read_csv(file_path)
    print(f"Loading return data: {file_path}, columns: {len(df.columns)}")
    
    # Print all column names for inspection
    print(f"All column names: {list(df.columns)}")
    
    # Process date column
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1)
    else:
        # If the first column is date but column name is not datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.rename(columns={df.columns[0]: 'date'})
    
    # Only select columns containing '_log_return_'
    return_columns = [col for col in df.columns if '_log_return_' in col]
    print(f"Found {len(return_columns)} log return columns")
    if len(return_columns) == 0:
        print("Warning: No log return columns found, please check data format")
        # Print first few column names for reference
        print(f"Column name examples: {list(df.columns)[:10]}")
    else:
        print(f"Log return column examples: {return_columns[:5]}")
    
    # Select date column and return columns
    df_selected = df[['date'] + return_columns]
    
    # Convert from wide format to long format
    df_long = df_selected.melt(id_vars=['date'], var_name='asset', value_name=value_name)
    df_long = df_long.dropna(subset=[value_name])
    print(f"After converting to long format: {len(df_long)} records")
    
    # Print some asset name examples
    if len(df_long) > 0:
        print(f"Asset name examples (before conversion): {df_long['asset'].iloc[:5].tolist()}")
        # Print number of unique asset names
        unique_assets = df_long['asset'].unique()
        print(f"Number of unique asset names (before conversion): {len(unique_assets)}")
        print(f"Unique asset name examples (before conversion): {sorted(unique_assets)[:10]}")
    else:
        print("Warning: Data is empty after conversion")
    
    # If token_whitelist is None, load token list from common_tokens.txt
    if token_whitelist is None:
        try:
            common_tokens_path = '/Users/queenwen/Desktop/QI_Paper/cleaned_data/common_tokens.txt'
            with open(common_tokens_path, 'r') as f:
                lines = f.readlines()
                # Skip the first few lines of description text, only extract token names
                token_list = []
                for line in lines:
                    if line.strip().startswith('- '):
                        token_list.append(line.strip()[2:])  # Remove '- ' prefix
            print(f"Loaded {len(token_list)} tokens from {common_tokens_path}")
            token_whitelist = token_list
        except Exception as e:
            print(f"Failed to load common_tokens.txt: {e}")
    
    # Create a mapping dictionary to map asset names to token names
    asset_to_token = {}
    for asset in df_long['asset'].unique():
        for token in token_whitelist:
            # Check if asset name contains token name (as prefix)
            if asset.startswith(token + '_'):
                asset_to_token[asset] = token
                break
    
    # Apply mapping
    df_long['original_asset'] = df_long['asset']  # Save original asset names
    df_long['asset'] = df_long['asset'].map(asset_to_token)
    
    # Print mapping results
    print(f"Asset name mapping examples: {dict(list(asset_to_token.items())[:5])}")
    print(f"Mapped asset examples: {df_long['asset'].iloc[:5].tolist() if len(df_long) > 0 else 'No data'}")
    print(f"Number of unique assets after mapping: {df_long['asset'].nunique()}")
    
    # Remove rows where mapping failed (asset is NaN)
    df_long = df_long.dropna(subset=['asset'])
    print(f"After removing failed mapping rows: {len(df_long)} records, {df_long['asset'].nunique()} unique assets")
    
    # Token filtering
    if token_whitelist is not None:
        df_long = df_long[df_long['asset'].isin(token_whitelist)]
        print(f"Return data after filtering retained {df_long['asset'].nunique()} tokens")
    
    # Outlier handling (Winsorization)
    if winsorize_pct is not None:
        from scipy.stats import mstats
        df_long[value_name] = mstats.winsorize(df_long[value_name], limits=[winsorize_pct/100, winsorize_pct/100])
        print(f"Applied {winsorize_pct}% Winsorization to return data")
    
    return df_long

def get_all_factor_files():
    """Get list of all factor files"""
    base_dir = 'normalized_factors'
    factor_files = []
    
    # Get factor file names from IC_analysis folder (all three analysis folders have the same files)
    ic_dir = os.path.join(base_dir, 'IC_analysis')
    if os.path.exists(ic_dir):
        for file in os.listdir(ic_dir):
            if file.endswith('.csv'):
                factor_name = file.replace('.csv', '')
                factor_files.append(factor_name)
    
    return factor_files

def calculate_market_return(return_period):
    """Load pre-calculated market returns"""
    market_return_path = f'/Users/queenwen/Desktop/QI_Paper/return_data/market_returns/market_log_return_{return_period}.csv'
    
    if os.path.exists(market_return_path):
        market_returns = pd.read_csv(market_return_path)
        market_returns['date'] = pd.to_datetime(market_returns['date'])
        print(f'Loaded market returns: {len(market_returns)} trading days')
        return market_returns
    else:
        print(f'Error: Market return file {market_return_path} does not exist')
        return pd.DataFrame(columns=['date', 'market_return'])

if __name__ == '__main__':
    # Set output directory
    output_dir = 'market_factor_validation_results'
    agent = MarketFactorValidationAgent(
        output_dir=output_dir,
        ic_windows=[20, 40, 60, 120],  # Set multiple rolling windows
        quantiles=5,
        add_timestamp=False  # Set to False to avoid creating new directory each run
    )
    
    # Get all factor files
    factor_files = get_all_factor_files()
    print(f'Found {len(factor_files)} factors: {factor_files}')
    
    # Load return data and calculate market returns
    try:
        # Define time periods to analyze
        return_periods = ['1d', '3d', '7d', '14d', '21d', '30d', '60d']
        
        # Analyze each time period
        all_results = []
        
        for period in return_periods:
            print(f'\n=== Starting to process {period} returns ===')
            
            # Load market returns for corresponding time period
            market_returns = calculate_market_return(period)
            
            # Validate each factor
            for factor_name in factor_files:
                print(f'\nProcessing factor: {factor_name} ({period})')
                
                try:
                    # Load return data for corresponding time period (using log returns)
                    # No need to pass token_whitelist parameter as it will be loaded from common_tokens.txt inside the function
                    return_data = load_and_prepare_return_data(f'return_data/returns_{period}.csv', 'return')
                    print(f'  Return data loading completed: {len(return_data)} records, {return_data["asset"].nunique()} unique assets')
                    print(f'  Return data asset list: {sorted(return_data["asset"].unique().tolist())[:5]}...')
                    
                    # Load corresponding factor data for each type of analysis
                    # No need to pass token_whitelist parameter as it will be loaded from common_tokens.txt inside the function
                    ic_factor_data = load_and_prepare_data(f'normalized_factors/IC_analysis/{factor_name}.csv', 'factor')
                    quantile_factor_data = load_and_prepare_data(f'normalized_factors/Quantile_analysis/{factor_name}.csv', 'factor')
                    capm_factor_data = load_and_prepare_data(f'normalized_factors/CAPM_regression/{factor_name}.csv', 'factor')
                    
                    print(f'  Factor data loading completed: IC({len(ic_factor_data)}), Quantile({len(quantile_factor_data)}), CAPM({len(capm_factor_data)}) records')
                    print(f'  Factor data asset list: {sorted(ic_factor_data["asset"].unique().tolist())[:5]}...')
                    print(f'  Factor data date range: {ic_factor_data["date"].min()} to {ic_factor_data["date"].max()}')
                    print(f'  Return data date range: {return_data["date"].min()} to {return_data["date"].max()}')
                    
                    # Check common dates and assets before merging
                    common_dates_ic = set(ic_factor_data['date']).intersection(set(return_data['date']))
                    common_assets_ic = set(ic_factor_data['asset']).intersection(set(return_data['asset']))
                    print(f'  Number of common dates before merging: {len(common_dates_ic)}, common assets: {len(common_assets_ic)}')
                    print(f'  Common asset examples: {list(common_assets_ic)[:5]}...')
                    
                    if len(common_assets_ic) == 0:
                        print(f'  Warning: Factor data and return data have no common assets!')
                        print(f'  Factor data asset examples: {sorted(ic_factor_data["asset"].unique())[:5]}...')
                        print(f'  Return data asset examples: {sorted(return_data["asset"].unique())[:5]}...')
                    
                    # Merge data separately
                    ic_merged_data = pd.merge(ic_factor_data, return_data, on=['date', 'asset'], how='inner')
                    quantile_merged_data = pd.merge(quantile_factor_data, return_data, on=['date', 'asset'], how='inner')
                    capm_merged_data = pd.merge(capm_factor_data, return_data, on=['date', 'asset'], how='inner')
                    capm_merged_data = pd.merge(capm_merged_data, market_returns, on='date', how='left')
                    
                    print(f'  IC data merge completed: {len(ic_merged_data)} records, {len(ic_merged_data["date"].unique())} trading days, {len(ic_merged_data["asset"].unique())} assets')
                    print(f'  Quantile data merge completed: {len(quantile_merged_data)} records, {len(quantile_merged_data["date"].unique())} trading days, {len(quantile_merged_data["asset"].unique())} assets')
                    print(f'  CAPM data merge completed: {len(capm_merged_data)} records, {len(capm_merged_data["date"].unique())} trading days, {len(capm_merged_data["asset"].unique())} assets')
                    
                    if len(ic_merged_data) > 0:
                        print(f'  Merged data date range: {ic_merged_data["date"].min()} to {ic_merged_data["date"].max()}')
                        print(f'  Merged asset examples: {sorted(ic_merged_data["asset"].unique())[:5]}...')
                    
                    if len(ic_merged_data) == 0 and len(quantile_merged_data) == 0 and len(capm_merged_data) == 0:
                        print(f'  Warning: {factor_name} ({period}) all merged data is empty, skipping processing')
                        all_results.append({
                            'factor_name': factor_name,
                            'return_period': period,
                            'ic_records': 0,
                            'quantile_records': 0,
                            'capm_records': 0,
                            'status': 'error: all merged data is empty'
                        })
                        continue
                    
                    # Run factor validation (using different data for three types of analysis)
                    ic_df, quantile_df, capm_df = agent.run_validation(
                        ic_merged_data,
                        quantile_merged_data, 
                        capm_merged_data,
                        'factor',
                        'return',
                        'market_return',
                        factor_name,
                        period,
                        use_multiple_windows=True  # Enable multiple rolling windows
                    )
                    
                    # Record results
                    result_summary = {
                        'factor_name': factor_name,
                        'return_period': period,
                        'ic_records': len(ic_df),
                        'quantile_records': len(quantile_df),
                        'capm_records': len(capm_df),
                        'status': 'success'
                    }
                    all_results.append(result_summary)
                    
                    print(f'  {factor_name} ({period}) validation completed: IC({len(ic_df)}), Quantile({len(quantile_df)}), CAPM({len(capm_df)}) results')
                    
                except Exception as e:
                    print(f'  Error processing factor {factor_name} ({period}): {e}')
                    all_results.append({
                        'factor_name': factor_name,
                        'return_period': period,
                        'ic_records': 0,
                        'quantile_records': 0,
                        'capm_records': 0,
                        'status': f'error: {e}'
                    })
        
        # Save summary report
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(os.path.join(output_dir, 'validation_summary.csv'), index=False)
        
        print(f'\nAll factor validation completed!')
        print(f'Successfully processed: {len([r for r in all_results if r["status"] == "success"])} factor-time period combinations')
        print(f'Failed to process: {len([r for r in all_results if r["status"] != "success"])} factor-time period combinations')
        print(f'Results saved to: {output_dir}')
        
    except FileNotFoundError as e:
        print(f'Return data file not found: {e}')
        print('Please ensure return data file exists at return_data/returns_1d.csv')