#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine individual CAPM analysis files into consolidated files
"""

import pandas as pd
import os
import glob
from pathlib import Path

def combine_capm_files():
    """
    Combine individual CAPM analysis files into consolidated files
    """
    capm_dir = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/CAPM_analysis'
    
    # 1. Combine pooled CAPM regression results
    print("Combining pooled CAPM regression results...")
    pooled_files = glob.glob(os.path.join(capm_dir, 'pooled_capm_*.csv'))
    pooled_dfs = []
    
    for file in pooled_files:
        df = pd.read_csv(file)
        # Extract factor name and return period from filename
        filename = os.path.basename(file)
        parts = filename.replace('pooled_capm_', '').replace('.csv', '').split('_')
        
        if len(parts) >= 2:
            return_period = parts[-1]  # Last part is return period
            factor_name = '_'.join(parts[:-1])  # Everything else is factor name
        else:
            return_period = 'unknown'
            factor_name = parts[0] if parts else 'unknown'
        
        df['factor_name'] = factor_name
        df['return_period'] = return_period
        pooled_dfs.append(df)
    
    if pooled_dfs:
        combined_pooled = pd.concat(pooled_dfs, ignore_index=True)
        combined_pooled.to_csv(os.path.join(capm_dir, 'combined_pooled_regression.csv'), index=False)
        print(f"✅ Combined {len(pooled_files)} pooled regression files")
    
    # 2. Combine portfolio stats
    print("Combining portfolio stats...")
    stats_files = glob.glob(os.path.join(capm_dir, 'portfolio_stats_*.csv'))
    stats_dfs = []
    
    for file in stats_files:
        df = pd.read_csv(file)
        # Extract factor name and return period from filename
        filename = os.path.basename(file)
        parts = filename.replace('portfolio_stats_portfolio_capm_', '').replace('.csv', '').split('_')
        
        if len(parts) >= 2:
            return_period = parts[-1]
            factor_name = '_'.join(parts[:-1])
        else:
            return_period = 'unknown'
            factor_name = parts[0] if parts else 'unknown'
        
        df['factor_name'] = factor_name
        df['return_period'] = return_period
        stats_dfs.append(df)
    
    if stats_dfs:
        combined_stats = pd.concat(stats_dfs, ignore_index=True)
        combined_stats.to_csv(os.path.join(capm_dir, 'combined_portfolio_stats.csv'), index=False)
        print(f"✅ Combined {len(stats_files)} portfolio stats files")
    
    # 3. Combine portfolio timeseries
    print("Combining portfolio timeseries...")
    ts_files = glob.glob(os.path.join(capm_dir, 'portfolio_timeseries_*.csv'))
    ts_dfs = []
    
    for file in ts_files:
        df = pd.read_csv(file)
        # Extract factor name and return period from filename
        filename = os.path.basename(file)
        parts = filename.replace('portfolio_timeseries_portfolio_capm_', '').replace('.csv', '').split('_')
        
        if len(parts) >= 2:
            return_period = parts[-1]
            factor_name = '_'.join(parts[:-1])
        else:
            return_period = 'unknown'
            factor_name = parts[0] if parts else 'unknown'
        
        df['factor_name'] = factor_name
        df['return_period'] = return_period
        ts_dfs.append(df)
    
    if ts_dfs:
        combined_ts = pd.concat(ts_dfs, ignore_index=True)
        combined_ts.to_csv(os.path.join(capm_dir, 'combined_portfolio_timeseries.csv'), index=False)
        print(f"✅ Combined {len(ts_files)} portfolio timeseries files")
    
    print("\n🎉 All CAPM files combined successfully!")

if __name__ == '__main__':
    combine_capm_files()