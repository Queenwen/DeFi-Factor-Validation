import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

# Set font for visualization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Load summary data
summary_file = '/Users/queenwen/Desktop/QI_Paper/factor_visualizations/all_factors_summary.csv'
all_summary = pd.read_csv(summary_file)

# Create comprehensive visualization directory
comparison_dir = '/Users/queenwen/Desktop/QI_Paper/factor_comparison_visualizations'
os.makedirs(comparison_dir, exist_ok=True)

# Return period list
return_periods = ['3d', '7d', '14d', '21d', '30d', '60d']
return_periods_cn = ['3D Forward Acc. Return', '7D Forward Acc. Return', '14D Forward Acc. Return', '21D Forward Acc. Return', '30D Forward Acc. Return', '60D Forward Acc. Return']

# 1. Plot average IC heatmap across different return periods
def plot_ic_heatmap():
    # Pivot table: factor x return period, values are average Spearman IC
    ic_pivot = all_summary.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='mean_spearman_ic',
        aggfunc='mean'
    )
    
    # Sort by return periods
    ic_pivot = ic_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(ic_pivot, annot=True, cmap='RdBu_r', center=0, fmt='.3f', linewidths=.5)
    plt.title('Average Spearman IC of Each Factor Across Different Return Periods')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'ic_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ IC heatmap saved: {save_path}")
    
    # Calculate ICIR and plot heatmap
    # First calculate ICIR (IC/IC standard deviation)
    all_summary['icir'] = all_summary['mean_spearman_ic'] / all_summary['std_spearman_ic']
    
    # Create ICIR pivot table
    icir_pivot = all_summary.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='icir',
        aggfunc='mean'
    )
    
    # Sort by return periods
    icir_pivot = icir_pivot[return_periods_cn]
    
    # Plot ICIR heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(icir_pivot, annot=True, cmap='RdBu_r', center=0, fmt='.4f', linewidths=.5)
    plt.title('ICIR (IC/IC Standard Deviation) of Each Factor Across Different Return Periods')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'icir_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ ICIR heatmap saved: {save_path}")

# 2. Plot quantile spread heatmap across different return periods
def plot_quantile_spread_heatmap():
    # Pivot table: factor x return period, values are quantile spread
    spread_pivot = all_summary.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='quantile_spread',
        aggfunc='mean'
    )
    
    # Sort by return periods
    spread_pivot = spread_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(spread_pivot, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
    plt.title('Quantile Spread of Each Factor Across Different Return Periods')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'quantile_spread_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Quantile spread heatmap saved: {save_path}")

# 3. Plot scatter plot of IC and quantile spread for factors under 7-day return period
def plot_ic_spread_scatter():
    # Filter data for 7-day return period
    df_7d = all_summary[all_summary['return_period'] == '7d'].copy()
    
    # Ensure correct data types
    df_7d['mean_spearman_ic'] = df_7d['mean_spearman_ic'].astype(float)
    df_7d['quantile_spread'] = df_7d['quantile_spread'].astype(float)
    
    # Plot scatter plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    scatter = plt.scatter(
        df_7d['mean_spearman_ic'], 
        df_7d['quantile_spread'],
        c=df_7d['mean_spearman_ic'].abs(),  # Color by IC absolute value
        cmap='viridis',
        s=100,  # Point size
        alpha=0.7
    )
    
    # Add factor labels
    for i, row in df_7d.iterrows():
        plt.annotate(
            row['factor_name'],
            (row['mean_spearman_ic'], row['quantile_spread']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # Add colorbar
    plt.colorbar(scatter, label='|Spearman IC|')
    
    # Add reference lines
    plt.axhline(y=df_7d['quantile_spread'].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Relationship between IC and Quantile Spread for Each Factor (7-Day Return Period)')
    plt.xlabel('Average Spearman IC')
    plt.ylabel('Quantile Spread')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'ic_spread_scatter_7d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ IC-spread scatter plot saved: {save_path}")

# 4. Plot factor ranking bar chart for 7-day return period
def plot_factor_ranking():
    # Filter data for 7-day return period
    df_7d = all_summary[all_summary['return_period'] == '7d'].copy()
    
    # Ensure correct data types
    df_7d['mean_spearman_ic'] = df_7d['mean_spearman_ic'].astype(float)
    df_7d['std_spearman_ic'] = df_7d['std_spearman_ic'].astype(float)
    
    # Calculate IC absolute value and sort
    df_7d['abs_ic'] = df_7d['mean_spearman_ic'].abs()
    
    # Calculate ICIR and ICIR absolute value
    df_7d['icir'] = df_7d['mean_spearman_ic'] / df_7d['std_spearman_ic']
    df_7d['abs_icir'] = df_7d['icir'].abs()
    
    # Create chart layout (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Plot IC absolute value ranking
    # Sort by IC absolute value
    ic_sorted = df_7d.sort_values('abs_ic', ascending=False)
    
    # Plot bar chart
    bars = axes[0].bar(
        range(len(ic_sorted)), 
        ic_sorted['abs_ic'],
        color=plt.cm.viridis(np.linspace(0, 1, len(ic_sorted)))
    )
    
    # Add IC value labels on bars
    for bar, ic, raw_ic in zip(bars, ic_sorted['abs_ic'], ic_sorted['mean_spearman_ic']):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.005,
            f'{ic:.3f}\n({raw_ic:.3f})',
            ha='center', va='bottom',
            fontsize=8
        )
    
    axes[0].set_title('IC Absolute Value Ranking of Each Factor (7-Day Return Period)')
    axes[0].set_xlabel('Factor')
    axes[0].set_ylabel('|Spearman IC|')
    axes[0].set_xticks(range(len(ic_sorted)))
    axes[0].set_xticklabels(ic_sorted['factor_name'], rotation=45, ha='right')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # 2. Plot ICIR absolute value ranking
    # Sort by ICIR absolute value
    icir_sorted = df_7d.sort_values('abs_icir', ascending=False)
    
    # Plot bar chart
    bars = axes[1].bar(
        range(len(icir_sorted)), 
        icir_sorted['abs_icir'],
        color=plt.cm.viridis(np.linspace(0, 1, len(icir_sorted)))
    )
    
    # Add ICIR value labels on bars
    for bar, icir, raw_icir in zip(bars, icir_sorted['abs_icir'], icir_sorted['icir']):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.05,
            f'{icir:.3f}\n({raw_icir:.3f})',
            ha='center', va='bottom',
            fontsize=8
        )
    
    axes[1].set_title('ICIR Absolute Value Ranking of Each Factor (7-Day Return Period)')
    axes[1].set_xlabel('Factor')
    axes[1].set_ylabel('|ICIR|')
    axes[1].set_xticks(range(len(icir_sorted)))
    axes[1].set_xticklabels(icir_sorted['factor_name'], rotation=45, ha='right')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'factor_ranking_7d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Factor ranking chart saved: {save_path}")

# 5. Plot IC and ICIR trend charts across different return periods
def plot_ic_trend():
    # Create chart layout (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # 1. Plot IC trend chart
    # Group by factor, calculate average IC for each return period
    ic_trend = all_summary.pivot_table(
        index='return_period_cn', 
        columns='factor_name',
        values='mean_spearman_ic',
        aggfunc='mean'
    )
    
    # Sort by return periods
    ic_trend = ic_trend.loc[return_periods_cn]
    
    # Plot a line for each factor
    for factor in ic_trend.columns:
        axes[0].plot(range(len(ic_trend.index)), ic_trend[factor], marker='o', label=factor)
    
    axes[0].set_title('IC Trend of Each Factor Across Different Return Periods')
    axes[0].set_xlabel('Return Period')
    axes[0].set_ylabel('Average Spearman IC')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_xticks(range(len(ic_trend.index)))
    axes[0].set_xticklabels(ic_trend.index)
    
    # 2. Plot ICIR trend chart
    # Group by factor, calculate average ICIR for each return period
    icir_trend = all_summary.pivot_table(
        index='return_period_cn', 
        columns='factor_name',
        values='icir',
        aggfunc='mean'
    )
    
    # Sort by return periods
    icir_trend = icir_trend.loc[return_periods_cn]
    
    # Plot a line for each factor
    for factor in icir_trend.columns:
        axes[1].plot(range(len(icir_trend.index)), icir_trend[factor], marker='o', label=factor)
    
    axes[1].set_title('ICIR Trend of Each Factor Across Different Return Periods')
    axes[1].set_xlabel('Return Period')
    axes[1].set_ylabel('ICIR (IC/IC Standard Deviation)')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xticks(range(len(icir_trend.index)))
    axes[1].set_xticklabels(icir_trend.index)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'ic_icir_trend.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ IC and ICIR trend chart saved: {save_path}")

# 6. Plot quantile spread trend chart across different return periods
def plot_spread_trend():
    # Group by factor, calculate quantile spread for each return period
    spread_trend = all_summary.pivot_table(
        index='return_period_cn', 
        columns='factor_name',
        values='quantile_spread',
        aggfunc='mean'
    )
    
    # Sort by return periods
    spread_trend = spread_trend.loc[return_periods_cn]
    
    # Plot trend chart
    plt.figure(figsize=(12, 8))
    
    # Plot a line for each factor
    for factor in spread_trend.columns:
        plt.plot(range(len(spread_trend.index)), spread_trend[factor], marker='o', label=factor)
    
    plt.title('Quantile Spread Trend of Each Factor Across Different Return Periods')
    plt.xlabel('Return Period')
    plt.ylabel('Quantile Spread')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(len(spread_trend.index)), spread_trend.index)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'spread_trend.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Spread trend chart saved: {save_path}")

# 7. Plot comparison chart between whale trading factors and other factors
def plot_whale_comparison():
    # Filter data for 7-day return period
    df_7d = all_summary[all_summary['return_period'] == '7d'].copy()
    
    # Ensure correct data types
    df_7d['mean_spearman_ic'] = df_7d['mean_spearman_ic'].astype(float)
    df_7d['std_spearman_ic'] = df_7d['std_spearman_ic'].astype(float)
    df_7d['quantile_spread'] = df_7d['quantile_spread'].astype(float)
    
    # Calculate ICIR
    df_7d['icir'] = df_7d['mean_spearman_ic'] / df_7d['std_spearman_ic']
    
    # Mark whale trading factors
    df_7d['is_whale'] = df_7d['factor_name'].str.contains('whale')
    
    # Calculate averages
    whale_avg_ic = df_7d[df_7d['is_whale']]['mean_spearman_ic'].mean()
    other_avg_ic = df_7d[~df_7d['is_whale']]['mean_spearman_ic'].mean()
    
    whale_avg_icir = df_7d[df_7d['is_whale']]['icir'].mean()
    other_avg_icir = df_7d[~df_7d['is_whale']]['icir'].mean()
    
    whale_avg_spread = df_7d[df_7d['is_whale']]['quantile_spread'].mean()
    other_avg_spread = df_7d[~df_7d['is_whale']]['quantile_spread'].mean()
    
    # Plot comparison bar charts
    plt.figure(figsize=(15, 6))
    
    # IC comparison
    plt.subplot(1, 3, 1)
    bars = plt.bar([0, 1], [whale_avg_ic, other_avg_ic], color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.005 if height >= 0 else height - 0.015,
            f'{height:.3f}',
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=10
        )
    
    plt.title('Average Spearman IC Comparison')
    plt.ylabel('Average Spearman IC')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks([0, 1], ['Whale Trading Factors', 'Other Factors'])
    
    # ICIR comparison
    plt.subplot(1, 3, 2)
    bars = plt.bar([0, 1], [whale_avg_icir, other_avg_icir], color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.005 if height >= 0 else height - 0.015,
            f'{height:.3f}',
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=10
        )
    
    plt.title('Average ICIR Comparison')
    plt.ylabel('Average ICIR (IC/IC Standard Deviation)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks([0, 1], ['Whale Trading Factors', 'Other Factors'])
    
    # Quantile spread comparison
    plt.subplot(1, 3, 3)
    bars = plt.bar([0, 1], [whale_avg_spread, other_avg_spread], color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.005,
            f'{height:.3f}',
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.title('Average Quantile Spread Comparison')
    plt.ylabel('Average Quantile Spread')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks([0, 1], ['Whale Trading Factors', 'Other Factors'])
    
    plt.suptitle('Performance Comparison between Whale Trading Factors and Other Factors (7-Day Return Period)', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'whale_comparison_7d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Whale trading factor comparison chart saved: {save_path}")

# 8. Plot CAPM regression results heatmap across different return periods
def plot_capm_heatmap():
    # Read CAPM regression results
    capm_file = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/CAPM_analysis/combined_pooled_regression.csv'
    capm_df = pd.read_csv(capm_file)
    
    # Add Chinese return periods
    period_map = {period: cn for period, cn in zip(return_periods, return_periods_cn)}
    capm_df['return_period_cn'] = capm_df['return_period'].map(period_map)
    
    # Use factor names directly
    # No need to map since we're using factor_name directly
    
    # Remove duplicate rows
    capm_df = capm_df.drop_duplicates(['factor_name', 'return_period'])
    
    # 1. Plot factor exposure coefficient (gamma) heatmap
    gamma_pivot = capm_df.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='gamma',
        aggfunc='mean'
    )
    
    # Sort by return periods
    gamma_pivot = gamma_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(gamma_pivot, annot=True, cmap='RdBu_r', center=0, fmt='.4f', linewidths=.5)
    plt.title('CAPM Factor Exposure Coefficient (Gamma) Across Different Return Periods')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'capm_gamma_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ CAPM factor exposure coefficient heatmap saved: {save_path}")
    
    # 2. Plot factor significance heatmap
    # Create significance markers
    capm_df['gamma_significance'] = np.where(capm_df['gamma_pvalue'] < 0.05, 1, 0)
    
    sig_pivot = capm_df.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='gamma_significance',
        aggfunc='mean'
    )
    
    # Sort by return periods
    sig_pivot = sig_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sig_pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.0f', linewidths=.5)
    plt.title('CAPM Factor Significance Across Different Return Periods (p<0.05)')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'capm_significance_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ CAPM factor significance heatmap saved: {save_path}")
    
    # 3. Plot R-squared heatmap
    r2_pivot = capm_df.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='r_squared',
        aggfunc='mean'
    )
    
    # Sort by return periods
    r2_pivot = r2_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(r2_pivot, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
    plt.title('CAPM Regression R² Across Different Return Periods')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'capm_r2_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ CAPM R² heatmap saved: {save_path}")

# 9. Plot CAPM regression results trend charts
def plot_capm_trend():
    # Read CAPM regression results
    capm_file = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/CAPM_analysis/combined_pooled_regression.csv'
    capm_df = pd.read_csv(capm_file)
    
    # Add Chinese return periods
    period_map = {period: cn for period, cn in zip(return_periods, return_periods_cn)}
    capm_df['return_period_cn'] = capm_df['return_period'].map(period_map)
    
    # Use factor names directly
    # No need to map since we're using factor_name directly
    
    # Remove duplicate rows
    capm_df = capm_df.drop_duplicates(['factor_name', 'return_period'])
    
    # 1. Plot factor exposure coefficient (gamma) trend chart
    gamma_trend = capm_df.pivot_table(
        index='return_period_cn', 
        columns='factor_name',
        values='gamma',
        aggfunc='mean'
    )
    
    # Sort by return periods
    gamma_trend = gamma_trend.loc[return_periods_cn]
    
    # Plot trend chart
    plt.figure(figsize=(12, 8))
    
    # Plot a line for each factor
    for factor in gamma_trend.columns:
        plt.plot(range(len(gamma_trend.index)), gamma_trend[factor], marker='o', label=factor)
    
    plt.title('CAPM Factor Exposure Coefficient (Gamma) Trend Across Different Return Periods')
    plt.xlabel('Return Period')
    plt.ylabel('Factor Exposure Coefficient (Gamma)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(len(gamma_trend.index)), gamma_trend.index)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'capm_gamma_trend.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ CAPM factor exposure coefficient trend chart saved: {save_path}")
    
    # 2. Plot R-squared trend chart
    r2_trend = capm_df.pivot_table(
        index='return_period_cn', 
        columns='factor_name',
        values='r_squared',
        aggfunc='mean'
    )
    
    # Sort by return periods
    r2_trend = r2_trend.loc[return_periods_cn]
    
    # Plot trend chart
    plt.figure(figsize=(12, 8))
    
    # Plot a line for each factor
    for factor in r2_trend.columns:
        plt.plot(range(len(r2_trend.index)), r2_trend[factor], marker='o', label=factor)
    
    plt.title('CAPM Regression R² Trend Across Different Return Periods')
    plt.xlabel('Return Period')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(len(r2_trend.index)), r2_trend.index)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'capm_r2_trend.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ CAPM R² trend chart saved: {save_path}")

# 10. Plot Portfolio CAPM analysis results visualization
def plot_portfolio_capm():
    # Read Portfolio CAPM statistical results
    portfolio_stats_file = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/CAPM_analysis/combined_portfolio_stats.csv'
    portfolio_stats = pd.read_csv(portfolio_stats_file)
    
    # Read Portfolio CAPM time series data
    portfolio_ts_file = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/CAPM_analysis/combined_portfolio_timeseries.csv'
    portfolio_ts = pd.read_csv(portfolio_ts_file)
    portfolio_ts['date'] = pd.to_datetime(portfolio_ts['date'])
    
    # Add Chinese return periods
    period_map = {period: cn for period, cn in zip(return_periods, return_periods_cn)}
    portfolio_stats['return_period_cn'] = portfolio_stats['return_period'].map(period_map)
    
    # Use factor names directly
    # No need to map since we're using factor_name directly
    
    # Remove duplicate rows
    portfolio_stats = portfolio_stats.drop_duplicates(['factor_name', 'return_period'])
    
    # 1. Plot Alpha, Beta and Gamma comparison chart for different factors
    plt.figure(figsize=(18, 15))
    
    # Select 7-day return period data
    stats_7d = portfolio_stats[portfolio_stats['return_period'] == '7d']
    
    # Sort by Gamma value (factor's explanatory power for returns)
    stats_7d = stats_7d.sort_values('gamma', ascending=False)
    
    # Plot Alpha bar chart
    plt.subplot(3, 1, 1)
    bars = plt.bar(stats_7d['factor_name'], stats_7d['alpha'], color='skyblue')
    
    # Mark significant Alpha
    for i, (is_sig, val) in enumerate(zip(stats_7d['alpha_significant'], stats_7d['alpha'])):
        color = 'red' if is_sig else 'skyblue'
        bars[i].set_color(color)
    
    plt.title('Portfolio Alpha of Each Factor (7-Day Return Period)')
    plt.ylabel('Alpha')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot Beta bar chart
    plt.subplot(3, 1, 2)
    plt.bar(stats_7d['factor_name'], stats_7d['beta'], color='lightgreen')
    plt.title('Portfolio Beta of Each Factor (7-Day Return Period)')
    plt.ylabel('Beta')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot Gamma bar chart (factor's explanatory power for returns)
    plt.subplot(3, 1, 3)
    gamma_bars = plt.bar(stats_7d['factor_name'], stats_7d['gamma'], color='lightsalmon')
    
    # Mark significant Gamma
    for i, (is_sig, val) in enumerate(zip(stats_7d['gamma_significant'], stats_7d['gamma'])):
        color = 'red' if is_sig else 'lightsalmon'
        gamma_bars[i].set_color(color)
    
    plt.title('Portfolio Gamma of Each Factor (Factor Explanatory Power for Returns, 7-Day Return Period)')
    plt.ylabel('Gamma')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'portfolio_alpha_beta_gamma_7d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Portfolio Alpha, Beta and Gamma comparison chart saved: {save_path}")
    
    # 2. Plot Alpha significance heatmap across different return periods
    alpha_sig_pivot = portfolio_stats.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='alpha_significant',
        aggfunc='mean'
    )
    
    # Sort by return periods
    alpha_sig_pivot = alpha_sig_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(alpha_sig_pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.0f', linewidths=.5)
    plt.title('Portfolio Alpha Significance of Each Factor Across Different Return Periods (p<0.05)')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'portfolio_alpha_significance_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Portfolio Alpha significance heatmap saved: {save_path}")
    
    # 3. Plot Gamma significance heatmap across different return periods (factor explanatory power for returns)
    gamma_sig_pivot = portfolio_stats.pivot_table(
        index='factor_name', 
        columns='return_period_cn',
        values='gamma_significant',
        aggfunc='mean'
    )
    
    # Sort by return periods
    gamma_sig_pivot = gamma_sig_pivot[return_periods_cn]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(gamma_sig_pivot, annot=True, cmap='YlOrRd', vmin=0, vmax=1, fmt='.0f', linewidths=.5)
    plt.title('Portfolio Gamma Significance of Each Factor Across Different Return Periods (p<0.05)')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'portfolio_gamma_significance_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Portfolio Gamma significance heatmap saved: {save_path}")
    
    # 4. Plot long-short portfolio returns time series (select representative factors)
    # Select representative factors
    selected_factors = ['whale_transaction_count_100k_usd_to_inf', 'social_volume_total', 'exchange_inflow_usd']
    
    # Select 7-day return period data
    ts_7d = portfolio_ts[(portfolio_ts['return_period'] == '7d') & 
                        (portfolio_ts['factor_name'].isin(selected_factors))]
    
    # Plot long-short portfolio returns time series
    plt.figure(figsize=(14, 8))
    
    for i, factor in enumerate(selected_factors):
        factor_ts = ts_7d[ts_7d['factor_name'] == factor]
        factor_name = selected_factors[i]
        
        # Calculate cumulative returns
        factor_ts['cumulative_return'] = (1 + factor_ts['long_short_return']).cumprod() - 1
        
        plt.plot(factor_ts['date'], factor_ts['cumulative_return'], label=factor)
    
    plt.title('Cumulative Returns of Representative Factors Long-Short Portfolios (7-Day Return Period)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(comparison_dir, 'portfolio_cumulative_returns_7d.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Portfolio cumulative returns time series chart saved: {save_path}")

# Execute all visualization functions
if __name__ == "__main__":
    print("Starting to generate factor comparison visualizations...")
    
    # Plot various comparison charts
    plot_ic_heatmap()
    plot_quantile_spread_heatmap()
    plot_ic_spread_scatter()
    plot_factor_ranking()
    plot_ic_trend()
    plot_spread_trend()
    plot_whale_comparison()
    plot_capm_heatmap()
    plot_capm_trend()
    plot_portfolio_capm()  # Add Portfolio CAPM visualization
    
    print("\n✅ All factor comparison visualizations completed!")