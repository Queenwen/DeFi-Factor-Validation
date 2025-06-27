import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, spearmanr
import glob
import os
from matplotlib import font_manager

# Set professional theme and styling
sns.set_theme(style="whitegrid", font_scale=1.2, rc={
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Professional color palette
PROFESSIONAL_COLORS = {
    'positive_significant': '#2E8B57',  # Sea Green
    'positive_insignificant': '#90EE90',  # Light Green
    'negative_significant': '#B22222',  # Fire Brick
    'negative_insignificant': '#FFB6C1',  # Light Pink
    'neutral': '#708090',  # Slate Gray
    'excellent': '#006400',  # Dark Green
    'good': '#32CD32',  # Lime Green
    'acceptable': '#FF8C00',  # Dark Orange
    'poor': '#DC143C'  # Crimson
}

# Set output directory
output_dir = '/Users/queenwen/Desktop/QI_Paper/strategy_template_results'
os.makedirs(output_dir, exist_ok=True)

# Define user-provided strategy combination templates
COMBO_TEMPLATES = [
    {'name': 'A', 'description': 'Ultra-Short Term Reversal Strategy', 'rolling_window': 3, 'return_period': '3d'},  # Adjusted to available shortest return period
    {'name': 'B', 'description': 'Short Term Trend Continuation Strategy', 'rolling_window': 7, 'return_period': '7d'},
    {'name': 'C', 'description': 'Medium Term Wave Strategy', 'rolling_window': 14, 'return_period': '14d'},  # Adjusted to close available return period
    {'name': 'D', 'description': 'Pullback After High Sentiment', 'rolling_window': 30, 'return_period': '30d'},  # Adjusted to available shortest return period
    {'name': 'E', 'description': 'Medium Term Reversal/Mean Reversion', 'rolling_window': 60, 'return_period': '60d'},
]

# Data source directories
data_dir = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results'
ic_dir = os.path.join(data_dir, 'IC_analysis')
quantile_dir = os.path.join(data_dir, 'Quantile_analysis')
capm_dir = os.path.join(data_dir, 'CAPM_analysis')  # Add CAPM directory

def get_available_factors():
    """Get available factors from IC files"""
    all_ic_files = glob.glob(os.path.join(ic_dir, 'ic_*.csv'))
    factors = []
    for file in all_ic_files:
        basename = os.path.basename(file)
        # Remove 'ic_' prefix and '.csv' suffix
        name_part = basename.replace('ic_', '').replace('.csv', '')
        # Find the part after the last underscore, if it's in the format like '3d', consider the part before it as the factor name
        if '_' in name_part:
            parts = name_part.split('_')
            last_part = parts[-1]
            if last_part.endswith('d') and last_part[:-1].isdigit():
                factor = '_'.join(parts[:-1])
                factors.append(factor)
    return list(set(factors))  # Remove duplicates

def get_available_return_periods():
    """Get available return periods from IC files"""
    all_ic_files = glob.glob(os.path.join(ic_dir, 'ic_*.csv'))
    periods = []
    for file in all_ic_files:
        basename = os.path.basename(file)
        # Look for patterns like '_3d.csv'
        if '_' in basename and basename.endswith('.csv'):
            parts = basename.split('_')
            last_part = parts[-1].replace('.csv', '')
            if last_part.endswith('d') and last_part[:-1].isdigit():
                periods.append(last_part)
    return list(set(periods))  # remove duplicates

def calculate_rolling_ic(ic_df, window):
    """Calculate rolling IC for specified window"""
    return ic_df['ic'].rolling(window).mean()

def analyze_combo_template(combo, factors, return_periods):
    """Analyze specific strategy combination template"""
    combo_name = combo['name']
    rolling_window = combo['rolling_window']
    return_period = combo['return_period']
    
    print(f"\n📊 Analyzing combo template {combo_name}: {combo['description']}")
    print(f"   Rolling Window = {rolling_window}, Return Period = {return_period}")
    results = []
    
    # Check if there is a matching return period
    period_str = return_period  # return_period is already in format like '3d'
    if period_str not in return_periods:
        print(f"❌ Return period {period_str} data not found")
        print(f"   Available return periods: {', '.join(return_periods)}")
        return None
    
    # Calculate IC for each factor in this combination
    for factor in factors:
        ic_file = os.path.join(ic_dir, f'ic_{factor}_{period_str}.csv')
        if not os.path.exists(ic_file):
            print(f"   ⚠️ Factor {factor} data for return period {period_str} not found")
            continue
        
        # Read IC file
        ic_df = pd.read_csv(ic_file)
        if 'ic' not in ic_df.columns:
            continue
        
        # Calculate rolling IC
        ic_df['rolling_ic'] = calculate_rolling_ic(ic_df, rolling_window)
        
        # Calculate statistics
        rolling_ic = ic_df['rolling_ic'].dropna()
        if len(rolling_ic) < 10:  # At least 10 valid observations needed
            continue
            
        mean_ic = rolling_ic.mean()
        std_ic = rolling_ic.std()
        t_stat, p_value = ttest_1samp(rolling_ic, 0)
        
        # Get quantile returns
        quantile_file = os.path.join(quantile_dir, f'quantile_returns_{factor}_{period_str}.csv')
        if os.path.exists(quantile_file):
            quantile_df = pd.read_csv(quantile_file)
            if 5 in quantile_df.columns and 1 in quantile_df.columns:
                quantile_df['high_minus_low'] = quantile_df[5] - quantile_df[1]
                mean_spread = quantile_df['high_minus_low'].mean()
                spread_t, spread_p = ttest_1samp(quantile_df['high_minus_low'].dropna(), 0)
            else:
                mean_spread = np.nan
                spread_t = np.nan
                spread_p = np.nan
        else:
            mean_spread = np.nan
            spread_t = np.nan
            spread_p = np.nan
        
        # Get CAPM analysis data
        capm_file = os.path.join(capm_dir, f'pooled_capm_{factor}_{period_str}.csv')
        if os.path.exists(capm_file):
            capm_df = pd.read_csv(capm_file)
            if len(capm_df) > 0:
                gamma = capm_df['gamma'].iloc[0]
                gamma_pvalue = capm_df['gamma_pvalue'].iloc[0]
                gamma_significant = capm_df['gamma_significant'].iloc[0]
                r_squared = capm_df['r_squared'].iloc[0]
            else:
                gamma = np.nan
                gamma_pvalue = np.nan
                gamma_significant = False
                r_squared = np.nan
        else:
            gamma = np.nan
            gamma_pvalue = np.nan
            gamma_significant = False
            r_squared = np.nan
        
        # Add result
        results.append({
            'combo': combo_name,
            'description': combo['description'],
            'factor': factor,
            'rolling_window': rolling_window,
            'return_period': return_period,
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_spread': mean_spread,
            'spread_t': spread_t,
            'spread_p': spread_p,
            'spread_significant': spread_p < 0.05 if not np.isnan(spread_p) else False,
            'gamma': gamma,
            'gamma_pvalue': gamma_pvalue,
            'gamma_significant': gamma_significant,
            'r_squared': r_squared,
            'overall_score': abs(mean_ic) * 0.3 + (abs(mean_spread) * 0.3 if not np.isnan(mean_spread) else 0) + (abs(gamma) * 0.4 if not np.isnan(gamma) else 0)
        })
    
    if results:
        return pd.DataFrame(results)
    else:
        return None

def create_combo_ic_chart(results_df):
    """Create professional IC bar chart for each combination"""
    # Get a consistent factor order based on overall average IC
    factor_order = results_df.groupby('factor')['mean_ic'].mean().abs().sort_values(ascending=False).index.tolist()
    
    for combo in results_df['combo'].unique():
        combo_data = results_df[results_df['combo'] == combo].copy()
        
        # Sort by the consistent factor order
        combo_data['factor'] = pd.Categorical(combo_data['factor'], categories=factor_order, ordered=True)
        combo_data = combo_data.sort_values('factor')
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Professional color mapping based on IC value and significance
        def get_bar_color(ic_value, is_significant):
            if ic_value > 0:
                return PROFESSIONAL_COLORS['positive_significant'] if is_significant else PROFESSIONAL_COLORS['positive_insignificant']
            else:
                return PROFESSIONAL_COLORS['negative_significant'] if is_significant else PROFESSIONAL_COLORS['negative_insignificant']
        
        colors = [get_bar_color(ic, sig) for ic, sig in zip(combo_data['mean_ic'], combo_data['significant'])]
        
        # Draw IC bar chart with enhanced styling
        bars = ax.bar(combo_data['factor'], combo_data['mean_ic'], 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
        
        # Add horizontal reference lines with professional styling
        ax.axhline(y=0, color='#2F2F2F', linestyle='-', alpha=0.8, linewidth=1.5)
        ax.axhline(y=0.05, color=PROFESSIONAL_COLORS['excellent'], linestyle='--', alpha=0.7, linewidth=2, label='Excellent IC (>0.05)')
        ax.axhline(y=-0.05, color=PROFESSIONAL_COLORS['excellent'], linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(y=0.02, color=PROFESSIONAL_COLORS['acceptable'], linestyle='--', alpha=0.7, linewidth=2, label='Acceptable IC (>0.02)')
        ax.axhline(y=-0.02, color=PROFESSIONAL_COLORS['acceptable'], linestyle='--', alpha=0.7, linewidth=2)
        
        # Enhanced title with two-line structure
        description = combo_data['description'].iloc[0]
        rolling_window = combo_data['rolling_window'].iloc[0]
        return_period = combo_data['return_period'].iloc[0]
        ax.set_title(f'Strategy Combination {combo}\n{description} (Rolling Window={rolling_window}, Return Period={return_period})', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Professional axis labels
        ax.set_xlabel('Factor Name', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Information Coefficient (IC)', fontsize=14, fontweight='bold')
        
        # Improved x-axis labels
        ax.set_xticklabels(combo_data['factor'], rotation=45, ha='right', fontsize=11)
        
        # Add performance annotations on bars
        for bar, ic, sig, p_val in zip(bars, combo_data['mean_ic'], combo_data['significant'], combo_data['p_value']):
            height = bar.get_height()
            
            # Determine significance stars
            if p_val < 0.01:
                stars = '***'
            elif p_val < 0.05:
                stars = '**'
            elif p_val < 0.1:
                stars = '*'
            else:
                stars = ''
            
            # Add value label with significance
            label_text = f'{height:.3f}{stars}'
            text_color = PROFESSIONAL_COLORS['positive_significant'] if (height > 0 and sig) else PROFESSIONAL_COLORS['negative_significant'] if (height < 0 and sig) else '#666666'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.008 if height >= 0 else -0.008),
                   label_text, ha='center', va='bottom' if height >= 0 else 'top',
                   color=text_color, fontweight='bold', fontsize=10)
            
            # Add performance indicator
            if abs(height) > 0.05:
                indicator = '↑ Excellent' if height > 0 else '↓ Excellent'
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.02),
                       indicator, ha='center', va='bottom' if height >= 0 else 'top',
                       color=PROFESSIONAL_COLORS['excellent'], fontweight='bold', fontsize=8)
        
        # Professional legend
        legend_elements = [
            plt.Line2D([0], [0], color=PROFESSIONAL_COLORS['excellent'], linestyle='--', linewidth=2, label='Excellent IC (>0.05)'),
            plt.Line2D([0], [0], color=PROFESSIONAL_COLORS['acceptable'], linestyle='--', linewidth=2, label='Acceptable IC (>0.02)'),
            plt.Rectangle((0,0),1,1, facecolor=PROFESSIONAL_COLORS['positive_significant'], alpha=0.8, label='Significant Positive'),
            plt.Rectangle((0,0),1,1, facecolor=PROFESSIONAL_COLORS['negative_significant'], alpha=0.8, label='Significant Negative'),
            plt.Rectangle((0,0),1,1, facecolor=PROFESSIONAL_COLORS['positive_insignificant'], alpha=0.8, label='Non-significant')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        # Add footnote
        fig.text(0.99, 0.01, '***p<0.01, **p<0.05, *p<0.1 | IC>0.05 indicates high-efficiency factor', 
                ha='right', va='bottom', fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        # Save with high quality
        save_path = os.path.join(output_dir, f'combo_{combo}_ic_professional.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✅ Professional strategy combination {combo} IC bar chart saved to: {save_path}")

def create_combo_spread_chart(results_df):
    """Create professional quantile spread bar chart for each combination"""
    # Get a consistent factor order based on overall average spread
    factor_order = results_df.dropna(subset=['mean_spread']).groupby('factor')['mean_spread'].mean().abs().sort_values(ascending=False).index.tolist()
    
    for combo in results_df['combo'].unique():
        combo_data = results_df[results_df['combo'] == combo].copy()
        
        # Filter out rows without spread data
        combo_data = combo_data.dropna(subset=['mean_spread'])
        if len(combo_data) == 0:
            continue
        
        # Sort by the consistent factor order
        combo_data['factor'] = pd.Categorical(combo_data['factor'], categories=factor_order, ordered=True)
        combo_data = combo_data.sort_values('factor')
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Professional color mapping for spread
        def get_spread_color(spread_value, is_significant):
            if spread_value > 0:
                return PROFESSIONAL_COLORS['positive_significant'] if is_significant else PROFESSIONAL_COLORS['positive_insignificant']
            else:
                return PROFESSIONAL_COLORS['negative_significant'] if is_significant else PROFESSIONAL_COLORS['negative_insignificant']
        
        colors = [get_spread_color(spread, sig) for spread, sig in zip(combo_data['mean_spread'], combo_data['spread_significant'])]
        
        # Draw spread bar chart with enhanced styling
        bars = ax.bar(combo_data['factor'], combo_data['mean_spread'], 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
        
        # Add horizontal reference lines
        ax.axhline(y=0, color='#2F2F2F', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # Dynamic benchmark lines based on data range
        max_spread = combo_data['mean_spread'].abs().max()
        if max_spread > 0.05:
            ax.axhline(y=0.05, color=PROFESSIONAL_COLORS['excellent'], linestyle='--', alpha=0.7, linewidth=2, label='Excellent Spread (>0.05)')
            ax.axhline(y=-0.05, color=PROFESSIONAL_COLORS['excellent'], linestyle='--', alpha=0.7, linewidth=2)
        if max_spread > 0.02:
            ax.axhline(y=0.02, color=PROFESSIONAL_COLORS['acceptable'], linestyle='--', alpha=0.7, linewidth=2, label='Acceptable Spread (>0.02)')
            ax.axhline(y=-0.02, color=PROFESSIONAL_COLORS['acceptable'], linestyle='--', alpha=0.7, linewidth=2)
        
        # Enhanced title
        description = combo_data['description'].iloc[0]
        rolling_window = combo_data['rolling_window'].iloc[0]
        return_period = combo_data['return_period'].iloc[0]
        ax.set_title(f'Strategy Combination {combo} - Quantile Spread Analysis\n{description} (Rolling Window={rolling_window}, Return Period={return_period})', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Professional axis labels
        ax.set_xlabel('Factor Name', fontsize=14, fontweight='bold')
        ax.set_ylabel('Highest Quantile - Lowest Quantile Average Return', fontsize=14, fontweight='bold')
        
        # Improved x-axis labels
        ax.set_xticklabels(combo_data['factor'], rotation=45, ha='right', fontsize=11)
        
        # Add enhanced value labels with significance
        for bar, spread, sig, p_val in zip(bars, combo_data['mean_spread'], combo_data['spread_significant'], combo_data['spread_p']):
            height = bar.get_height()
            
            # Determine significance stars
            if not pd.isna(p_val):
                if p_val < 0.01:
                    stars = '***'
                elif p_val < 0.05:
                    stars = '**'
                elif p_val < 0.1:
                    stars = '*'
                else:
                    stars = ''
            else:
                stars = ''
            
            # Add value label with significance
            label_text = f'{height:.3f}{stars}'
            text_color = PROFESSIONAL_COLORS['positive_significant'] if (height > 0 and sig) else PROFESSIONAL_COLORS['negative_significant'] if (height < 0 and sig) else '#666666'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.008 if height >= 0 else -0.008),
                   label_text, ha='center', va='bottom' if height >= 0 else 'top',
                   color=text_color, fontweight='bold', fontsize=10)
            
            # Add performance indicator for excellent spreads
            if abs(height) > 0.05:
                indicator = '↑ Excellent' if height > 0 else '↓ Excellent'
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.02),
                       indicator, ha='center', va='bottom' if height >= 0 else 'top',
                       color=PROFESSIONAL_COLORS['excellent'], fontweight='bold', fontsize=8)
        
        # Professional legend
        legend_elements = []
        if max_spread > 0.05:
            legend_elements.append(plt.Line2D([0], [0], color=PROFESSIONAL_COLORS['excellent'], linestyle='--', linewidth=2, label='Excellent Spread (>0.05)'))
        if max_spread > 0.02:
            legend_elements.append(plt.Line2D([0], [0], color=PROFESSIONAL_COLORS['acceptable'], linestyle='--', linewidth=2, label='Acceptable Spread (>0.02)'))
        
        legend_elements.extend([
            plt.Rectangle((0,0),1,1, facecolor=PROFESSIONAL_COLORS['positive_significant'], alpha=0.8, label='Significant Positive'),
            plt.Rectangle((0,0),1,1, facecolor=PROFESSIONAL_COLORS['negative_significant'], alpha=0.8, label='Significant Negative'),
            plt.Rectangle((0,0),1,1, facecolor=PROFESSIONAL_COLORS['positive_insignificant'], alpha=0.8, label='Non-significant')
        ])
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        # Add footnote
        fig.text(0.99, 0.01, '***p<0.01, **p<0.05, *p<0.1 | Spread>0.05 indicates strong stratification effect', 
                ha='right', va='bottom', fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        # Save with high quality
        save_path = os.path.join(output_dir, f'combo_{combo}_spread_professional.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✅ Professional strategy combination {combo} quantile spread chart saved to: {save_path}")

def create_ic_heatmap(results_df):
    """Create professional IC heatmap for all combinations"""
    # Get a consistent factor order based on overall average IC
    factor_order = results_df.groupby('factor')['mean_ic'].mean().abs().sort_values(ascending=False).index.tolist()
    
    # Create pivot table: combo x factor
    pivot_data = results_df.pivot_table(
        index='factor',
        columns='combo',
        values='mean_ic'
    )
    
    # Reorder factors
    pivot_data = pivot_data.reindex(factor_order)
    
    # Create professional heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Use professional colormap with better contrast
    cmap = sns.diverging_palette(250, 10, as_cmap=True, center='light')
    
    # Create heatmap with enhanced styling
    heatmap = sns.heatmap(pivot_data, 
                         annot=True, 
                         cmap=cmap, 
                         center=0, 
                         fmt='.3f',
                         linewidths=0.5,
                         linecolor='white',
                         cbar_kws={'label': 'Average Information Coefficient (IC)', 'shrink': 0.8},
                         annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                         ax=ax)
    
    # Enhanced title
    ax.set_title('Factor IC Heatmap - Strategy Combination Comparison\nFactor Information Coefficient Heatmap Across Strategy Combinations', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Professional axis labels
    ax.set_xlabel('Strategy Combination', fontsize=14, fontweight='bold')
    ax.set_ylabel('Factor Name', fontsize=14, fontweight='bold')
    
    # Improve readability
    ax.set_xticklabels([f'Combo {combo}' for combo in pivot_data.columns], fontsize=12, rotation=0)
    ax.set_yticklabels(pivot_data.index, fontsize=11, rotation=0)
    
    # Add colorbar label styling
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Average Information Coefficient (IC)', fontsize=12, fontweight='bold')
    
    # Add footnote
    fig.text(0.99, 0.01, 'IC>0.05: Excellent | IC>0.02: Acceptable | |IC|<0.02: Weak Effect', 
            ha='right', va='bottom', fontsize=11, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save with high quality
    save_path = os.path.join(output_dir, 'combo_ic_heatmap_professional.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Professional strategy combination IC heatmap saved to: {save_path}")
    
    # Create significance table instead of heatmap
    pvalue_pivot = results_df.pivot_table(
        index='factor',
        columns='combo',
        values='p_value'
    )
    
    # Reorder factors
    pvalue_pivot = pvalue_pivot.reindex(factor_order)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(18, 12))  # Increased width for better factor name display
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for factor in pvalue_pivot.index:
        # Use full factor names without truncation
        row = [factor]
        for combo in pvalue_pivot.columns:
            p_val = pvalue_pivot.loc[factor, combo]
            if pd.isna(p_val):
                row.append('N/A')
            else:
                # Format p-value with significance indicator
                if p_val < 0.01:
                    row.append(f'{p_val:.4f}***')
                elif p_val < 0.05:
                    row.append(f'{p_val:.4f}**')
                elif p_val < 0.1:
                    row.append(f'{p_val:.4f}*')
                else:
                    row.append(f'{p_val:.4f}')
        table_data.append(row)
    
    # Create table
    columns = ['Factor'] + [f'Combo {combo}' for combo in pvalue_pivot.columns]
    table = ax.table(cellText=table_data, colLabels=columns, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Larger font for better readability
    table.scale(1.0, 2.5)  # Increased height for better readability
    
    # Set different column widths - much wider for factor names, narrower for data columns
    for i in range(len(table_data) + 1):  # +1 for header
        table[(i, 0)].set_width(0.6)  # Factor name column - much wider
        for j in range(1, len(columns)):
            table[(i, j)].set_width(0.12)  # Data columns - narrower
    
    # Set white background for the figure
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Color cells based on significance with lighter color scheme
    for i, factor in enumerate(pvalue_pivot.index):
        for j, combo in enumerate(pvalue_pivot.columns):
            p_val = pvalue_pivot.loc[factor, combo]
            if not pd.isna(p_val):
                if p_val < 0.01:
                    table[(i+1, j+1)].set_facecolor('#E8F5E8')  # Very light green
                elif p_val < 0.05:
                    table[(i+1, j+1)].set_facecolor('#F0F8F0')  # Lighter green
                elif p_val < 0.1:
                    table[(i+1, j+1)].set_facecolor('#FFF8E1')  # Very light amber
                else:
                    table[(i+1, j+1)].set_facecolor('#FFF3E0')  # Very light orange
    
    # Style header row with lighter colors
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#E3F2FD')  # Light blue
        table[(0, j)].set_text_props(weight='bold', color='black')
    
    # Style factor name column with very light background
    for i in range(1, len(pvalue_pivot.index) + 1):
        table[(i, 0)].set_facecolor('#F5F5F5')  # Very light gray
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Factor IC Significance Table Across Different Strategy Combinations\n(***p<0.01, **p<0.05, *p<0.1)', 
              fontsize=16, pad=20, weight='bold')
    
    # Save the figure
    save_path = os.path.join(output_dir, 'combo_ic_significance_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Strategy combination IC significance table saved to: {save_path}")

def create_spread_heatmap(results_df):
    """Create quantile spread heatmap for all combinations"""
    # Filter out rows without spread data
    spread_df = results_df.dropna(subset=['mean_spread'])
    if len(spread_df) == 0:
        print("❌ Not enough quantile spread data to create heatmap")
        return
    
    # Get a consistent factor order based on overall average spread
    factor_order = spread_df.groupby('factor')['mean_spread'].mean().abs().sort_values(ascending=False).index.tolist()
    
    # Create pivot table: combo x factor
    pivot_data = spread_df.pivot_table(
        index='factor',
        columns='combo',
        values='mean_spread'
    )
    
    # Reorder factors
    pivot_data = pivot_data.reindex(factor_order)
    
    # Create professional heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Use professional colormap with better contrast
    cmap = sns.diverging_palette(250, 10, as_cmap=True, center='light')
    
    # Create heatmap with enhanced styling
    heatmap = sns.heatmap(pivot_data, 
                         annot=True, 
                         cmap=cmap, 
                         center=0, 
                         fmt='.4f',
                         linewidths=0.5,
                         linecolor='white',
                         cbar_kws={'label': 'Quantile Spread', 'shrink': 0.8},
                         annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                         ax=ax)
    
    # Enhanced title
    ax.set_title('Quantile Spread Heatmap - Strategy Combination Comparison\nQuantile Spread Heatmap Across Strategy Combinations', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Professional axis labels
    ax.set_xlabel('Strategy Combination', fontsize=14, fontweight='bold')
    ax.set_ylabel('Factor Name', fontsize=14, fontweight='bold')
    
    # Improve readability
    ax.set_xticklabels([f'Combo {combo}' for combo in pivot_data.columns], fontsize=12, rotation=0)
    ax.set_yticklabels(pivot_data.index, fontsize=11, rotation=0)
    
    # Add colorbar label styling
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Quantile Spread', fontsize=12, fontweight='bold')
    
    # Add footnote
    fig.text(0.99, 0.01, 'Spread>0: Long strategy effective | Spread<0: Short strategy effective', 
            ha='right', va='bottom', fontsize=11, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save with high quality
    save_path = os.path.join(output_dir, 'combo_spread_heatmap_professional.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Professional quantile spread heatmap saved to: {save_path}")
    
    # Create significance table instead of heatmap
    spread_pvalue_pivot = spread_df.pivot_table(
        index='factor',
        columns='combo',
        values='spread_p'
    )
    
    # Reorder factors
    spread_pvalue_pivot = spread_pvalue_pivot.reindex(factor_order)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(18, 12))  # Increased width for better factor name display
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for factor in spread_pvalue_pivot.index:
        # Use full factor names without truncation
        row = [factor]
        for combo in spread_pvalue_pivot.columns:
            p_val = spread_pvalue_pivot.loc[factor, combo]
            if pd.isna(p_val):
                row.append('N/A')
            else:
                # Format p-value with significance indicator
                if p_val < 0.01:
                    row.append(f'{p_val:.4f}***')
                elif p_val < 0.05:
                    row.append(f'{p_val:.4f}**')
                elif p_val < 0.1:
                    row.append(f'{p_val:.4f}*')
                else:
                    row.append(f'{p_val:.4f}')
        table_data.append(row)
    
    # Create table
    columns = ['Factor'] + [f'Combo {combo}' for combo in spread_pvalue_pivot.columns]
    table = ax.table(cellText=table_data, colLabels=columns, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Larger font for better readability
    table.scale(1.0, 2.5)  # Increased height for better readability
    
    # Set different column widths - much wider for factor names, narrower for data columns
    for i in range(len(table_data) + 1):  # +1 for header
        table[(i, 0)].set_width(0.6)  # Factor name column - much wider
        for j in range(1, len(columns)):
            table[(i, j)].set_width(0.12)  # Data columns - narrower
    
    # Set white background for the figure
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Color cells based on significance with lighter color scheme
    for i, factor in enumerate(spread_pvalue_pivot.index):
        for j, combo in enumerate(spread_pvalue_pivot.columns):
            p_val = spread_pvalue_pivot.loc[factor, combo]
            if not pd.isna(p_val):
                if p_val < 0.01:
                    table[(i+1, j+1)].set_facecolor('#E8F5E8')  # Very light green
                elif p_val < 0.05:
                    table[(i+1, j+1)].set_facecolor('#F0F8F0')  # Lighter green
                elif p_val < 0.1:
                    table[(i+1, j+1)].set_facecolor('#FFF8E1')  # Very light amber
                else:
                    table[(i+1, j+1)].set_facecolor('#FFF3E0')  # Very light orange
    
    # Style header row with lighter colors
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#E3F2FD')  # Light blue
        table[(0, j)].set_text_props(weight='bold', color='black')
    
    # Style factor name column with very light background
    for i in range(1, len(spread_pvalue_pivot.index) + 1):
        table[(i, 0)].set_facecolor('#F5F5F5')  # Very light gray
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Factor Quantile Spread Significance Table Across Different Strategy Combinations\n(***p<0.01, **p<0.05, *p<0.1)', 
              fontsize=16, pad=20, weight='bold')
    
    # Save the figure
    save_path = os.path.join(output_dir, 'combo_spread_significance_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Strategy combination quantile spread significance table saved to: {save_path}")

def create_best_factors_summary(results_df):
    """Create summary of best factors for each combination"""
    # Get a consistent factor order based on overall score
    overall_factor_ranking = results_df.groupby('factor')['overall_score'].mean().sort_values(ascending=False)
    
    # Analyze each combination
    best_factors = []
    
    for combo in results_df['combo'].unique():
        combo_data = results_df[results_df['combo'] == combo].copy()
        
        # Sort by overall score
        combo_data = combo_data.sort_values('overall_score', ascending=False)
        
        # Get top 5 factors
        top_factors = combo_data.head(5)
        
        for _, row in top_factors.iterrows():
            best_factors.append({
                'combo': row['combo'],
                'description': row['description'],
                'factor': row['factor'],
                'rolling_window': row['rolling_window'],
                'return_period': row['return_period'],
                'mean_ic': row['mean_ic'],
                'ic_significant': row['significant'],
                'mean_spread': row['mean_spread'],
                'spread_significant': row['spread_significant'],
                'overall_score': row['overall_score'],
                'rank': len(best_factors) % 5 + 1  # Ranking within each combination
            })
    
    # Create summary DataFrame and save
    if best_factors:
        best_df = pd.DataFrame(best_factors)
        best_df.to_csv(os.path.join(output_dir, 'best_factors_by_combo.csv'), index=False)
        print(f"\n✅ Best factors summary by strategy combination saved to: {os.path.join(output_dir, 'best_factors_by_combo.csv')}")
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Use the same y-axis limits for all subplots
        max_score = best_df['overall_score'].max() * 1.2  # Add 20% margin
        
        for i, combo in enumerate(best_df['combo'].unique()):
            combo_best = best_df[best_df['combo'] == combo]
            
            plt.subplot(len(best_df['combo'].unique()), 1, i+1)
            
            # Create horizontal bar plot
            bars = plt.barh(combo_best['factor'], combo_best['overall_score'], alpha=0.7)
            
            # Set consistent x-axis limits
            plt.xlim(0, max_score)
            
            # For each bar add IC and quantile spread tags
            for j, (_, row) in enumerate(combo_best.iterrows()):
                plt.text(row['overall_score'] + 0.01, j, 
                        f"IC: {row['mean_ic']:.3f} | Spread: {row['mean_spread']:.3f}", 
                        va='center')
            
            # Add title
            description = combo_best['description'].iloc[0]
            plt.title(f'Combo {combo}: {description} - Best Factors')
            
            plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(output_dir, 'best_factors_by_combo.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Best factors chart by strategy combination saved to: {save_path}")
        
        return best_df
    
    return None

def create_camp_gamma_chart(results_df):
    """Create professional CAPM gamma line chart showing trends across different combinations"""
    # Filter out rows without gamma data
    gamma_df = results_df.dropna(subset=['gamma'])
    if len(gamma_df) == 0:
        print("❌ Not enough CAPM gamma data to create line chart")
        return
    
    # Get a consistent factor order based on overall average gamma
    factor_order = gamma_df.groupby('factor')['gamma'].mean().abs().sort_values(ascending=False).index.tolist()
    
    # Create professional line chart
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Define professional markers for better distinction
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    combo_list = sorted(gamma_df['combo'].unique())
    
    for i, factor in enumerate(factor_order[:10]):  # Show top 10 factors to avoid overcrowding
        factor_data = gamma_df[gamma_df['factor'] == factor]
        
        if len(factor_data) < 2:  # Need at least 2 points for a line
            continue
            
        # Prepare data for line plot
        combo_values = []
        gamma_values = []
        significance = []
        
        for combo in combo_list:
            combo_factor_data = factor_data[factor_data['combo'] == combo]
            if len(combo_factor_data) > 0:
                combo_values.append(combo)
                gamma_values.append(combo_factor_data['gamma'].iloc[0])
                significance.append(combo_factor_data['gamma_significant'].iloc[0])
        
        if len(combo_values) >= 2:
            # Create factor abbreviation for cleaner display
            factor_abbrev = factor.replace('_', ' ').title()[:20] + ('...' if len(factor) > 20 else '')
            
            # Use professional colors and styling
            color_list = ['#2E8B57', '#B22222', '#FF8C00', '#006400', '#32CD32', '#DC143C', '#708090', '#90EE90', '#FFB6C1']
            color = color_list[i % len(color_list)]
            marker = markers[i % len(markers)]
            
            # Plot line with enhanced styling
            ax.plot(combo_values, gamma_values, 
                   color=color, marker=marker, linewidth=3, markersize=10,
                   label=factor_abbrev, alpha=0.85,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)
            
            # Mark significant points with enhanced styling
            for j, (combo, gamma, sig) in enumerate(zip(combo_values, gamma_values, significance)):
                if sig:
                    # Significant points: filled with bold border
                    ax.scatter(combo, gamma, color=color, s=120, marker=marker, 
                              edgecolors='black', linewidth=2, zorder=5, alpha=0.9)
                else:
                    # Non-significant points: hollow with colored border
                    ax.scatter(combo, gamma, color='white', s=120, marker=marker, 
                              edgecolors=color, linewidth=3, zorder=5, alpha=0.8)
    
    # Add enhanced reference lines
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.6, linewidth=2, label='Zero Baseline')
    
    # Enhanced title
    ax.set_title('CAPM Gamma Trend Analysis - Strategy Combination Comparison\nCAPM Gamma Trends Across Strategy Combinations\n(Filled markers: Significant | Hollow markers: Non-significant)', 
                fontsize=18, fontweight='bold', pad=30)
    
    # Professional axis labels
    ax.set_xlabel('Strategy Combination', fontsize=16, fontweight='bold')
    ax.set_ylabel('CAPM Gamma Coefficient', fontsize=16, fontweight='bold')
    
    # Enhanced legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                      fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    # Professional grid and styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Enhanced tick styling
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add footnote
    fig.text(0.99, 0.01, 'Gamma>0: Positive factor exposure | Gamma<0: Negative factor exposure | Significance level: p<0.05', 
            ha='right', va='bottom', fontsize=12, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save with high quality
    save_path = os.path.join(output_dir, 'combo_capm_gamma_trends_professional.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Professional CAPM Gamma trend chart saved to: {save_path}")

def create_capm_r2_chart(results_df):
    """Create CAPM R² bar chart for each combination"""
    # Get a consistent factor order based on overall average R²
    factor_order = results_df.dropna(subset=['r_squared']).groupby('factor')['r_squared'].mean().sort_values(ascending=False).index.tolist()
    
    for combo in results_df['combo'].unique():
        combo_data = results_df[results_df['combo'] == combo].copy()
        
        # Filter out rows without R² data
        combo_data = combo_data.dropna(subset=['r_squared'])
        if len(combo_data) == 0:
            continue
        
        # Sort by the consistent factor order
        combo_data['factor'] = pd.Categorical(combo_data['factor'], categories=factor_order, ordered=True)
        combo_data = combo_data.sort_values('factor')
        
        plt.figure(figsize=(12, 8))
        
        # Draw R² bar chart
        bars = plt.bar(combo_data['factor'], combo_data['r_squared'], alpha=0.7)
        
        # Add title and label
        description = combo_data['description'].iloc[0]
        rolling_window = combo_data['rolling_window'].iloc[0]
        return_period = combo_data['return_period'].iloc[0]
        plt.title(f'Combo {combo}: {description} - CAPM R² (Rolling Window={rolling_window}, Return Period={return_period})')
        plt.xlabel('Factor')
        plt.ylabel('CAPM R²')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, r2 in zip(bars, combo_data['r_squared']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(output_dir, f'combo_{combo}_capm_r2.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Combo {combo} CAPM R² bar chart saved to: {save_path}")

def create_capm_gamma_heatmap(results_df):
    """Create professional CAMP gamma heatmap for all combinations"""
    # Filter out rows without gamma data
    gamma_df = results_df.dropna(subset=['gamma'])
    if len(gamma_df) == 0:
        print("❌ Not enough CAPM gamma data to create heatmap")
        return
    
    # Get a consistent factor order based on overall average gamma
    factor_order = gamma_df.groupby('factor')['gamma'].mean().abs().sort_values(ascending=False).index.tolist()
    
    # Create pivot table: combo x factor
    pivot_data = gamma_df.pivot_table(
        index='factor',
        columns='combo',
        values='gamma'
    )
    
    # Reorder factors
    pivot_data = pivot_data.reindex(factor_order)
    
    # Create professional heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Use professional colormap with better contrast
    cmap = sns.diverging_palette(10, 250, as_cmap=True, center='light')
    
    # Create heatmap with enhanced styling
    heatmap = sns.heatmap(pivot_data, 
                         annot=True, 
                         cmap=cmap, 
                         center=0, 
                         fmt='.6f',
                         linewidths=0.5,
                         linecolor='white',
                         cbar_kws={'label': 'CAPM Gamma Coefficient', 'shrink': 0.8},
                         annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                         ax=ax)
    
    # Enhanced title
    ax.set_title('CAPM Gamma Heatmap - Strategy Combination Comparison\nCAPM Gamma Heatmap Across Strategy Combinations', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Professional axis labels
    ax.set_xlabel('Strategy Combination', fontsize=14, fontweight='bold')
    ax.set_ylabel('Factor Name', fontsize=14, fontweight='bold')
    
    # Improve readability
    ax.set_xticklabels([f'Combo {combo}' for combo in pivot_data.columns], fontsize=12, rotation=0)
    ax.set_yticklabels(pivot_data.index, fontsize=11, rotation=0)
    
    # Add colorbar label styling
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('CAPM Gamma Coefficient', fontsize=12, fontweight='bold')
    
    # Add footnote
    fig.text(0.99, 0.01, 'Gamma>0: Positive factor exposure | Gamma<0: Negative factor exposure', 
            ha='right', va='bottom', fontsize=11, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save with high quality
    save_path = os.path.join(output_dir, 'combo_capm_gamma_heatmap_professional.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Professional CAPM Gamma heatmap saved to: {save_path}")
    
    # Create significance table instead of heatmap
    gamma_pvalue_pivot = gamma_df.pivot_table(
        index='factor',
        columns='combo',
        values='gamma_pvalue'
    )
    
    # Reorder factors
    gamma_pvalue_pivot = gamma_pvalue_pivot.reindex(factor_order)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(18, 12))  # Increased width for better factor name display
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for factor in gamma_pvalue_pivot.index:
        # Use full factor names without truncation
        row = [factor]
        for combo in gamma_pvalue_pivot.columns:
            p_val = gamma_pvalue_pivot.loc[factor, combo]
            if pd.isna(p_val):
                row.append('N/A')
            else:
                # Format p-value with significance indicator
                if p_val < 0.01:
                    row.append(f'{p_val:.4f}***')
                elif p_val < 0.05:
                    row.append(f'{p_val:.4f}**')
                elif p_val < 0.1:
                    row.append(f'{p_val:.4f}*')
                else:
                    row.append(f'{p_val:.4f}')
        table_data.append(row)
    
    # Create table
    columns = ['Factor'] + [f'Combo {combo}' for combo in gamma_pvalue_pivot.columns]
    table = ax.table(cellText=table_data, colLabels=columns, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Larger font for better readability
    table.scale(1.0, 2.5)  # Increased height for better readability
    
    # Set different column widths - much wider for factor names, narrower for data columns
    for i in range(len(table_data) + 1):  # +1 for header
        table[(i, 0)].set_width(0.6)  # Factor name column - much wider
        for j in range(1, len(columns)):
            table[(i, j)].set_width(0.12)  # Data columns - narrower
    
    # Set white background for the figure
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Color cells based on significance with lighter color scheme
    for i, factor in enumerate(gamma_pvalue_pivot.index):
        for j, combo in enumerate(gamma_pvalue_pivot.columns):
            p_val = gamma_pvalue_pivot.loc[factor, combo]
            if not pd.isna(p_val):
                if p_val < 0.01:
                    table[(i+1, j+1)].set_facecolor('#E8F5E8')  # Very light green
                elif p_val < 0.05:
                    table[(i+1, j+1)].set_facecolor('#F0F8F0')  # Lighter green
                elif p_val < 0.1:
                    table[(i+1, j+1)].set_facecolor('#FFF8E1')  # Very light amber
                else:
                    table[(i+1, j+1)].set_facecolor('#FFF3E0')  # Very light orange
    
    # Style header row with lighter colors
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#E3F2FD')  # Light blue
        table[(0, j)].set_text_props(weight='bold', color='black')
    
    # Style factor name column with very light background
    for i in range(1, len(gamma_pvalue_pivot.index) + 1):
        table[(i, 0)].set_facecolor('#F5F5F5')  # Very light gray
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Factor CAPM Gamma Significance Table Across Different Strategy Combinations\n(***p<0.01, **p<0.05, *p<0.1)', 
              fontsize=16, pad=20, weight='bold')
    
    # Save the figure
    save_path = os.path.join(output_dir, 'combo_capm_gamma_significance_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Strategy combination CAPM gamma significance table saved to: {save_path}")

def create_capm_r2_heatmap(results_df):
    """Create CAPM R² heatmap for all combinations"""
    # Filter out rows without R² data
    r2_df = results_df.dropna(subset=['r_squared'])
    if len(r2_df) == 0:
        print("❌ Not enough CAPM R² data to create heatmap")
        return
    
    # Get a consistent factor order based on overall average R²
    factor_order = r2_df.groupby('factor')['r_squared'].mean().sort_values(ascending=False).index.tolist()
    
    # Create pivot table: combo x factor
    pivot_data = r2_df.pivot_table(
        index='factor',
        columns='combo',
        values='r_squared'
    )
    
    # Reorder factors
    pivot_data = pivot_data.reindex(factor_order)
    
    # Draw heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, annot=True, cmap='Blues', fmt='.3f')
    plt.title('Factor CAPM R² Heatmap Across Different Strategy Combinations')
    plt.xlabel('Strategy Combination')
    plt.ylabel('Factor')
    
    # Save the figure
    save_path = os.path.join(output_dir, 'combo_capm_r2_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Strategy combination CAPM R² heatmap saved to: {save_path}")

def create_factor_matrix_heatmaps(results_df):
    """Create 2D matrix heatmaps for each factor showing metrics across different window and return period combinations"""
    # Get unique factors, rolling windows, and return periods
    factors = results_df['factor'].unique()
    rolling_windows = sorted(results_df['rolling_window'].unique())
    return_periods = sorted(results_df['return_period'].unique())
    
    # Create a directory for factor matrix heatmaps
    factor_matrix_dir = os.path.join(output_dir, 'factor_matrix_heatmaps')
    os.makedirs(factor_matrix_dir, exist_ok=True)
    
    print(f"\n📊 Creating factor matrix heatmaps for {len(factors)} factors...")
    
    # For each factor, create a set of heatmaps
    for factor in factors:
        factor_data = results_df[results_df['factor'] == factor].copy()
        
        if len(factor_data) < 2:  # Need at least 2 data points for a meaningful heatmap
            print(f"⚠️ Skipping factor {factor}: insufficient data points")
            continue
            
        # Create a figure with 2x2 subplots for the 4 metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Factor Analysis Matrix: {factor}', fontsize=16)
        
        # 1. Average IC Heatmap
        ax = axes[0, 0]
        try:
            pivot_ic = factor_data.pivot_table(
                index='rolling_window',
                columns='return_period',
                values='mean_ic'
            )
            if not pivot_ic.empty and not pivot_ic.isnull().all().all():
                sns.heatmap(pivot_ic, annot=True, cmap='RdBu_r', center=0, fmt='.3f', ax=ax)
                ax.set_title('Average IC')
                ax.set_xlabel('Return Period (days)')
                ax.set_ylabel('Rolling Window (days)')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                ax.set_title('Average IC (No Data)')
                ax.axis('off')
        except Exception as e:
            print(f"⚠️ Error creating IC heatmap for {factor}: {e}")
            ax.text(0.5, 0.5, 'Error creating heatmap', ha='center', va='center', fontsize=12)
            ax.set_title('Average IC (Error)')
            ax.axis('off')
        
        # 2. Q5-Q1 Spread Heatmap
        ax = axes[0, 1]
        try:
            # Filter out NaN values for spread
            spread_data = factor_data.dropna(subset=['mean_spread'])
            if len(spread_data) >= 2:
                pivot_spread = spread_data.pivot_table(
                    index='rolling_window',
                    columns='return_period',
                    values='mean_spread'
                )
                if not pivot_spread.empty and not pivot_spread.isnull().all().all():
                    sns.heatmap(pivot_spread, annot=True, cmap='RdBu_r', center=0, fmt='.3f', ax=ax)
                    ax.set_title('Q5-Q1 Average Return')
                    ax.set_xlabel('Return Period (days)')
                    ax.set_ylabel('Rolling Window (days)')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                    ax.set_title('Q5-Q1 Average Return (No Data)')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                ax.set_title('Q5-Q1 Average Return (No Data)')
                ax.axis('off')
        except Exception as e:
            print(f"⚠️ Error creating spread heatmap for {factor}: {e}")
            ax.text(0.5, 0.5, 'Error creating heatmap', ha='center', va='center', fontsize=12)
            ax.set_title('Q5-Q1 Average Return (Error)')
            ax.axis('off')
        
        # 3. CAPM Alpha (Gamma) Heatmap
        ax = axes[1, 0]
        try:
            # Filter out NaN values for gamma
            gamma_data = factor_data.dropna(subset=['gamma'])
            if len(gamma_data) >= 2:
                pivot_gamma = gamma_data.pivot_table(
                    index='rolling_window',
                    columns='return_period',
                    values='gamma'
                )
                if not pivot_gamma.empty and not pivot_gamma.isnull().all().all():
                    sns.heatmap(pivot_gamma, annot=True, cmap='RdBu_r', center=0, fmt='.6f', ax=ax)
                    ax.set_title('CAPM Alpha (Gamma)')
                    ax.set_xlabel('Return Period (days)')
                    ax.set_ylabel('Rolling Window (days)')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                    ax.set_title('CAPM Alpha (Gamma) (No Data)')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                ax.set_title('CAPM Alpha (Gamma) (No Data)')
                ax.axis('off')
        except Exception as e:
            print(f"⚠️ Error creating gamma heatmap for {factor}: {e}")
            ax.text(0.5, 0.5, 'Error creating heatmap', ha='center', va='center', fontsize=12)
            ax.set_title('CAPM Alpha (Gamma) (Error)')
            ax.axis('off')
        
        # 4. Calculate and plot Sharpe ratio (using mean_spread / std_ic as a proxy)
        ax = axes[1, 1]
        try:
            # Calculate Sharpe ratio and handle potential division by zero or NaN values
            factor_data['sharpe_ratio'] = factor_data.apply(
                lambda row: row['mean_spread'] / row['std_ic'] if not np.isnan(row['mean_spread']) and not np.isnan(row['std_ic']) and row['std_ic'] > 0 else np.nan, 
                axis=1
            )
            
            # Filter out NaN values for Sharpe ratio
            sharpe_data = factor_data.dropna(subset=['sharpe_ratio'])
            if len(sharpe_data) >= 2:
                pivot_sharpe = sharpe_data.pivot_table(
                    index='rolling_window',
                    columns='return_period',
                    values='sharpe_ratio'
                )
                if not pivot_sharpe.empty and not pivot_sharpe.isnull().all().all():
                    sns.heatmap(pivot_sharpe, annot=True, cmap='RdBu_r', center=0, fmt='.3f', ax=ax)
                    ax.set_title('Sharpe Ratio (Q5-Q1 Return / StdDev)')
                    ax.set_xlabel('Return Period (days)')
                    ax.set_ylabel('Rolling Window (days)')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                    ax.set_title('Sharpe Ratio (No Data)')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
                ax.set_title('Sharpe Ratio (No Data)')
                ax.axis('off')
        except Exception as e:
            print(f"⚠️ Error creating Sharpe ratio heatmap for {factor}: {e}")
            ax.text(0.5, 0.5, 'Error creating heatmap', ha='center', va='center', fontsize=12)
            ax.set_title('Sharpe Ratio (Error)')
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save the figure
        save_path = os.path.join(factor_matrix_dir, f'factor_matrix_{factor}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Factor matrix heatmap for {factor} saved to: {save_path}")

def main():
    print("\n🚀 Starting Strategy Combination Template Analysis...")
    
    # Get available factors and return periods
    factors = get_available_factors()
    return_periods = get_available_return_periods()
    
    print(f"\n📋 Found {len(factors)} available factors")
    print(f"📋 Found {len(return_periods)} available return periods: {', '.join(return_periods)}")
    
    all_results = []
    
    # Analyze each combination template
    for combo in COMBO_TEMPLATES:
        results = analyze_combo_template(combo, factors, return_periods)
        if results is not None:
            all_results.append(results)
    
    if all_results:
        # Merge all results
        combined_results = pd.concat(all_results)
        combined_results.to_csv(os.path.join(output_dir, 'all_combo_results.csv'), index=False)
        print(f"\n✅ All strategy combination results saved to: {os.path.join(output_dir, 'all_combo_results.csv')}")
        
        # Create various visualizations
        create_combo_ic_chart(combined_results)
        create_combo_spread_chart(combined_results)
        create_ic_heatmap(combined_results)
        create_spread_heatmap(combined_results)
        create_best_factors_summary(combined_results)
        
        # Create CAPM visualizations
        create_camp_gamma_chart(combined_results)
        create_capm_r2_chart(combined_results)
        create_capm_gamma_heatmap(combined_results)
        create_capm_r2_heatmap(combined_results)
        
        # Create factor matrix heatmaps
        create_factor_matrix_heatmaps(combined_results)
    else:
        print("❌ Failed to generate any strategy combination results")
    
    print("\n✅ Strategy combination template analysis completed!")

if __name__ == "__main__":
    main()