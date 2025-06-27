import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import re
import numpy as np
from matplotlib.table import Table

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建可视化结果目录
visualization_dir = '/Users/queenwen/Desktop/QI_Paper/factor_visualizations'
os.makedirs(visualization_dir, exist_ok=True)

# 因子名称映射（英文到中文）
factor_name_mapping = {
    'amount_in_top_holders': '大户持仓量',
    'dev_activity_1d': '开发活跃度(日)',
    'dev_activity_contributors_count': '开发贡献者数量',
    'exchange_balance': '交易所余额',
    'exchange_inflow_usd': '交易所流入(USD)',
    'exchange_outflow_usd': '交易所流出(USD)',
    'github_activity_1d': 'GitHub活跃度(日)',
    'sentiment_weighted_total_1d': '情绪加权总分(日)',
    'social_volume_total': '社交媒体总量',
    'whale_transaction_count_100k_usd_to_inf': '鲸鱼交易数量',
    'whale_transaction_volume_100k_usd_to_inf': '鲸鱼交易量'
}

# 收益周期映射
return_period_mapping = {
    '1d': '1D Forward Return',
    '3d': '3D Forward Acc. Return',
    '7d': '7D Forward Acc. Return',
    '14d': '14D Forward Acc. Return',
    '21d': '21D Forward Acc. Return',
    '30d': '30D Forward Acc. Return',
    '60d': '60D Forward Acc. Return'
}

def plot_ic_curve(ic_df, factor_name, return_period, save_dir):
    """绘制IC曲线图"""
    plt.figure(figsize=(10, 5))
    
    # 确保日期列是日期时间类型
    ic_df['date'] = pd.to_datetime(ic_df['date'])
    
    # 计算滚动IC（如果数据中没有）
    if ic_df['rolling_spearman_ic'].isnull().all():
        ic_df['rolling_spearman_ic'] = ic_df['spearman_ic'].rolling(window=20).mean()
    
    # 绘制IC曲线
    plt.plot(ic_df['date'], ic_df['rolling_spearman_ic'], label='滚动Spearman IC', color='blue')
    plt.axhline(0, color='gray', linestyle=':')
    
    # 获取中文因子名称和收益周期
    cn_factor_name = factor_name_mapping.get(factor_name, factor_name)
    cn_return_period = return_period_mapping.get(return_period, return_period)
    
    plt.title(f"{cn_factor_name} - IC曲线 ({cn_return_period})")
    plt.xlabel("日期")
    plt.ylabel("信息系数")
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{factor_name}_ic_curve_{return_period}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ IC曲线已保存: {save_path}")

def plot_quantile_returns(quantile_df, factor_name, return_period, save_dir):
    """绘制分位数收益柱状图"""
    # 确保日期列是日期时间类型
    if 'date' in quantile_df.columns:
        date_col = 'date'
    else:
        # 找到日期列（可能是最后一列）
        for col in quantile_df.columns:
            if pd.api.types.is_datetime64_any_dtype(quantile_df[col]) or 'date' in col.lower():
                date_col = col
                break
        else:
            # 如果没有找到日期列，假设最后一列是日期
            date_col = quantile_df.columns[-1]
    
    # 计算每个分位数的平均收益
    quantile_cols = [col for col in quantile_df.columns if col != date_col]
    quantile_means = quantile_df[quantile_cols].mean()
    
    # 确保分位数是字符串类型
    quantile_means.index = quantile_means.index.astype(str)
    
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=quantile_means.index, y=quantile_means.values, palette='viridis')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(quantile_means.values):
        ax.text(i, v + (0.01 if v >= 0 else -0.01), 
                f"{v:.4f}", ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=8)
    
    # 获取中文因子名称和收益周期
    cn_factor_name = factor_name_mapping.get(factor_name, factor_name)
    cn_return_period = return_period_mapping.get(return_period, return_period)
    
    plt.title(f"{cn_factor_name} - 分位数平均收益 ({cn_return_period})")
    plt.xlabel("分位数")
    plt.ylabel("平均收益")
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{factor_name}_quantile_bar_{return_period}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 分位数收益图已保存: {save_path}")

def summarize_ic_and_quantile(ic_df, quantile_df, factor_name, return_period, save_dir):
    """汇总IC和分位数分析结果"""
    # IC汇总
    ic_summary = {
        'factor_name': factor_name,
        'return_period': return_period,
        'mean_spearman_ic': ic_df['spearman_ic'].mean(),
        'std_spearman_ic': ic_df['spearman_ic'].std()
    }
    
    # 分位数汇总
    # 确定日期列
    if 'date' in quantile_df.columns:
        date_col = 'date'
    else:
        # 找到日期列（可能是最后一列）
        for col in quantile_df.columns:
            if pd.api.types.is_datetime64_any_dtype(quantile_df[col]) or 'date' in col.lower():
                date_col = col
                break
        else:
            # 如果没有找到日期列，假设最后一列是日期
            date_col = quantile_df.columns[-1]
    
    # 计算分位数平均收益
    quantile_cols = [col for col in quantile_df.columns if col != date_col]
    q_mean = quantile_df[quantile_cols].mean()
    
    # 计算分位数价差（最高分位数与最低分位数的收益差）
    quantile_spread = q_mean.max() - q_mean.min()
    ic_summary['quantile_spread'] = quantile_spread
    ic_summary['top_quantile_return'] = q_mean[q_mean.idxmax()]
    ic_summary['bottom_quantile_return'] = q_mean[q_mean.idxmin()]
    ic_summary['top_quantile'] = str(q_mean.idxmax())
    ic_summary['bottom_quantile'] = str(q_mean.idxmin())
    
    # 创建汇总DataFrame并保存
    summary_df = pd.DataFrame([ic_summary])
    summary_path = os.path.join(save_dir, f"{factor_name}_summary_{return_period}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ 汇总数据已保存: {summary_path}")
    
    return summary_df

def process_all_factors():
    """处理所有因子的IC和分位数分析"""
    # 数据目录
    ic_dir = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/IC_analysis'
    quantile_dir = '/Users/queenwen/Desktop/QI_Paper/market_factor_validation_results/Quantile_analysis'
    
    # 获取所有IC文件
    ic_files = glob.glob(os.path.join(ic_dir, 'ic_*.csv'))
    
    # 创建一个列表来存储所有汇总数据
    all_summaries = []
    
    # 处理每个IC文件
    for ic_file in ic_files:
        # 从文件名中提取因子名称和收益周期
        file_name = os.path.basename(ic_file)
        match = re.match(r'ic_(.+)_(\d+d)\.csv', file_name)
        
        if match:
            factor_name = match.group(1)
            return_period = match.group(2)
            
            # 构建对应的分位数文件名
            quantile_file = os.path.join(quantile_dir, f"quantile_returns_{factor_name}_{return_period}.csv")
            
            # 检查分位数文件是否存在
            if os.path.exists(quantile_file):
                print(f"\n处理因子: {factor_name}, 收益周期: {return_period}")
                
                # 读取IC和分位数数据
                ic_df = pd.read_csv(ic_file)
                quantile_df = pd.read_csv(quantile_file)
                
                # 为该因子创建子目录
                factor_dir = os.path.join(visualization_dir, factor_name)
                os.makedirs(factor_dir, exist_ok=True)
                
                # 绘制IC曲线
                plot_ic_curve(ic_df, factor_name, return_period, factor_dir)
                
                # 绘制分位数收益图
                plot_quantile_returns(quantile_df, factor_name, return_period, factor_dir)
                
                # 汇总数据
                summary = summarize_ic_and_quantile(ic_df, quantile_df, factor_name, return_period, factor_dir)
                all_summaries.append(summary)
            else:
                print(f"⚠️ 未找到分位数文件: {quantile_file}")
    
    # 合并所有汇总数据并保存
    if all_summaries:
        all_summary_df = pd.concat(all_summaries, ignore_index=True)
        
        # 添加中文因子名称和收益周期
        all_summary_df['factor_name_cn'] = all_summary_df['factor_name'].map(factor_name_mapping)
        all_summary_df['return_period_cn'] = all_summary_df['return_period'].map(return_period_mapping)
        
        # 保存总汇总文件
        all_summary_path = os.path.join(visualization_dir, 'all_factors_summary.csv')
        all_summary_df.to_csv(all_summary_path, index=False)
        print(f"\n✅ 所有因子汇总数据已保存: {all_summary_path}")
        
        # 生成所有因子的验证摘要表格图片
        generate_all_factors_validation_summary(all_summary_df)

def plot_factor_validation_summary_table(factor_name, summary_df, save_dir):
    """为指定因子生成验证摘要表格图片"""
    # 创建一个新的DataFrame，只包含需要的列和行
    table_data = []
    
    # 按照收益周期排序（1d, 3d, 7d, 14d, 21d, 30d, 60d）
    period_order = ['1d', '3d', '7d', '14d', '21d', '30d', '60d']
    sorted_df = summary_df.sort_index(key=lambda x: pd.Categorical(x, categories=period_order))
    
    # 遍历每个收益周期
    for _, row in sorted_df.iterrows():
        return_period = row['return_period']
        # 计算ICIR (Information Coefficient Information Ratio)
        icir = row['mean_spearman_ic'] / row['std_spearman_ic'] if row['std_spearman_ic'] != 0 else 0
        
        # 获取分位数收益
        q1_ret = row['bottom_quantile_return'] if row['bottom_quantile'] == '1' else None
        q2_ret = None
        q3_ret = None
        q4_ret = None
        q5_ret = None
        
        # 根据top_quantile和bottom_quantile确定各分位数的收益
        for i in range(1, 6):
            if str(i) == row['top_quantile']:
                if i == 1:
                    q1_ret = row['top_quantile_return']
                elif i == 2:
                    q2_ret = row['top_quantile_return']
                elif i == 3:
                    q3_ret = row['top_quantile_return']
                elif i == 4:
                    q4_ret = row['top_quantile_return']
                elif i == 5:
                    q5_ret = row['top_quantile_return']
            
            if str(i) == row['bottom_quantile']:
                if i == 1:
                    q1_ret = row['bottom_quantile_return']
                elif i == 2:
                    q2_ret = row['bottom_quantile_return']
                elif i == 3:
                    q3_ret = row['bottom_quantile_return']
                elif i == 4:
                    q4_ret = row['bottom_quantile_return']
                elif i == 5:
                    q5_ret = row['bottom_quantile_return']
        
        # 如果某些分位数的收益未知，使用线性插值估计
        all_returns = [q1_ret, q2_ret, q3_ret, q4_ret, q5_ret]
        known_indices = [i for i, ret in enumerate(all_returns) if ret is not None]
        known_values = [all_returns[i] for i in known_indices]
        
        for i in range(5):
            if all_returns[i] is None:
                # 简单线性插值
                if len(known_indices) >= 2:
                    all_returns[i] = np.interp(i, known_indices, known_values)
                else:
                    all_returns[i] = 0  # 如果无法插值，设为0
        
        q1_ret, q2_ret, q3_ret, q4_ret, q5_ret = all_returns
        
        # 添加到表格数据
        table_data.append([
            return_period_mapping.get(return_period, return_period),
            row['mean_spearman_ic'],
            icir,
            q1_ret,
            q2_ret,
            q3_ret,
            q4_ret,
            q5_ret
        ])
    
    # 创建表格
    fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(table_data)))
    ax.axis('off')
    
    # 表格标题
    cn_factor_name = factor_name_mapping.get(factor_name, factor_name)
    plt.suptitle(f"Factor Validation Summary: {cn_factor_name} (Lag1 Z-Score)", fontsize=14, y=0.95)
    
    # 表格列标题
    columns = ['Return Period', 'IC Mean', 'ICIR', 'Q1 Ret', 'Q2 Ret', 'Q3 Ret', 'Q4 Ret', 'Q5 Ret']
    
    # 创建表格
    table = ax.table(
        cellText=[[f"{x:.4f}" if isinstance(x, float) else x for x in row] for row in table_data],
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#4472C4'] * len(columns),
        colLoc='center'
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 设置表格文本颜色为白色（表头）
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_text_props(color='white', fontweight='bold')
    
    # 设置表格边框和列宽
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if key[0] == 0:  # 表头行
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_facecolor('#4472C4')
        else:  # 数据行
            if key[1] == 0:  # 第一列（收益周期）
                cell.set_text_props(fontweight='bold')
    
    # 调整第一列的宽度
    table.auto_set_column_width([0])
    
    # 手动设置列宽，第一列宽度增加
    col_widths = [0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    for i, width in enumerate(col_widths):
        for row in range(len(table_data) + 1):
            table.get_celld()[(row, i)].set_width(width)
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{factor_name}_validation_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 因子验证摘要表格已保存: {save_path}")

def generate_all_factors_validation_summary(all_summary_df):
    """为所有因子生成验证摘要表格图片"""
    print("\n开始生成所有因子的验证摘要表格...")
    
    # 获取所有唯一的因子名称
    factor_names = all_summary_df['factor_name'].unique()
    
    for factor_name in factor_names:
        # 获取该因子的所有收益周期数据
        factor_df = all_summary_df[all_summary_df['factor_name'] == factor_name]
        
        # 为该因子创建子目录
        factor_dir = os.path.join(visualization_dir, factor_name)
        os.makedirs(factor_dir, exist_ok=True)
        
        # 生成验证摘要表格图片
        plot_factor_validation_summary_table(factor_name, factor_df, factor_dir)
    
    print(f"\n✅ 所有因子的验证摘要表格已生成!")

# 执行主函数
if __name__ == "__main__":
    print("开始生成因子IC和分位数可视化...")
    process_all_factors()
    print("\n✅ 所有因子的IC和分位数可视化已完成!")