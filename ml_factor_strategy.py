import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import glob
from datetime import datetime, timedelta
import os

# 从环境变量获取参数，如果没有则使用默认值
def get_env_var(var_name, default_value):
    return os.environ.get(var_name, default_value)

# 设置输出目录
output_dir = get_env_var('OUTPUT_DIR', '/Users/queenwen/Desktop/QI_Paper/ml_strategy_results')
os.makedirs(output_dir, exist_ok=True)

# 数据源目录
data_dir = '/Users/queenwen/Desktop/QI_Paper'
normalized_data_dir = os.path.join(data_dir, 'normalized_factors', 'IC_analysis')
return_data_dir = os.path.join(data_dir, 'return_data')
market_return_dir = os.path.join(return_data_dir, 'market_returns')

# 从环境变量获取参数
return_period = get_env_var('HOLDING_PERIOD', '3d')
model_type = get_env_var('MODEL_TYPE', 'lasso')
rolling_window = int(get_env_var('ROLLING_WINDOW', '30'))
prediction_window = int(get_env_var('PREDICTION_WINDOW', '3'))
top_pct = float(get_env_var('TOP_PCT', '0.2'))
bottom_pct = float(get_env_var('BOTTOM_PCT', '0.2'))

# 获取筛选后的高解释力因子
def get_available_factors():
    """获取筛选后的高解释力因子"""
    # 筛选出的高解释力因子列表
    selected_factors = [
        'exchange_inflow_usd',
        'social_volume_total', 
        'whale_transaction_count_100k_usd_to_inf',
        'sentiment_weighted_total_1d',
        'whale_transaction_volume_100k_usd_to_inf',
        'exchange_outflow_usd'  # 反向因子
    ]
    
    factors = []
    
    # 只加载筛选出的因子
    for factor_name in selected_factors:
        factor_file = f"{factor_name}.csv"
        factor_path = os.path.join(normalized_data_dir, factor_file)
        
        # 检查文件是否存在
        if os.path.exists(factor_path):
            factors.append((factor_name, factor_path))
            print(f"✅ 找到因子文件: {factor_name}")
        else:
            print(f"⚠️  警告: 因子文件不存在: {factor_path}")
    
    print(f"\n📊 共筛选出 {len(factors)} 个高解释力因子用于机器学习")
    return factors

# 加载因子数据
def load_factor_data(factor_path):
    """加载因子数据"""
    df = pd.read_csv(factor_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

# 加载收益数据
def load_return_data(return_period):
    """加载收益数据"""
    return_file = os.path.join(return_data_dir, f'returns_{return_period}.csv')
    df = pd.read_csv(return_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # 只保留log_return列
    log_return_cols = [col for col in df.columns if f'log_return_{return_period}' in col]
    return df[log_return_cols]

# 加载市场收益数据
def load_market_return_data(return_period):
    """加载市场收益数据"""
    market_return_file = os.path.join(market_return_dir, f'market_log_return_{return_period}.csv')
    df = pd.read_csv(market_return_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# 准备模型训练数据
def prepare_model_data(
    factors_data, return_data, rolling_window=30, fixed_tokens=None):
    """准备模型训练数据"""
    # 检查输入数据
    if not factors_data or return_data.empty:
        print("错误: 输入数据为空，无法准备模型数据")
        return np.array([]).reshape(0, 0), np.array([]), [], [], None
    
    # 合并所有因子数据
    factor_dfs = []
    
    for factor_name, factor_df in factors_data:
        # 重命名列以包含因子名称
        renamed_df = factor_df.copy()
        renamed_df.columns = [f'{factor_name}_{col.split("_")[0]}' for col in factor_df.columns]
        factor_dfs.append(renamed_df)
    
    # 使用concat一次性合并所有因子数据
    all_factors = pd.concat(factor_dfs, axis=1, sort=False)
    all_factors = all_factors.loc[:, ~all_factors.columns.duplicated()]  # 去除重复列
    
    # 对因子数据进行清理和标准化处理
    print(f"清理因子数据，原始形状: {all_factors.shape}")
    # 替换无穷大值和极大值
    all_factors = all_factors.replace([np.inf, -np.inf], np.nan)
    # 对极大值进行截断（99.9%分位数）
    for col in all_factors.columns:
        if all_factors[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            upper_bound = all_factors[col].quantile(0.999)
            lower_bound = all_factors[col].quantile(0.001)
            all_factors[col] = all_factors[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 填充缺失值
    all_factors = all_factors.fillna(0)
    
    # 标准化处理
    scaler = StandardScaler()
    all_factors_scaled = pd.DataFrame(
        scaler.fit_transform(all_factors),
        index=all_factors.index,
        columns=all_factors.columns
    )
    
    # 再次检查并清理标准化后的数据
    all_factors_scaled = all_factors_scaled.replace([np.inf, -np.inf], 0)
    all_factors_scaled = all_factors_scaled.fillna(0)
    
    # 合并因子数据和收益数据
    merged_data = pd.merge(all_factors_scaled, return_data, left_index=True, right_index=True, how='inner')
    
    # 如果没有提供固定代币列表，则从数据中提取
    if fixed_tokens is None:
        # 获取所有可用的代币
        return_columns = [col for col in merged_data.columns if 'log_return' in col]
        fixed_tokens = sorted(list(set([col.split('_')[0] for col in return_columns])))
        print(f"检测到 {len(fixed_tokens)} 个代币: {fixed_tokens[:10]}...")
    
    # 创建训练集
    X = []
    y = []
    dates = []
    tokens = []
    
    # 检查数据是否足够
    if len(merged_data.index) <= rolling_window:
        print(f"警告: 数据量不足，无法准备训练数据。数据长度: {len(merged_data.index)}, 需要至少: {rolling_window + 1}")
        return np.array([]).reshape(0, 0), np.array([]), [], [], fixed_tokens
    
    for date in merged_data.index[rolling_window:]:
        try:
            # 获取当前日期之前的rolling_window天的数据作为特征
            window_data = merged_data.loc[:date].iloc[-rolling_window-1:-1]
            
            # 获取当前日期的收益作为目标
            current_returns = merged_data.loc[date, [col for col in merged_data.columns if 'log_return' in col]]
            
            for token in fixed_tokens:
                return_col = f'{token}_log_return'
                if return_col in current_returns.index:
                    # 获取该代币在窗口期内的所有因子值，按固定顺序
                    token_feature_cols = []
                    for factor_name, _ in factors_data:
                        factor_col = f'{factor_name}_{token}'
                        if factor_col in window_data.columns:
                            token_feature_cols.append(factor_col)
                    
                    if token_feature_cols:
                        token_features = window_data[token_feature_cols]
                        feature_vector = token_features.values.flatten()
                        
                        # 检查特征向量是否包含NaN
                        if not np.isnan(feature_vector).any() and not np.isnan(current_returns[return_col]):
                            X.append(feature_vector)
                            y.append(current_returns[return_col])
                            dates.append(date)
                            tokens.append(token)
        except Exception as e:
            print(f"处理日期 {date} 时出错: {str(e)}")
    
    # 检查是否有足够的样本
    if len(X) == 0:
        print("警告: 没有有效的训练样本")
        return np.array([]).reshape(0, 0), np.array([]), [], [], fixed_tokens
    
    # 确保所有特征向量长度一致
    feature_lengths = [len(x) for x in X]
    if len(set(feature_lengths)) > 1:
        print(f"警告: 特征向量长度不一致: {set(feature_lengths)}")
        # 找出最常见的长度
        from collections import Counter
        most_common_length = Counter(feature_lengths).most_common(1)[0][0]
        print(f"使用最常见的特征长度: {most_common_length}")
        # 只保留长度一致的样本
        valid_indices = [i for i, length in enumerate(feature_lengths) if length == most_common_length]
        X = [X[i] for i in valid_indices]
        y = [y[i] for i in valid_indices]
        dates = [dates[i] for i in valid_indices]
        tokens = [tokens[i] for i in valid_indices]
    
    # 转换为numpy数组
    X_array = np.array(X)
    y_array = np.array(y)
    
    print(f"准备了 {len(X_array)} 个训练样本，特征维度: {X_array.shape[1] if X_array.shape[0] > 0 else 0}")
    
    return X_array, y_array, dates, tokens, fixed_tokens

# 训练Lasso模型
def train_lasso_model(X, y, alpha=0.01):
    """训练Lasso模型"""
    # 检查输入数据
    if X is None or y is None:
        print("错误: 输入数据为空，无法训练Lasso模型")
        return None
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("输入数据为空")
    
    # 检查X和y的长度是否一致
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X和y的长度不一致: X.shape={X.shape}, y.shape={y.shape}")
    
    try:
        # 尝试不同的alpha值，如果模型不收敛
        alphas = [alpha, 0.1, 1.0, 10.0]
        for a in alphas:
            try:
                model = Lasso(alpha=a, max_iter=10000, tol=1e-3)
                model.fit(X, y)
                # 检查模型是否收敛
                if model.n_iter_ < 10000:
                    print(f"Lasso模型使用alpha={a}收敛，迭代次数: {model.n_iter_}")
                    return model
                else:
                    print(f"Lasso模型使用alpha={a}未收敛，尝试更大的alpha值")
            except Exception as e:
                print(f"使用alpha={a}训练Lasso模型失败: {str(e)}")
        
        # 如果所有alpha值都不收敛，使用最后一个尝试的alpha值
        print(f"所有alpha值都未能使模型收敛，使用alpha={alphas[-1]}")
        model = Lasso(alpha=alphas[-1], max_iter=10000, tol=1e-2)
        model.fit(X, y)
        return model
    except Exception as e:
        raise ValueError(f"训练Lasso模型失败: {str(e)}")

# 训练LightGBM模型
def train_lightgbm_model(X, y):
    """训练LightGBM模型"""
    try:
        # 条件导入lightgbm
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM未安装或无法导入。请安装LightGBM: pip install lightgbm")
        
        # 检查输入数据
        if X is None or y is None:
            print("错误: 输入数据为空，无法训练LightGBM模型")
            return None
        
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("输入数据为空")
        
        # 检查X和y的长度是否一致
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y的长度不一致: X.shape={X.shape}, y.shape={y.shape}")
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # 如果样本数量较少，减少模型复杂度
        if X.shape[0] < 100:
            print(f"样本数量较少 ({X.shape[0]}), 减少模型复杂度")
            params.update({
                'num_leaves': 7,
                'min_data_in_leaf': 3,
                'learning_rate': 0.01
            })
        
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=100)
        return model
    except Exception as e:
        raise ValueError(f"训练LightGBM模型失败: {str(e)}")

# 滚动训练和预测
def rolling_train_predict(factors_data, return_data, model_type='lasso', rolling_window=30, prediction_window=30):
    """滚动训练和预测"""
    print("开始准备因子数据...")
    # 合并所有因子数据
    factor_dfs = []
    
    for factor_name, factor_df in factors_data:
        # 重命名列以包含因子名称
        renamed_df = factor_df.copy()
        renamed_df.columns = [f'{factor_name}_{col.split("_")[0]}' for col in factor_df.columns]
        factor_dfs.append(renamed_df)
    
    # 使用concat一次性合并所有因子数据
    all_factors = pd.concat(factor_dfs, axis=1, sort=False)
    all_factors = all_factors.loc[:, ~all_factors.columns.duplicated()]  # 去除重复列
    
    # 对因子数据进行清理和标准化处理
    print(f"清理因子数据，原始形状: {all_factors.shape}")
    # 替换无穷大值和极大值
    all_factors = all_factors.replace([np.inf, -np.inf], np.nan)
    # 对极大值进行截断（99.9%分位数）
    for col in all_factors.columns:
        if all_factors[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            upper_bound = all_factors[col].quantile(0.999)
            lower_bound = all_factors[col].quantile(0.001)
            all_factors[col] = all_factors[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 填充缺失值
    all_factors = all_factors.fillna(0)
    
    # 标准化处理
    print(f"对因子数据进行标准化处理，形状: {all_factors.shape}...")
    scaler = StandardScaler()
    all_factors_scaled = pd.DataFrame(
        scaler.fit_transform(all_factors),
        index=all_factors.index,
        columns=all_factors.columns
    )
    
    # 再次检查并清理标准化后的数据
    all_factors_scaled = all_factors_scaled.replace([np.inf, -np.inf], 0)
    all_factors_scaled = all_factors_scaled.fillna(0)
    
    # 合并因子数据和收益数据
    print("合并因子数据和收益数据...")
    merged_data = pd.merge(all_factors_scaled, return_data, left_index=True, right_index=True, how='inner')
    print(f"合并后数据形状: {merged_data.shape}")
    
    # 初始化结果列表
    predictions = []
    actual_returns = []
    dates = []
    tokens = []
    model_weights = []
    
    # 检查数据是否足够进行滚动训练
    if len(merged_data.index) <= rolling_window + prediction_window:
        print(f"警告: 数据量不足，无法进行滚动训练。数据长度: {len(merged_data.index)}, 需要至少: {rolling_window + prediction_window + 1}")
        # 创建空的结果DataFrame
        results = pd.DataFrame({
            'date': [],
            'token': [],
            'predicted_return': [],
            'actual_return': []
        })
        return results, []
    
    # 滚动训练和预测
    print("开始滚动训练和预测...")
    successful_windows = 0
    total_windows = (len(merged_data.index) - rolling_window) // prediction_window
    print(f"将执行 {total_windows} 个滚动窗口的训练和预测")
    
    for i in range(rolling_window, len(merged_data.index) - prediction_window, prediction_window):
        try:
            print(f"训练窗口: {i-rolling_window} 到 {i}")
            train_end_idx = i
            test_start_idx = i
            test_end_idx = min(i + prediction_window, len(merged_data.index))
            
            train_data = merged_data.iloc[:train_end_idx]
            test_data = merged_data.iloc[test_start_idx:test_end_idx]
            
            # 准备训练数据
            X_train, y_train, train_dates, train_tokens, fixed_tokens = prepare_model_data(
                [(factor_name, factor_df.loc[train_data.index]) for factor_name, factor_df in factors_data],
                return_data.loc[train_data.index],
                rolling_window
            )
            
            # 检查训练数据是否为空
            if len(X_train) == 0 or len(y_train) == 0:
                print("警告: 训练数据为空，跳过此训练窗口")
                continue
                
            print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # 检查是否有NaN值
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                print("警告: 训练数据包含NaN值，尝试删除或填充...")
                # 找出包含NaN的行
                nan_rows = np.isnan(X_train).any(axis=1) | np.isnan(y_train)
                print(f"发现 {nan_rows.sum()} 行包含NaN值")
                
                # 删除包含NaN的行
                X_train = X_train[~nan_rows]
                y_train = y_train[~nan_rows]
                
                # 再次检查数据长度是否足够
                if len(X_train) < 10:  # 至少需要10个样本进行训练
                    print(f"警告: 删除NaN后训练样本不足，跳过此窗口")
                    continue
            
            # 训练模型
            try:
                if model_type == 'lasso':
                    model = train_lasso_model(X_train, y_train)
                    # 保存模型权重
                    current_weights = model.coef_
                elif model_type == 'lightgbm':
                    model = train_lightgbm_model(X_train, y_train)
                    # 保存模型权重（特征重要性）
                    current_weights = model.feature_importance(importance_type='gain')
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
                
                model_weights.append(current_weights)
                successful_windows += 1
            except Exception as e:
                print(f"模型训练失败: {str(e)}")
                continue
            
            # 对测试集中的每一天进行预测
            print(f"预测测试集: {test_start_idx} 到 {test_end_idx}")
            for test_date in test_data.index:
                # 获取当前日期之前的rolling_window天的数据作为特征
                try:
                    window_end_date = test_data.index[test_data.index < test_date][-1] if any(test_data.index < test_date) else train_data.index[-1]
                    window_start_idx = list(merged_data.index).index(window_end_date) - rolling_window + 1
                    window_end_idx = list(merged_data.index).index(window_end_date) + 1
                    window_data = merged_data.iloc[window_start_idx:window_end_idx]
                    
                    # 获取当前日期的实际收益
                    current_returns = test_data.loc[test_date, [col for col in test_data.columns if 'log_return' in col]]
                    
                    # 对每个固定代币进行预测
                    for token in fixed_tokens:
                        return_col = f'{token}_log_return'
                        if return_col in current_returns.index:
                            # 获取该代币在窗口期内的所有因子值，按固定顺序
                            token_feature_cols = []
                            for factor_name, _ in factors_data:
                                factor_col = f'{factor_name}_{token}'
                                if factor_col in window_data.columns:
                                    token_feature_cols.append(factor_col)
                            
                            if token_feature_cols:
                                try:
                                    token_features = window_data[token_feature_cols]
                                    features_flat = token_features.values.flatten()
                                    # 确保特征向量是二维的
                                    features_2d = features_flat.reshape(1, -1)
                                    
                                    # 检查特征向量是否包含NaN
                                    if np.isnan(features_2d).any():
                                        continue
                                    
                                    # 检查特征维度是否与训练时一致
                                    if features_2d.shape[1] != X_train.shape[1]:
                                        continue
                                    
                                    if model_type == 'lasso':
                                        pred = model.predict(features_2d)[0]
                                    elif model_type == 'lightgbm':
                                        pred = model.predict(features_2d)[0]
                                    
                                    predictions.append(pred)
                                    actual_returns.append(current_returns[return_col])
                                    dates.append(test_date)
                                    tokens.append(token)
                                except Exception as e:
                                    continue  # 静默跳过错误，避免过多输出
                except Exception as e:
                    print(f"处理测试日期 {test_date} 时出错: {str(e)}")
            
            # 每10个窗口打印一次进度
            if successful_windows % 10 == 0 or successful_windows == total_windows:
                print(f"进度: {successful_windows}/{total_windows} 个窗口完成")
                
        except Exception as e:
            print(f"处理训练窗口 {i} 时出错: {str(e)}")
            continue
    
    # 检查是否有成功的预测
    if len(predictions) == 0:
        print("警告: 没有成功的预测结果，返回空DataFrame")
        results = pd.DataFrame({
            'date': [],
            'token': [],
            'predicted_return': [],
            'actual_return': []
        })
        return results, model_weights
    
    # 创建结果DataFrame
    print(f"预测完成，共 {len(predictions)} 条预测记录")
    results = pd.DataFrame({
        'date': dates,
        'token': tokens,
        'predicted_return': predictions,
        'actual_return': actual_returns
    })
    
    return results, model_weights

# 构建多空组合
def build_long_short_portfolio(predictions_df, top_pct=0.2, bottom_pct=0.2):
    """构建多空组合"""
    portfolio_returns = []
    dates = []
    
    # 检查预测结果是否为空
    if predictions_df.empty:
        print("警告: 预测结果为空，无法构建投资组合")
        # 返回空的DataFrame，但包含必要的列
        return pd.DataFrame(columns=['long_return', 'short_return', 'long_short_return'])
    
    print(f"构建多空组合: 做多前 {top_pct*100}%，做空后 {bottom_pct*100}%")
    
    # 按日期分组
    for date, group in predictions_df.groupby('date'):
        try:
            # 检查当前日期的样本数量
            n_tokens = len(group)
            if n_tokens < 5:  # 至少需要5个代币才能构建有意义的组合
                print(f"警告: 日期 {date} 的代币数量不足 ({n_tokens})，跳过")
                continue
            
            # 按预测收益排序
            sorted_group = group.sort_values('predicted_return', ascending=False)
            
            # 计算分位数位置
            top_n = max(1, int(n_tokens * top_pct))  # 至少选择1个代币
            bottom_n = max(1, int(n_tokens * bottom_pct))  # 至少选择1个代币
            
            print(f"日期 {date}: 总代币数 {n_tokens}, 做多 {top_n} 个, 做空 {bottom_n} 个")
            
            # 选择做多和做空的代币
            long_tokens = sorted_group.iloc[:top_n]
            short_tokens = sorted_group.iloc[-bottom_n:]
            
            # 计算多空组合收益（等权重）
            if not long_tokens.empty and not short_tokens.empty:
                # 检查是否有NaN值
                if long_tokens['actual_return'].isna().any() or short_tokens['actual_return'].isna().any():
                    print(f"警告: 日期 {date} 的收益数据包含NaN值，使用dropna处理")
                    long_return = long_tokens['actual_return'].dropna().mean()
                    short_return = short_tokens['actual_return'].dropna().mean()
                else:
                    long_return = long_tokens['actual_return'].mean()
                    short_return = short_tokens['actual_return'].mean()
                
                # 如果仍然有NaN值，跳过这个日期
                if np.isnan(long_return) or np.isnan(short_return):
                    print(f"警告: 日期 {date} 的收益计算结果为NaN，跳过")
                    continue
                
                long_short_return = long_return - short_return
                
                portfolio_returns.append({
                    'date': date,
                    'long_return': long_return,
                    'short_return': short_return,
                    'long_short_return': long_short_return
                })
                dates.append(date)
        except Exception as e:
            print(f"处理日期 {date} 时出错: {str(e)}")
    
    # 检查是否有足够的交易日
    if len(portfolio_returns) == 0:
        print("警告: 没有足够的交易日构建投资组合")
        return pd.DataFrame(columns=['long_return', 'short_return', 'long_short_return'])
    
    # 创建结果DataFrame
    portfolio_df = pd.DataFrame(portfolio_returns)
    portfolio_df.set_index('date', inplace=True)
    portfolio_df.sort_index(inplace=True)  # 确保按日期排序
    
    print(f"多空组合构建完成，共 {len(portfolio_df)} 个交易日")
    return portfolio_df

# 计算累积收益
def calculate_cumulative_returns(portfolio_df):
    """计算累积收益"""
    portfolio_df['cum_long_return'] = np.exp(np.cumsum(portfolio_df['long_return'])) - 1
    portfolio_df['cum_short_return'] = np.exp(np.cumsum(portfolio_df['short_return'])) - 1
    portfolio_df['cum_long_short_return'] = np.exp(np.cumsum(portfolio_df['long_short_return'])) - 1
    
    return portfolio_df

# 绘制累积收益图
def plot_cumulative_returns(portfolio_df, title="Cumulative Returns"):
    """绘制累积收益图"""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['cum_long_return'], label='Long Portfolio')
    plt.plot(portfolio_df.index, portfolio_df['cum_short_return'], label='Short Portfolio')
    plt.plot(portfolio_df.index, portfolio_df['cum_long_short_return'], label='Long-Short Portfolio')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    save_path = os.path.join(output_dir, 'cumulative_returns.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 累积收益图已保存至: {save_path}")

# 计算投资组合统计指标
def calculate_portfolio_stats(portfolio_df, market_returns):
    """计算投资组合统计指标"""
    # 检查输入数据
    if portfolio_df is None or portfolio_df.empty:
        print("警告: 投资组合数据为空，无法计算统计指标")
        return pd.DataFrame({
            'Portfolio': ['Long', 'Short', 'Long-Short', 'Market'],
            'Annual Return': [np.nan, np.nan, np.nan, np.nan],
            'Annual Volatility': [np.nan, np.nan, np.nan, np.nan],
            'Sharpe Ratio': [np.nan, np.nan, np.nan, np.nan],
            'Max Drawdown': [np.nan, np.nan, np.nan, np.nan]
        })
    
    if market_returns is None or market_returns.empty:
        print("警告: 市场收益数据为空，无法计算统计指标")
        return pd.DataFrame({
            'Portfolio': ['Long', 'Short', 'Long-Short', 'Market'],
            'Annual Return': [np.nan, np.nan, np.nan, np.nan],
            'Annual Volatility': [np.nan, np.nan, np.nan, np.nan],
            'Sharpe Ratio': [np.nan, np.nan, np.nan, np.nan],
            'Max Drawdown': [np.nan, np.nan, np.nan, np.nan]
        })
    
    try:
        # 合并投资组合收益和市场收益
        merged_df = pd.merge(portfolio_df, market_returns, left_index=True, right_index=True, how='inner')
        
        # 检查合并后的数据是否为空
        if merged_df.empty:
            print("警告: 投资组合和市场收益没有共同的日期")
            return pd.DataFrame({
                'Portfolio': ['Long', 'Short', 'Long-Short', 'Market'],
                'Annual Return': [np.nan, np.nan, np.nan, np.nan],
                'Annual Volatility': [np.nan, np.nan, np.nan, np.nan],
                'Sharpe Ratio': [np.nan, np.nan, np.nan, np.nan],
                'Max Drawdown': [np.nan, np.nan, np.nan, np.nan]
            })
        
        # 计算年化收益率
        try:
            n_days = (merged_df.index[-1] - merged_df.index[0]).days
            if n_days <= 0:  # 如果只有一天数据
                n_days = 1
            annualized_factor = 365 / n_days
        except Exception as e:
            print(f"计算交易天数时出错: {str(e)}，使用数据点数量代替")
            n_days = len(merged_df)
            annualized_factor = 365 / n_days if n_days > 0 else 0
        
        # 计算各组合的年化收益率
        long_annual_return = (np.exp(np.sum(merged_df['long_return'])) - 1) * annualized_factor
        short_annual_return = (np.exp(np.sum(merged_df['short_return'])) - 1) * annualized_factor
        long_short_annual_return = (np.exp(np.sum(merged_df['long_short_return'])) - 1) * annualized_factor
        market_annual_return = (np.exp(np.sum(merged_df['market_return'])) - 1) * annualized_factor
        
        # 计算波动率
        long_volatility = np.std(merged_df['long_return']) * np.sqrt(365)
        short_volatility = np.std(merged_df['short_return']) * np.sqrt(365)
        long_short_volatility = np.std(merged_df['long_short_return']) * np.sqrt(365)
        market_volatility = np.std(merged_df['market_return']) * np.sqrt(365)
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        long_sharpe = (long_annual_return - risk_free_rate) / long_volatility if long_volatility != 0 else 0
        short_sharpe = (short_annual_return - risk_free_rate) / short_volatility if short_volatility != 0 else 0
        long_short_sharpe = (long_short_annual_return - risk_free_rate) / long_short_volatility if long_short_volatility != 0 else 0
        market_sharpe = (market_annual_return - risk_free_rate) / market_volatility if market_volatility != 0 else 0
        
        # 计算最大回撤
        try:
            long_cum_returns = np.exp(np.cumsum(merged_df['long_return'])) - 1
            short_cum_returns = np.exp(np.cumsum(merged_df['short_return'])) - 1
            long_short_cum_returns = np.exp(np.cumsum(merged_df['long_short_return'])) - 1
            market_cum_returns = np.exp(np.cumsum(merged_df['market_return'])) - 1
            
            long_max_drawdown = calculate_max_drawdown(long_cum_returns)
            short_max_drawdown = calculate_max_drawdown(short_cum_returns)
            long_short_max_drawdown = calculate_max_drawdown(long_short_cum_returns)
            market_max_drawdown = calculate_max_drawdown(market_cum_returns)
        except Exception as e:
            print(f"计算最大回撤时出错: {str(e)}")
            long_max_drawdown = np.nan
            short_max_drawdown = np.nan
            long_short_max_drawdown = np.nan
            market_max_drawdown = np.nan
        
        # 创建统计指标DataFrame
        stats = pd.DataFrame({
            'Portfolio': ['Long', 'Short', 'Long-Short', 'Market'],
            'Annual Return': [long_annual_return, short_annual_return, long_short_annual_return, market_annual_return],
            'Annual Volatility': [long_volatility, short_volatility, long_short_volatility, market_volatility],
            'Sharpe Ratio': [long_sharpe, short_sharpe, long_short_sharpe, market_sharpe],
            'Max Drawdown': [long_max_drawdown, short_max_drawdown, long_short_max_drawdown, market_max_drawdown]
        })
        
        print(f"投资组合统计指标计算完成: 多空组合年化收益={long_short_annual_return:.4f}, 夏普比率={long_short_sharpe:.4f}")
        return stats
        
    except Exception as e:
        print(f"计算投资组合统计指标时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({
            'Portfolio': ['Long', 'Short', 'Long-Short', 'Market'],
            'Annual Return': [np.nan, np.nan, np.nan, np.nan],
            'Annual Volatility': [np.nan, np.nan, np.nan, np.nan],
            'Sharpe Ratio': [np.nan, np.nan, np.nan, np.nan],
            'Max Drawdown': [np.nan, np.nan, np.nan, np.nan]
        })

# 计算最大回撤
def calculate_max_drawdown(cum_returns):
    """计算最大回撤"""
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / (running_max + 1e-10)  # 避免除以零
    max_drawdown = np.min(drawdown)
    return max_drawdown

# 进行CAPM回归分析
def perform_capm_analysis(portfolio_df, market_returns):
    """进行CAPM回归分析"""
    # 检查输入数据
    if portfolio_df.empty:
        print("警告: 投资组合数据为空，无法进行CAPM回归分析")
        return pd.DataFrame(), None
    
    if market_returns.empty:
        print("警告: 市场收益数据为空，无法进行CAPM回归分析")
        return pd.DataFrame(), None
    
    try:
        # 合并投资组合收益和市场收益
        merged_df = pd.merge(portfolio_df, market_returns, left_index=True, right_index=True, how='inner')
        
        # 检查合并后的数据
        if len(merged_df) < 10:  # 至少需要10个数据点进行有意义的回归
            print(f"警告: 投资组合和市场收益只有 {len(merged_df)} 个共同日期，不足以进行回归分析")
            return pd.DataFrame(), None
        
        # 进行CAPM回归
        X = add_constant(merged_df['market_return'])
        y = merged_df['long_short_return']
        
        model = OLS(y, X).fit()
        
        # 提取回归结果
        alpha = model.params[0]
        beta = model.params[1]
        alpha_pvalue = model.pvalues[0]
        beta_pvalue = model.pvalues[1]
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        alpha_significant = alpha_pvalue < 0.05
        
        # 创建结果DataFrame
        capm_results = pd.DataFrame({
            'alpha': [alpha],
            'beta': [beta],
            'alpha_pvalue': [alpha_pvalue],
            'beta_pvalue': [beta_pvalue],
            'r_squared': [r_squared],
            'adj_r_squared': [adj_r_squared],
            'alpha_significant': [alpha_significant],
            'observations': [len(merged_df)]
        })
        
        print(f"CAPM回归分析完成: Alpha={alpha:.6f} (p={alpha_pvalue:.4f}), Beta={beta:.4f}, R²={r_squared:.4f}")
        return capm_results, model
        
    except Exception as e:
        print(f"CAPM回归分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None

# 绘制CAPM回归图
def plot_capm_regression(portfolio_df, market_returns, model, title="CAPM Regression"):
    """绘制CAPM回归图"""
    # 检查输入数据
    if portfolio_df is None or portfolio_df.empty or market_returns is None or market_returns.empty or model is None:
        print(f"警告: 无法绘制CAPM回归图，数据或模型为空")
        # 创建一个空白图像
        plt.figure(figsize=(10, 6))
        plt.title(f"{title}\n(无足够数据进行回归分析)")
        plt.xlabel('Market Return')
        plt.ylabel('Long-Short Portfolio Return')
        plt.grid(True)
        
        # 保存空白图像
        save_path = os.path.join(output_dir, 'capm_regression.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ CAPM回归图(空白)已保存至: {save_path}")
        return
    
    try:
        # 合并投资组合收益和市场收益
        merged_df = pd.merge(portfolio_df, market_returns, left_index=True, right_index=True, how='inner')
        
        if merged_df.empty:
            print(f"警告: 投资组合和市场收益没有共同的日期，无法绘制CAPM回归图")
            plt.figure(figsize=(10, 6))
            plt.title(f"{title}\n(无共同日期数据)")
            plt.xlabel('Market Return')
            plt.ylabel('Long-Short Portfolio Return')
            plt.grid(True)
            
            # 保存空白图像
            save_path = os.path.join(output_dir, 'capm_regression.png')
            plt.savefig(save_path)
            plt.close()
            print(f"✅ CAPM回归图(无共同日期)已保存至: {save_path}")
            return
        
        # 绘制散点图和回归线
        plt.figure(figsize=(10, 6))
        plt.scatter(merged_df['market_return'], merged_df['long_short_return'], alpha=0.5)
        
        # 添加回归线
        x_range = np.linspace(merged_df['market_return'].min(), merged_df['market_return'].max(), 100)
        y_pred = model.params[0] + model.params[1] * x_range
        plt.plot(x_range, y_pred, 'r-', linewidth=2)
        
        # 添加标题和标签
        alpha = model.params[0]
        beta = model.params[1]
        r_squared = model.rsquared
        alpha_pvalue = model.pvalues[0]
        
        plt.title(f"{title}\nAlpha: {alpha:.4f} (p-value: {alpha_pvalue:.4f}), Beta: {beta:.4f}, R²: {r_squared:.4f}")
        plt.xlabel('Market Return')
        plt.ylabel('Long-Short Portfolio Return')
        plt.grid(True)
        
        # 添加零线
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # 保存图表
        save_path = os.path.join(output_dir, 'capm_regression.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ CAPM回归图已保存至: {save_path}")
    except Exception as e:
        print(f"绘制CAPM回归图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 创建一个错误信息图像
        plt.figure(figsize=(10, 6))
        plt.title(f"{title}\n(绘图过程中出错: {str(e)})")
        plt.xlabel('Market Return')
        plt.ylabel('Long-Short Portfolio Return')
        plt.grid(True)
        
        # 保存错误信息图像
        save_path = os.path.join(output_dir, 'capm_regression.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ CAPM回归图(错误信息)已保存至: {save_path}")

# 绘制模型特征重要性图
def plot_feature_importance(model_weights, factors_data, model_type='lasso', top_n=20):
    """绘制模型特征重要性图"""
    # 检查输入数据
    if model_weights is None or len(model_weights) == 0:
        print("警告: 模型权重为空，无法绘制特征重要性图")
        # 创建一个空白图像
        plt.figure(figsize=(12, 8))
        plt.title(f"{model_type.capitalize()} 模型特征重要性\n(无模型权重数据)")
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(True)
        
        # 保存空白图像
        output_dir = os.environ.get('OUTPUT_DIR', 'ml_strategy_results')
        save_path = os.path.join(output_dir, f'feature_importance_{model_type}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 特征重要性图(空白)已保存至: {save_path}")
        return
    
    if factors_data is None or len(factors_data) == 0:
        print("警告: 因子数据为空，无法绘制特征重要性图")
        # 创建一个空白图像
        plt.figure(figsize=(12, 8))
        plt.title(f"{model_type.capitalize()} 模型特征重要性\n(无因子数据)")
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(True)
        
        # 保存空白图像
        output_dir = os.environ.get('OUTPUT_DIR', 'ml_strategy_results')
        save_path = os.path.join(output_dir, f'feature_importance_{model_type}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 特征重要性图(空白)已保存至: {save_path}")
        return
    
    try:
        # 获取最后一个模型的权重
        weights = model_weights[-1]
        
        # 获取特征名称
        feature_names = []
        for factor_name, factor_df in factors_data:
            for col in factor_df.columns:
                token = col.split('_')[0]  # 提取代币名称
                feature_names.append(f"{factor_name}_{token}")
        
        # 检查特征名称和权重长度是否匹配
        if len(feature_names) != len(weights):
            print(f"警告: 特征名称数量({len(feature_names)})与权重数量({len(weights)})不匹配")
            # 调整长度以匹配
            min_len = min(len(feature_names), len(weights))
            feature_names = feature_names[:min_len]
            weights = weights[:min_len]
        
        # 创建特征重要性DataFrame
        if model_type == 'lasso':
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(weights)
            })
        elif model_type == 'lightgbm':
            importance_df = pd.DataFrame({
                'Feature': feature_names,  # 已确保长度匹配
                'Importance': weights
            })
        
        # 按重要性排序并选择前top_n个特征
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
    except Exception as e:
        print(f"准备特征重要性数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 创建一个错误信息图像
        plt.figure(figsize=(12, 8))
        plt.title(f"{model_type.capitalize()} 模型特征重要性\n(处理数据时出错: {str(e)})")
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(True)
        
        # 保存错误信息图像
        output_dir = os.environ.get('OUTPUT_DIR', 'ml_strategy_results')
        save_path = os.path.join(output_dir, f'feature_importance_{model_type}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 特征重要性图(错误信息)已保存至: {save_path}")
        return
    try:
        # 绘制条形图
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        
        plt.title(f'Top {top_n} Feature Importance ({model_type.capitalize()} Model)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(output_dir, f'{model_type}_feature_importance.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 特征重要性图已保存至: {save_path}")
    except Exception as e:
        print(f"绘制特征重要性图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 创建一个错误信息图像
        plt.figure(figsize=(12, 8))
        plt.title(f"{model_type.capitalize()} 模型特征重要性\n(绘图过程中出错: {str(e)})")
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(True)
        
        # 保存错误信息图像
        save_path = os.path.join(output_dir, f'{model_type}_feature_importance.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 特征重要性图(错误信息)已保存至: {save_path}")

# 主函数
def main():
    try:
        print("\n📊 开始机器学习因子策略回测")
        
        # 加载因子数据
        print("\n🔍 加载因子数据...")
        factors = get_available_factors()
        factors_data = [(factor_name, load_factor_data(factor_path)) for factor_name, factor_path in factors]
        print(f"✅ 已加载 {len(factors_data)} 个因子")
        
        # 加载收益数据
        print("\n🔍 加载收益数据...")
        return_data = load_return_data(return_period)
        print(f"✅ 已加载 {return_period} 收益数据，共 {len(return_data)} 条记录")
        
        # 加载市场收益数据
        print("\n🔍 加载市场收益数据...")
        market_returns = load_market_return_data(return_period)
        print(f"✅ 已加载市场收益数据，共 {len(market_returns)} 条记录")
        
        # 滚动训练和预测
        print(f"\n🔍 使用 {model_type.capitalize()} 模型进行滚动训练和预测...")
        predictions, model_weights = rolling_train_predict(
            factors_data, 
            return_data, 
            model_type=model_type,
            rolling_window=rolling_window,
            prediction_window=prediction_window
        )
        print(f"✅ 预测完成，共 {len(predictions)} 条预测记录")
        
        # 保存预测结果
        predictions_file = os.path.join(output_dir, f'predictions_{model_type}_{return_period}.csv')
        predictions.to_csv(predictions_file)
        print(f"✅ 预测结果已保存至: {predictions_file}")
        
        # 构建多空组合
        print("\n🔍 构建多空组合...")
        portfolio_df = build_long_short_portfolio(predictions, top_pct=top_pct, bottom_pct=bottom_pct)
        print(f"✅ 多空组合构建完成，共 {len(portfolio_df)} 个交易日")
        
        # 计算累积收益
        portfolio_df = calculate_cumulative_returns(portfolio_df)
        
        # 保存组合收益
        portfolio_file = os.path.join(output_dir, f'portfolio_returns_{model_type}_{return_period}.csv')
        portfolio_df.to_csv(portfolio_file)
        print(f"✅ 组合收益已保存至: {portfolio_file}")
        
        # 绘制累积收益图
        plot_cumulative_returns(portfolio_df, title=f"{model_type.capitalize()} Model Strategy Cumulative Returns ({return_period})")
        
        # 计算投资组合统计指标
        print("\n🔍 计算投资组合统计指标...")
        stats = calculate_portfolio_stats(portfolio_df, market_returns)
        
        # 保存统计指标
        stats_file = os.path.join(output_dir, f'portfolio_stats_{model_type}_{return_period}.csv')
        stats.to_csv(stats_file, index=False)
        print(f"✅ 投资组合统计指标已保存至: {stats_file}")
        print(stats)
        
        # 进行CAPM回归分析
        print("\n🔍 进行CAPM回归分析...")
        capm_results, capm_model = perform_capm_analysis(portfolio_df, market_returns)
        
        # 保存CAPM回归结果
        capm_file = os.path.join(output_dir, f'capm_results_{model_type}_{return_period}.csv')
        capm_results.to_csv(capm_file, index=False)
        print(f"✅ CAPM回归结果已保存至: {capm_file}")
        print(capm_results)
        
        # 绘制CAPM回归图
        plot_capm_regression(portfolio_df, market_returns, capm_model, title=f"{model_type.capitalize()} Model Strategy CAPM Regression ({return_period})")
        
        # 绘制模型特征重要性图
        plot_feature_importance(model_weights, factors_data, model_type=model_type)
        
        print("\n✅ 机器学习因子策略回测完成!")
        
        return portfolio_df, stats, capm_results, model_weights
    except Exception as e:
        print(f"\n❌ 策略执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    main()