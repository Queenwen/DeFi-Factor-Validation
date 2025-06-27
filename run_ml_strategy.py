#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行机器学习因子策略回测

此脚本用于执行机器学习因子策略的回测，包括：
1. 构建机器学习模型（Lasso或LightGBM）预测未来3天收益
2. 使用模型输出的因子权重构建合成得分
3. 每天用得分进行排序，构建多空组合（做多top 20%，做空bottom 20%）
4. 对多空组合进行回测，持有期为3天
5. 使用CAPM回归验证策略是否具有显著的alpha
"""

import os
import sys
import argparse
import traceback
from ml_factor_strategy import main as run_strategy

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行机器学习因子策略回测")
    parser.add_argument(
        "--model", 
        type=str, 
        default="lasso", 
        choices=["lasso", "lightgbm"],
        help="选择使用的机器学习模型类型 (默认: lasso)"
    )
    parser.add_argument(
        "--holding", 
        type=int, 
        default=3,
        help="持有期天数 (默认: 3天)"
    )
    parser.add_argument(
        "--rolling", 
        type=int, 
        default=60,
        help="滚动窗口大小，即用于训练模型的历史数据天数 (默认: 60天)"
    )
    parser.add_argument(
        "--prediction", 
        type=int, 
        default=30,
        help="预测窗口大小，即多少天重新训练一次模型 (默认: 30天)"
    )
    parser.add_argument(
        "--top", 
        type=float, 
        default=0.2,
        help="做多的百分比 (默认: 0.2，即做多排名前20%的资产)"
    )
    parser.add_argument(
        "--bottom", 
        type=float, 
        default=0.2,
        help="做空的百分比 (默认: 0.2，即做空排名后20%的资产)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="ml_strategy_results",
        help="输出目录 (默认: ml_strategy_results)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式，打印更多信息"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    try:
        args = parse_args()
        
        # 打印参数信息
        print("="*80)
        print("机器学习因子策略回测")
        print("="*80)
        print(f"模型类型: {args.model}")
        print(f"持有期: {args.holding}天")
        print(f"滚动窗口: {args.rolling}天")
        print(f"预测窗口: {args.prediction}天")
        print(f"做多比例: {args.top*100}%")
        print(f"做空比例: {args.bottom*100}%")
        print(f"输出目录: {args.output}")
        print(f"调试模式: {'启用' if args.debug else '禁用'}")
        print("="*80)
        
        # 参数验证
        if args.holding <= 0:
            print("错误: 持有期必须大于0")
            return 1
        
        if args.rolling <= 0:
            print("错误: 滚动窗口必须大于0")
            return 1
        
        if args.prediction <= 0:
            print("错误: 预测窗口必须大于0")
            return 1
        
        if args.top <= 0 or args.top > 1:
            print("错误: 做多比例必须在(0,1]范围内")
            return 1
        
        if args.bottom <= 0 or args.bottom > 1:
            print("错误: 做空比例必须在(0,1]范围内")
            return 1
        
        # 设置环境变量
        os.environ["MODEL_TYPE"] = args.model
        os.environ["HOLDING_PERIOD"] = str(args.holding)
        os.environ["ROLLING_WINDOW"] = str(args.rolling)
        os.environ["PREDICTION_WINDOW"] = str(args.prediction)
        os.environ["TOP_PCT"] = str(args.top)
        os.environ["BOTTOM_PCT"] = str(args.bottom)
        os.environ["OUTPUT_DIR"] = args.output
        
        if args.debug:
            os.environ["DEBUG"] = "1"
        
        # 创建输出目录
        try:
            os.makedirs(args.output, exist_ok=True)
            print(f"输出目录已创建/确认: {args.output}")
        except Exception as e:
            print(f"创建输出目录时出错: {str(e)}")
            return 1
        
        # 运行策略
        print("\n开始执行策略...\n")
        results = run_strategy()
        
        if results is None:
            print("\n策略执行失败，请检查错误信息")
            return 1
        else:
            print("\n策略执行成功!")
            return 0
            
    except KeyboardInterrupt:
        print("\n用户中断执行")
        return 130
    except Exception as e:
        print(f"\n执行过程中发生未处理的异常: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())