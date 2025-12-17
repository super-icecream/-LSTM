# -*- coding: utf-8 -*-
"""
时间顺序切分模块
按天（自然日）将数据切分为 train / val / test，保证时间顺序不被破坏
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SplitResult:
    """切分结果，包含三个子集和统计信息"""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_days: int
    val_days: int
    test_days: int
    train_rows: int
    val_rows: int
    test_rows: int
    total_days: int
    day_list: List  # 所有日期列表（按顺序）


def split_by_day(
    df: pd.DataFrame,
    time_column: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> SplitResult:
    """
    按天（自然日）切分数据为 train / val / test
    
    为什么按天切分：
    1. 项目技术路线明确要求"以天为最小单位切分"
    2. 保证时间顺序不被破坏，train < val < test（时间上）
    3. 同一天的数据不会同时出现在不同子集中
    4. 后续"不跨天滑窗"能正常工作
    
    严格约束：
    - 禁止随机打乱
    - 必须保持时间顺序
    
    Args:
        df: 原始数据 DataFrame
        time_column: 时间列名
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        SplitResult: 包含切分后的子集和统计信息
    """
    # 验证比例之和约等于 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "切分比例之和必须等于 1"
    
    # 确保数据已按时间排序
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # 提取日期（自然日）
    # 为什么用 dt.date：按自然日（年-月-日）分组，忽略时分秒
    df['_date'] = df[time_column].dt.date
    
    # 获取唯一日期列表（已排序）
    unique_dates = df['_date'].unique().tolist()
    total_days = len(unique_dates)
    
    # 计算切分点
    train_end_idx = int(total_days * train_ratio)
    val_end_idx = int(total_days * (train_ratio + val_ratio))
    
    # 切分日期列表
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    # 根据日期归属切分数据
    train_df = df[df['_date'].isin(train_dates)].drop(columns=['_date']).reset_index(drop=True)
    val_df = df[df['_date'].isin(val_dates)].drop(columns=['_date']).reset_index(drop=True)
    test_df = df[df['_date'].isin(test_dates)].drop(columns=['_date']).reset_index(drop=True)
    
    # 移除临时列
    if '_date' in df.columns:
        df.drop(columns=['_date'], inplace=True)
    
    # 构建结果
    result = SplitResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_days=len(train_dates),
        val_days=len(val_dates),
        test_days=len(test_dates),
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        total_days=total_days,
        day_list=unique_dates
    )
    
    return result


def print_split_summary(result: SplitResult) -> None:
    """打印切分摘要"""
    print("\n" + "=" * 60)
    print("数据切分摘要（按天）")
    print("=" * 60)
    print(f"总天数: {result.total_days}")
    print("-" * 40)
    print(f"{'子集':<10} {'天数':<10} {'行数':<10} {'占比':<10}")
    print("-" * 40)
    
    total_rows = result.train_rows + result.val_rows + result.test_rows
    for name, days, rows in [
        ('train', result.train_days, result.train_rows),
        ('val', result.val_days, result.val_rows),
        ('test', result.test_days, result.test_rows)
    ]:
        ratio = days / result.total_days * 100
        print(f"{name:<10} {days:<10} {rows:<10} {ratio:.1f}%")
    
    print("-" * 40)
    print(f"{'总计':<10} {result.total_days:<10} {total_rows:<10}")
    print("=" * 60)
