# -*- coding: utf-8 -*-
"""
白天筛选模块
根据 DNI (Direct Normal Irradiance) 判断白天/夜间，支持 drop 和 mask 两种模式
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DaylightStats:
    """白天筛选统计信息"""
    total_rows: int
    daylight_rows: int
    night_rows: int
    daylight_ratio: float
    night_ratio: float
    # 每天白天点数统计
    daily_daylight_min: int
    daily_daylight_median: float
    daily_daylight_max: int


@dataclass
class DaylightResult:
    """白天筛选结果"""
    df: pd.DataFrame  # 添加了 is_daylight 列的 DataFrame
    stats: DaylightStats
    filtered_df: Optional[pd.DataFrame] = None  # drop 模式下过滤后的 DataFrame
    rows_before_filter: int = 0
    rows_after_filter: int = 0


def add_daylight_flag(
    df: pd.DataFrame,
    time_column: str,
    dni_col: str,
    threshold: float = 5.0
) -> DaylightResult:
    """
    为数据添加 is_daylight 布尔列
    
    白天判定规则：DNI >= threshold 视为白天
    
    为什么用 DNI 而不是时间：
    1. DNI 直接反映太阳辐照情况，比固定时间段更准确
    2. 不同季节日出日落时间不同，用 DNI 可以自适应
    3. 阴天等情况下 DNI 也会很低，这时光伏出力也低，归类为"无效时段"是合理的
    
    数据泄漏风险分析：
    - is_daylight 仅由当前时刻的 DNI 计算，不使用未来信息
    - 筛选发生在样本生成前，不影响预测逻辑
    
    Args:
        df: 原始数据 DataFrame
        time_column: 时间列名
        dni_col: DNI 列名
        threshold: DNI 阈值
    
    Returns:
        DaylightResult: 包含添加 is_daylight 列后的 DataFrame 和统计信息
    """
    df = df.copy()
    
    # 添加 is_daylight 列
    df['is_daylight'] = df[dni_col] >= threshold
    
    # 统计
    total_rows = len(df)
    daylight_rows = df['is_daylight'].sum()
    night_rows = total_rows - daylight_rows
    
    # 计算每天白天点数统计
    df['_date'] = pd.to_datetime(df[time_column]).dt.date
    daily_daylight = df.groupby('_date')['is_daylight'].sum()
    
    daily_daylight_min = int(daily_daylight.min())
    daily_daylight_median = float(daily_daylight.median())
    daily_daylight_max = int(daily_daylight.max())
    
    # 移除临时列
    df.drop(columns=['_date'], inplace=True)
    
    stats = DaylightStats(
        total_rows=total_rows,
        daylight_rows=int(daylight_rows),
        night_rows=int(night_rows),
        daylight_ratio=daylight_rows / total_rows if total_rows > 0 else 0,
        night_ratio=night_rows / total_rows if total_rows > 0 else 0,
        daily_daylight_min=daily_daylight_min,
        daily_daylight_median=daily_daylight_median,
        daily_daylight_max=daily_daylight_max
    )
    
    return DaylightResult(
        df=df,
        stats=stats,
        rows_before_filter=total_rows,
        rows_after_filter=total_rows
    )


def filter_daylight_rows(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int]:
    """
    过滤掉夜间行（drop 模式）
    
    注意：过滤后时间戳会不连续，后续滑窗时需要继续检查时间连续性
    
    Args:
        df: 包含 is_daylight 列的 DataFrame
    
    Returns:
        (过滤后的 DataFrame, 过滤前行数, 过滤后行数)
    """
    rows_before = len(df)
    filtered_df = df[df['is_daylight'] == True].copy().reset_index(drop=True)
    rows_after = len(filtered_df)
    
    return filtered_df, rows_before, rows_after


def print_daylight_summary(result: DaylightResult, mode: str = "mask", dni_col: str = None) -> None:
    """打印白天筛选摘要"""
    stats = result.stats
    print("\n" + "=" * 60)
    print("白天筛选摘要")
    print("=" * 60)
    if dni_col:
        print(f"判定列: {dni_col}")
    print(f"总行数: {stats.total_rows}")
    print(f"白天行数: {stats.daylight_rows} ({stats.daylight_ratio*100:.1f}%)")
    print(f"夜间行数: {stats.night_rows} ({stats.night_ratio*100:.1f}%)")
    print("-" * 40)
    print("每天白天点数统计:")
    print(f"  - 最小值: {stats.daily_daylight_min}")
    print(f"  - 中位数: {stats.daily_daylight_median:.1f}")
    print(f"  - 最大值: {stats.daily_daylight_max}")
    
    if mode == "drop" and result.filtered_df is not None:
        print("-" * 40)
        print(f"过滤模式: drop")
        print(f"过滤前行数: {result.rows_before_filter}")
        print(f"过滤后行数: {result.rows_after_filter}")
        print(f"过滤比例: {(1 - result.rows_after_filter/result.rows_before_filter)*100:.1f}%")
    elif mode == "mask":
        print("-" * 40)
        print(f"过滤模式: mask (保留所有行，滑窗时检查)")
    
    print("=" * 60)
