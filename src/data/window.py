# -*- coding: utf-8 -*-
"""
不跨天滑窗样本生成器
生成用于时间序列预测的 (X, Y) 样本对，严格保证不跨天、时间连续
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class WindowSample:
    """单个滑窗样本"""
    X: np.ndarray           # 输入窗口 [Lx, features]
    Y: np.ndarray           # 输出窗口 [Hmax]，仅功率
    anchor_time: pd.Timestamp  # 锚点时间（预测起始时刻）
    anchor_idx: int         # 锚点在原数据中的索引
    date: object            # 所属日期


@dataclass
class WindowStats:
    """滑窗生成统计信息"""
    total_samples: int = 0
    skipped_insufficient_input: int = 0   # 输入窗口不足
    skipped_insufficient_output: int = 0  # 输出窗口不足
    skipped_cross_day: int = 0            # 跨天
    skipped_discontinuous: int = 0        # 时间不连续
    skipped_contains_night: int = 0       # 锚点为夜间（mask 模式）


@dataclass
class WindowResult:
    """滑窗生成结果"""
    samples: List[WindowSample]
    stats: WindowStats


def generate_windows(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    input_length: int = 16,
    max_horizon: int = 16,
    time_interval_minutes: int = 15,
    daylight_mask_mode: bool = False
) -> WindowResult:
    """
    生成不跨天的滑窗样本
    
    为什么不跨天：
    1. 项目技术路线明确要求"滑动窗口不允许跨天"
    2. 夜间光伏功率为 0，跨天窗口会引入不连续性
    3. 避免模型学到"夜间到白天"的伪模式
    4. 保证每个样本的物理意义一致
    
    为什么检查时间连续：
    - 若出现缺口或跨天，窗口立即中断
    - 这是技术路线中的硬约束
    
    滑窗逻辑：
    - 对于锚点 t，输入窗口为 [t-Lx, t)，输出窗口为 [t, t+Hmax)
    - 锚点 t 是预测的起始时刻，即我们站在 t-1 时刻，预测 t, t+1, ..., t+Hmax-1
    
    Args:
        df: 数据 DataFrame
        time_column: 时间列名
        target_column: 目标列名（功率）
        input_length: 输入窗口长度 Lx
        max_horizon: 最大预测步长 Hmax
        time_interval_minutes: 时间间隔（分钟）
        daylight_mask_mode: 是否启用 mask 模式（只要求锚点处是白天）
    
    Returns:
        WindowResult: 包含样本列表和统计信息
    """
    samples = []
    stats = WindowStats()
    
    # 确保数据已排序
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # 提取日期
    df['_date'] = df[time_column].dt.date
    
    # 期望的时间间隔
    expected_interval = pd.Timedelta(minutes=time_interval_minutes)
    
    # 按天分组处理
    # 为什么按天分组：确保滑窗不会跨天
    for date, day_df in df.groupby('_date'):
        day_df = day_df.reset_index(drop=True)
        n = len(day_df)
        
        # 遍历所有可能的锚点
        # 锚点 t 需要满足：
        # - 前面至少有 input_length 个点（输入窗口）
        # - 后面至少有 max_horizon 个点（输出窗口，包含锚点本身）
        for anchor_local_idx in range(input_length, n - max_horizon + 1):
            # 输入窗口索引：[anchor_local_idx - input_length, anchor_local_idx)
            input_start = anchor_local_idx - input_length
            input_end = anchor_local_idx
            
            # 输出窗口索引：[anchor_local_idx, anchor_local_idx + max_horizon)
            output_start = anchor_local_idx
            output_end = anchor_local_idx + max_horizon
            
            # 检查输入窗口时间连续性
            input_times = day_df[time_column].iloc[input_start:input_end].values
            input_diffs = pd.to_datetime(input_times[1:]) - pd.to_datetime(input_times[:-1])
            if not all(diff == expected_interval for diff in input_diffs):
                stats.skipped_discontinuous += 1
                continue
            
            # 检查输出窗口时间连续性
            output_times = day_df[time_column].iloc[output_start:output_end].values
            output_diffs = pd.to_datetime(output_times[1:]) - pd.to_datetime(output_times[:-1])
            if not all(diff == expected_interval for diff in output_diffs):
                stats.skipped_discontinuous += 1
                continue
            
            # 检查输入到输出的连续性（锚点前后）
            bridge_diff = pd.to_datetime(output_times[0]) - pd.to_datetime(input_times[-1])
            if bridge_diff != expected_interval:
                stats.skipped_discontinuous += 1
                continue
            
            # mask 模式：检查锚点是否为白天
            # 放宽条件：只要求锚点（预测起始时刻）是白天即可
            # 允许输入窗口包含夜间点（清晨日出场景）和输出窗口包含夜间点（黄昏日落场景）
            # 这样模型可以学习日出日落过渡时段的预测模式
            if daylight_mask_mode and 'is_daylight' in day_df.columns:
                anchor_is_daylight = day_df['is_daylight'].iloc[anchor_local_idx]
                if not anchor_is_daylight:
                    stats.skipped_contains_night += 1
                    continue
            
            # 提取输入窗口数据（所有特征列，排除时间、日期和 is_daylight）
            feature_cols = [c for c in day_df.columns if c not in [time_column, '_date', 'is_daylight']]
            X = day_df[feature_cols].iloc[input_start:input_end].values
            
            # 提取输出窗口数据（仅功率）
            Y = day_df[target_column].iloc[output_start:output_end].values
            
            # 创建样本
            sample = WindowSample(
                X=X,
                Y=Y,
                anchor_time=day_df[time_column].iloc[anchor_local_idx],
                anchor_idx=anchor_local_idx,
                date=date
            )
            samples.append(sample)
    
    stats.total_samples = len(samples)
    
    # 清理临时列
    if '_date' in df.columns:
        df.drop(columns=['_date'], inplace=True)
    
    return WindowResult(samples=samples, stats=stats)


def print_window_summary(result: WindowResult, daylight_mode: str = None) -> None:
    """打印滑窗生成摘要"""
    stats = result.stats
    print("\n" + "=" * 60)
    print("滑窗样本生成摘要")
    print("=" * 60)
    print(f"生成样本数: {stats.total_samples}")
    print("-" * 40)
    print("跳过原因统计:")
    print(f"  - 输入窗口不足: {stats.skipped_insufficient_input}")
    print(f"  - 输出窗口不足: {stats.skipped_insufficient_output}")
    print(f"  - 跨天: {stats.skipped_cross_day}")
    print(f"  - 时间不连续: {stats.skipped_discontinuous}")
    if daylight_mode == 'mask':
        print(f"  - 锚点在夜间: {stats.skipped_contains_night}")
    print("=" * 60)
