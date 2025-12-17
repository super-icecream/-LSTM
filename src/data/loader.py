# -*- coding: utf-8 -*-
"""
数据读取模块
负责从 Excel 文件读取光伏功率数据，并进行时间解析与质量检查
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DataLoadResult:
    """数据读取结果，包含数据和诊断信息"""
    df: pd.DataFrame
    time_column: str
    time_dtype: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    total_rows: int
    is_sorted: bool
    duplicate_count: int
    gap_count: int  # 非 15min 间隔的缺口数量


def load_solar_data(
    file_path: str,
    time_column: str = "Time(year-month-day h:m:s)",
    time_format: str = "%Y-%m-%d %H:%M:%S",
    time_interval_minutes: int = 15,
    project_root: Optional[Path] = None
) -> DataLoadResult:
    """
    读取光伏功率数据 Excel 文件
    
    为什么这样做：
    1. 时间序列预测必须保证时间顺序正确
    2. 必须检测重复时间戳和时间缺口，避免后续滑窗出错
    3. 严格解析时间格式，确保数据质量
    
    Args:
        file_path: Excel 文件路径（相对于 project_root 或绝对路径）
        time_column: 时间列名
        time_format: 时间格式字符串
        time_interval_minutes: 期望的时间间隔（分钟）
        project_root: 项目根目录，用于解析相对路径
    
    Returns:
        DataLoadResult: 包含数据和诊断信息的结果对象
    """
    # 解析文件路径
    if project_root is not None:
        full_path = project_root / file_path
    else:
        full_path = Path(file_path)
    
    # 读取 Excel 文件
    df = pd.read_excel(full_path, engine='openpyxl')
    
    # 严格解析时间列
    # 为什么用 pd.to_datetime 而不是直接读取：确保时间格式一致，便于后续操作
    df[time_column] = pd.to_datetime(df[time_column], format=time_format)
    
    # 检查排序前的状态
    is_sorted_before = df[time_column].is_monotonic_increasing
    
    # 按时间升序排序
    # 为什么必须排序：时间序列切分和滑窗都依赖正确的时间顺序
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # 检查重复时间戳
    # 为什么检查重复：重复时间戳会导致滑窗样本异常
    duplicate_count = df[time_column].duplicated().sum()
    
    # 计算时间间隔，检测非标准间隔（缺口）
    # 为什么检查缺口：时间不连续会导致滑窗样本包含无效数据
    time_diffs = df[time_column].diff().dropna()
    expected_interval = pd.Timedelta(minutes=time_interval_minutes)
    gap_count = (time_diffs != expected_interval).sum()
    
    # 构建结果
    result = DataLoadResult(
        df=df,
        time_column=time_column,
        time_dtype=str(df[time_column].dtype),
        start_time=df[time_column].iloc[0],
        end_time=df[time_column].iloc[-1],
        total_rows=len(df),
        is_sorted=is_sorted_before,
        duplicate_count=int(duplicate_count),
        gap_count=int(gap_count)
    )
    
    return result


def print_load_summary(result: DataLoadResult) -> None:
    """打印数据读取摘要"""
    print("\n" + "=" * 60)
    print("数据读取摘要")
    print("=" * 60)
    print(f"时间列名: {result.time_column}")
    print(f"时间 dtype: {result.time_dtype}")
    print(f"起始时间: {result.start_time}")
    print(f"结束时间: {result.end_time}")
    print(f"总行数: {result.total_rows}")
    print(f"原始数据是否已排序: {'是' if result.is_sorted else '否'}")
    print(f"重复时间戳数量: {result.duplicate_count}")
    print(f"非15min间隔缺口数量: {result.gap_count}")
    print("=" * 60)
