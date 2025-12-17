# -*- coding: utf-8 -*-
"""
对比修改前后 daylight_mask_mode 的样本数量变化
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

from src.data.loader import load_solar_data
from src.data.splitter import split_by_day
from src.data.daylight import add_daylight_flag


@dataclass
class WindowSample:
    X: np.ndarray
    Y: np.ndarray
    anchor_time: pd.Timestamp
    anchor_idx: int
    date: object


def generate_windows_old_logic(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    input_length: int = 16,
    max_horizon: int = 16,
    time_interval_minutes: int = 15,
    daylight_mask_mode: bool = False
):
    """旧逻辑: 要求输入+输出窗口所有32个点都是白天"""
    samples = []
    skipped_night = 0
    
    df = df.sort_values(by=time_column).reset_index(drop=True)
    df['_date'] = df[time_column].dt.date
    expected_interval = pd.Timedelta(minutes=time_interval_minutes)
    
    for date, day_df in df.groupby('_date'):
        day_df = day_df.reset_index(drop=True)
        n = len(day_df)
        
        for anchor_local_idx in range(input_length, n - max_horizon + 1):
            input_start = anchor_local_idx - input_length
            input_end = anchor_local_idx
            output_start = anchor_local_idx
            output_end = anchor_local_idx + max_horizon
            
            # 检查时间连续性
            input_times = day_df[time_column].iloc[input_start:input_end].values
            input_diffs = pd.to_datetime(input_times[1:]) - pd.to_datetime(input_times[:-1])
            if not all(diff == expected_interval for diff in input_diffs):
                continue
            
            output_times = day_df[time_column].iloc[output_start:output_end].values
            output_diffs = pd.to_datetime(output_times[1:]) - pd.to_datetime(output_times[:-1])
            if not all(diff == expected_interval for diff in output_diffs):
                continue
            
            bridge_diff = pd.to_datetime(output_times[0]) - pd.to_datetime(input_times[-1])
            if bridge_diff != expected_interval:
                continue
            
            # 旧逻辑: 要求所有32个点都是白天
            if daylight_mask_mode and 'is_daylight' in day_df.columns:
                input_daylight = day_df['is_daylight'].iloc[input_start:input_end].values
                output_daylight = day_df['is_daylight'].iloc[output_start:output_end].values
                if not (input_daylight.all() and output_daylight.all()):
                    skipped_night += 1
                    continue
            
            feature_cols = [c for c in day_df.columns if c not in [time_column, '_date', 'is_daylight']]
            X = day_df[feature_cols].iloc[input_start:input_end].values
            Y = day_df[target_column].iloc[output_start:output_end].values
            
            sample = WindowSample(
                X=X, Y=Y,
                anchor_time=day_df[time_column].iloc[anchor_local_idx],
                anchor_idx=anchor_local_idx,
                date=date
            )
            samples.append(sample)
    
    if '_date' in df.columns:
        df.drop(columns=['_date'], inplace=True)
    
    return samples, skipped_night


def generate_windows_new_logic(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    input_length: int = 16,
    max_horizon: int = 16,
    time_interval_minutes: int = 15,
    daylight_mask_mode: bool = False
):
    """新逻辑: 只要求锚点是白天"""
    samples = []
    skipped_night = 0
    
    df = df.sort_values(by=time_column).reset_index(drop=True)
    df['_date'] = df[time_column].dt.date
    expected_interval = pd.Timedelta(minutes=time_interval_minutes)
    
    for date, day_df in df.groupby('_date'):
        day_df = day_df.reset_index(drop=True)
        n = len(day_df)
        
        for anchor_local_idx in range(input_length, n - max_horizon + 1):
            input_start = anchor_local_idx - input_length
            input_end = anchor_local_idx
            output_start = anchor_local_idx
            output_end = anchor_local_idx + max_horizon
            
            # 检查时间连续性
            input_times = day_df[time_column].iloc[input_start:input_end].values
            input_diffs = pd.to_datetime(input_times[1:]) - pd.to_datetime(input_times[:-1])
            if not all(diff == expected_interval for diff in input_diffs):
                continue
            
            output_times = day_df[time_column].iloc[output_start:output_end].values
            output_diffs = pd.to_datetime(output_times[1:]) - pd.to_datetime(output_times[:-1])
            if not all(diff == expected_interval for diff in output_diffs):
                continue
            
            bridge_diff = pd.to_datetime(output_times[0]) - pd.to_datetime(input_times[-1])
            if bridge_diff != expected_interval:
                continue
            
            # 新逻辑: 只要求锚点是白天
            if daylight_mask_mode and 'is_daylight' in day_df.columns:
                anchor_is_daylight = day_df['is_daylight'].iloc[anchor_local_idx]
                if not anchor_is_daylight:
                    skipped_night += 1
                    continue
            
            feature_cols = [c for c in day_df.columns if c not in [time_column, '_date', 'is_daylight']]
            X = day_df[feature_cols].iloc[input_start:input_end].values
            Y = day_df[target_column].iloc[output_start:output_end].values
            
            sample = WindowSample(
                X=X, Y=Y,
                anchor_time=day_df[time_column].iloc[anchor_local_idx],
                anchor_idx=anchor_local_idx,
                date=date
            )
            samples.append(sample)
    
    if '_date' in df.columns:
        df.drop(columns=['_date'], inplace=True)
    
    return samples, skipped_night


def analyze_transition_samples(samples_old, samples_new, df_with_daylight, time_column):
    """分析新增样本的时段分布"""
    old_anchors = set(s.anchor_time for s in samples_old)
    new_samples_only = [s for s in samples_new if s.anchor_time not in old_anchors]
    
    if not new_samples_only:
        return {}
    
    # 统计新增样本的小时分布
    hour_dist = {}
    for s in new_samples_only:
        hour = s.anchor_time.hour
        hour_dist[hour] = hour_dist.get(hour, 0) + 1
    
    return hour_dist


def main():
    print("\n" + "=" * 70)
    print("daylight_mask_mode 修改前后样本数量对比")
    print("=" * 70)
    
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    daylight_config = config.get('daylight_filter', {})
    
    # 加载数据
    print("\n[1] 加载数据...")
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    # 添加白天标记
    daylight_result = add_daylight_flag(
        df=load_result.df,
        time_column=data_config['time_column'],
        dni_col=daylight_config['dni_col'],
        threshold=daylight_config['threshold']
    )
    
    # 按天切分
    split_result = split_by_day(
        df=daylight_result.df,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    print(f"  Train days: {split_result.train_days}, Val days: {split_result.val_days}, Test days: {split_result.test_days}")
    
    # 对比两种逻辑
    print("\n[2] 生成样本 (旧逻辑: 所有32点为白天)...")
    train_old, train_skip_old = generate_windows_old_logic(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )
    val_old, val_skip_old = generate_windows_old_logic(
        df=split_result.val_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )
    test_old, test_skip_old = generate_windows_old_logic(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )
    
    print("\n[3] 生成样本 (新逻辑: 只要求锚点为白天)...")
    train_new, train_skip_new = generate_windows_new_logic(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )
    val_new, val_skip_new = generate_windows_new_logic(
        df=split_result.val_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )
    test_new, test_skip_new = generate_windows_new_logic(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )
    
    # 输出对比结果
    print("\n" + "=" * 70)
    print("样本数量对比")
    print("=" * 70)
    print(f"{'数据集':<10} {'旧逻辑':>12} {'新逻辑':>12} {'增加':>10} {'增幅':>10}")
    print("-" * 56)
    
    train_diff = len(train_new) - len(train_old)
    val_diff = len(val_new) - len(val_old)
    test_diff = len(test_new) - len(test_old)
    
    train_pct = train_diff / len(train_old) * 100 if len(train_old) > 0 else 0
    val_pct = val_diff / len(val_old) * 100 if len(val_old) > 0 else 0
    test_pct = test_diff / len(test_old) * 100 if len(test_old) > 0 else 0
    
    print(f"{'Train':<10} {len(train_old):>12} {len(train_new):>12} {train_diff:>+10} {train_pct:>+9.1f}%")
    print(f"{'Val':<10} {len(val_old):>12} {len(val_new):>12} {val_diff:>+10} {val_pct:>+9.1f}%")
    print(f"{'Test':<10} {len(test_old):>12} {len(test_new):>12} {test_diff:>+10} {test_pct:>+9.1f}%")
    print("-" * 56)
    
    total_old = len(train_old) + len(val_old) + len(test_old)
    total_new = len(train_new) + len(val_new) + len(test_new)
    total_diff = total_new - total_old
    total_pct = total_diff / total_old * 100 if total_old > 0 else 0
    print(f"{'Total':<10} {total_old:>12} {total_new:>12} {total_diff:>+10} {total_pct:>+9.1f}%")
    
    # 跳过统计
    print("\n" + "=" * 70)
    print("跳过样本数对比 (因夜间过滤)")
    print("=" * 70)
    print(f"{'数据集':<10} {'旧逻辑跳过':>15} {'新逻辑跳过':>15} {'减少':>12}")
    print("-" * 56)
    print(f"{'Train':<10} {train_skip_old:>15} {train_skip_new:>15} {train_skip_old - train_skip_new:>12}")
    print(f"{'Val':<10} {val_skip_old:>15} {val_skip_new:>15} {val_skip_old - val_skip_new:>12}")
    print(f"{'Test':<10} {test_skip_old:>15} {test_skip_new:>15} {test_skip_old - test_skip_new:>12}")
    
    # 新增样本的时段分布
    print("\n" + "=" * 70)
    print("新增样本的锚点时段分布 (Train)")
    print("=" * 70)
    
    hour_dist = analyze_transition_samples(train_old, train_new, split_result.train_df, data_config['time_column'])
    if hour_dist:
        print(f"{'小时':<8} {'样本数':>10} {'占比':>10}")
        print("-" * 30)
        total_new_only = sum(hour_dist.values())
        for hour in sorted(hour_dist.keys()):
            count = hour_dist[hour]
            pct = count / total_new_only * 100
            print(f"{hour:02d}:00    {count:>10} {pct:>9.1f}%")
        print("-" * 30)
        print(f"{'合计':<8} {total_new_only:>10}")
    else:
        print("无新增样本")
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"修改后样本总量从 {total_old} 增加到 {total_new} (+{total_pct:.1f}%)")
    print("新增样本主要来自清晨和黄昏的过渡时段")
    print("=" * 70)


if __name__ == "__main__":
    main()
