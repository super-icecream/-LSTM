# -*- coding: utf-8 -*-
"""
诊断 drop 模式的时间连续性问题
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 读取数据
data_path = PROJECT_ROOT / "datas/2. 甘肃光伏功率预测数据集/data_processed/solar_stations/Solar station site 1 (Nominal capacity-50MW).xlsx"
df = pd.read_excel(data_path)
time_col = 'Time(year-month-day h:m:s)'
df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S')
df = df.sort_values(by=time_col).reset_index(drop=True)

# 添加 is_daylight
df['is_daylight'] = df['Total solar irradiance (W/m2)'] >= 5

print('=== 原始数据 ===')
print(f'总行数: {len(df)}')
print(f'白天行数: {df["is_daylight"].sum()}')

# drop 过滤
filtered_df = df[df['is_daylight'] == True].copy().reset_index(drop=True)
print(f'\n=== drop 过滤后 ===')
print(f'过滤后行数: {len(filtered_df)}')

# 计算时间差
time_diffs = filtered_df[time_col].diff().dropna()
expected = pd.Timedelta(minutes=15)
non_15min_count = (time_diffs != expected).sum()

print(f'\n=== 时间连续性诊断 ===')
print(f'时间差 != 15min 的数量: {non_15min_count}')
print(f'时间差 == 15min 的数量: {(time_diffs == expected).sum()}')

# 显示部分不连续的例子
non_15min_mask = time_diffs != expected
non_15min_indices = time_diffs[non_15min_mask].index[:10].tolist()
print(f'\n前10个不连续点:')
for idx in non_15min_indices:
    prev_time = filtered_df[time_col].iloc[idx-1]
    curr_time = filtered_df[time_col].iloc[idx]
    diff = curr_time - prev_time
    print(f'  idx={idx}: {prev_time} -> {curr_time}, diff={diff}')

# ========== 关键诊断：按天分组后，每天内部是否连续 ==========
print('\n=== 按天分组后的连续性诊断 ===')
filtered_df['_date'] = filtered_df[time_col].dt.date

intra_day_discontinuous = 0
for date, day_df in filtered_df.groupby('_date'):
    day_df = day_df.reset_index(drop=True)
    if len(day_df) < 2:
        continue
    day_diffs = day_df[time_col].diff().dropna()
    day_non_15min = (day_diffs != expected).sum()
    if day_non_15min > 0:
        intra_day_discontinuous += day_non_15min
        print(f'  日期 {date}: 天内不连续 {day_non_15min} 次')

print(f'\n天内不连续总数: {intra_day_discontinuous}')
if intra_day_discontinuous == 0:
    print('结论: 每天内部的白天行是连续的，window.py 按天分组后检测不到不连续')
    print('这解释了为什么 drop 模式下"时间不连续: 0"')

# ========== 验证两种模式样本数是否应该一致 ==========
print('\n=== 验证 drop vs mask 样本数（仅 test 集，后 15% 天数）===')
input_length = 16
max_horizon = 16

# 按天切分，取后 15% 作为 test
df['_date'] = df[time_col].dt.date
unique_dates = sorted(df['_date'].unique())
total_days = len(unique_dates)
test_start_idx = int(total_days * 0.85)
test_dates = unique_dates[test_start_idx:]
print(f'总天数: {total_days}, test 天数: {len(test_dates)}')

# mask 模式：用原始数据
test_df_full = df[df['_date'].isin(test_dates)].copy()
mask_samples = 0
mask_skipped_night = 0

for date, day_df in test_df_full.groupby('_date'):
    day_df = day_df.reset_index(drop=True)
    n = len(day_df)
    
    for anchor in range(input_length, n - max_horizon + 1):
        input_daylight = day_df['is_daylight'].iloc[anchor - input_length:anchor].values
        output_daylight = day_df['is_daylight'].iloc[anchor:anchor + max_horizon].values
        if input_daylight.all() and output_daylight.all():
            mask_samples += 1
        else:
            mask_skipped_night += 1

# drop 模式：用过滤后的数据
test_df_drop = filtered_df[filtered_df['_date'].isin(test_dates)].copy()
drop_samples = 0
drop_skipped_discontinuous = 0

for date, day_df in test_df_drop.groupby('_date'):
    day_df = day_df.reset_index(drop=True)
    n = len(day_df)
    
    # 检查天内连续性
    if n < 2:
        continue
    day_diffs = day_df[time_col].diff().dropna()
    
    for anchor in range(input_length, n - max_horizon + 1):
        # 检查输入窗口连续性
        input_diffs = day_diffs.iloc[anchor - input_length:anchor - 1]
        output_diffs = day_diffs.iloc[anchor:anchor + max_horizon - 1]
        
        input_continuous = (input_diffs == expected).all() if len(input_diffs) > 0 else True
        output_continuous = (output_diffs == expected).all() if len(output_diffs) > 0 else True
        
        if input_continuous and output_continuous:
            drop_samples += 1
        else:
            drop_skipped_discontinuous += 1

print(f'\nmask 模式:')
print(f'  - 样本数: {mask_samples}')
print(f'  - 因锚点在夜间跳过: {mask_skipped_night}')

print(f'\ndrop 模式:')
print(f'  - 样本数: {drop_samples}')
print(f'  - 因时间不连续跳过: {drop_skipped_discontinuous}')

if mask_samples == drop_samples:
    print(f'\n结论: 两种模式样本数相同 ({mask_samples})，逻辑正确')
    print('原因: 每天内部的白天行是连续的，drop 后天内无缺口')
else:
    print(f'\n警告: 两种模式样本数不同！需检查逻辑')
