# -*- coding: utf-8 -*-
"""
B0 阶段主入口脚本
串联数据读取、白天筛选、切分、滑窗生成、Persistence Baseline、Rolling Evaluation
并输出详细的终端摘要
支持白天筛选功能（drop/mask 两种模式）
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np

from src.data.loader import load_solar_data, print_load_summary
from src.data.splitter import split_by_day, print_split_summary
from src.data.window import generate_windows, print_window_summary
from src.data.daylight import add_daylight_flag, filter_daylight_rows, print_daylight_summary
from src.baselines.persistence import PersistenceBaseline
from src.evaluation.rolling_eval import RollingEvaluator, print_eval_table
from src.utils.io import save_results
from src.utils.seed import set_seed, print_seed_info


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """B0 阶段主流程"""
    print("\n" + "=" * 70)
    print("B0 阶段 - Persistence Baseline 实验")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/7] 配置加载完成: {config_path}")
    
    # 获取各模块配置
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    eval_config = config['evaluation']
    output_config = config['output']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    train_config = config.get('training', {})
    
    # 随机种子设置 (可复现性)
    seed = train_config.get('seed', 42)
    set_seed(seed, deterministic=True)
    print_seed_info(seed, deterministic=True)
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'drop') if daylight_enabled else None
    
    if daylight_enabled:
        print(f"  - 白天筛选: 启用 (mode={daylight_mode}, threshold={daylight_config['threshold']})")
    else:
        print(f"  - 白天筛选: 禁用")
    
    # =========================================================================
    # 2. 数据读取
    # =========================================================================
    print("\n[2/7] 读取数据...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    # 打印数据读取摘要
    print_load_summary(load_result)
    
    # -------------------------------------------------------------------------
    # 自检1: Power 单位确认
    # -------------------------------------------------------------------------
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    
    # 筛选白天样本（Power > 1MW 视为有效白天样本）
    daylight_power = load_result.df[load_result.df[target_col] > 1][target_col].values
    
    print("\n" + "-" * 60)
    print("[自检1] Power 单位确认 (白天有效样本)")
    print("-" * 60)
    print(f"Power 列名: {target_col}")
    print(f"装机容量: {capacity} MW")
    print(f"Power (MW) 白天前5条: {np.round(daylight_power[:5], 2)}")
    power_pu = daylight_power / capacity
    print(f"Power (p.u.) 白天前5条: {np.round(power_pu[:5], 4)}")
    
    # =========================================================================
    # 3. 白天筛选（可选）
    # =========================================================================
    print("\n[3/7] 白天筛选...")
    
    df_for_split = load_result.df
    original_rows = len(df_for_split)
    
    if daylight_enabled:
        # 添加 is_daylight 列
        daylight_result = add_daylight_flag(
            df=load_result.df,
            time_column=data_config['time_column'],
            dni_col=daylight_config['dni_col'],
            threshold=daylight_config['threshold']
        )
        
        # 打印白天筛选统计
        print_daylight_summary(daylight_result, mode=daylight_mode, dni_col=daylight_config['dni_col'])
        
        if daylight_mode == 'drop':
            # drop 模式：直接过滤夜间行
            filtered_df, rows_before, rows_after = filter_daylight_rows(daylight_result.df)
            daylight_result.filtered_df = filtered_df
            daylight_result.rows_before_filter = rows_before
            daylight_result.rows_after_filter = rows_after
            
            df_for_split = filtered_df
            print(f"\n[drop 模式] 过滤后行数: {rows_after} / {rows_before} ({rows_after/rows_before*100:.1f}%)")
            
            # 诊断：drop 后的时间连续性
            import pandas as pd
            time_diffs = filtered_df[data_config['time_column']].diff().dropna()
            expected_interval = pd.Timedelta(minutes=data_config['time_interval_minutes'])
            global_discontinuous = (time_diffs != expected_interval).sum()
            
            # 按天分组检查天内连续性
            filtered_df_temp = filtered_df.copy()
            filtered_df_temp['_date'] = filtered_df_temp[data_config['time_column']].dt.date
            intra_day_discontinuous = 0
            for date, day_df in filtered_df_temp.groupby('_date'):
                if len(day_df) < 2:
                    continue
                day_diffs = day_df[data_config['time_column']].diff().dropna()
                intra_day_discontinuous += (day_diffs != expected_interval).sum()
            
            print(f"[drop 诊断] 全局时间不连续: {global_discontinuous} (跨天跳跃)")
            print(f"[drop 诊断] 天内时间不连续: {intra_day_discontinuous}")
            if intra_day_discontinuous == 0:
                print("[drop 诊断] 每天内部白天行连续，按天分组后滑窗不会触发时间不连续")
        else:
            # mask 模式：保留所有行，滑窗时检查
            df_for_split = daylight_result.df
            print(f"\n[mask 模式] 保留所有行，滑窗时检查锚点是否为白天")
    else:
        print("白天筛选已禁用，跳过此步骤")
    
    # =========================================================================
    # 4. 按天切分 train / val / test
    # =========================================================================
    print("\n[4/7] 按天切分数据...")
    
    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    # 打印切分摘要
    print_split_summary(split_result)
    
    # -------------------------------------------------------------------------
    # 自检2: test 集 Power 分布统计
    # -------------------------------------------------------------------------
    import pandas as pd
    test_power = split_result.test_df[target_col]
    
    print("\n" + "-" * 60)
    print("[自检2] test 集 Power 分布统计")
    print("-" * 60)
    print(f"count: {len(test_power)}")
    print(f"mean:  {test_power.mean():.4f} MW")
    print(f"std:   {test_power.std():.4f} MW")
    print(f"min:   {test_power.min():.4f} MW")
    print(f"P5:    {test_power.quantile(0.05):.4f} MW")
    print(f"P25:   {test_power.quantile(0.25):.4f} MW")
    print(f"median:{test_power.median():.4f} MW")
    print(f"P75:   {test_power.quantile(0.75):.4f} MW")
    print(f"P95:   {test_power.quantile(0.95):.4f} MW")
    print(f"max:   {test_power.max():.4f} MW")
    
    # Power>0 和 Power<1MW 的比例
    ratio_gt0 = (test_power > 0).mean() * 100
    ratio_lt1 = (test_power < 1).mean() * 100
    print(f"\nPower > 0 占比:  {ratio_gt0:.1f}%")
    print(f"Power < 1MW 占比: {ratio_lt1:.1f}% (夜晚/低功率)")
    
    # |deltaPower| 15min差分分布
    delta_power = test_power.diff().abs().dropna()
    print(f"\n|deltaPower| (15min差分/ramp):")
    print(f"  mean: {delta_power.mean():.4f} MW")
    print(f"  P95:  {delta_power.quantile(0.95):.4f} MW")
    print(f"  max:  {delta_power.max():.4f} MW")
    
    # =========================================================================
    # 5. 生成滑窗样本（仅对 test 集）
    # =========================================================================
    print("\n[5/7] 生成滑窗样本（test 集）...")
    
    # 为什么只对 test 集生成样本：
    # B0 阶段 Persistence Baseline 不需要训练，只需评估
    # 后续 B1+ 阶段会对 train/val 也生成样本
    
    # 判断是否使用 mask 模式
    use_mask_mode = daylight_enabled and daylight_mode == 'mask'
    
    test_window_result = generate_windows(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=data_config['target_column'],
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    # 打印滑窗摘要
    print_window_summary(test_window_result, daylight_mode=daylight_mode if daylight_enabled else None)
    
    # 打印样本数变化（如果启用了白天筛选）
    if daylight_enabled:
        print(f"\n剔除夜间后 test 样本数: {test_window_result.stats.total_samples}")
        if daylight_mode == 'drop':
            print(f"  - 因时间不连续跳过: {test_window_result.stats.skipped_discontinuous}")
        elif daylight_mode == 'mask':
            print(f"  - 因锚点在夜间跳过: {test_window_result.stats.skipped_contains_night}")
    
    # =========================================================================
    # 6. Persistence Baseline + Rolling Evaluation
    # =========================================================================
    print("\n[6/7] 执行 Rolling Evaluation...")
    
    # 创建 Persistence Baseline
    baseline = PersistenceBaseline(max_horizon=window_config['max_horizon'])
    print(f"Model: {baseline}")
    
    # 创建评估器
    evaluator = RollingEvaluator(
        horizons=eval_config['horizons'],
        horizon_names=eval_config['horizon_names'],
        nominal_capacity=data_config['nominal_capacity_mw']
    )
    
    # 执行评估
    eval_result = evaluator.evaluate(
        samples=test_window_result.samples,
        predict_fn=baseline.predict
    )
    
    # 打印评估结果表格
    model_display_name = "Persistence Baseline"
    if daylight_enabled:
        model_display_name += f" (daylight={daylight_mode})"
    print_eval_table(eval_result, model_name=model_display_name)
    
    # -------------------------------------------------------------------------
    # 自检3: nRMSE 计算校验 & 量级解释
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[自检3] nRMSE 计算校验")
    print("-" * 60)
    print(f"注: 评估表中 MAE/RMSE 单位为 MW (原始功率)")
    
    # 取 4h horizon 校验
    hr_4h = [hr for hr in eval_result.horizon_results if hr.horizon == 16][0]
    rmse_mw = hr_4h.metrics.rmse
    nrmse_reported = hr_4h.metrics.nrmse
    nrmse_calc = rmse_mw / capacity
    diff = abs(nrmse_reported - nrmse_calc)
    
    print(f"4h RMSE_MW: {rmse_mw:.4f}")
    print(f"4h nRMSE (reported): {nrmse_reported:.4f}")
    print(f"4h nRMSE (RMSE_MW / {capacity}): {nrmse_calc:.4f}")
    print(f"差值: {diff:.2e} {'(OK)' if diff < 1e-6 else '(WARNING: 不一致!)'}")
    
    # 量级解释
    print("\n" + "-" * 60)
    print("[量级解释]")
    print("-" * 60)
    # 获取 test 集白天 Power 的 P95
    test_power_daylight = split_result.test_df[split_result.test_df[target_col] > 1][target_col]
    p95_delta = test_power_daylight.diff().abs().dropna().quantile(0.95)
    
    print(f"4h RMSE: {rmse_mw:.2f} MW, nRMSE: {nrmse_reported*100:.1f}%")
    print(f"test 白天 |deltaPower| P95: {p95_delta:.2f} MW")
    
    if nrmse_reported > 0.4:
        print("-> Persistence 在 4h 误差较大属正常: 光伏出力变化快，4h 后难以用当前值预测")
    
    # =========================================================================
    # 7. 保存结果
    # =========================================================================
    print("\n[7/7] 保存结果...")
    
    saved_files = save_results(
        result=eval_result,
        output_dir=str(PROJECT_ROOT / output_config['results_dir']),
        model_name="persistence",
        experiment_name="b0",
        daylight_config=daylight_config
    )
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("B0 阶段完成！")
    if daylight_enabled:
        print(f"  - 白天筛选模式: {daylight_mode}")
        print(f"  - 最终 test 样本数: {eval_result.total_samples}")
    print("=" * 70)
    
    return eval_result


if __name__ == "__main__":
    main()
