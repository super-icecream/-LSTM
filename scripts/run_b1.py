# -*- coding: utf-8 -*-
"""
B1 阶段主入口脚本
Global LSTM 多步预测（不分型）
复用 B0 的数据流水线和评估模块，仅将预测器从 Persistence 换成 Global LSTM
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import numpy as np
import torch
from datetime import datetime

from src.data.loader import load_solar_data, print_load_summary
from src.data.splitter import split_by_day, print_split_summary
from src.data.window import generate_windows, print_window_summary
from src.data.daylight import add_daylight_flag, filter_daylight_rows, print_daylight_summary
from src.models.lstm import GlobalLSTM
from src.training.trainer import LSTMTrainer, TrainConfig
from src.evaluation.rolling_eval import RollingEvaluator, print_eval_table
from src.utils.io import save_results
from src.utils.seed import set_seed, print_seed_info


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def samples_to_arrays(samples, nominal_capacity):
    """
    将 WindowSample 列表转换为训练用的 numpy 数组
    
    Args:
        samples: List[WindowSample]
        nominal_capacity: 装机容量 (MW)
    
    Returns:
        X: (n_samples, Lx, n_features) - 输入特征 (Power_pu)
        y: (n_samples, Hmax) - 目标 Power_pu
    """
    X_list = []
    y_list = []
    
    for sample in samples:
        # 输入特征：X 已经是 (Lx, n_features) 的 numpy 数组
        X_list.append(sample.X)
        
        # 目标：Y 已经是 (Hmax,) 的 numpy 数组
        y_list.append(sample.Y)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # 归一化到 p.u.
    X = X / nominal_capacity
    y = y / nominal_capacity
    
    return X, y


def main():
    """B1 阶段主流程"""
    print("\n" + "=" * 70)
    print("B1 阶段 - Global LSTM 多步预测")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/9] 配置加载完成: {config_path}")
    
    # 获取各模块配置
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    eval_config = config['evaluation']
    output_config = config['output']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    model_config = config.get('model', {})
    train_config_dict = config.get('training', {})
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    
    # 特征列（B1 阶段仅使用 Power_pu 作为输入）
    feature_columns = [target_col]
    
    print(f"  - 白天筛选: {'启用 (mode=' + daylight_mode + ')' if daylight_enabled else '禁用'}")
    print(f"  - 输入长度 Lx: {window_config['input_length']}")
    print(f"  - 输出长度 Hmax: {window_config['max_horizon']}")
    print(f"  - 装机容量: {capacity} MW")
    
    # =========================================================================
    # 2. 数据读取
    # =========================================================================
    print("\n[2/9] 读取数据...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    print_load_summary(load_result)
    
    # =========================================================================
    # 3. 白天筛选（可选）
    # =========================================================================
    print("\n[3/9] 白天筛选...")
    
    df_for_split = load_result.df
    
    if daylight_enabled:
        daylight_result = add_daylight_flag(
            df=load_result.df,
            time_column=data_config['time_column'],
            dni_col=daylight_config['dni_col'],
            threshold=daylight_config['threshold']
        )
        
        print_daylight_summary(daylight_result, mode=daylight_mode, dni_col=daylight_config['dni_col'])
        
        if daylight_mode == 'drop':
            filtered_df, rows_before, rows_after = filter_daylight_rows(daylight_result.df)
            df_for_split = filtered_df
            print(f"\n[drop 模式] 过滤后行数: {rows_after} / {rows_before}")
        else:
            df_for_split = daylight_result.df
            print(f"\n[mask 模式] 保留所有行，滑窗时检查锚点是否为白天")
    else:
        print("白天筛选已禁用，跳过此步骤")
    
    # =========================================================================
    # 4. 按天切分 train / val / test
    # =========================================================================
    print("\n[4/9] 按天切分数据...")
    
    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    print_split_summary(split_result)
    
    # =========================================================================
    # 5. 生成滑窗样本（train / val / test）
    # =========================================================================
    print("\n[5/9] 生成滑窗样本...")
    
    use_mask_mode = daylight_enabled and daylight_mode == 'mask'
    
    # Train
    train_window_result = generate_windows(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    # Val
    val_window_result = generate_windows(
        df=split_result.val_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    # Test
    test_window_result = generate_windows(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    print(f"Train 样本数: {train_window_result.stats.total_samples}")
    print(f"Val 样本数:   {val_window_result.stats.total_samples}")
    print(f"Test 样本数:  {test_window_result.stats.total_samples}")
    
    if use_mask_mode:
        print(f"  (因锚点在夜间跳过: train={train_window_result.stats.skipped_contains_night}, "
              f"val={val_window_result.stats.skipped_contains_night}, "
              f"test={test_window_result.stats.skipped_contains_night})")
    
    # =========================================================================
    # 6. 准备训练数据（转换为 numpy 数组，归一化到 p.u.）
    # =========================================================================
    print("\n[6/9] 准备训练数据...")
    
    # 转换为数组（已归一化到 p.u.）
    X_train, y_train = samples_to_arrays(train_window_result.samples, capacity)
    X_val, y_val = samples_to_arrays(val_window_result.samples, capacity)
    X_test, y_test = samples_to_arrays(test_window_result.samples, capacity)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    
    # 自检：打印 p.u. 范围
    print(f"\n[自检] Power_pu 范围:")
    print(f"  y_train: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  y_val:   [{y_val.min():.4f}, {y_val.max():.4f}]")
    print(f"  y_test:  [{y_test.min():.4f}, {y_test.max():.4f}]")
    
    # =========================================================================
    # 7. 训练 Global LSTM
    # =========================================================================
    print("\n[7/9] 训练 Global LSTM...")
    
    # 模型配置
    input_size = X_train.shape[2]  # n_features
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    output_steps = window_config['max_horizon']
    
    model = GlobalLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_steps=output_steps,
        dropout=dropout
    )
    print(f"Model: {model}")
    
    # 随机种子设置 (可复现性)
    seed = train_config_dict.get('seed', 42)
    deterministic = train_config_dict.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    print_seed_info(seed, deterministic)
    
    # 设备配置
    device_config = train_config_dict.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    # 注意: 如果 deterministic=True, cudnn.benchmark 会被设为 False
    cudnn_benchmark = train_config_dict.get('cudnn_benchmark', False)
    if cudnn_benchmark and device == "cuda" and not deterministic:
        torch.backends.cudnn.benchmark = True
    
    # 打印设备信息
    print(f"\n[设备信息]")
    print(f"  Device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    
    # 训练配置
    train_cfg = TrainConfig(
        lr=train_config_dict.get('lr', 1e-3),
        batch_size=train_config_dict.get('batch_size', 256),
        max_epochs=train_config_dict.get('max_epochs', 100),
        patience=train_config_dict.get('patience', 10),
        device=device,
        seed=seed
    )
    
    # 创建训练器
    trainer = LSTMTrainer(
        model=model,
        config=train_cfg,
        nominal_capacity=capacity,
        horizons=eval_config['horizons'],
        horizon_names=eval_config['horizon_names']
    )
    
    # 训练
    train_result = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=True
    )
    
    # =========================================================================
    # 8. 测试集评估（Rolling Evaluation）
    # =========================================================================
    print("\n[8/9] 执行 Rolling Evaluation (Test)...")
    
    # 创建评估器
    evaluator = RollingEvaluator(
        horizons=eval_config['horizons'],
        horizon_names=eval_config['horizon_names'],
        nominal_capacity=capacity
    )
    
    # 定义预测函数（返回 MW）
    def lstm_predict_fn(input_window_values: np.ndarray) -> np.ndarray:
        """
        预测函数
        
        Args:
            input_window_values: (Lx, n_features) 输入特征 (MW)
        
        Returns:
            (Hmax,) 预测的 Power (MW)
        """
        # 转换为 p.u.
        x_pu = input_window_values.reshape(1, -1, input_window_values.shape[-1]) / capacity
        x_tensor = torch.FloatTensor(x_pu).to(train_cfg.device)
        
        model.eval()
        with torch.no_grad():
            y_pred_pu = model(x_tensor)
        
        # 转换回 MW
        y_pred_mw = y_pred_pu.cpu().numpy().flatten() * capacity
        return y_pred_mw
    
    # 执行评估
    eval_result = evaluator.evaluate(
        samples=test_window_result.samples,
        predict_fn=lstm_predict_fn
    )
    
    # 打印评估结果表格
    model_display_name = "Global LSTM"
    if daylight_enabled:
        model_display_name += f" (daylight={daylight_mode})"
    print_eval_table(eval_result, model_name=model_display_name)
    
    # 自检：nRMSE 校验
    print("\n" + "-" * 60)
    print("[自检] nRMSE 计算校验")
    print("-" * 60)
    hr_4h = [hr for hr in eval_result.horizon_results if hr.horizon == 16][0]
    rmse_mw = hr_4h.metrics.rmse
    nrmse_reported = hr_4h.metrics.nrmse
    nrmse_calc = rmse_mw / capacity
    diff = abs(nrmse_reported - nrmse_calc)
    print(f"4h RMSE_MW: {rmse_mw:.4f}")
    print(f"4h nRMSE (reported): {nrmse_reported:.4f}")
    print(f"4h nRMSE (RMSE_MW / {capacity}): {nrmse_calc:.4f}")
    print(f"差值: {diff:.2e} {'(OK)' if diff < 1e-6 else '(WARNING!)'}")
    
    # =========================================================================
    # 9. 保存结果
    # =========================================================================
    print("\n[9/9] 保存结果...")
    
    # 保存模型
    model_path = PROJECT_ROOT / output_config['results_dir'] / "model_global.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # 保存评估结果
    saved_files = save_results(
        result=eval_result,
        output_dir=str(PROJECT_ROOT / output_config['results_dir']),
        model_name="lstm_global",
        experiment_name="b1",
        daylight_config=daylight_config
    )
    
    # 保存训练日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / output_config['logs_dir'] / f"train_log_b1_{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_log = {
        "config_snapshot": {
            "Lx": window_config['input_length'],
            "Hmax": window_config['max_horizon'],
            "daylight_enabled": daylight_enabled,
            "daylight_mode": daylight_mode,
            "feature_columns": feature_columns,
            "nominal_capacity_mw": capacity,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": train_cfg.lr,
            "batch_size": train_cfg.batch_size,
            "max_epochs": train_cfg.max_epochs,
            "patience": train_cfg.patience
        },
        "train_result": {
            "best_epoch": train_result.best_epoch,
            "best_val_loss": float(train_result.best_val_loss),
            "best_val_rmse_4h_pu": float(train_result.best_val_rmse_4h_pu),
            "stopped_early": train_result.stopped_early,
            "total_epochs": len(train_result.epoch_results)
        },
        "test_metrics": {
            hr.horizon_name: {
                "mae_mw": float(hr.metrics.mae),
                "rmse_mw": float(hr.metrics.rmse),
                "nrmse": float(hr.metrics.nrmse)
            }
            for hr in eval_result.horizon_results
        }
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)
    print(f"Train log saved to: {log_path}")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("B1 阶段完成!")
    print(f"  - 模型: Global LSTM (hidden={hidden_size}, layers={num_layers})")
    print(f"  - Best epoch: {train_result.best_epoch}")
    print(f"  - Test 样本数: {eval_result.total_samples}")
    print(f"  - Test 4h nRMSE: {nrmse_reported:.4f} ({nrmse_reported*100:.1f}%)")
    print("=" * 70)
    
    return eval_result


if __name__ == "__main__":
    main()
