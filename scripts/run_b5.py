# -*- coding: utf-8 -*-
"""
B5 阶段主入口脚本
Hard vs Soft 路由对比实验

与 B4 的区别：
- B4: 训练 3 个专家模型，仅使用 Hard 路由 (argmax)
- B5: 加载 B4 已训练的专家模型，对比 Hard 与 Soft 路由

流程：
1. 复用 B3/B4 的 Router: 加载 scaler_g.pkl, pca.pkl, fcm_centers.npy
2. 加载 B4 已训练的专家模型 (model_expert_0/1/2.pt)
3. 对 test 每个样本:
   - 计算 g_t -> scaler_g -> PCA -> FCM 隶属度 u_t
   - Hard: 使用 expert[argmax(u_t)] 预测
   - Soft: 对所有专家预测加权融合 sum_k u_t[k] * hatY_t^(k)
4. 输出 Hard/Soft 整体和按簇误差对比
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from src.data.loader import load_solar_data, print_load_summary
from src.data.splitter import split_by_day, print_split_summary
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows, print_daylight_summary
from src.models.lstm import GlobalLSTM
from src.features.scaler import FeatureScaler, FeatureScalerG
from src.utils.seed import set_seed, print_seed_info


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# 状态向量 g_t 构建 (与 B3 完全一致)
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """
    计算单个样本的状态向量 g_t (与 B3 完全一致)
    
    Returns:
        g_t: 状态向量 (D,)
        g_names: 特征名列表
    """
    X = sample.X  # (Lx, n_features)
    
    g_values = []
    g_names = []
    
    def get_col_idx(partial_name):
        for i, name in enumerate(feature_names):
            if partial_name.lower() in name.lower():
                return i
        return None
    
    tsi_idx = get_col_idx('Total solar')
    dni_idx = get_col_idx('Direct normal')
    ghi_idx = get_col_idx('Global horizontal')
    temp_idx = get_col_idx('temperature')
    atm_idx = get_col_idx('Atmosphere')
    power_idx = get_col_idx('Power')
    
    # 1. 辐照统计 (TSI/DNI/GHI): mean/std/min/max
    irr_indices = [('TSI', tsi_idx), ('DNI', dni_idx), ('GHI', ghi_idx)]
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            g_values.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
            g_names.extend([f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max'])
    
    # 2. 辐照变化强度: mean(|delta|), std(delta), max(delta)
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            delta = np.diff(vals)
            if len(delta) > 0:
                g_values.extend([np.mean(np.abs(delta)), np.std(delta), np.max(np.abs(delta))])
                g_names.extend([f'{name}_delta_mean_abs', f'{name}_delta_std', f'{name}_delta_max_abs'])
    
    # 3. 历史功率 (Power_pu): mean/std/mean(|delta|)
    if power_idx is not None:
        power_pu = X[:, power_idx] / capacity
        g_values.extend([power_pu.mean(), power_pu.std()])
        g_names.extend(['Power_pu_mean', 'Power_pu_std'])
        
        delta_power = np.diff(power_pu)
        if len(delta_power) > 0:
            g_values.append(np.mean(np.abs(delta_power)))
            g_names.append('Power_pu_delta_mean_abs')
    
    # 4. 气温、气压: mean/std
    if temp_idx is not None:
        temp = X[:, temp_idx]
        g_values.extend([temp.mean(), temp.std()])
        g_names.extend(['Temp_mean', 'Temp_std'])
    
    if atm_idx is not None:
        atm = X[:, atm_idx]
        g_values.extend([atm.mean(), atm.std()])
        g_names.extend(['Atm_mean', 'Atm_std'])
    
    return np.array(g_values, dtype=np.float32), g_names


def compute_fcm_membership(z: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    """
    计算 FCM 隶属度向量 (与 B4 一致，使用平方距离)
    
    Args:
        z: (n_features,) 单个样本的 PCA 降维后向量
        centers: (K, n_features) FCM 聚类中心
        m: FCM 模糊指数
    
    Returns:
        u: (K,) 隶属度向量，sum(u) = 1
    """
    K = centers.shape[0]
    
    # 计算到各中心的平方距离 (与 B4 一致)
    distances = np.zeros(K)
    for k in range(K):
        diff = z - centers[k]
        distances[k] = np.sum(diff ** 2)
    
    # 避免除零
    distances = np.maximum(distances, 1e-10)
    
    # FCM 隶属度公式
    power = 2.0 / (m - 1)
    u = np.zeros(K)
    
    for k in range(K):
        u[k] = 1.0 / np.sum((distances[k] / distances) ** power)
    
    return u


def adjust_membership(u: np.ndarray, temperature: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """
    调整隶属度向量: 先温度缩放，再幂次调整
    
    Args:
        u: (K,) 原始隶属度向量
        temperature: 温度参数 T, u = softmax(log(u+eps)/T)
                     T=1.0 时不做温度缩放; T<1 锐化, T>1 平滑
        alpha: 幂次参数, u = (u**alpha) / sum(u**alpha)
               alpha=1.0 时不做幂次调整; alpha>1 锐化, alpha<1 平滑
    
    Returns:
        调整后的隶属度向量
    """
    eps = 1e-10
    u_adjusted = u.copy()
    
    # 1. 温度缩放 (当 T != 1.0 时启用)
    if temperature != 1.0:
        log_u = np.log(u_adjusted + eps)
        scaled = log_u / temperature
        scaled = scaled - np.max(scaled)  # 数值稳定
        exp_scaled = np.exp(scaled)
        u_adjusted = exp_scaled / np.sum(exp_scaled)
    
    # 2. 幂次调整 (当 alpha != 1.0 时启用)
    if alpha != 1.0:
        u_pow = np.power(u_adjusted, alpha)
        u_adjusted = u_pow / np.sum(u_pow)
    
    return u_adjusted


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("B5 阶段 - Hard vs Soft 路由对比实验")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/8] 配置加载完成: {config_path}")
    
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    eval_config = config['evaluation']
    output_config = config['output']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    n_clusters = 3
    
    # 软路由配置 (从 config.yaml 读取)
    soft_routing_config = config.get('soft_routing', {})
    soft_temperature = soft_routing_config.get('temperature', 1.0)
    soft_alpha = soft_routing_config.get('alpha', 1.0)
    
    # 随机种子设置 (可复现性)
    seed = train_config.get('seed', 42)
    deterministic = train_config.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    print_seed_info(seed, deterministic)
    
    # 设备配置
    device_config = train_config.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    print(f"  - Device: {device}")
    if device == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - n_clusters: {n_clusters}")
    print(f"  - Soft routing: T={soft_temperature}, alpha={soft_alpha}")
    
    # =========================================================================
    # 2. 加载 Router 组件 + 专家模型 (从 B3/B4 产物)
    # =========================================================================
    print("\n[2/8] 加载 Router 组件 + 专家模型...")
    
    results_dir = PROJECT_ROOT / output_config['results_dir']
    
    # 加载 router_config.json (包含簇语义映射)
    with open(results_dir / "router_config.json", 'r', encoding='utf-8') as f:
        router_config_data = json.load(f)
    
    cluster_semantic_map = {int(k): v for k, v in router_config_data['cluster_semantic_map'].items()}
    cluster_stats = router_config_data.get('cluster_stats', {})
    
    print(f"\n  簇语义映射 (来自 B3 训练集):")
    print(f"  {'Cluster':<12} {'Semantic':>15} {'Power_pu_mean':>14} {'Power_delta':>12}")
    print(f"  {'-'*55}")
    for k in range(n_clusters):
        semantic = cluster_semantic_map.get(k, 'unknown')
        stats = cluster_stats.get(str(k), {})
        power_mean = stats.get('Power_pu_mean', 0)
        power_delta = stats.get('Power_pu_delta_mean_abs', 0)
        print(f"  Cluster {k:<3} {semantic:>15} {power_mean:>14.4f} {power_delta:>12.4f}")
    
    # 加载 Router 组件
    with open(results_dir / "scaler_g.pkl", 'rb') as f:
        scaler_g = pickle.load(f)
    print(f"  Loaded scaler_g.pkl")
    
    with open(results_dir / "pca.pkl", 'rb') as f:
        pca = pickle.load(f)
    print(f"  Loaded pca.pkl (n_components={pca.n_components_})")
    
    fcm_centers = np.load(results_dir / "fcm_centers.npy")
    print(f"  Loaded fcm_centers.npy (shape={fcm_centers.shape})")
    
    with open(results_dir / "scaler_X.pkl", 'rb') as f:
        scaler_X = pickle.load(f)
    print(f"  Loaded scaler_X.pkl")
    
    # 加载专家模型 (从 B4)
    input_size = 6
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    output_steps = window_config['max_horizon']
    
    experts = {}
    for k in range(n_clusters):
        model_path = results_dir / f"model_expert_{k}.pt"
        if not model_path.exists():
            print(f"  [Warning] model_expert_{k}.pt not found, skipping...")
            continue
        
        model = GlobalLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=output_steps,
            dropout=dropout
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        experts[k] = model
        print(f"  Loaded model_expert_{k}.pt")
    
    if len(experts) < n_clusters:
        print(f"  [Error] Not all experts loaded. Expected {n_clusters}, got {len(experts)}")
        return
    
    # =========================================================================
    # 3. 数据读取
    # =========================================================================
    print("\n[3/8] 读取数据...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    feature_names = list(load_result.df.select_dtypes(include=[np.number]).columns)
    feature_names = [c for c in feature_names if c != 'is_daylight']
    print(f"  Features: {len(feature_names)}")
    
    # =========================================================================
    # 4. 白天筛选 + 按天切分
    # =========================================================================
    print("\n[4/8] 白天筛选 + 按天切分...")
    
    df_for_split = load_result.df
    
    if daylight_enabled:
        daylight_result = add_daylight_flag(
            df=load_result.df,
            time_column=data_config['time_column'],
            dni_col=daylight_config['dni_col'],
            threshold=daylight_config['threshold']
        )
        
        if daylight_mode == 'drop':
            filtered_df, _, _ = filter_daylight_rows(daylight_result.df)
            df_for_split = filtered_df
        else:
            df_for_split = daylight_result.df
    
    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    print(f"  Train days: {split_result.train_days}, "
          f"Val days: {split_result.val_days}, "
          f"Test days: {split_result.test_days}")
    
    # =========================================================================
    # 5. 生成滑窗样本 (仅 test)
    # =========================================================================
    print("\n[5/8] 生成滑窗样本 (Test only)...")
    
    test_window_result = generate_windows(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=(daylight_mode == 'mask') if daylight_enabled else False
    )
    
    test_samples = test_window_result.samples
    print(f"  Test samples: {len(test_samples)}")
    
    # =========================================================================
    # 6. Rolling Evaluation: Hard vs Soft
    # =========================================================================
    print("\n[6/8] Rolling Evaluation (Hard vs Soft)...")
    
    horizons = eval_config['horizons']
    horizon_names = eval_config['horizon_names']
    
    # 存储预测结果
    hard_preds = []
    soft_preds = []
    targets = []
    cluster_labels = []
    memberships = []
    
    with torch.no_grad():
        for sample in test_samples:
            # 1. 计算状态向量 g_t
            g_t, _ = compute_state_vector(sample, feature_names, capacity)
            
            # 2. scaler_g -> PCA -> FCM membership
            g_scaled = scaler_g.transform(g_t.reshape(1, -1))[0]
            z_t = pca.transform(g_scaled.reshape(1, -1))[0]
            u_t = compute_fcm_membership(z_t, fcm_centers, m=2.0)
            
            # 3. 调整隶属度 (先温度缩放，再幂次调整)
            u_adjusted = adjust_membership(u_t, 
                                           temperature=soft_temperature, 
                                           alpha=soft_alpha)
            
            # 4. 硬标签
            k_t = np.argmax(u_t)
            
            # 5. 准备输入
            X_raw = sample.X  # (Lx, n_features)
            X_scaled = scaler_X.transform(X_raw)
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(device)  # (1, Lx, n_features)
            
            # 6. Hard 预测
            hard_pred_pu = experts[k_t](X_tensor).cpu().numpy().flatten()  # (Hmax,)
            hard_pred_mw = hard_pred_pu * capacity
            
            # 7. Soft 预测 (加权融合)
            soft_pred_pu = np.zeros(output_steps)
            for k in range(n_clusters):
                expert_pred_pu = experts[k](X_tensor).cpu().numpy().flatten()
                soft_pred_pu += u_adjusted[k] * expert_pred_pu
            soft_pred_mw = soft_pred_pu * capacity
            
            # 8. 真实值
            y_true_mw = sample.Y  # (Hmax,) in MW
            
            # 存储
            hard_preds.append(hard_pred_mw)
            soft_preds.append(soft_pred_mw)
            targets.append(y_true_mw)
            cluster_labels.append(k_t)
            memberships.append(u_t)
    
    hard_preds = np.array(hard_preds)  # (N, Hmax)
    soft_preds = np.array(soft_preds)  # (N, Hmax)
    targets = np.array(targets)  # (N, Hmax)
    cluster_labels = np.array(cluster_labels)  # (N,)
    memberships = np.array(memberships)  # (N, K)
    
    print(f"  Completed: {len(targets)} samples")
    
    # =========================================================================
    # 7. 计算误差指标
    # =========================================================================
    print("\n[7/8] 计算误差指标...")
    
    def compute_metrics(preds, targets, horizons, horizon_names, capacity):
        """计算 MAE, RMSE, nRMSE"""
        results = {}
        for h in horizons:
            name = horizon_names.get(h, f"{h}step") if isinstance(horizon_names, dict) else horizon_names[horizons.index(h)]
            pred_h = preds[:, :h]
            target_h = targets[:, :h]
            
            errors = pred_h - target_h
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            nrmse = rmse / capacity
            
            results[name] = {'mae': mae, 'rmse': rmse, 'nrmse': nrmse}
        return results
    
    # 整体指标
    hard_metrics = compute_metrics(hard_preds, targets, horizons, horizon_names, capacity)
    soft_metrics = compute_metrics(soft_preds, targets, horizons, horizon_names, capacity)
    
    # 按簇指标
    hard_cluster_metrics = {}
    soft_cluster_metrics = {}
    
    for k in range(n_clusters):
        mask = cluster_labels == k
        n_samples = np.sum(mask)
        
        if n_samples > 0:
            hard_cluster_metrics[k] = {
                'n_samples': int(n_samples),
                'semantic': cluster_semantic_map.get(k, 'unknown'),
                'avg_membership': float(np.mean(memberships[mask, k])),
                **compute_metrics(hard_preds[mask], targets[mask], horizons, horizon_names, capacity)
            }
            soft_cluster_metrics[k] = {
                'n_samples': int(n_samples),
                'semantic': cluster_semantic_map.get(k, 'unknown'),
                'avg_membership': float(np.mean(memberships[mask, k])),
                **compute_metrics(soft_preds[mask], targets[mask], horizons, horizon_names, capacity)
            }
    
    # =========================================================================
    # 8. 打印汇总表 + 保存结果
    # =========================================================================
    print("\n[8/8] 汇总表 + 保存产物...")
    
    print("\n" + "=" * 100)
    print("B5 汇总表 - Hard vs Soft 路由对比")
    print("=" * 100)
    
    # 整体指标表
    print("\n[整体误差]")
    print(f"{'Horizon':<10} {'Hard MAE':>12} {'Soft MAE':>12} {'Hard RMSE':>12} {'Soft RMSE':>12} "
          f"{'Hard nRMSE':>12} {'Soft nRMSE':>12} {'Delta nRMSE':>12}")
    print("-" * 100)
    
    overall_comparison = {}
    for h in horizons:
        name = horizon_names.get(h, f"{h}step") if isinstance(horizon_names, dict) else str(h)
        h_hard = hard_metrics[name]
        h_soft = soft_metrics[name]
        delta_nrmse = h_soft['nrmse'] - h_hard['nrmse']
        
        print(f"{name:<10} {h_hard['mae']:>12.4f} {h_soft['mae']:>12.4f} "
              f"{h_hard['rmse']:>12.4f} {h_soft['rmse']:>12.4f} "
              f"{h_hard['nrmse']:>12.4f} {h_soft['nrmse']:>12.4f} {delta_nrmse:>+12.4f}")
        
        overall_comparison[name] = {
            'hard': h_hard,
            'soft': h_soft,
            'delta_nrmse': delta_nrmse
        }
    
    # 按簇指标表
    print("\n[按簇误差 - Hard]")
    print(f"{'Cluster':<10} {'Semantic':>15} {'N':>8} {'1h nRMSE':>12} {'2h nRMSE':>12} {'4h nRMSE':>12} {'Avg Memb':>10}")
    print("-" * 85)
    
    for k in range(n_clusters):
        if k in hard_cluster_metrics:
            m = hard_cluster_metrics[k]
            print(f"Cluster {k:<2} {m['semantic']:>15} {m['n_samples']:>8} "
                  f"{m['1h']['nrmse']:>12.4f} {m['2h']['nrmse']:>12.4f} {m['4h']['nrmse']:>12.4f} "
                  f"{m['avg_membership']:>10.4f}")
    
    print("\n[按簇误差 - Soft]")
    print(f"{'Cluster':<10} {'Semantic':>15} {'N':>8} {'1h nRMSE':>12} {'2h nRMSE':>12} {'4h nRMSE':>12} {'Avg Memb':>10}")
    print("-" * 85)
    
    for k in range(n_clusters):
        if k in soft_cluster_metrics:
            m = soft_cluster_metrics[k]
            print(f"Cluster {k:<2} {m['semantic']:>15} {m['n_samples']:>8} "
                  f"{m['1h']['nrmse']:>12.4f} {m['2h']['nrmse']:>12.4f} {m['4h']['nrmse']:>12.4f} "
                  f"{m['avg_membership']:>10.4f}")
    
    # Delta 表 (Soft - Hard)
    print("\n[按簇 Delta nRMSE (Soft - Hard)]")
    print(f"{'Cluster':<10} {'Semantic':>15} {'1h Delta':>12} {'2h Delta':>12} {'4h Delta':>12}")
    print("-" * 65)
    
    cluster_deltas = {}
    for k in range(n_clusters):
        if k in hard_cluster_metrics and k in soft_cluster_metrics:
            h_m = hard_cluster_metrics[k]
            s_m = soft_cluster_metrics[k]
            d1h = s_m['1h']['nrmse'] - h_m['1h']['nrmse']
            d2h = s_m['2h']['nrmse'] - h_m['2h']['nrmse']
            d4h = s_m['4h']['nrmse'] - h_m['4h']['nrmse']
            
            print(f"Cluster {k:<2} {h_m['semantic']:>15} {d1h:>+12.4f} {d2h:>+12.4f} {d4h:>+12.4f}")
            
            cluster_deltas[k] = {'1h': d1h, '2h': d2h, '4h': d4h}
    
    print("=" * 100)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    metrics_data = {
        'timestamp': timestamp,
        'config': {
            'temperature_T': soft_temperature,
            'alpha': soft_alpha,
            'n_clusters': n_clusters,
            'n_test_samples': len(targets)
        },
        'cluster_semantic_map': cluster_semantic_map,
        'overall': overall_comparison,
        'hard_by_cluster': {str(k): v for k, v in hard_cluster_metrics.items()},
        'soft_by_cluster': {str(k): v for k, v in soft_cluster_metrics.items()},
        'delta_by_cluster': {str(k): v for k, v in cluster_deltas.items()}
    }
    
    json_path = results_dir / "metrics_B5.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: metrics_B5.json")
    
    # CSV
    csv_path = results_dir / "metrics_B5.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Type,Cluster,Semantic,N,1h_MAE,1h_RMSE,1h_nRMSE,2h_MAE,2h_RMSE,2h_nRMSE,4h_MAE,4h_RMSE,4h_nRMSE,Avg_Membership\n")
        
        # Overall Hard
        f.write(f"Hard,Overall,,-,{hard_metrics['1h']['mae']:.4f},{hard_metrics['1h']['rmse']:.4f},{hard_metrics['1h']['nrmse']:.4f},"
                f"{hard_metrics['2h']['mae']:.4f},{hard_metrics['2h']['rmse']:.4f},{hard_metrics['2h']['nrmse']:.4f},"
                f"{hard_metrics['4h']['mae']:.4f},{hard_metrics['4h']['rmse']:.4f},{hard_metrics['4h']['nrmse']:.4f},-\n")
        
        # Overall Soft
        f.write(f"Soft,Overall,,-,{soft_metrics['1h']['mae']:.4f},{soft_metrics['1h']['rmse']:.4f},{soft_metrics['1h']['nrmse']:.4f},"
                f"{soft_metrics['2h']['mae']:.4f},{soft_metrics['2h']['rmse']:.4f},{soft_metrics['2h']['nrmse']:.4f},"
                f"{soft_metrics['4h']['mae']:.4f},{soft_metrics['4h']['rmse']:.4f},{soft_metrics['4h']['nrmse']:.4f},-\n")
        
        # By cluster
        for k in range(n_clusters):
            if k in hard_cluster_metrics:
                m = hard_cluster_metrics[k]
                f.write(f"Hard,{k},{m['semantic']},{m['n_samples']},"
                        f"{m['1h']['mae']:.4f},{m['1h']['rmse']:.4f},{m['1h']['nrmse']:.4f},"
                        f"{m['2h']['mae']:.4f},{m['2h']['rmse']:.4f},{m['2h']['nrmse']:.4f},"
                        f"{m['4h']['mae']:.4f},{m['4h']['rmse']:.4f},{m['4h']['nrmse']:.4f},{m['avg_membership']:.4f}\n")
            if k in soft_cluster_metrics:
                m = soft_cluster_metrics[k]
                f.write(f"Soft,{k},{m['semantic']},{m['n_samples']},"
                        f"{m['1h']['mae']:.4f},{m['1h']['rmse']:.4f},{m['1h']['nrmse']:.4f},"
                        f"{m['2h']['mae']:.4f},{m['2h']['rmse']:.4f},{m['2h']['nrmse']:.4f},"
                        f"{m['4h']['mae']:.4f},{m['4h']['rmse']:.4f},{m['4h']['nrmse']:.4f},{m['avg_membership']:.4f}\n")
    
    print(f"  Saved: metrics_B5.csv")
    
    # 运行日志
    log_path = PROJECT_ROOT / output_config['logs_dir'] / f"train_log_b5_{timestamp}.json"
    log_data = {
        'timestamp': timestamp,
        'stage': 'B5',
        'temperature_T': soft_temperature,
        'alpha': soft_alpha,
        'n_test_samples': len(targets),
        'hard_4h_nrmse': hard_metrics['4h']['nrmse'],
        'soft_4h_nrmse': soft_metrics['4h']['nrmse'],
        'delta_4h_nrmse': soft_metrics['4h']['nrmse'] - hard_metrics['4h']['nrmse']
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {log_path.name}")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("B5 阶段完成!")
    print(f"  - Soft routing: T={soft_temperature}, alpha={soft_alpha}")
    print(f"  - Test 样本数: {len(targets)}")
    print(f"  - Hard 4h nRMSE: {hard_metrics['4h']['nrmse']:.4f} ({hard_metrics['4h']['nrmse']*100:.2f}%)")
    print(f"  - Soft 4h nRMSE: {soft_metrics['4h']['nrmse']:.4f} ({soft_metrics['4h']['nrmse']*100:.2f}%)")
    delta = soft_metrics['4h']['nrmse'] - hard_metrics['4h']['nrmse']
    delta_sign = "+" if delta >= 0 else ""
    print(f"  - Delta (Soft - Hard): {delta_sign}{delta:.4f} ({delta_sign}{delta*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
