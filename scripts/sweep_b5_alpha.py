# -*- coding: utf-8 -*-
"""
B5 Alpha Sweep 脚本
遍历不同 alpha 值，对比 Soft 融合效果

在保持同一套 router + 同一组 B4 专家权重 + 同一 test 滑窗样本不变的前提下，
依次遍历 alpha_list，对每个 alpha 在推理阶段将隶属度做幂次调整后进行 Soft 融合预测。
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
from typing import List, Dict, Tuple

from src.data.loader import load_solar_data
from src.data.splitter import split_by_day
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows
from src.models.lstm import GlobalLSTM
from src.utils.seed import set_seed, print_seed_info


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# 状态向量 g_t 构建 (与 B3/B4/B5 完全一致)
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """计算单个样本的状态向量 g_t (与 B3 完全一致)"""
    X = sample.X
    
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
    """计算 FCM 隶属度向量 (与 B4 一致，使用平方距离)"""
    K = centers.shape[0]
    
    distances = np.zeros(K)
    for k in range(K):
        diff = z - centers[k]
        distances[k] = np.sum(diff ** 2)
    
    distances = np.maximum(distances, 1e-10)
    
    power = 2.0 / (m - 1)
    u = np.zeros(K)
    
    for k in range(K):
        u[k] = 1.0 / np.sum((distances[k] / distances) ** power)
    
    return u


def apply_temperature_scaling(u: np.ndarray, temperature: float) -> np.ndarray:
    """温度缩放: u = softmax(log(u+eps)/T)"""
    if temperature == 1.0:
        return u
    eps = 1e-10
    log_u = np.log(u + eps)
    scaled = log_u / temperature
    scaled = scaled - np.max(scaled)  # 数值稳定
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)


def adjust_membership_power(u: np.ndarray, alpha: float) -> np.ndarray:
    """幂次调整隶属度: u_alpha = (u**alpha) / sum(u**alpha)"""
    if alpha == 1.0:
        return u
    u_pow = np.power(u, alpha)
    return u_pow / np.sum(u_pow)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, 
                    horizons: List[int], capacity: float) -> Dict[str, Dict[str, float]]:
    """计算 MAE, RMSE, nRMSE"""
    horizon_names = {4: '1h', 8: '2h', 16: '4h'}
    results = {}
    
    for h in horizons:
        name = horizon_names.get(h, f"{h}step")
        pred_h = preds[:, :h]
        target_h = targets[:, :h]
        
        errors = pred_h - target_h
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        nrmse = rmse / capacity
        
        results[name] = {'mae': mae, 'rmse': rmse, 'nrmse': nrmse}
    
    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("B5 Alpha Sweep - 幂次调整参数扫描")
    print("=" * 80)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/6] 配置加载完成")
    
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
    horizons = eval_config['horizons']
    
    # 软路由配置 (从 config.yaml 读取)
    soft_routing_config = config.get('soft_routing', {})
    soft_temperature = soft_routing_config.get('temperature', 1.0)
    alpha_list = soft_routing_config.get('alpha_list', [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0])
    
    # 专家模型配置
    use_warm_experts = soft_routing_config.get('use_warm_experts', False)
    experts_suffix = soft_routing_config.get('experts_suffix', '')
    if use_warm_experts and not experts_suffix:
        experts_suffix = '_warm'
    
    # 随机种子设置
    seed = train_config.get('seed', 42)
    deterministic = train_config.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    
    # 设备配置
    device_config = train_config.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    experts_type = "Warm Experts" if use_warm_experts else "B4 Experts"
    print(f"  - Device: {device}")
    print(f"  - Experts: {experts_type} (suffix='{experts_suffix}')")
    print(f"  - Temperature T: {soft_temperature}")
    print(f"  - Alpha list: {alpha_list}")
    
    # =========================================================================
    # 2. 加载 Router 组件 + 专家模型
    # =========================================================================
    print("\n[2/6] 加载 Router 组件 + 专家模型...")
    
    results_dir = PROJECT_ROOT / output_config['results_dir']
    
    # 加载 router_config.json
    with open(results_dir / "router_config.json", 'r', encoding='utf-8') as f:
        router_config_data = json.load(f)
    
    cluster_semantic_map = {int(k): v for k, v in router_config_data['cluster_semantic_map'].items()}
    
    # 加载 Router 组件
    with open(results_dir / "scaler_g.pkl", 'rb') as f:
        scaler_g = pickle.load(f)
    
    with open(results_dir / "pca.pkl", 'rb') as f:
        pca = pickle.load(f)
    
    fcm_centers = np.load(results_dir / "fcm_centers.npy")
    
    with open(results_dir / "scaler_X.pkl", 'rb') as f:
        scaler_X = pickle.load(f)
    
    print(f"  Loaded: scaler_g, pca, fcm_centers, scaler_X")
    
    # 加载专家模型
    input_size = 6
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    output_steps = window_config['max_horizon']
    
    experts = {}
    expert_model_names = []
    for k in range(n_clusters):
        model_name = f"model_expert_{k}{experts_suffix}.pt"
        model_path = results_dir / model_name
        model = GlobalLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=output_steps,
            dropout=dropout
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        experts[k] = model
        expert_model_names.append(model_name)
    
    print(f"  Loaded: {len(experts)} expert models ({experts_type})")
    for name in expert_model_names:
        print(f"    - {name}")
    
    # =========================================================================
    # 3. 数据读取 + 生成 Test 滑窗样本
    # =========================================================================
    print("\n[3/6] 数据读取 + 生成 Test 滑窗样本...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    feature_names = list(load_result.df.select_dtypes(include=[np.number]).columns)
    feature_names = [c for c in feature_names if c != 'is_daylight']
    
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
    # 4. 预计算: 状态向量 + 隶属度 + 各专家预测
    # =========================================================================
    print("\n[4/6] 预计算状态向量 + 隶属度 + 各专家预测...")
    
    all_memberships = []
    all_expert_preds = {k: [] for k in range(n_clusters)}
    all_targets = []
    all_cluster_labels = []
    
    with torch.no_grad():
        for sample in test_samples:
            # 计算状态向量
            g_t, _ = compute_state_vector(sample, feature_names, capacity)
            
            # scaler_g -> PCA -> FCM membership
            g_scaled = scaler_g.transform(g_t.reshape(1, -1))[0]
            z_t = pca.transform(g_scaled.reshape(1, -1))[0]
            u_t = compute_fcm_membership(z_t, fcm_centers, m=2.0)
            
            # 硬标签
            k_t = np.argmax(u_t)
            
            # 准备输入
            X_raw = sample.X
            X_scaled = scaler_X.transform(X_raw)
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(device)
            
            # 各专家预测
            for k in range(n_clusters):
                expert_pred_pu = experts[k](X_tensor).cpu().numpy().flatten()
                all_expert_preds[k].append(expert_pred_pu * capacity)
            
            # 真实值
            y_true_mw = sample.Y
            
            all_memberships.append(u_t)
            all_targets.append(y_true_mw)
            all_cluster_labels.append(k_t)
    
    all_memberships = np.array(all_memberships)  # (N, K)
    all_targets = np.array(all_targets)  # (N, Hmax)
    all_cluster_labels = np.array(all_cluster_labels)  # (N,)
    for k in range(n_clusters):
        all_expert_preds[k] = np.array(all_expert_preds[k])  # (N, Hmax)
    
    print(f"  Precomputed: {len(all_targets)} samples x {n_clusters} experts")
    
    # =========================================================================
    # 5. Hard 基线计算
    # =========================================================================
    print("\n[5/6] 计算 Hard 基线 + Alpha Sweep...")
    
    # Hard 预测
    hard_preds = np.zeros_like(all_targets)
    for i in range(len(all_targets)):
        k_t = all_cluster_labels[i]
        hard_preds[i] = all_expert_preds[k_t][i]
    
    hard_overall = compute_metrics(hard_preds, all_targets, horizons, capacity)
    
    # Hard 按簇指标
    hard_by_cluster = {}
    for k in range(n_clusters):
        mask = all_cluster_labels == k
        if np.sum(mask) > 0:
            hard_by_cluster[k] = {
                'n_samples': int(np.sum(mask)),
                'semantic': cluster_semantic_map.get(k, 'unknown'),
                'avg_membership': float(np.mean(all_memberships[mask, k])),
                **compute_metrics(hard_preds[mask], all_targets[mask], horizons, capacity)
            }
    
    # =========================================================================
    # 6. Alpha Sweep
    # =========================================================================
    sweep_results = {}
    
    for alpha in alpha_list:
        # Soft 预测
        soft_preds = np.zeros_like(all_targets)
        
        for i in range(len(all_targets)):
            # 1. 先温度缩放 (当 T != 1.0 时启用)
            u_temp = apply_temperature_scaling(all_memberships[i], soft_temperature)
            # 2. 再幂次调整
            u_adjusted = adjust_membership_power(u_temp, alpha)
            for k in range(n_clusters):
                soft_preds[i] += u_adjusted[k] * all_expert_preds[k][i]
        
        # Overall 指标
        soft_overall = compute_metrics(soft_preds, all_targets, horizons, capacity)
        
        # 按簇指标
        soft_by_cluster = {}
        for k in range(n_clusters):
            mask = all_cluster_labels == k
            if np.sum(mask) > 0:
                soft_by_cluster[k] = {
                    'n_samples': int(np.sum(mask)),
                    'semantic': cluster_semantic_map.get(k, 'unknown'),
                    'avg_membership': float(np.mean(all_memberships[mask, k])),
                    **compute_metrics(soft_preds[mask], all_targets[mask], horizons, capacity)
                }
        
        # Delta (Soft - Hard)
        delta_overall = {}
        for h_name in ['1h', '2h', '4h']:
            delta_overall[h_name] = {
                'delta_nrmse': soft_overall[h_name]['nrmse'] - hard_overall[h_name]['nrmse']
            }
        
        delta_by_cluster = {}
        for k in range(n_clusters):
            if k in hard_by_cluster and k in soft_by_cluster:
                delta_by_cluster[k] = {}
                for h_name in ['1h', '2h', '4h']:
                    delta_by_cluster[k][h_name] = {
                        'delta_nrmse': soft_by_cluster[k][h_name]['nrmse'] - hard_by_cluster[k][h_name]['nrmse']
                    }
        
        sweep_results[alpha] = {
            'soft_overall': soft_overall,
            'soft_by_cluster': {str(k): v for k, v in soft_by_cluster.items()},
            'delta_overall': delta_overall,
            'delta_by_cluster': {str(k): v for k, v in delta_by_cluster.items()}
        }
    
    print(f"  Completed: {len(alpha_list)} alpha values")
    
    # =========================================================================
    # 7. 保存结果
    # =========================================================================
    print("\n[6/6] 保存结果 + 打印汇总表...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建完整结果
    full_results = {
        'timestamp': timestamp,
        'use_warm_experts': use_warm_experts,
        'experts_suffix': experts_suffix,
        'expert_models': expert_model_names,
        'temperature_T': soft_temperature,
        'n_test_samples': len(all_targets),
        'alpha_list': alpha_list,
        'hard_baseline': {
            'overall': hard_overall,
            'by_cluster': {str(k): v for k, v in hard_by_cluster.items()}
        },
        'sweep_results': {str(a): v for a, v in sweep_results.items()}
    }
    
    # JSON
    json_path = results_dir / "alpha_sweep_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path.name}")
    
    # CSV
    csv_path = results_dir / "alpha_sweep_results.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Alpha,Type,Cluster,Semantic,N,1h_nRMSE,2h_nRMSE,4h_nRMSE,1h_Delta,2h_Delta,4h_Delta\n")
        
        # Hard baseline
        f.write(f"Hard,Overall,,,{len(all_targets)},"
                f"{hard_overall['1h']['nrmse']:.6f},{hard_overall['2h']['nrmse']:.6f},{hard_overall['4h']['nrmse']:.6f},"
                f"0,0,0\n")
        for k in range(n_clusters):
            if k in hard_by_cluster:
                m = hard_by_cluster[k]
                f.write(f"Hard,Cluster,{k},{m['semantic']},{m['n_samples']},"
                        f"{m['1h']['nrmse']:.6f},{m['2h']['nrmse']:.6f},{m['4h']['nrmse']:.6f},"
                        f"0,0,0\n")
        
        # Alpha sweep
        for alpha in alpha_list:
            res = sweep_results[alpha]
            so = res['soft_overall']
            do = res['delta_overall']
            f.write(f"{alpha},Overall,,,{len(all_targets)},"
                    f"{so['1h']['nrmse']:.6f},{so['2h']['nrmse']:.6f},{so['4h']['nrmse']:.6f},"
                    f"{do['1h']['delta_nrmse']:.6f},{do['2h']['delta_nrmse']:.6f},{do['4h']['delta_nrmse']:.6f}\n")
            
            for k in range(n_clusters):
                if str(k) in res['soft_by_cluster']:
                    sc = res['soft_by_cluster'][str(k)]
                    dc = res['delta_by_cluster'][str(k)]
                    f.write(f"{alpha},Cluster,{k},{sc['semantic']},{sc['n_samples']},"
                            f"{sc['1h']['nrmse']:.6f},{sc['2h']['nrmse']:.6f},{sc['4h']['nrmse']:.6f},"
                            f"{dc['1h']['delta_nrmse']:.6f},{dc['2h']['delta_nrmse']:.6f},{dc['4h']['delta_nrmse']:.6f}\n")
    
    print(f"  Saved: {csv_path.name}")
    
    # =========================================================================
    # 打印对比表
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"Alpha Sweep 汇总表 - {experts_type}")
    print("=" * 100)
    
    # Overall 对比表
    print("\n[Overall]")
    print(f"{'Alpha':<10} {'1h nRMSE':>12} {'2h nRMSE':>12} {'4h nRMSE':>12} "
          f"{'1h Delta':>12} {'2h Delta':>12} {'4h Delta':>12}")
    print("-" * 82)
    
    # Hard baseline
    print(f"{'Hard':<10} {hard_overall['1h']['nrmse']:>12.4f} {hard_overall['2h']['nrmse']:>12.4f} "
          f"{hard_overall['4h']['nrmse']:>12.4f} {'-':>12} {'-':>12} {'-':>12}")
    
    # Find best alpha
    best_alpha = None
    best_4h_nrmse = float('inf')
    
    for alpha in sorted(alpha_list):
        res = sweep_results[alpha]
        so = res['soft_overall']
        do = res['delta_overall']
        
        if so['4h']['nrmse'] < best_4h_nrmse:
            best_4h_nrmse = so['4h']['nrmse']
            best_alpha = alpha
        
        marker = " *" if alpha == best_alpha else ""
        print(f"{alpha:<10} {so['1h']['nrmse']:>12.4f} {so['2h']['nrmse']:>12.4f} "
              f"{so['4h']['nrmse']:>12.4f} {do['1h']['delta_nrmse']:>+12.4f} "
              f"{do['2h']['delta_nrmse']:>+12.4f} {do['4h']['delta_nrmse']:>+12.4f}{marker}")
    
    print("-" * 82)
    print(f"* Best alpha = {best_alpha} (4h nRMSE = {best_4h_nrmse:.4f})")
    
    # 按簇对比表 (仅显示 4h)
    print("\n[按簇 4h nRMSE]")
    header = f"{'Alpha':<10}"
    for k in range(n_clusters):
        semantic = cluster_semantic_map.get(k, f'C{k}')
        header += f" {semantic:>15}"
    print(header)
    print("-" * (10 + 16 * n_clusters))
    
    # Hard baseline
    row = f"{'Hard':<10}"
    for k in range(n_clusters):
        if k in hard_by_cluster:
            row += f" {hard_by_cluster[k]['4h']['nrmse']:>15.4f}"
    print(row)
    
    for alpha in sorted(alpha_list):
        res = sweep_results[alpha]
        row = f"{alpha:<10}"
        for k in range(n_clusters):
            if str(k) in res['soft_by_cluster']:
                row += f" {res['soft_by_cluster'][str(k)]['4h']['nrmse']:>15.4f}"
        print(row)
    
    print("=" * 100)
    
    # 最终总结
    print("\n" + "=" * 70)
    print(f"Alpha Sweep 完成! ({experts_type})")
    print(f"  - Experts: {experts_type} (suffix='{experts_suffix}')")
    print(f"  - Temperature T: {soft_temperature}")
    print(f"  - 测试样本数: {len(all_targets)}")
    print(f"  - Alpha 范围: {min(alpha_list)} ~ {max(alpha_list)}")
    print(f"  - Hard 4h nRMSE: {hard_overall['4h']['nrmse']:.4f} ({hard_overall['4h']['nrmse']*100:.2f}%)")
    print(f"  - Best alpha: {best_alpha}")
    print(f"  - Best 4h nRMSE: {best_4h_nrmse:.4f} ({best_4h_nrmse*100:.2f}%)")
    delta = best_4h_nrmse - hard_overall['4h']['nrmse']
    print(f"  - Improvement: {delta:+.4f} ({delta*100:+.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
